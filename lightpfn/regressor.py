"""Quantile regression with cached context and range-checked mixed precision.

Two members share one encode: y and log(y) only for positive targets with sample
skewness >= log_skew, otherwise y and -y. The quantile curves are averaged.
Decoder moments use float32 and target units float64; CPU and Vulkan inference use float32.
On Vulkan, category statistics, target normalization and quantile head readout run on the host;
cell embeddings, categorical-statistics projections, column/row stages, ICL and decoder attention run on the GPU.
"""

import copy
import dataclasses
import logging
import os
import warnings
from contextlib import nullcontext
from numbers import Integral

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from lightpfn.device import kind, resolve_device
from lightpfn.model.layers import fuse_enabled
from lightpfn.model.regression import TAUS
from lightpfn.precision import shared_fp16_view
from lightpfn.sklearn import (
    CPU_CHUNK_CELLS_PER_THREAD,
    FP16_MIN_CELLS,
    GPU_CHUNK_CELLS,
    _context_isfinite,
    _encode_frame,
    _fast_fp16,
    _fit_categories,
    _is_frame,
    auto_categorical,
    resolve_device_was_auto,
)


def _fp32_cache(obj):
    """Promote autocast caches for retries, preserving float64 target mean/scale exactly."""
    if isinstance(obj, torch.Tensor):
        return obj.float() if obj.dtype in (torch.float16, torch.bfloat16) else obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.replace(obj, **{f.name: _fp32_cache(getattr(obj, f.name))
                                           for f in dataclasses.fields(obj) if f.init})
    if isinstance(obj, dict):
        return {k: _fp32_cache(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_fp32_cache(v) for v in obj)
    return obj


def load_regression_model(path, device="cpu"):
    """The EMA weights of a training checkpoint, or a weights file, with the quantiles head; None otherwise
    (a v2 checkpoint, or a model without the head)."""
    from lightpfn.checkpoint import load_model

    ck = torch.load(path, map_location="cpu", weights_only=True) if str(path).endswith(".pt") else None
    if ck is not None and "unique_blocks" in ck.get("config", {}):
        return None
    model = load_model(path, device)
    return model if model.reg is not None else None


class V15Regressor:
    def __init__(self, model, device="cpu", max_context=50000, seed=0, query_rows=4096, chunk_cells="auto",
                 n_estimators=2, precision="auto", log_skew=2.0, cache_context=True):
        if getattr(model, "reg", None) is None or not isinstance(max_context, int) or max_context < 1:
            raise ValueError("a model with the quantiles head and a positive max_context are required")
        if not isinstance(query_rows, int) or query_rows < 1:
            raise ValueError("query_rows must be a positive integer")
        if not (chunk_cells is None or isinstance(chunk_cells, str) and chunk_cells == "auto"
                or isinstance(chunk_cells, int) and not isinstance(chunk_cells, bool) and chunk_cells > 0):
            raise ValueError('chunk_cells must be "auto", a positive integer or None')
        if isinstance(n_estimators, bool) or not isinstance(n_estimators, Integral) or n_estimators not in (1, 2):
            raise ValueError("n_estimators must be 1 or 2")
        if log_skew is not None and (isinstance(log_skew, bool) or not isinstance(log_skew, (int, float))
                                     or np.isnan(log_skew)):
            raise ValueError("log_skew must be a number or None")
        if not isinstance(cache_context, bool):
            raise ValueError("cache_context must be a boolean")
        self.cache_context = cache_context
        self.log_skew = log_skew
        self.n_estimators = n_estimators
        self.precision = precision
        self.model = model
        self.device = torch.device(device) if isinstance(device, torch.device) or (
            isinstance(device, str) and kind(device.strip().lower()) in ("cpu", "cuda", "mps")) else device
        self.max_context, self.seed, self.query_rows, self.chunk_cells = max_context, seed, query_rows, chunk_cells

    def _tensor_device(self):
        dev = getattr(self, "fit_device_", self.device)
        return torch.device("cpu" if dev is None or kind(str(dev).strip().lower()) in ("vulkan", "auto") else dev)

    def _autocast(self):
        if self.precision_ == "fp16":
            return torch.autocast("cuda", dtype=torch.float16)
        # Preserve the old CPU wrapper's disabled-autocast semantics even inside a caller's AMP scope.
        tdev = self._tensor_device().type
        return (torch.autocast(tdev, enabled=False)
                if torch.is_autocast_enabled(tdev) else nullcontext())

    def _inference_net(self):
        if kind(getattr(self, "fit_device_", self.device)) == "vulkan":
            return self._vk_net
        if self.precision_ != "fp16" or not fuse_enabled("halfw"):
            return self.model
        if not hasattr(self, "_fp16_net"):
            self._fp16_net = shared_fp16_view(self.model)
        return self._fp16_net

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_fp16_net", None)
        state.pop("_vk_net", None)
        state.pop("_vk_key", None)
        return state

    def _precision_fallback(self, block):
        self.n_precision_fallbacks_ += 1
        logging.getLogger("lightpfn").debug("Non-finite fp16 %s; retrying in fp32", block)

    def _chunk_cells(self):
        """Same frontend budget as LightPFNClassifier; None disables chunking."""
        if self.chunk_cells != "auto":
            return self.chunk_cells
        dev = getattr(self, "fit_device_", None) or self.device
        return CPU_CHUNK_CELLS_PER_THREAD * torch.get_num_threads() if kind(str(dev).strip().lower()) == "cpu" else GPU_CHUNK_CELLS

    def _x(self, X):
        if _is_frame(X):
            if self._frame_columns is not None and not self._frame_columns.equals(X.columns):
                raise ValueError("The feature names must match those passed during fit, in the same order.")
            X = _encode_frame(X, self.categories_).to_numpy(dtype=np.float32, copy=True)
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != len(self.cat_) or not len(X) or np.isinf(X).any():
            raise ValueError("invalid regression X")
        return torch.as_tensor(X, device=self._tensor_device())[None].expand(self._members, -1, -1).contiguous()

    def _use_cpu(self, msg):
        if not resolve_device_was_auto(self.device):
            raise MemoryError(msg)
        warnings.warn(msg + ": running this fit on the CPU", RuntimeWarning, stacklevel=2)
        self.fit_device_ = "cpu"
        self.model = self.model.cpu()

    @torch.no_grad()
    def fit(self, X, y, cat=None):
        """cat: None, "auto" or (m,) bool, the categorical columns used by the adapter.
        DataFrame categories are encoded with the fit-time levels; unseen levels become NaN."""
        if self.precision not in ("auto", "fp32", "fp16"):
            raise ValueError('precision must be "auto", "fp32" or "fp16".')
        self.device_ = self.fit_device_ = resolve_device(self.device)
        if self.precision == "fp16" and kind(self.device_) != "cuda":
            raise ValueError('precision="fp16" requires a CUDA/ROCm fit device.')
        if isinstance(cat, str):
            if cat != "auto":
                raise ValueError('cat must be None, "auto" or a boolean mask')
            cat = auto_categorical(X) if self.model.cfg.cat_adapter else None
        self.categories_ = {}
        self._frame_columns = X.columns.copy() if _is_frame(X) else None
        if _is_frame(X):
            self.categories_ = _fit_categories(X)
            X = _encode_frame(X, self.categories_).to_numpy(dtype=np.float32, copy=True)
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] < 1 or y.shape != (len(X),) or not len(y) or not np.isfinite(y).all() or np.isinf(X).any():
            raise ValueError("finite regression targets must match X")
        self.cat_ = np.zeros(X.shape[1], bool) if cat is None else np.asarray(cat, bool)
        if self.cat_.shape != (X.shape[1],):
            raise ValueError("invalid categorical mask")
        if len(y) > self.max_context:
            keep = np.sort(np.random.default_rng(self.seed).choice(len(y), self.max_context, replace=False))
            X, y = X[keep], y[keep]
        self.n_precision_fallbacks_ = 0
        self.__dict__.pop("_fp16_net", None)
        self.__dict__.pop("ctx_", None)
        self._uncached_data = None
        self.model.eval()
        # Adjusted Fisher-Pearson sample skewness, as pandas.Series.skew().
        centered = y - y.mean()
        m2 = np.mean(centered ** 2)
        skew = (np.sqrt(len(y) * (len(y) - 1)) / (len(y) - 2) * np.mean(centered ** 3) / m2**1.5
                if len(y) >= 3 and m2 > 0 else 0.0)
        self.skew_ = float(skew)
        use_log = self.log_skew is not None and y.min() > 0 and skew >= self.log_skew
        self.second_ = None if self.n_estimators == 1 else ("log" if use_log else "flip")
        self._members = 1 if self.second_ is None else 2
        if kind(self.device_) == "vulkan":
            from lightpfn.vulkan import VulkanLightPFN

            index = self.device_.partition(":")[2]
            if getattr(self, "_vk_key", None) != (id(self.model), self.device_):
                self._vk_net = VulkanLightPFN(self.model, adapter=int(index) if index else None)
                self._vk_key = (id(self.model), self.device_)
            if self._vk_net.train_capacity(len(y), X.shape[1]) < self._members:
                self._use_cpu(f"{self._members} x {len(y)} x {X.shape[1]} training cells exceed a Vulkan "
                              "cache/scratch buffer limit; lower max_context or use device='cpu'")
        self.precision_ = "fp16" if kind(self.fit_device_) == "cuda" and (
            self.precision == "fp16" or self.precision == "auto" and _fast_fp16(self.fit_device_) and
            X.size >= FP16_MIN_CELLS) else "fp32"
        self.y_range_ = (float(y.min()), float(y.max()))
        ys = np.stack([y] + ([] if self.second_ is None else [np.log(y) if self.second_ == "log" else -y]))
        if not getattr(self, "cache_context", True):
            # Fold models need the context only once: keep the inputs, encode in predict (_uncached_blocks).
            self._uncached_data = (X.copy(), ys)
            return self
        self.ctx_ = self._encode(X, ys)
        return self

    def _cat_tensor(self):
        return (torch.as_tensor(self.cat_, device=self._tensor_device())[None].expand(self._members, -1)
                if self.cat_.any() else None)

    def _encode(self, X, ys):
        """The encoded context of the (n, m) float32 inputs and the (members, n) targets of fit."""
        Xt, yt = self._x(X), torch.as_tensor(ys, device=self._tensor_device())
        kw = dict(chunk_cells=self._chunk_cells(), cat=self._cat_tensor(), regression=True)
        try:
            with self._autocast():
                ctx = self._inference_net().encode(Xt, yt, **kw)
        except MemoryError as exc:
            if kind(self.fit_device_) != "vulkan":
                raise
            msg = str(exc)
            exc.__traceback__ = None
            eng = self._vk_net.eng
            eng.ops.clear()
            eng.pending = 0.0
            eng.scratch.clear()
            eng.finish()
            self._use_cpu(msg)
            Xt, yt = Xt.cpu(), yt.cpu()
            kw.update(chunk_cells=self._chunk_cells(), cat=self._cat_tensor())
            ctx = self._inference_net().encode(Xt, yt, **kw)
        if kind(self.fit_device_) == "vulkan":
            self._vk_net.eng.finish()
        if self.precision_ == "fp16":
            if not _context_isfinite(ctx):
                self._precision_fallback("context group")
                with torch.autocast("cuda", enabled=False):
                    ctx = self.model.encode(Xt, yt, **kw)
            ctx = _fp32_cache(ctx)
        return ctx

    def _combine(self, q):
        """(members, M, 999) original-unit quantiles of the members -> (M, 999) of the ensemble."""
        if self.second_ is None:
            return q[0]
        second = q[1]
        if self.second_ == "log":
            lo, hi = self.y_range_
            second = np.clip(np.exp(np.clip(second, -700, 700)), max(lo - (hi - lo), 0.), hi + (hi - lo))
        else:
            second = -second[:, ::-1]
        return np.sort((q[0] + second) / 2, axis=1)

    def _one_pass(self, n, n_test, m):
        """cache_context=False on CUDA/ROCm: encode and predict together when no piece would be chunked."""
        cells = self._chunk_cells()
        return (kind(self.fit_device_) == "cuda" and fuse_enabled("onepass") and 0 < n_test <= self.query_rows
                and (cells is None or self._members * (n + n_test) * m <= cells))

    def _uncached_blocks(self, X):
        Xc, ys = self._uncached_data
        Xq = self._x(X)
        if not self._one_pass(len(Xc), Xq.shape[1], Xc.shape[1]):
            # Larger tables: the cached path, with a context that lives only for this prediction.
            self.ctx_ = self._encode(Xc, ys)
            try:
                yield from self._cached_blocks(Xq)
            finally:
                del self.ctx_
            return
        kw = dict(cat=self._cat_tensor())
        Xt, yt = self._x(Xc), torch.as_tensor(ys, device=self._tensor_device())
        with self._autocast():
            q = self._inference_net().encode_predict_quantiles(Xt, yt, Xq, **kw)
        if self.precision_ == "fp16" and not torch.isfinite(q).all():
            self._precision_fallback("one-pass group")
            with torch.autocast("cuda", enabled=False):
                q = self.model.encode_predict_quantiles(Xt, yt, Xq, **kw)
        yield self._combine(q.cpu().numpy())

    def _quantile_blocks(self, X):
        """Yield one NumPy query block, without retaining the preceding blocks."""
        if getattr(self, "_uncached_data", None) is not None:
            yield from self._uncached_blocks(X)
            return
        if not hasattr(self, "ctx_"):
            raise ValueError("fit the regressor first")
        yield from self._cached_blocks(self._x(X))

    def _cached_blocks(self, Xt):
        for i in range(0, Xt.shape[1], self.query_rows):
            query = Xt[:, i:i + self.query_rows]
            with self._autocast():
                q = self._inference_net().predict_quantiles(self.ctx_, query, self._chunk_cells())
            if self.precision_ == "fp16" and not torch.isfinite(q).all():
                self._precision_fallback("prediction chunk")
                with torch.autocast("cuda", enabled=False):
                    q = self.model.predict_quantiles(self.ctx_, query, self._chunk_cells())
            yield self._combine(q.cpu().numpy())
            del q

    @torch.no_grad()
    def predict_quantiles(self, X, quantiles=None):
        if quantiles is None:
            return np.concatenate(list(self._quantile_blocks(X)))
        levels = np.asarray(quantiles, dtype=np.float64)
        if levels.ndim != 1 or not np.isfinite(levels).all() or np.any((levels < 0) | (levels > 1)):
            raise ValueError("quantile levels must be in [0,1]")
        taus = TAUS.double().numpy()
        return np.concatenate([np.array([np.interp(levels, taus, row) for row in q])
                               for q in self._quantile_blocks(X)])

    @torch.no_grad()
    def predict(self, X):
        integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        return np.concatenate([.001 * q[:, 0] + integrate(q, dx=.001, axis=1) + .001 * q[:, -1]
                               for q in self._quantile_blocks(X)])


class LightPFNRegressor(RegressorMixin, BaseEstimator):
    """Pretrained quantile regressor with the scikit-learn estimator interface.

    Weights come from checkpoint (a local safetensors directory or .pt file),
    model (an unfused network), or the pinned Hugging Face regressor subfolder.
    cat="auto" detects DataFrame categories and integer features with 2..20 levels.
    n_estimators is 1 or 2; log_skew=None always uses a sign-flipped second member.
    predict_quantiles(X) returns the 999 levels .001 through .999; supply quantiles
    to interpolate other levels in [0, 1]. predict integrates the quantile curve.
    precision="auto" follows the classifier policy on CUDA/ROCm; CPU and Vulkan use fp32.
    fold=True uses a folded inference network; False preserves the original CPU operations.
    cache_context=False keeps the fit inputs and encodes them at each prediction, without storing the encoded
    context (fold models predicted once); on CUDA/ROCm a small table is then encoded and predicted in one pass.
    device="auto" tries CUDA/ROCm, then Vulkan, then the CPU (LIGHTPFN_DEVICE overrides it), as the classifier.
    Vulkan supports marked categorical columns; category statistics, target normalization and quantile head
    readout stay on the host. A fit exceeding Vulkan buffer limits falls back to CPU with a warning under
    auto and raises MemoryError with explicit Vulkan. to() moves only between torch devices.
    """

    def __init__(self, model=None, checkpoint=None, device="auto", precision="auto", n_estimators=2,
                 max_context=50000, random_state=0, log_skew=2.0, query_rows=4096, chunk_cells="auto",
                 n_threads=None, repo_id=None, revision=None, cache_dir=None, local_files_only=False,
                 fold=False, cache_context=True):
        self.model = model
        self.checkpoint = checkpoint
        self.device = device
        self.precision = precision
        self.n_estimators = n_estimators
        self.max_context = max_context
        self.random_state = random_state
        self.log_skew = log_skew
        self.query_rows = query_rows
        self.chunk_cells = chunk_cells
        self.n_threads = n_threads
        self.repo_id = repo_id
        self.revision = revision
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.fold = fold
        self.cache_context = cache_context

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.regressor_tags.poor_score = True
        return tags

    def _initialize_backend(self):
        """Load the data-independent network once, also usable by AutoGluon SharedWeights.

        A weightless AutoGluon pickle retains the key but needs the network rebuilt.
        Fitted contexts belong to each estimator, never to the shared network.
        """
        from lightpfn.checkpoint import load_model, load_pretrained
        from lightpfn.device import kind, resolve_device

        if self.model is not None and self.checkpoint is not None:
            raise ValueError("Provide either model or checkpoint, not both.")
        # Same device policy as the classifier: auto (or LIGHTPFN_DEVICE) tries CUDA/ROCm, then Vulkan, then CPU.
        # On Vulkan the torch network stays on the CPU and V15Regressor builds its Vulkan copy at fit.
        device = resolve_device(self.device)
        torch_device = "cpu" if kind(device) == "vulkan" else device
        key = (id(self.model), self.checkpoint, device, self.fold, self.repo_id,
               self.revision, self.cache_dir, self.local_files_only,
               os.environ.get("LIGHTPFN_WEIGHTS_DIR"))
        if getattr(self, "_backend_key", None) == key and getattr(self, "model_", None) is not None:
            return
        if self.model is not None:
            model = copy.deepcopy(self.model).to(torch_device).eval()
        elif self.checkpoint is not None:
            model = load_model(self.checkpoint, torch_device)
        else:
            model = load_pretrained(task="regressor", repo_id=self.repo_id, revision=self.revision,
                                    device=torch_device, cache_dir=self.cache_dir,
                                    local_files_only=self.local_files_only)
        self.model_, self.device_ = model.folded() if self.fold and kind(device) != "vulkan" else model, device
        self._backend_key = key

    def fit(self, X, y, cat="auto"):
        self.__dict__.pop("regressor_", None)
        if self.n_threads is not None:
            if isinstance(self.n_threads, bool) or not isinstance(self.n_threads, Integral) or self.n_threads < 1:
                raise ValueError("n_threads must be a positive integer or None")
            torch.set_num_threads(self.n_threads)
        self._initialize_backend()
        seed = self.random_state
        if isinstance(seed, np.random.RandomState):
            seed = seed.randint(2**32)
        # auto stays auto so a Vulkan fit too large for the GPU buffers falls back to the CPU with a warning
        engine = V15Regressor(self.model_, device="auto" if resolve_device_was_auto(self.device) else self.device_,
                              max_context=self.max_context, seed=seed,
                              query_rows=self.query_rows, chunk_cells=self.chunk_cells,
                              n_estimators=self.n_estimators, precision=self.precision, log_skew=self.log_skew,
                              cache_context=self.cache_context)
        engine.fit(X, y, cat=cat)
        self.regressor_ = engine
        self.n_features_in_ = len(engine.cat_)
        self.__dict__.pop("feature_names_in_", None)
        if _is_frame(X) and all(isinstance(c, str) for c in X.columns):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def to(self, device):
        """Move this estimator's network and fitted context to an explicit torch device."""
        from lightpfn.device import kind, resolve_device
        from lightpfn.sklearn import _to_device

        device = resolve_device(device)
        if kind(device) == "vulkan":
            raise ValueError("to() moves between torch devices; refit with device='vulkan' to use Vulkan.")
        check_is_fitted(self, "regressor_")
        self.model_.to(device)
        engine = self.regressor_
        engine.model = self.model_
        engine.device = torch.device(device)
        engine.device_ = engine.fit_device_ = device
        if hasattr(engine, "ctx_"):  # cache_context=False keeps NumPy inputs instead
            context = _fp32_cache(engine.ctx_) if kind(device) == "cpu" else engine.ctx_
            engine.ctx_ = _to_device(context, device)
        engine.__dict__.pop("_fp16_net", None)
        if kind(device) == "cpu":
            engine.precision_ = "fp32"
        self.device_ = self.device = device
        self._backend_key = None
        return self

    @property
    def second_(self):
        check_is_fitted(self, "regressor_")
        return self.regressor_.second_

    @property
    def precision_(self):
        check_is_fitted(self, "regressor_")
        return self.regressor_.precision_

    @property
    def n_precision_fallbacks_(self):
        check_is_fitted(self, "regressor_")
        return self.regressor_.n_precision_fallbacks_

    def predict(self, X):
        check_is_fitted(self, "regressor_")
        return self.regressor_.predict(X)

    def predict_quantiles(self, X, quantiles=None):
        check_is_fitted(self, "regressor_")
        return self.regressor_.predict_quantiles(X, quantiles)
