"""scikit-learn style wrapper: fit() encodes the training set once (column statistics, inducing
states, ICL key/value cache), predict_proba() only runs the test rows, in chunks.

Estimators beyond the first use a random feature order and a random class -> label-slot map;
their probabilities are averaged. Training sets larger than max_context are subsampled per
estimator with stratification, so every class (even one with a single row) stays in the context.
"""

import copy
import dataclasses
import warnings
from numbers import Integral

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

from lightpfn.checkpoint import load_model, load_pretrained
from lightpfn.device import kind, resolve_device
from lightpfn.model.lightpfn import Config, LightPFN

# Inference pieces, in cells (rows x features, times the estimators of a batch); measured on an 8-core
# CPU and an RTX 5090 (inference section of the technical report).
CPU_CHUNK_CELLS_PER_THREAD = 1024
GPU_CHUNK_CELLS = 1 << 20
CPU_BATCH_CELLS = 0  # one estimator at a time: batching saves nothing once pieces fit the cache
GPU_BATCH_CELLS = 1 << 22


def _is_frame(X):
    return hasattr(X, "columns") and hasattr(X, "dtypes") and hasattr(X, "iloc")


def _is_categorical(dtype):
    import pandas as pd

    return (isinstance(dtype, pd.CategoricalDtype) or dtype == object or pd.api.types.is_string_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype))


def _encode_frame(X, categories):
    """A DataFrame of float32 columns with the same names: categorical columns (category, string, object or
    bool dtype) become the ordinal codes of `categories` (column position -> categories seen in training),
    with missing and unseen values as NaN; the other columns are converted to numbers."""
    import pandas as pd

    cols = {}
    for j, name in enumerate(X.columns):
        s = X.iloc[:, j]
        if j in categories:
            codes = categories[j].get_indexer(s).astype(np.float32)
            codes[codes < 0] = np.nan
            cols[j] = codes
        else:
            cols[j] = pd.to_numeric(s).to_numpy(dtype=np.float32, na_value=np.nan)
    out = pd.DataFrame(cols, index=X.index)
    out.columns = X.columns
    return out


def stratified_subsample(rng, y, n, min_per_class=5):
    """Exactly n of the len(y) > n rows: class proportions kept, and min(count, min_per_class) rows
    of every class, taken from the largest classes (with fewer classes guaranteed if the minimums
    alone would exceed n)."""
    classes, counts = np.unique(y, return_counts=True)
    if n < len(classes):
        raise ValueError("max_context must be at least the number of classes.")
    mins = np.minimum(counts, min_per_class)
    if mins.sum() > n:
        mins = np.minimum(counts, max(1, n // len(classes)))
    take = np.maximum(np.floor(counts * n / len(y)).astype(int), mins)
    while take.sum() > n:
        take[np.argmax(take - mins)] -= 1
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), size=k, replace=False) for c, k in zip(classes, take)])
    if len(idx) < n:
        rest = np.setdiff1d(np.arange(len(y)), idx)
        idx = np.concatenate([idx, rng.choice(rest, size=n - len(idx), replace=False)])
    return np.sort(idx)


class LightPFNClassifier(ClassifierMixin, BaseEstimator):
    """device: "auto" (default: CUDA/ROCm through torch, else a Vulkan GPU of any vendor, else the CPU; the
    environment variable LIGHTPFN_DEVICE can replace it), or explicitly "cuda[:i]", "vulkan[:i]", "cpu".
    On "vulkan" the network runs on lightpfn.vulkan (wgpu); a training set too large for the GPU's buffers
    falls back to the CPU with a warning when the device was chosen by "auto", and raises otherwise.
    Weights and devices are initialized on fit, so construction and sklearn clone
    perform no downloads or GPU work. Without model/checkpoint the pinned release
    is downloaded from Hugging Face. X is a numeric array or
    a pandas DataFrame; NaN is supported. In a DataFrame, columns of category, string,
    object or bool dtype become ordinal codes of the categories seen in fit, with
    missing and unseen values as NaN (categories_, is_categorical_). The model was
    trained for 2-10 classes.

    n_estimators averages feature/label permutations; max_context bounds the
    stratified context per member. chunk_rows bounds prediction batches;
    chunk_cells/batch_cells control cache blocking and estimator batching.
    cache_context=False builds contexts only during each prediction and releases them afterwards.
    n_threads sets PyTorch's process-wide CPU thread count. seed is the original
    RNG parameter; random_state is its sklearn alias (use one of them).
    fold enables equivalent inference folding. A supplied torch model is copied
    during fit, leaving the constructor parameter and its device unchanged.
    """

    def __init__(self, model=None, checkpoint=None, device="auto", n_estimators=1, max_context=20000,
                 chunk_rows=2048, n_threads=None, seed=0, *, chunk_cells="auto", batch_cells="auto", fold=True,
                 random_state=None, repo_id=None, revision=None, cache_dir=None, local_files_only=False,
                 cache_context=True):
        self.model = model
        self.checkpoint = checkpoint
        self.device = device
        self.fold = fold
        self.n_estimators = n_estimators
        self.max_context = max_context
        self.chunk_rows = chunk_rows
        self.chunk_cells = chunk_cells
        self.batch_cells = batch_cells
        self.n_threads = n_threads
        self.seed = seed
        self.random_state = random_state
        self.repo_id = repo_id
        self.revision = revision
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.cache_context = cache_context

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        # A pretrained network is not optimized on sklearn's toy check datasets.
        tags.classifier_tags.poor_score = True
        return tags

    def _initialize_backend(self):
        key = (id(self.model), self.checkpoint, self.device, self.fold, self.repo_id,
               self.revision, self.cache_dir, self.local_files_only)
        # a pickle without the network (AutoGluon shared weights) keeps the key but not model_ and net_
        if getattr(self, "_backend_key", None) == key and getattr(self, "net_", None) is not None:
            return
        self.device_ = resolve_device(self.device)
        torch_device = "cpu" if kind(self.device_) == "vulkan" else self.device_
        if self.model is not None and self.checkpoint is not None:
            raise ValueError("Provide either model or checkpoint, not both.")
        if self.model is not None:
            self.model_ = copy.deepcopy(self.model) if isinstance(self.model, torch.nn.Module) else self.model
        elif self.checkpoint is not None:
            self.model_ = load_model(self.checkpoint, torch_device)
        else:
            self.model_ = load_pretrained(repo_id=self.repo_id, revision=self.revision, device=torch_device,
                                         cache_dir=self.cache_dir, local_files_only=self.local_files_only)
        if isinstance(self.model_, torch.nn.Module):
            self.model_ = self.model_.to(torch_device).eval()
        # fold=True predicts with LightPFN.folded(), the same function with fewer memory passes
        self.cpu_net_ = None
        if kind(self.device_) == "vulkan":
            from lightpfn.vulkan import VulkanLightPFN

            index = self.device_.partition(":")[2]
            self.net_ = VulkanLightPFN(self.model_, adapter=int(index) if index else None)
        else:
            self.net_ = self.model_.folded() if self.fold else self.model_
        self._backend_key = key

    def _validate_parameters(self):
        for name in ("n_estimators", "max_context", "chunk_rows"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.n_threads is not None and (isinstance(self.n_threads, bool) or
                not isinstance(self.n_threads, Integral) or self.n_threads < 1):
            raise ValueError("n_threads must be None or a positive integer.")
        for name in ("chunk_cells", "batch_cells"):
            value = getattr(self, name)
            if value == "auto" or (name == "chunk_cells" and value is None):
                continue
            minimum = 1 if name == "chunk_cells" else 0
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be 'auto' or an integer >= {minimum}." )
        if not isinstance(self.fold, bool):
            raise ValueError("fold must be a boolean.")
        if not isinstance(self.cache_context, bool):
            raise ValueError("cache_context must be a boolean.")
        if self.random_state is not None and self.seed not in (0, None):
            raise ValueError("Use random_state or seed, not both.")

    def __sklearn_is_fitted__(self):
        return getattr(self, "_is_fitted", False)

    def _auto(self, value, cpu, gpu):
        if value != "auto":
            return value
        return cpu if kind(self.fit_device_) == "cpu" else gpu

    def _chunk_cells(self):
        """Cells per piece of the cell stages: small pieces stay in the CPU cache (about 3x faster on
        wide tables), large pieces bound GPU memory. The predictions do not depend on it."""
        return self._auto(self.chunk_cells, CPU_CHUNK_CELLS_PER_THREAD * torch.get_num_threads(), GPU_CHUNK_CELLS)

    def _group(self, n, m):
        """Estimators encoded together as one batch: as many as fit in batch_cells training cells
        (fewer, larger operations; the same predictions as one at a time)."""
        return max(1, min(self.n_estimators, self._auto(self.batch_cells, CPU_BATCH_CELLS, GPU_BATCH_CELLS) // max(1, n * m)))

    @torch.inference_mode()
    def fit(self, X, y, cat=None):
        self._is_fitted = False
        self._validate_parameters()
        self.categories_ = {}
        self._frame_columns = X.columns.copy() if _is_frame(X) else None
        if _is_frame(X):
            # categories of the training rows, in pandas order (a category dtype keeps its own order)
            self.categories_ = {j: X.iloc[:, j].astype("category").cat.categories
                                for j, dtype in enumerate(X.dtypes) if _is_categorical(dtype)}
            X = _encode_frame(X, self.categories_)
        X, y = validate_data(self, X, y, dtype=np.float32, ensure_all_finite="allow-nan")
        check_classification_targets(y)
        self.is_categorical_ = np.zeros(X.shape[1], bool)
        self.is_categorical_[list(self.categories_)] = True
        if cat is not None:
            cat = np.asarray(cat)
            if cat.shape != (X.shape[1],) or cat.dtype != np.bool_:
                raise ValueError("cat must be a boolean mask with one entry per feature.")
        if self.n_threads is not None:
            torch.set_num_threads(self.n_threads)
        self.classes_, y = np.unique(np.asarray(y), return_inverse=True)
        if len(self.classes_) > 10:
            raise ValueError("LightPFN supports at most 10 classes (trained on 2-10 classes).")
        if self.max_context < len(self.classes_):
            raise ValueError("max_context must be at least the number of classes.")
        self._initialize_backend()
        if len(self.classes_) > self.net_.cfg.label_slots:
            raise ValueError("Number of classes exceeds the model's label slots.")
        self.cat_ = None
        adapter = getattr(self.net_.cfg, "cat_adapter", False)
        if cat is None and adapter and self.is_categorical_.any():
            self.cat_ = self.is_categorical_
        elif cat is not None and cat.any():
            if adapter:
                self.cat_ = cat
            else:
                warnings.warn("cat does not enable native categorical handling without a categorical adapter; "
                              "columns are treated as numeric codes. Encode categories before fit.",
                              FutureWarning, stacklevel=2)
        random_state = self.seed if self.random_state is None else self.random_state
        if isinstance(random_state, np.random.RandomState):
            random_state = random_state.randint(2**32)
        rng = np.random.default_rng(random_state)
        S = self.net_.cfg.label_slots
        plans = []
        for e in range(self.n_estimators):
            rows = np.arange(len(y))
            if len(rows) > self.max_context:
                rows = stratified_subsample(rng, y, self.max_context)
            feats = np.arange(X.shape[1]) if e == 0 else rng.permutation(X.shape[1])
            slots = np.arange(S) if e == 0 else rng.permutation(S)
            plans.append((rows, feats, slots, None))
        if self.cat_ is not None:
            # the row order of the categorical adapter's ordered statistics, per estimator, drawn after the
            # plans: the plain model's draws stay the same
            plans = [(rows, feats, slots, rng.permutation(len(rows))) for rows, feats, slots, _ in plans]
        self._choose_backend(len(plans[0][0]), X.shape[1])
        if self.cat_ is not None and kind(self.fit_device_) == "vulkan":
            self._use_cpu("the Vulkan backend has no categorical adapter")
        self.members_ = []
        self._uncached_data = None
        if self.cache_context:
            self._build_members(X, y, plans)
        else:
            # Fold models need the context only once. Keep owned input arrays on the CPU;
            # predict builds and releases the encoded context, including after an error.
            self._uncached_data = (X.copy(), y.copy(), plans)
        self._is_fitted = True
        return self

    def _build_members(self, X, y, plans):
        try:
            self._encode_members(X, y, plans)
        except MemoryError as exc:
            if kind(self.fit_device_) != "vulkan":
                raise
            # Drop partial contexts and unsubmitted work before a retry or a later fit.
            msg = str(exc)
            exc.__traceback__ = None
            self.members_.clear()
            eng = self.fit_net_.eng
            eng.ops.clear()
            eng.pending = 0.0
            eng.scratch.clear()
            eng.finish()
            self._use_cpu(msg)
            self._encode_members(X, y, plans)
        if kind(self.fit_device_) == "cuda":
            torch.cuda.synchronize(self.fit_device_)  # fit time includes the GPU work, not only its launch
        elif kind(self.fit_device_) == "vulkan":
            self.fit_net_.eng.finish()

    def _encode_members(self, X, y, plans):
        group = self._group(len(plans[0][0]), X.shape[1])
        if kind(self.fit_device_) == "vulkan":
            group = min(group, self.fit_net_.train_capacity(len(plans[0][0]), X.shape[1]))
        tdev = self._tensor_device()
        self.members_ = []
        for g in range(0, len(plans), group):
            part = plans[g : g + group]
            Xt = torch.from_numpy(np.stack([X[rows][:, feats] for rows, feats, _, _ in part])).to(tdev)
            yt = torch.from_numpy(np.stack([y[rows] for rows, _, _, _ in part])).long().to(tdev)
            st = torch.from_numpy(np.stack([slots for _, _, slots, _ in part])).long().to(tdev)
            kw = {}
            if self.cat_ is not None:
                kw["cat"] = torch.from_numpy(np.stack([self.cat_[feats] for _, feats, _, _ in part])).to(tdev)
                kw["cat_perm"] = torch.from_numpy(np.stack([order for _, _, _, order in part])).to(tdev)
            ctx = self.fit_net_.encode(Xt, yt, slots=st, n_classes=len(self.classes_), chunk_cells=self._chunk_cells(),
                                       **kw)
            # (feature order, context): one estimator per entry as before, or (G, m) orders for a batch of G
            feats = part[0][1] if len(part) == 1 else np.stack([feats for _, feats, _, _ in part])
            self.members_.append((feats, ctx))

    def _choose_backend(self, n, m):
        """The network and device of this fit: the classifier's, or the CPU when a Vulkan device chosen by
        "auto" cannot hold one estimator's caches and scratch buffers."""
        self.fit_device_, self.fit_net_ = self.device_, self.net_
        if kind(self.device_) == "vulkan" and self.net_.train_capacity(n, m) < 1:
            self._use_cpu(f"{n} x {m} training cells exceed a Vulkan cache/scratch buffer limit; "
                          "lower max_context or use device='cpu'")

    def _use_cpu(self, msg):
        if not resolve_device_was_auto(self.device):
            raise MemoryError(msg)
        warnings.warn(msg + ": running this fit on the CPU", RuntimeWarning, stacklevel=3)
        if self.cpu_net_ is None:
            self.cpu_net_ = self.model_.folded() if self.fold else self.model_
        self.fit_device_, self.fit_net_ = "cpu", self.cpu_net_

    def _tensor_device(self):
        return "cpu" if kind(self.fit_device_) == "vulkan" else self.fit_device_

    @torch.inference_mode()
    def predict_proba(self, X):
        check_is_fitted(self)
        if _is_frame(X) and self._frame_columns is not None and not self._frame_columns.equals(X.columns):
            # Check before ordinal encoding: integer names are not checked by sklearn,
            # and a reordered numeric column must never use another column's categories.
            raise ValueError("The feature names must match those passed during fit, in the same order.")
        if _is_frame(X) and X.shape[1] == len(self.is_categorical_):
            X = _encode_frame(X, self.categories_)
        X = validate_data(self, X, reset=False, dtype=np.float32, ensure_all_finite="allow-nan")
        try:
            if self._uncached_data is not None:
                self._build_members(*self._uncached_data)
            return self._predict_encoded(X)
        finally:
            if self._uncached_data is not None:
                self.members_.clear()

    def _predict_encoded(self, X):
        P = np.zeros((len(X), len(self.classes_)))
        for feats, ctx in self.members_:
            for i in range(0, len(X), self.chunk_rows):
                Xt = torch.from_numpy(np.stack([X[i : i + self.chunk_rows][:, f] for f in np.atleast_2d(feats)])).to(self._tensor_device())
                probs = torch.softmax(self.fit_net_.predict_logits(ctx, Xt, self._chunk_cells()).float(), -1).cpu().numpy()
                for p in probs:  # one estimator at a time, in float64 as before
                    P[i : i + self.chunk_rows] += p
        # float32 softmax rows sum to 1 only within ~1e-7, which sklearn's log_loss flags
        return P / P.sum(1, keepdims=True)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(1)]

    @torch.inference_mode()  # the network and contexts were built in inference mode
    def to(self, device):
        """Move a fitted classifier (network and encoded training contexts) to a PyTorch device, "cpu" or
        "cuda[:i]", and return it. The constructor's device parameter is unchanged; the next fit
        resolves it again. A classifier on the Vulkan backend cannot be moved: refit it instead."""
        check_is_fitted(self)
        requested = str(device).strip().lower()
        if kind(self.device_) == "vulkan" or kind(requested) == "vulkan":
            raise ValueError("Vulkan contexts live in GPU buffers; refit with the device you need.")
        if kind(requested) not in ("cpu", "cuda"):
            raise ValueError("to() supports only 'cpu' or 'cuda[:i]'.")
        device = resolve_device(requested)
        self.model_ = self.model_.to(device)
        self.net_ = self.net_.to(device)
        self.members_ = [(feats, _to_device(ctx, device)) for feats, ctx in self.members_]
        self.device_ = self.fit_device_ = device
        self.fit_net_ = self.net_
        self._backend_key = None
        return self


def _to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.replace(obj, **{f.name: _to_device(getattr(obj, f.name), device)
                                           for f in dataclasses.fields(obj) if f.init})
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


def resolve_device_was_auto(device):
    return device is None or str(device).strip().lower() == "auto"
