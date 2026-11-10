"""Minimal sklearn-style classifier around LightPFNV2 for the training evaluations (lite, D1-D3).

Not the public API: no downloads, no CPU/Vulkan fallbacks. Members differ only in the column UIDs,
which changes the canonical column order (grouping, RoPE); labels are already equivariant.
On GPU the members run as one batch when they fit batch_cells (as v1's batch_cells): the same
predictions in fp32, bf16 rounding differences up to ~5e-4 in probability (runs/v2_arch/perf/SOL).
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from lightpfn.sklearn import stratified_subsample

from .types import Table


def load_v2(path, device="cpu"):
    """EMA weights of a C2v2 training checkpoint (ema_step*.pt), or None for a v1 checkpoint or weights directory."""
    from .config import Config
    from .model import LightPFNV2

    if Path(path).is_dir() or Path(path).suffix != ".pt":
        return None
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if "unique_blocks" not in ck.get("config", {}):
        return None
    model = LightPFNV2(Config(**ck["config"]))
    model.load_state_dict(ck["ema"])
    return model.to(device).eval()


def to_table(X, cat, device, uid0=0, column_uid=None):
    """(n, m) array -> one-task Table. cat marks nominal columns holding integer codes (NaN = missing)."""
    X = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)[None]
    missing = torch.isnan(X)
    mask = torch.as_tensor(np.asarray(cat, dtype=bool), device=device)[None]
    codes = torch.where(missing | ~mask[:, None], 0, X.nan_to_num()).round().to(torch.int32)
    n, m = X.shape[1:]
    rows = torch.arange(uid0, uid0 + n, device=device)[None]
    cols = torch.arange(m, device=device)[None] if column_uid is None else column_uid[None]
    return Table(X, codes, mask, missing, rows, cols)


class V2Classifier:
    def __init__(self, model, device="cuda", n_estimators=4, max_context=20000, seed=0, categorical=False,
                 batch_cells="auto"):
        # categorical=False: nominal columns enter as numbers, as for models trained on pools without
        # nominal IDs (graph v3/v4); True once the v2 prior writes int32 categorical IDs.
        self.model, self.device, self.categorical = model, torch.device(device), categorical
        self.n_estimators, self.max_context, self.seed = n_estimators, max_context, seed
        # Context + query cells of the members run together; 0 = one member at a time.
        # "auto": 2**22 on GPU (v1's GPU_BATCH_CELLS), serial on CPU.
        self.batch_cells = (1 << 22 if self.device.type == "cuda" else 0) if batch_cells == "auto" else batch_cells

    def fit(self, X, y, cat=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if not 2 <= len(self.classes_) <= 10:
            raise ValueError("C2v2 supports 2..10 classes")
        rng = np.random.default_rng(self.seed)
        if len(y) > self.max_context:
            keep = np.sort(stratified_subsample(rng, y, self.max_context))
            X, y = X[keep], y[keep]
        self.X_, self.y_ = X, np.searchsorted(self.classes_, y)
        self.cat_ = (np.asarray(cat, bool) if cat is not None and self.categorical
                     else np.zeros(X.shape[1], bool))
        self.uids_ = [torch.as_tensor(np.arange(X.shape[1]) if k == 0 else rng.permutation(X.shape[1]),
                                      device=self.device) for k in range(self.n_estimators)]
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        was_training = self.model.training
        self.model.eval()
        n_ctx, C = len(self.y_), len(self.classes_)
        y = torch.as_tensor(self.y_, dtype=torch.long, device=self.device)[None]
        P = 0
        try:
            ctx = to_table(self.X_, self.cat_, self.device)
            q = to_table(X, self.cat_, self.device, n_ctx)
            cells = max(1, (n_ctx + q.numerical.shape[1]) * self.X_.shape[1])
            step = max(1, min(len(self.uids_), self.batch_cells // cells))

            def members(table, uids):
                batch = len(uids)
                return replace(table, **{name: getattr(table, name).expand(batch, *getattr(table, name).shape[1:])
                                         for name in ("numerical", "categorical", "categorical_mask", "missing", "row_uid")},
                               column_uid=torch.stack(uids))

            for i in range(0, len(self.uids_), step):
                uids = self.uids_[i:i + step]
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    out = self.model(members(ctx, uids), y.expand(len(uids), -1), members(q, uids), n_classes=C)
                # Keep the original accumulation order, including in serial mode.
                for member in out:
                    P = P + member.float().exp()
        finally:
            self.model.train(was_training)
        P = (P / len(self.uids_)).cpu().numpy()
        return P / P.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class V2Regressor:
    """Experimental one-member regressor; original-unit quantiles, no downloads.

    predict integrates the quantile curve, extending the .001/.999 quantiles
    constantly to 0/1. predict_quantiles interpolates on that same convention.
    Nominal int32 IDs are preserved before the numerical float32 conversion.
    """

    def __init__(self, model, device="cpu", max_context=20000, seed=0):
        if model.regression_head is None or not isinstance(max_context, int) or max_context < 1:
            raise ValueError("a regression configuration and positive max_context are required")
        self.model, self.device = model, torch.device(device)
        self.max_context, self.seed = max_context, seed

    def _table(self, X, uid0=0):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.cat_) or not len(X) or np.isinf(X).any():
            raise ValueError("invalid regression X")
        nominal = X[:, self.cat_]
        finite = nominal[np.isfinite(nominal)]
        if (np.any(finite != np.floor(finite)) or np.any(finite < 0)
                or np.any(finite > np.iinfo(np.int32).max)):
            raise ValueError("nominal IDs must be nonnegative int32, NaN for missing")
        codes = np.zeros(X.shape, np.int32)
        codes[:, self.cat_] = np.where(np.isnan(nominal), -1, nominal).astype(np.int32)
        numerical = X.copy()
        numerical[:, self.cat_] = 0
        table = to_table(numerical, self.cat_, self.device, uid0)
        return replace(table, categorical=torch.as_tensor(codes, device=self.device)[None],
                       missing=torch.as_tensor(np.isnan(X), device=self.device)[None])

    def fit(self, X, y, cat=None):
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] < 1 or y.shape != (len(X),) or not len(y) or not np.isfinite(y).all():
            raise ValueError("finite regression targets must match X")
        self.cat_ = np.zeros(X.shape[1], bool) if cat is None else np.asarray(cat, bool)
        if self.cat_.shape != (X.shape[1],):
            raise ValueError("invalid categorical mask")
        self._table(X)  # validate before sampling
        keep = (np.sort(np.random.default_rng(self.seed).choice(len(y), self.max_context, replace=False))
                if len(y) > self.max_context else np.arange(len(y)))
        self.X_, self.y_ = X[keep].copy(), y[keep].copy()
        return self

    @torch.no_grad()
    def predict_quantiles(self, X, quantiles=None):
        if not hasattr(self, "y_"):
            raise ValueError("fit the regressor first")
        training = self.model.training
        self.model.eval()
        try:
            context, query = self._table(self.X_), self._table(X, len(self.y_))
            y = torch.as_tensor(self.y_, device=self.device)[None]
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                q = self.model(context, y, query, regression=True)[0].double().cpu().numpy()
        finally:
            self.model.train(training)
        if quantiles is None:
            return q
        levels = np.asarray(quantiles, dtype=np.float64)
        if levels.ndim != 1 or not np.isfinite(levels).all() or np.any((levels < 0) | (levels > 1)):
            raise ValueError("quantile levels must be in [0,1]")
        return np.array([np.interp(levels, np.arange(1, 1000) / 1000, row) for row in q])

    def predict(self, X):
        q = self.predict_quantiles(X)
        return (.001 * q[:, 0] + np.trapezoid(q, dx=.001, axis=1) + .001 * q[:, -1])
