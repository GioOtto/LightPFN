"""Baseline classifiers behind one interface for the eval harness.

Every model is built by `make_model(name, n_threads, seed)` and exposes
  fit(X, y, cat)       X float32 with NaN for missing values and ordinal codes in the categorical
                       columns (cat is a bool mask), y contiguous labels 0..k-1
  predict_proba(X)     (n, k) probabilities
Baselines run with library defaults: the reference point is what a user gets out of the box.
"""

import numpy as np
import pandas as pd


class RandomForest:
    def __init__(self, n_threads, seed):
        from sklearn.ensemble import RandomForestClassifier

        self.model = RandomForestClassifier(n_jobs=n_threads, random_state=seed)

    def fit(self, X, y, cat):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class LightGBM:
    def __init__(self, n_threads, seed):
        import lightgbm as lgb

        self.model = lgb.LGBMClassifier(n_jobs=n_threads, random_state=seed, verbose=-1)

    def fit(self, X, y, cat):
        self.model.fit(X, y, categorical_feature=np.flatnonzero(cat).tolist())
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class XGBoost:
    def __init__(self, n_threads, seed):
        import xgboost as xgb

        self.model = xgb.XGBClassifier(n_jobs=n_threads, random_state=seed, enable_categorical=True, tree_method="hist")

    def _frame(self, X):
        df = pd.DataFrame(X)
        for j in np.flatnonzero(self.cat):
            # integer codes of the categories seen in training; unseen or missing values -> -1
            cats = self.categories[j]
            codes = np.full(len(X), -1)
            if len(cats):
                pos = np.minimum(np.searchsorted(cats, X[:, j]), len(cats) - 1)
                codes = np.where(cats[pos] == X[:, j], pos, -1)
            df[j] = pd.Categorical.from_codes(codes, categories=np.arange(len(cats)))
        return df

    def fit(self, X, y, cat):
        self.cat = cat
        self.categories = {j: np.unique(X[~np.isnan(X[:, j]), j]) for j in np.flatnonzero(cat)}
        self.model.fit(self._frame(X), y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self._frame(X))


class CatBoost:
    def __init__(self, n_threads, seed):
        from catboost import CatBoostClassifier

        self.model = CatBoostClassifier(thread_count=n_threads, random_seed=seed, verbose=0, allow_writing_files=False)

    def _frame(self, X):
        df = pd.DataFrame(X)
        for j in np.flatnonzero(self.cat):
            df[j] = np.where(np.isnan(X[:, j]), -1, X[:, j]).astype(np.int64)
        return df

    def fit(self, X, y, cat):
        self.cat = cat
        self.model.fit(self._frame(X), y, cat_features=np.flatnonzero(cat).tolist())
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self._frame(X))


class TabICL:
    """TabICLv2 (28M params), the reference foundation model. Categorical codes are passed as
    numbers, like the ordinal encoding TabICL applies itself."""

    def __init__(self, n_threads, seed, device="cpu"):
        import torch
        from tabicl import TabICLClassifier

        from lightpfn.eval.data import ROOT

        torch.set_num_threads(n_threads)
        # large tables (APSFailure, kddcup09) need ~21 GB of column embeddings: more than the GPU,
        # and pinned host memory fails under ROCm, so TabICL's "auto" offload must be able to
        # fall back to memory-mapped files on the data disk
        offload_dir = ROOT / "data" / "tabicl_offload"
        offload_dir.mkdir(parents=True, exist_ok=True)
        self.model = TabICLClassifier(device=device, random_state=seed, n_jobs=n_threads,
                                      disk_offload_dir=str(offload_dir))

    def fit(self, X, y, cat):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


MODELS = {
    "rf": RandomForest,
    "lightgbm": LightGBM,
    "xgboost": XGBoost,
    "catboost": CatBoost,
    "tabicl": TabICL,
    "tabicl_gpu": lambda n_threads, seed: TabICL(n_threads, seed, device="cuda"),
}
BASELINES = ["rf", "lightgbm", "xgboost", "catboost"]


def make_model(name, n_threads, seed):
    return MODELS[name](n_threads, seed)
