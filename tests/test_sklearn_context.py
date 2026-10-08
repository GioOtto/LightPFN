"""The context subsample of LightPFNClassifier keeps every class, even with a single row."""
from types import SimpleNamespace

import numpy as np
import torch

from lightpfn.sklearn import LightPFNClassifier


class RecordingModel:
    """Stand-in for LightPFN: records the context labels passed to encode()."""

    def __init__(self):
        self.cfg = SimpleNamespace(label_slots=16)
        self.contexts = []

    def folded(self):
        return self

    def encode(self, X, y, slots, n_classes, chunk_cells=None):
        for Xb, yb in zip(X, y):  # one entry per estimator, also when estimators are batched
            self.contexts.append((Xb.numpy().copy(), yb.numpy().copy(), n_classes))
        return None


def rare_task(seed=0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(60_000, int), np.ones(20_000, int), np.full(3, 2), np.full(1, 3), np.full(7, 4)]
    y = rng.permutation(y)
    X = np.c_[y.astype(np.float32), rng.normal(size=(len(y), 2)).astype(np.float32)]
    return X, y


def test_large_training_set_context_keeps_every_class():
    X, y = rare_task()
    model = RecordingModel()
    LightPFNClassifier(model=model, device="cpu", max_context=2000, n_estimators=8, seed=3).fit(X, y)
    assert len(model.contexts) == 8
    for Xc, yc, n_classes in model.contexts:
        counts = np.bincount(yc, minlength=5)
        assert len(yc) == 2000 and n_classes == 5
        assert counts[2] == 3 and counts[3] == 1 and counts[4] == 5, counts
        assert len(np.unique(Xc, axis=0)) == len(Xc)  # no duplicated rows
        # the frequent classes keep their proportion (3:1)
        assert abs(counts[0] / counts[1] - 3) < 0.1


def test_context_rows_are_aligned_with_labels():
    X, y = rare_task(1)
    model = RecordingModel()
    LightPFNClassifier(model=model, device="cpu", max_context=500, n_estimators=1, seed=0).fit(X, y)
    Xc, yc, _ = model.contexts[0]
    assert np.array_equal(Xc[:, 0].astype(int), yc)  # first estimator keeps the feature order


def test_small_training_set_uses_all_rows_in_order():
    X, y = rare_task()
    X, y = X[:900], y[:900]
    model = RecordingModel()
    LightPFNClassifier(model=model, device="cpu", max_context=1000, n_estimators=1).fit(X, y)
    Xc, yc, _ = model.contexts[0]
    assert np.array_equal(yc, np.unique(y, return_inverse=True)[1]) and np.array_equal(Xc, X)
