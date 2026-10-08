"""The fast inference paths compute the same function as the plain one: cell stages run by pieces
(chunk_cells), estimators encoded as one batch (batch_cells) and the folded model (LightPFN.folded).
Run: python -m pytest tests/test_inference_paths.py -q"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightpfn.model.lightpfn import Config, LightPFN
from lightpfn.sklearn import LightPFNClassifier, stratified_subsample

torch.set_num_threads(4)
CONFIGS = {
    "default": {},
    "B4": dict(row_refine=True, row_mode="summary", icl_drop_blocks=1),
    "refine_self_attention": dict(row_refine=True, icl_drop_blocks=1),
    "summary": dict(row_mode="summary"),
    "rbf": dict(cell_embed="rbf", icl_ff_reallocation=4),
}
LOGIT_TOL = dict(rtol=0, atol=2e-5)


def _model(cfg):
    torch.manual_seed(0)
    model = LightPFN(Config(**cfg)).eval()
    for p in model.parameters():  # zero-initialized outputs would hide differences: randomize them
        torch.nn.init.normal_(p, std=0.05)
    return model


def _task(B=2, n_train=60, n_test=25, m=7, C=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(B, n_train + n_test, m, generator=g)
    X[0, 5, min(2, m - 1)] = float("nan")
    X[-1, :, m // 2] = 1.0  # constant column
    y = torch.randint(0, C, (B, n_train), generator=g)
    y[:, :C] = torch.arange(C)
    return X[:, :n_train], y, X[:, n_train:]


def _caches(ctx):
    return ctx.col_kv + (ctx.col_refine_kv or [])


@pytest.mark.parametrize("name", list(CONFIGS))
@torch.no_grad()
def test_chunked_encode_and_predict_match_plain(name):
    model = _model(CONFIGS[name])
    Xtr, y, Xte = _task()
    slots = torch.stack([torch.randperm(16, generator=torch.Generator().manual_seed(s)) for s in range(2)])
    plain = model.encode(Xtr, y, slots=slots, n_classes=3)
    base = model.predict_logits(plain, Xte)
    for cells in (1, 50, 400, 10**6):  # one cell, one column or a few rows, several columns, everything
        ctx = model.encode(Xtr, y, slots=slots, n_classes=3, chunk_cells=cells)
        for (k0, v0), (k1, v1) in zip(_caches(plain), _caches(ctx), strict=True):
            torch.testing.assert_close(k1, k0, rtol=0, atol=1e-5)
            torch.testing.assert_close(v1, v0, rtol=0, atol=1e-5)
        torch.testing.assert_close(model.predict_logits(ctx, Xte), base, **LOGIT_TOL)
        torch.testing.assert_close(model.predict_logits(plain, Xte, chunk_cells=cells), base, **LOGIT_TOL)


@pytest.mark.parametrize("shape", [(1, 30, 1), (1, 1, 5), (3, 40, 2)])
@torch.no_grad()
def test_chunked_encode_small_shapes_and_float64(shape):
    B, n, m = shape
    model = _model(CONFIGS["B4"])
    g = torch.Generator().manual_seed(1)
    Xtr, Xte = torch.randn(B, n, m, generator=g), torch.randn(B, 9, m, generator=g)
    y = torch.randint(0, 2, (B, n), generator=g)
    base = model.predict_logits(model.encode(Xtr, y, n_classes=2), Xte)
    ctx = model.encode(Xtr.double(), y, n_classes=2, chunk_cells=3)
    torch.testing.assert_close(model.predict_logits(ctx, Xte, chunk_cells=3), base, **LOGIT_TOL)


@pytest.mark.parametrize("name", ["B4", "default"])
@torch.no_grad()
def test_chunked_encode_with_padded_features(name):
    model = _model(CONFIGS[name])
    Xtr, y, Xte = _task(m=5)
    d = torch.tensor([5, 3])
    Xtr, Xte = Xtr.clone(), Xte.clone()
    Xtr[1, :, 3:], Xte[1, :, 3:] = 0.0, 0.0  # task 1 has 3 real features, zero-padded to 5
    plain = model.encode(Xtr, y, d=d, n_classes=3, has_padding=True)
    base = model.predict_logits(plain, Xte)
    ctx = model.encode(Xtr, y, d=d, n_classes=3, has_padding=True, chunk_cells=40)
    torch.testing.assert_close(model.predict_logits(ctx, Xte, chunk_cells=20), base, **LOGIT_TOL)


def test_chunked_encode_refuses_autograd():
    model = _model(CONFIGS["B4"])
    Xtr, y, _ = _task()
    with pytest.raises(RuntimeError, match="inference"):
        model.encode(Xtr, y, n_classes=3, chunk_cells=50)


@pytest.mark.parametrize("name", list(CONFIGS))
@torch.no_grad()
def test_folded_model_matches_original(name):
    model = _model(CONFIGS[name])
    Xtr, y, Xte = _task()
    fast = model.folded()
    assert fast.folded() is fast  # folding twice is a no-op
    base = model.predict_logits(model.encode(Xtr, y, n_classes=3), Xte)
    torch.testing.assert_close(fast.predict_logits(fast.encode(Xtr, y, n_classes=3), Xte), base, **LOGIT_TOL)
    torch.testing.assert_close(fast.predict_logits(fast.encode(Xtr, y, n_classes=3, chunk_cells=50), Xte, chunk_cells=30),
                               base, **LOGIT_TOL)
    assert model.row.rope_interleaved is False and not getattr(model, "is_folded", False)  # the original is untouched
    torch.testing.assert_close(model.predict_logits(model.encode(Xtr, y, n_classes=3), Xte), base, rtol=0, atol=0)


@torch.no_grad()
def test_folded_model_runs_under_bf16_autocast():
    model = _model(CONFIGS["B4"])
    Xtr, y, Xte = _task()
    fast = model.folded()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        base = model.predict_logits(model.encode(Xtr, y, n_classes=3), Xte).float()
        out = fast.predict_logits(fast.encode(Xtr, y, n_classes=3), Xte).float()
    torch.testing.assert_close(out, base, rtol=0, atol=0.1)  # bf16 rounding of a different but equal computation


def _wrapper_task(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(150, 6)).astype(np.float32)
    X[rng.random(X.shape) < 0.05] = np.nan
    y = (np.nan_to_num(X[:, 0]) + rng.normal(size=150) > 0).astype(int) + (X[:, 1] > 1)
    return X, y


@pytest.mark.parametrize("name", ["B4", "default"])
def test_batched_chunked_folded_estimators_match_plain(name):
    model = _model(CONFIGS[name])
    X, y = _wrapper_task()
    plain = LightPFNClassifier(model=model, device="cpu", n_estimators=3, max_context=80, seed=1, chunk_cells=None, batch_cells=0, fold=False)
    P = plain.fit(X[:100], y[:100]).predict_proba(X[100:])
    for kw in (dict(chunk_cells=None, batch_cells=10**9, fold=False), dict(chunk_cells=37, batch_cells=0, fold=False),
               dict(chunk_cells=37, batch_cells=10**9, fold=False), dict(chunk_cells=37, batch_cells=2 * 80 * 6, fold=False),
               dict(chunk_cells=None, batch_cells=0), dict(chunk_cells=37, batch_cells=10**9), {}):
        clf = LightPFNClassifier(model=model, device="cpu", n_estimators=3, max_context=80, seed=1, **kw).fit(X[:100], y[:100])
        Q = clf.predict_proba(X[100:])
        np.testing.assert_allclose(Q, P, rtol=0, atol=1e-6, err_msg=str(kw))
        assert np.array_equal(Q.argmax(1), P.argmax(1)), kw


class RecordingModel:
    """Stand-in for LightPFN: records what encode() receives, one entry per estimator."""

    def __init__(self):
        self.cfg = SimpleNamespace(label_slots=16)
        self.calls = []

    def folded(self):
        return self

    def encode(self, X, y, slots, n_classes, chunk_cells=None):
        for Xb, yb, sb in zip(X, y, slots):
            self.calls.append((Xb.numpy().copy(), yb.numpy().copy(), sb.numpy().copy()))


def _old_plans(X, y, n_estimators, max_context, seed, S=16):
    """The estimator planning of the wrapper before batching (commit 997455e), written out."""
    rng = np.random.default_rng(seed)
    out = []
    for e in range(n_estimators):
        rows = np.arange(len(y))
        if len(rows) > max_context:
            rows = stratified_subsample(rng, y, max_context)
        feats = np.arange(X.shape[1]) if e == 0 else rng.permutation(X.shape[1])
        slots = np.arange(S) if e == 0 else rng.permutation(S)
        out.append((X[rows][:, feats], y[rows], slots))
    return out


@pytest.mark.parametrize("device_batch", [0, 10**9, 2 * 80 * 6])
def test_estimator_plans_unchanged(device_batch):
    X, y = _wrapper_task(3)
    model = RecordingModel()
    LightPFNClassifier(model=model, device="cpu", n_estimators=5, max_context=80, seed=7, batch_cells=device_batch).fit(X, y)
    expected = _old_plans(X, np.unique(y, return_inverse=True)[1], 5, 80, 7)
    assert len(model.calls) == len(expected)
    for (Xc, yc, sc), (Xe, ye, se) in zip(model.calls, expected):
        assert np.array_equal(Xc, Xe, equal_nan=True) and np.array_equal(yc, ye) and np.array_equal(sc, se)
