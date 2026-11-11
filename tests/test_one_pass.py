"""One-pass encode + predict of uncached fits (LightPFN.encode_predict_logits, 2026-11-11)."""
import numpy as np
import pandas as pd
import pytest
import torch

from dataclasses import replace

import lightpfn.model.lightpfn as LM
from lightpfn.sklearn import LightPFNClassifier
from test_release import small_model


def task(categorical=False):
    rng = np.random.default_rng(43)
    x = rng.normal(size=(43, 4)).astype(np.float32)
    x[1, 2] = np.nan
    if categorical:
        x[:, 0] = np.arange(len(x)) % 3
        x = pd.DataFrame(x)
        x[0] = x[0].astype('category')
    return x, np.arange(len(x)) % 3


def model(categorical=False, regression=False):
    cfg = replace(small_model().cfg, cat_adapter=categorical, head='quantiles' if regression else 'classification',
                  row_mode='summary', row_refine=True, row_refine_rounds=1)
    torch.manual_seed(13)
    net = LM.LightPFN(cfg).eval()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * .05)
    return net

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA/ROCm is unavailable')


def inputs(B=3, n=40, M=17, m=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(B, n + M, m, generator=g)
    X[:, :, 1] = torch.randint(0, 4, (B, n + M), generator=g).float()
    X[0, 3, 2] = float('nan')
    X[1, n + 2, 1] = float('nan')
    y = torch.randint(0, 3, (B, n), generator=g)
    cat = torch.zeros(B, m, dtype=torch.bool)
    cat[:, 1] = True
    perm = torch.stack([torch.randperm(n, generator=g) for _ in range(B)])
    slots = torch.stack([torch.randperm(16, generator=g) for _ in range(B)])
    return X[:, :n], y, X[:, n:], dict(slots=slots, n_classes=3), dict(cat=cat, cat_perm=perm)


@pytest.mark.parametrize('folded', [False, True])
@pytest.mark.parametrize('categorical', [False, True])
def test_one_pass_equals_encode_then_predict(folded, categorical):
    net = model(categorical)
    net = net.folded() if folded else net
    Xc, y, Xq, kw, ckw = inputs()
    if categorical:
        kw.update(ckw)
    with torch.inference_mode():
        for chunk in (None, 1 << 20):
            ctx = net.encode(Xc, y, chunk_cells=chunk, **kw)
            expected = net.predict_logits(ctx, Xq, chunk)
            got = net.encode_predict_logits(Xc, y, Xq, **kw)
            torch.testing.assert_close(got, expected, rtol=0, atol=2e-5)


def test_one_pass_guards():
    net = model(False)
    Xc, y, Xq, kw, _ = inputs()
    with pytest.raises(RuntimeError, match='inference'):
        net.encode_predict_logits(Xc, y, Xq, **kw)
    with torch.inference_mode(), pytest.raises(ValueError, match='regression'):
        model(False, regression=True).encode_predict_logits(Xc, y.float(), Xq)


def test_one_pass_only_on_cuda_uncached():
    x, y = task(True)
    clf = LightPFNClassifier(model=model(True), device='cpu', cache_context=False).fit(x, y)
    assert not clf._one_pass(len(x))  # the CPU keeps its cache-sized pieces
    clf.fit_device_ = 'cuda'
    assert clf._one_pass(len(x)) and not clf._one_pass(clf.chunk_rows + 1) and not clf._one_pass(0)
    clf.fit_device_ = 'cpu'


@cuda
@pytest.mark.parametrize('precision', ['fp32', 'fp16'])
@pytest.mark.parametrize('categorical', [False, True])
@pytest.mark.parametrize('n_estimators', [1, 4])
def test_cuda_one_pass_matches_cached_classifier(monkeypatch, precision, categorical, n_estimators):
    x, y = task(categorical)
    kw = dict(model=model(categorical), device='cuda', precision=precision, n_estimators=n_estimators)
    cached = LightPFNClassifier(**kw).fit(x, y).predict_proba(x)
    one = LightPFNClassifier(cache_context=False, **kw).fit(x, y)
    assert one._one_pass(len(x))
    got = one.predict_proba(x)
    assert one.members_ == [] and one.n_precision_fallbacks_ == 0
    np.testing.assert_allclose(got, cached, rtol=0, atol=1e-5 if precision == 'fp32' else 2e-3)
    np.testing.assert_allclose(got.sum(1), 1, atol=1e-12)
    monkeypatch.setenv('LIGHTPFN_FUSE_DISABLE', 'onepass')
    two = LightPFNClassifier(cache_context=False, **kw).fit(x, y)
    assert not two._one_pass(len(x))
    np.testing.assert_array_equal(two.predict_proba(x), cached)


@cuda
def test_cuda_one_pass_fp32_retry(monkeypatch):
    x, y = task(True)
    clf = LightPFNClassifier(model=model(True), device='cuda', precision='fp16', cache_context=False).fit(x, y)
    reference = LightPFNClassifier(model=model(True), device='cuda', precision='fp32').fit(x, y).predict_proba(x)
    half = clf._inference_net()
    original = type(half).encode_predict_logits

    def poisoned(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        return out if self is not half else out.fill_(float('nan'))

    monkeypatch.setattr(type(half), 'encode_predict_logits', poisoned)
    got = clf.predict_proba(x)
    assert clf.n_precision_fallbacks_ == 1
    np.testing.assert_allclose(got, reference, rtol=0, atol=1e-5)


@cuda
def test_cuda_one_pass_large_falls_back_to_chunked_path():
    rng = np.random.default_rng(1)
    x = pd.DataFrame(rng.normal(size=(300, 6)).astype(np.float32))
    y = np.arange(300) % 2
    clf = LightPFNClassifier(model=model(False), device='cuda', cache_context=False, chunk_cells=1000).fit(x, y)
    assert not clf._one_pass(len(x))
    cached = LightPFNClassifier(model=model(False), device='cuda', chunk_cells=1000).fit(x, y).predict_proba(x)
    np.testing.assert_array_equal(clf.predict_proba(x), cached)


def reg_inputs(B=2, n=40, M=17, m=5, seed=1):
    Xc, _, Xq, _, ckw = inputs(B, n, M, m, seed)
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(B, n, generator=g, dtype=torch.float64) * 3 + 10
    return Xc, y, Xq, ckw


@pytest.mark.parametrize('folded', [False, True])
@pytest.mark.parametrize('categorical', [False, True])
def test_one_pass_quantiles_equal_encode_then_predict(folded, categorical):
    net = model(categorical, regression=True)
    net = net.folded() if folded else net
    Xc, y, Xq, ckw = reg_inputs()
    kw = ckw if categorical else {}
    with torch.inference_mode():
        for chunk in (None, 1 << 20):
            ctx = net.encode(Xc, y, chunk_cells=chunk, regression=True, **kw)
            expected = net.predict_quantiles(ctx, Xq, chunk)
            got = net.encode_predict_quantiles(Xc, y, Xq, **kw)
            assert got.dtype == torch.float64
            torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
    with pytest.raises(RuntimeError, match='inference'):
        net.encode_predict_quantiles(Xc, y, Xq)


def reg_task():
    rng = np.random.default_rng(5)
    x = pd.DataFrame(rng.normal(size=(60, 4)).astype(np.float32))
    x[3] = pd.Categorical(rng.choice(list('abc'), 60))
    y = np.exp(x[0].to_numpy() + rng.normal(size=60) * .3) + 1
    return x, y


def test_regressor_uncached_cpu_matches_cached_and_pickles():
    import pickle
    from lightpfn import LightPFNRegressor
    x, y = reg_task()
    kw = dict(model=model(True, regression=True), device='cpu')
    cached = LightPFNRegressor(**kw).fit(x, y)
    lazy = LightPFNRegressor(cache_context=False, **kw).fit(x, y)
    assert not hasattr(lazy.regressor_, 'ctx_') and lazy.regressor_._uncached_data is not None
    np.testing.assert_array_equal(lazy.predict(x), cached.predict(x))  # CPU: the cached path, built per call
    assert not hasattr(lazy.regressor_, 'ctx_')
    restored = pickle.loads(pickle.dumps(lazy))
    np.testing.assert_array_equal(restored.predict_quantiles(x[:7]), cached.predict_quantiles(x[:7]))
    lazy.to('cpu')
    np.testing.assert_array_equal(lazy.predict(x), cached.predict(x))


@cuda
@pytest.mark.parametrize('precision', ['fp32', 'fp16'])
@pytest.mark.parametrize('n_estimators', [1, 2])
def test_cuda_regressor_one_pass_matches_cached(monkeypatch, precision, n_estimators):
    from lightpfn import LightPFNRegressor
    x, y = reg_task()
    kw = dict(model=model(True, regression=True), device='cuda', precision=precision, n_estimators=n_estimators)
    cached = LightPFNRegressor(**kw).fit(x, y)
    lazy = LightPFNRegressor(cache_context=False, **kw).fit(x, y)
    assert lazy.regressor_._one_pass(len(x), len(x), x.shape[1])
    tol = dict(rtol=1e-5, atol=1e-5) if precision == 'fp32' else dict(rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(lazy.predict_quantiles(x), cached.predict_quantiles(x), **tol)
    assert lazy.n_precision_fallbacks_ == 0 and not hasattr(lazy.regressor_, 'ctx_')
    monkeypatch.setenv('LIGHTPFN_FUSE_DISABLE', 'onepass')
    np.testing.assert_array_equal(lazy.predict(x), cached.predict(x))
