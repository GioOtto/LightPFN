"""Public regressor contract, including the fitted second-member policy."""
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone, is_regressor
from sklearn.exceptions import NotFittedError

from lightpfn import Config, LightPFN, LightPFNRegressor, save_model


def network():
    torch.manual_seed(13)
    model = LightPFN(Config(col_dim=16, col_heads=2, col_blocks=1, n_inducing=3,
                            row_blocks=1, row_heads=2, n_cls=2, icl_blocks=2, icl_heads=2,
                            decoder_heads=2, label_slots=4, n_thinking=2, n_freq=2,
                            n_ecdf_freq=1, row_mode="summary", cat_adapter=True, head="quantiles")).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * .02)
    return model


def task():
    x = np.random.default_rng(3).normal(size=(30, 4)).astype(np.float32)
    return x, np.linspace(1, 3, 30)


def test_api_clone_and_fitted_checks(tmp_path):
    model = network()
    path = save_model(model, tmp_path / 'regressor')
    r = LightPFNRegressor(checkpoint=path, device='cpu', precision='fp32', random_state=7)
    assert is_regressor(r)
    assert clone(r).get_params() == r.get_params()
    with pytest.raises(NotFittedError):
        r.predict(np.zeros((2, 4)))
    x, y = task()
    assert r.fit(x, y) is r
    assert r.n_features_in_ == 4 and r.precision_ == 'fp32' and r.second_ == 'flip'
    assert r.predict(x[:3]).shape == (3,)
    q = r.predict_quantiles(x[:3])
    assert q.shape == (3, 999) and np.all(np.diff(q, axis=1) >= 0)
    assert r.predict_quantiles(x[:3], [.1, .5, .9]).shape == (3, 3)
    assert np.isfinite(r.score(x, y))
    assert r.set_params(n_estimators=1).fit(x, y).second_ is None
    with pytest.raises(ValueError):
        r.predict(x[:, :2])


@pytest.mark.parametrize('target,threshold,expected', [
    ('symmetric', 2., 'flip'), ('skew', 2., 'log'), ('negative', 2., 'flip'),
    ('skew', None, 'flip'), ('skew', 100., 'flip'), ('constant', 2., 'flip')])
def test_second_policy(target, threshold, expected):
    x, y = task()
    if target == 'skew':
        y[-1] = 1000
    elif target == 'negative':
        y[-1] = 1000
        y[0] = -1
    elif target == 'constant':
        y[:] = 1
    r = LightPFNRegressor(model=network(), device='cpu', log_skew=threshold).fit(x, y)
    assert r.second_ == expected
    # a constant target has no defined skewness (pandas versions differ: 0 or NaN); the regressor records 0
    expected_skew = 0.0 if target == 'constant' else pd.Series(y).skew()
    assert r.regressor_.skew_ == pytest.approx(expected_skew, abs=1e-12)


def test_subsampled_skew_and_names():
    x, y = task()
    y[-1] = 1000
    frame = pd.DataFrame(x, columns=list('abcd'))
    r = LightPFNRegressor(model=network(), device='cpu', max_context=10, random_state=0).fit(frame, y)
    keep = np.sort(np.random.default_rng(0).choice(30, 10, replace=False))
    assert r.regressor_.skew_ == pytest.approx(pd.Series(y[keep]).skew())
    assert list(r.feature_names_in_) == list('abcd')
    with pytest.raises(ValueError, match='feature names'):
        r.predict(frame[list('bacd')])


@pytest.mark.parametrize('target', [np.ones((30, 1)), np.ones((30, 2)), np.ones(29), np.full(30, np.inf)])
def test_target_shape(target):
    with pytest.raises(ValueError, match='targets'):
        LightPFNRegressor(model=network(), device='cpu').fit(task()[0], target)


@pytest.mark.parametrize('value', ['bad', True, float('nan')])
def test_invalid_log_skew(value):
    with pytest.raises(ValueError, match='log_skew'):
        LightPFNRegressor(model=network(), device='cpu', log_skew=value).fit(*task())


def test_dataframe_categories_and_pipeline():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    x, y = task()
    frame = pd.DataFrame(x, columns=list('abcd'))
    frame['a'] = pd.Categorical(['a', 'b', 'c'] * 10)
    r = LightPFNRegressor(model=network(), device='cpu').fit(frame, y)
    assert r.regressor_.cat_[0]
    query = frame.iloc[:3].copy()
    query['a'] = pd.Categorical(['unseen', 'a', None])
    assert np.isfinite(r.predict(query)).all()
    pipe = make_pipeline(StandardScaler(), LightPFNRegressor(model=network(), device='cpu'))
    assert pipe.fit(x, y).predict(x[:3]).shape == (3,)


def test_to_cpu_updates_all_device_metadata():
    x, y = task()
    r = LightPFNRegressor(model=network(), device='cpu').fit(x, y)
    expected = r.predict(x[:3])
    # Reproduce stale CUDA fit metadata without requiring a GPU for this contract.
    r.regressor_.fit_device_ = 'cuda:0'
    r.to('cpu')
    assert r.regressor_.fit_device_ == r.regressor_.device_ == r.device_ == 'cpu'
    np.testing.assert_array_equal(expected, r.predict(x[:3]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA/ROCm unavailable')
@pytest.mark.parametrize('precision', ['fp32', 'fp16'])
def test_cuda_cpu_cuda_round_trip(precision):
    x, y = task()
    r = LightPFNRegressor(model=network(), device='cuda', precision=precision).fit(x, y)
    before = r.predict(x[:3])
    r.to('cpu')
    assert str(r.regressor_._tensor_device()) == 'cpu'
    cpu = r.predict(x[:3])
    r.to('cuda')
    after = r.predict(x[:3])
    assert np.isfinite(cpu).all() and np.isfinite(after).all()
    np.testing.assert_allclose(before, after, rtol=1e-3, atol=1e-3)
