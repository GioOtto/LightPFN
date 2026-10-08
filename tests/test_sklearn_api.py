"""Sklearn integration exercises real inference, cloning and input validation."""

import numpy as np
import pickle
from dataclasses import asdict
import pytest
import torch
from sklearn.base import clone, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator

from lightpfn import LightPFNClassifier
from test_release import small_model

torch.set_num_threads(4)


@pytest.fixture
def task():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(32, 3)).astype(np.float32)
    y = np.where(X[:, 0] > 0, "yes", "no")
    return X, y


def test_construction_and_clone_do_not_load_weights(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("constructor performed IO")
    monkeypatch.setattr("lightpfn.sklearn.load_pretrained", fail)
    monkeypatch.setattr("lightpfn.sklearn.resolve_device", fail)
    clf = LightPFNClassifier(device="auto", n_estimators=3, random_state=12)
    cloned = clone(clf)
    assert cloned.get_params() == clf.get_params()
    assert is_classifier(cloned)
    assert not hasattr(cloned, "net_")
    with pytest.raises(NotFittedError):
        cloned.predict([[0, 1]])


def test_pipeline_grid_search_and_set_params(task):
    X, y = task
    clf = LightPFNClassifier(model=small_model(), device="cpu", random_state=3)
    pipe = Pipeline([("scale", StandardScaler()), ("model", clf)])
    search = GridSearchCV(pipe, {"model__n_estimators": [1, 2]}, cv=2).fit(X, y)
    assert search.predict(X[:4]).shape == (4,)
    fitted = search.best_estimator_.named_steps["model"]
    assert fitted.n_features_in_ == 3
    assert 0 <= fitted.score(X, y) <= 1
    changed = clone(clf).set_params(n_estimators=2).fit(X, y)
    assert len(changed.members_) == 2
    assert not hasattr(clone(changed), "members_")


def test_input_errors_and_nan_support(task):
    X, y = task
    clf = LightPFNClassifier(model=small_model(), device="cpu")
    with pytest.raises(ValueError, match="inconsistent"):
        clf.fit(X, y[:-1])
    with pytest.raises(ValueError, match="Unknown label|continuous"):
        clf.fit(X, np.linspace(0, 1, len(y)))
    X[0, 1] = np.nan
    clf.fit(X, y)
    P = clf.predict_proba(X)
    assert np.isfinite(P).all()
    np.testing.assert_allclose(P.sum(1), 1, atol=1e-6)
    with pytest.raises(ValueError, match="features"):
        clf.predict(X[:, :2])
    with pytest.raises(ValueError, match="infinity"):
        clf.predict([[0, np.inf, 1]])
    with pytest.raises(ValueError, match="10 classes"):
        clf.fit(X, np.arange(len(y)))
    with pytest.raises(ValueError, match="max_context"):
        clf.set_params(max_context=1).fit(X, y)
    with pytest.raises(NotFittedError):
        clf.predict(X)


def test_categorical_mask_warns_and_preserves_numeric_behavior(task):
    X, y = task
    model = small_model()
    plain = LightPFNClassifier(model=model, device="cpu").fit(X, y)
    other = LightPFNClassifier(model=model, device="cpu")
    with pytest.warns(FutureWarning, match="native categorical"):
        other.fit(X, y, cat=np.array([True, False, False]))
    np.testing.assert_array_equal(plain.predict_proba(X), other.predict_proba(X))
    with pytest.raises(ValueError, match="boolean mask"):
        other.fit(X, y, cat=[0])


def test_refit_and_single_class(task):
    X, y = task
    model = small_model().train()
    clf = LightPFNClassifier(model=model, device="cpu").fit(X, y)
    assert model.training and not clf.model_.training
    clf.fit(X[:, :2], np.full(len(y), "only"))
    assert clf.n_features_in_ == 2
    np.testing.assert_array_equal(clf.predict_proba(X[:, :2]), np.ones((len(y), 1)))


def test_feature_name_validation(task):
    pd = pytest.importorskip("pandas")
    X, y = task
    frame = pd.DataFrame(X, columns=["a", "b", "c"])
    clf = LightPFNClassifier(model=small_model(), device="cpu").fit(frame, y)
    np.testing.assert_array_equal(clf.feature_names_in_, frame.columns)
    with pytest.raises(ValueError, match="Feature names|feature names"):
        clf.predict(frame[["c", "b", "a"]])


def test_sklearn_estimator_checks(tmp_path):
    # joblib hashes torch storage identity, so raw nn.Module parameters cannot
    # satisfy its deepcopy/hash comparison. Exercise the distribution's path API.
    model = small_model()
    path = tmp_path / "model.pt"
    torch.save(dict(config=asdict(model.cfg), model=model.state_dict()), path)
    check_estimator(
        LightPFNClassifier(checkpoint=path, device="cpu", n_threads=4),
        expected_failed_checks={
            "check_methods_sample_order_invariance": "FP32 GPU/CPU kernels can differ by ~6e-8 after row permutation.",
            "check_methods_subset_invariance": "FP32 kernels can differ slightly when query batch shapes change.",
        },
    )


def test_dataframe_categorical_columns_are_ordinal_codes(task):
    pd = pytest.importorskip("pandas")
    import warnings

    X, y = task
    rng = np.random.default_rng(3)
    color = rng.choice(["red", "green", "blue"], size=len(y)).astype(object)
    color[[2, 5]] = None
    size = pd.Categorical(rng.choice(["S", "M", "L"], size=len(y)), categories=["S", "M", "L"])
    frame = pd.DataFrame({"x": X[:, 0], "color": color, "size": size, "flag": X[:, 1] > 0,
                          "count": pd.array(rng.integers(0, 5, len(y)), dtype="Int64")})
    frame.loc[3, "count"] = pd.NA
    model = small_model()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        clf = LightPFNClassifier(model=model, device="cpu", n_estimators=2).fit(frame, y)
    assert clf.is_categorical_.tolist() == [False, True, True, True, False]
    assert list(clf.categories_[1]) == ["blue", "green", "red"]
    assert list(clf.categories_[2]) == ["S", "M", "L"]  # a category dtype keeps its order

    def codes(f):
        out = np.stack([f.x.to_numpy(np.float32),
                        f.color.map({"blue": 0, "green": 1, "red": 2}).to_numpy(np.float32, na_value=np.nan),
                        f["size"].map({"S": 0, "M": 1, "L": 2}).astype(object).to_numpy(np.float32, na_value=np.nan),
                        f.flag.to_numpy(np.float32),
                        f["count"].to_numpy(np.float32, na_value=np.nan)], axis=1)
        return out.astype(np.float32)

    reference = LightPFNClassifier(model=model, device="cpu", n_estimators=2).fit(codes(frame), y)
    test = frame.iloc[:6].copy()
    test["color"] = ["red", "purple", None, "blue", "green", "red"]  # "purple" was not seen in fit
    test["size"] = pd.Categorical(["L", "S", "M", "XL", "S", "M"], categories=["XL", "L", "M", "S"])
    expected = codes(test)
    expected[1, 1] = np.nan
    expected[3, 2] = np.nan
    np.testing.assert_allclose(clf.predict_proba(test), reference.predict_proba(expected), rtol=0, atol=1e-6)
    np.testing.assert_array_equal(clf.predict(test), reference.predict(expected))


def test_to_moves_a_fitted_classifier(task):
    X, y = task
    clf = LightPFNClassifier(model=small_model(), device="cpu", n_estimators=2)
    with pytest.raises(NotFittedError):
        clf.to("cpu")
    before = clf.fit(X, y).predict_proba(X)
    assert clf.to("cpu") is clf and clf.device_ == "cpu"
    np.testing.assert_array_equal(clf.predict_proba(X), before)
    np.testing.assert_allclose(clf.predict_proba(X).sum(1), 1, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="Vulkan"):
        clf.to("vulkan")
    with pytest.raises(ValueError, match="only"):
        clf.to("meta")
    np.testing.assert_array_equal(clf.predict_proba(X), before)


@pytest.mark.parametrize("names", [[0, 1, 2], ["a", "b", "c"]])
def test_dataframe_names_checked_before_encoding(task, names):
    pd = pytest.importorskip("pandas")
    X, y = task
    frame = pd.DataFrame(X, columns=names)
    frame[names[0]] = np.where(X[:, 0] > 0, "yes", "no")
    clf = LightPFNClassifier(model=small_model(), device="cpu").fit(frame, y)
    for other in (frame.iloc[:, ::-1], frame.rename(columns={names[0]: "other"}), frame.iloc[:, :2]):
        with pytest.raises(ValueError, match="feature names"):
            clf.predict_proba(other)
    # A refit on an array resets the DataFrame schema.
    clf.fit(X, y)
    with pytest.warns(UserWarning, match="feature names"):
        np.testing.assert_array_equal(clf.predict_proba(pd.DataFrame(X, columns=["x", "y", "z"])),
                                      clf.predict_proba(X))


def test_nullable_and_mixed_categorical_values():
    pd = pytest.importorskip("pandas")
    from lightpfn.sklearn import _encode_frame, _is_categorical

    frame = pd.DataFrame({
        "text": pd.array(["z", "a", pd.NA, "z"], dtype="string"),
        "flag": pd.array([True, False, pd.NA, True], dtype="boolean"),
        "mixed": pd.Series([1, "1", None, 2], dtype=object),
        "empty": pd.Series([pd.NA, None, np.nan, None], dtype=object),
        "number": pd.array([1.5, pd.NA, 2.5, 0], dtype="Float64"),
        "ordered": pd.Categorical(["b", "a", None, "b"], categories=["b", "unused", "a"], ordered=True),
    })
    categories = {j: frame.iloc[:, j].astype("category").cat.categories
                  for j, dtype in enumerate(frame.dtypes) if _is_categorical(dtype)}
    expected = np.array([[1, 1, 0, np.nan, 1.5, 0], [0, 0, 2, np.nan, np.nan, 2],
                         [np.nan, np.nan, np.nan, np.nan, 2.5, np.nan], [1, 1, 1, np.nan, 0, 0]], np.float32)
    np.testing.assert_array_equal(_encode_frame(frame, categories), expected)
    query = frame.iloc[:2].copy()
    query["mixed"] = ["unseen", 1]
    query["empty"] = ["unseen", None]
    query["ordered"] = ["unused", "unseen"]
    encoded = _encode_frame(query, categories)
    assert np.isnan(encoded.mixed.iloc[0]) and encoded.mixed.iloc[1] == 0
    assert encoded["empty"].isna().all()
    assert encoded.ordered.iloc[0] == 1 and np.isnan(encoded.ordered.iloc[1])


@pytest.mark.parametrize("fold", [False, True])
def test_fitted_dataframe_pickle_and_cpu_transfer(task, fold):
    pd = pytest.importorskip("pandas")
    X, y = task
    frame = pd.DataFrame({"x": X[:, 0], "kind": pd.array(np.where(X[:, 1] > 0, "a", "b"), dtype="string")})
    clf = LightPFNClassifier(model=small_model(), device="cpu", fold=fold,
                             n_estimators=2, batch_cells=1000).fit(frame, y)
    expected = clf.predict_proba(frame)
    restored = pickle.loads(pickle.dumps(clf))
    assert restored.to("cpu") is restored
    np.testing.assert_array_equal(restored.predict_proba(frame), expected)
    # Movement invalidates the loader key; the next fit resolves the constructor device.
    assert restored._backend_key is None
    restored.fit(frame, y)
    np.testing.assert_array_equal(restored.predict_proba(frame), expected)


@pytest.mark.parametrize("cache_context", [False, True])
def test_dataframe_adapter_matches_explicit_mask(task, cache_context):
    pd = pytest.importorskip("pandas")
    from lightpfn import Config, LightPFN
    from lightpfn.sklearn import _encode_frame

    X, y = task
    cfg = asdict(small_model().cfg)
    model = LightPFN(Config(**{**cfg, "cat_adapter": True})).eval()
    # Nonzero adapter projections ensure the test exercises categorical statistics.
    with torch.no_grad():
        for p in model.parameters():
            torch.nn.init.normal_(p, std=0.05)
    frame = pd.DataFrame({"x": X[:, 0], "kind": pd.array(np.where(X[:, 1] > 0, "a", "b"), dtype="string")})
    inferred = LightPFNClassifier(model=model, device="cpu", n_estimators=2, cache_context=cache_context).fit(frame, y)
    explicit = LightPFNClassifier(model=model, device="cpu", n_estimators=2, cache_context=cache_context).fit(
        _encode_frame(frame, inferred.categories_).to_numpy(), y, cat=np.array([False, True]))
    assert inferred.cat_.tolist() == [False, True]
    query = frame.iloc[:4].copy()
    query["kind"] = ["a", "b", "unseen", pd.NA]
    np.testing.assert_array_equal(inferred.predict_proba(query),
                                  explicit.predict_proba(_encode_frame(query, inferred.categories_).to_numpy()))


def test_loader_rebuilds_after_network_removed(task, tmp_path, monkeypatch):
    from lightpfn import save_model
    import lightpfn.sklearn as wrapper

    X, y = task
    path = save_model(small_model(), tmp_path / "weights")
    clf = LightPFNClassifier(checkpoint=path, device="cpu").fit(X, y)
    expected = clf.predict_proba(X)
    old_key = clf._backend_key
    old_net = clf.net_
    del clf.model_, clf.net_
    calls = []
    loader = wrapper.load_model

    def load(*args, **kwargs):
        calls.append(args)
        return loader(*args, **kwargs)

    monkeypatch.setattr(wrapper, "load_model", load)
    clf.fit(X, y)
    assert len(calls) == 1 and clf._backend_key == old_key and clf.net_ is not old_net
    np.testing.assert_array_equal(clf.predict_proba(X), expected)
    clf.fit(X, y)
    assert len(calls) == 1  # ordinary refits reuse their network


@pytest.mark.parametrize("fold,batch_cells", [(False, 0), (True, 10000)])
def test_uncached_context_matches_cached_after_pickle_and_transfer(task, fold, batch_cells):
    X, y = task
    params = dict(model=small_model(), device="cpu", n_estimators=3, max_context=17,
                  random_state=4, fold=fold, batch_cells=batch_cells, chunk_rows=5)
    cached = LightPFNClassifier(**params).fit(X, y)
    uncached = LightPFNClassifier(**params, cache_context=False).fit(X, y)
    assert uncached.members_ == []
    query = X[:11].copy()
    expected = cached.predict_proba(query)
    np.testing.assert_array_equal(uncached.predict_proba(query), expected)
    assert uncached.members_ == []
    restored = pickle.loads(pickle.dumps(uncached)).to("cpu")
    np.testing.assert_array_equal(restored.predict_proba(query), expected)
    assert restored.members_ == []
    # The fitted estimator owns its inputs, even without an encoded cache.
    X[:] = 123
    y[:] = "no"
    np.testing.assert_array_equal(uncached.predict_proba(query), expected)
    assert not np.all(uncached._uncached_data[0] == 123)


def test_uncached_context_is_released_on_prediction_error(task, monkeypatch):
    X, y = task
    clf = LightPFNClassifier(model=small_model(), device="cpu", cache_context=False).fit(X, y)
    predict = clf.fit_net_.predict_logits

    def fail(*args, **kwargs):
        raise RuntimeError("prediction failure")

    monkeypatch.setattr(clf.fit_net_, "predict_logits", fail)
    with pytest.raises(RuntimeError, match="prediction failure"):
        clf.predict_proba(X[:3])
    assert clf.members_ == []
    monkeypatch.setattr(clf.fit_net_, "predict_logits", predict)
    assert clf.predict_proba(X[:3]).shape == (3, 2)
    clf.set_params(cache_context=True).fit(X, y)
    assert clf._uncached_data is None and clf.members_
