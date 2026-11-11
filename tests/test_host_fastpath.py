"""Bulk host conversions (2026-11-11) against the per-column reference implementations of 1.5.0."""
import numpy as np
import pandas as pd
import pytest

from lightpfn.sklearn import _encode_frame, _is_categorical, _is_frame, auto_categorical


def ref_auto(X, max_levels=20):
    frame = _is_frame(X)
    if not frame:
        X = np.asarray(X)
    out = np.zeros(X.shape[1], bool)
    dtypes = X.dtypes if frame else None
    for j in range(X.shape[1]):
        if frame and _is_categorical(dtypes.iloc[j]):
            out[j] = True
            continue
        if frame:
            column, dtype = X.iloc[:, j], dtypes.iloc[j]
            v = (column.to_numpy() if isinstance(dtype, np.dtype) and dtype.kind in "biuf"
                 else column.dropna().to_numpy())
        else:
            v = X[:, j]
        if v.dtype.kind not in "biuf":
            v = np.asarray(v, dtype=np.float64)
        if v.dtype.kind == "f":
            v = v[~np.isnan(v)]
            if not np.isfinite(v).all() or not np.all(v == np.floor(v)):
                continue
        if not len(v):
            continue
        lo, hi = int(v.min()), int(v.max())
        if lo == hi:
            continue
        if hi - lo < max_levels:
            out[j] = True
            continue
        out[j] = len(np.unique(v)) <= max_levels
    return out


def ref_encode(X, categories):
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


def frame(rng, n=300):
    X = pd.DataFrame({
        "f_small": rng.integers(0, 5, n).astype(float),
        "f_nan": np.where(rng.random(n) < .2, np.nan, rng.integers(0, 7, n)).astype(np.float64),
        "f_real": rng.normal(size=n),
        "f_inf": np.where(np.arange(n) == 3, np.inf, rng.integers(0, 3, n)).astype(float),
        "f_allnan": np.full(n, np.nan),
        "f_const": np.full(n, 2.0),
        "f_sparse": rng.choice([0., 100., 5000., 7.], n),
        "f_wide": rng.choice(np.arange(0., 300., 10.), n),
        "f32": rng.integers(0, 4, n).astype(np.float32),
        "i64_small": rng.integers(-3, 8, n),
        "i64_big": rng.integers(0, 10**6, n),
        "i32_sparse": rng.choice([1, 1000, 99999], n).astype(np.int32),
        "u8": rng.integers(0, 2, n).astype(np.uint8),
        "i64_huge": rng.choice([2**60, 2**60 + 3], n),
        "Int64": pd.array(np.where(rng.random(n) < .1, None, rng.integers(0, 4, n)), dtype="Int64"),
        "Float64": pd.array(rng.integers(0, 40, n).astype(float), dtype="Float64"),
        "cat": pd.Categorical(rng.choice(list("abc"), n)),
        "obj": rng.choice(["x", "y", None], n).astype(object),
        "bool": rng.random(n) < .5,
        "str": pd.array(rng.choice(["u", "v"], n), dtype="string"),
    })
    return X


@pytest.mark.parametrize("seed", range(4))
def test_auto_categorical_matches_reference(seed):
    X = frame(np.random.default_rng(seed))
    assert (auto_categorical(X) == ref_auto(X)).all()
    assert (auto_categorical(X, 3) == ref_auto(X, 3)).all()
    num = X.select_dtypes("number").drop(columns=["Int64", "Float64"]).to_numpy(np.float64)
    assert (auto_categorical(num) == ref_auto(num)).all()
    assert (auto_categorical(X.iloc[:0]) == ref_auto(X.iloc[:0])).all()


@pytest.mark.parametrize("seed", range(4))
def test_encode_frame_matches_reference(seed):
    rng = np.random.default_rng(seed)
    X = frame(rng)
    X.index = rng.permutation(len(X)) + 10
    cats = {j: X.iloc[:, j].astype("category").cat.categories
            for j, dtype in enumerate(X.dtypes) if _is_categorical(dtype)}
    new, ref = _encode_frame(X, cats), ref_encode(X, cats)
    pd.testing.assert_frame_equal(new, ref)
    unseen = X.copy()
    unseen["cat"] = pd.Categorical(rng.choice(list("abd"), len(X)))
    pd.testing.assert_frame_equal(_encode_frame(unseen, cats), ref_encode(unseen, cats))


def test_categorical_codes_and_categories_match_reference():
    from lightpfn.sklearn import _fit_categories

    rng = np.random.default_rng(7)
    n = 200
    fit = pd.DataFrame({
        "c": pd.Categorical(rng.choice(list("abc"), n), categories=list("cbaz")),  # unused level, own order
        "num": rng.integers(0, 3, n),
        "ci": pd.Categorical(rng.choice([3, 1, 2], n)),
        "o": rng.choice(["p", "q"], n).astype(object),
        "e": pd.Categorical([None] * n, categories=[]),
    })
    cats = _fit_categories(fit)
    ref = {j: fit.iloc[:, j].astype("category").cat.categories for j, d in enumerate(fit.dtypes) if _is_categorical(d)}
    assert cats.keys() == ref.keys() and all(cats[j].equals(ref[j]) for j in ref)
    test = pd.DataFrame({
        "c": pd.Categorical(rng.choice(list("abdz") + [None], n), categories=list("zdab")),
        "num": rng.integers(0, 3, n),
        "ci": pd.Categorical(rng.choice([1.0, 2.0, 7.0], n)),
        "o": pd.Categorical(rng.choice(["q", "r"], n)),
        "e": pd.Categorical(rng.choice(["x", None], n)),
    })
    for frame in (fit, test):
        pd.testing.assert_frame_equal(_encode_frame(frame, cats), ref_encode(frame, cats))
