"""Tree-based classification prior in the spirit of Mitra's fitted-tree priors.

Mitra (Amazon, 2025) found that mixing an SCM prior with priors whose p(y|x) comes from fitted tree
ensembles gives the largest gains (SCM + ExtraTrees + GradientBoosting: +76 Elo over SCM alone),
while directly sampled random trees add little because they overlap with the SCM. The TabICLv2
graph prior already contains random oblivious trees, so here the trees are *fitted*:

  1. features from a Gaussian copula: correlated columns with mixed marginals (normal, uniform,
     heavy-tailed, log-normal, Zipf-like counts, low-cardinality ordinal categories)
  2. a signal: a random sparse function of a few features (linear, pairwise products, thresholds
     or a small random MLP) plus noise, one output per class
  3. an ExtraTrees or gradient boosting ensemble fitted to the signal with sampled depth/size on
     an independent sample of the same feature distribution; its axis-aligned, piecewise-constant
     predictions on new rows are the class logits (as in Mitra's Algorithm 2, so the labels do not
     inherit the fit's overfitting to its own rows)
  4. labels: argmax of the logits + Gumbel noise (random signal-to-noise), with per-class biases
     balanced so every class is present; or, for ordinal targets, quantile cuts of one score

`generate_tree_task` returns (X, y) like the graph prior; class imbalance, the train/test split
and grouping are applied by lightpfn.prior.generate as for the SCM tasks.
"""

import math

import numpy as np
from scipy.special import ndtr
from scipy.stats import t as student_t


def _loguniform(rng, lo, hi):
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def sample_features(rng, n, d):
    # correlation from a random low-rank factor model
    rank = int(rng.integers(1, max(2, d // 2) + 1))
    L = rng.normal(size=(d, rank)) * rng.uniform(0.0, 1.5)
    Z = rng.normal(size=(n, rank)) @ L.T + rng.normal(size=(n, d))
    # Population variance, not the realized variance of the combined fit/task rows:
    # each sample is now independent conditional on the shared distribution parameters.
    Z /= np.sqrt((L**2).sum(1) + 1)[None, :]
    U = np.clip(ndtr(Z), 1e-6, 1 - 1e-6)

    X = np.empty((n, d))
    kinds = rng.choice(["normal", "uniform", "t", "lognormal", "zipf", "cat"], size=d, p=[0.3, 0.1, 0.15, 0.15, 0.1, 0.2])
    for j, kind in enumerate(kinds):
        u = U[:, j]
        if kind == "normal":
            X[:, j] = Z[:, j]
        elif kind == "uniform":
            X[:, j] = u
        elif kind == "t":  # heavy tails
            X[:, j] = student_t.ppf(u, rng.uniform(1.5, 5.0))
        elif kind == "lognormal":
            X[:, j] = np.exp(Z[:, j] * rng.uniform(0.3, 1.5))
        elif kind == "zipf":  # skewed counts
            X[:, j] = np.floor((1 - u) ** (-1 / rng.uniform(0.5, 2.0)))
        else:  # ordinal codes of a categorical column, categories in random order
            k = int(rng.integers(2, 11))
            X[:, j] = rng.permutation(k)[np.minimum((u * k).astype(int), k - 1)]
    return X, kinds


def sample_signal(rng, X, n_out):
    """Random sparse function of a few standardized features, (n, n_out)."""
    n, d = X.shape
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    k = int(round(_loguniform(rng, 1, min(d, 8) + 0.999)))
    S = rng.choice(d, size=min(k, d), replace=False)
    Xk = Xs[:, S]
    kind = rng.choice(["linear", "product", "threshold", "mlp"], p=[0.1, 0.3, 0.35, 0.25])
    if kind == "linear":
        F = Xk @ rng.normal(size=(len(S), n_out))
    elif kind == "product":
        a, b = rng.integers(0, len(S), size=(2, n_out))
        F = Xk[:, a] * Xk[:, b] + 0.5 * Xk @ rng.normal(size=(len(S), n_out))
    elif kind == "threshold":
        t = rng.normal(size=len(S))
        F = (Xk > t).astype(float) @ rng.normal(size=(len(S), n_out))
    else:
        h = int(rng.integers(4, 33))
        H = np.tanh(Xk @ rng.normal(size=(len(S), h)) * rng.uniform(0.5, 2.0))
        F = H @ rng.normal(size=(h, n_out))
    F /= F.std(0, keepdims=True) + 1e-9
    # 0 = the trees fit a clean signal; larger = the partition follows noise (fully random
    # partitions of deep trees give unlearnable labels, so noise stays below 0.6)
    noise = rng.uniform(0.0, 0.6)
    return math.sqrt(1 - noise**2) * F + noise * rng.normal(size=F.shape)


def fit_trees(rng, X_fit, T, X):
    """Fits a tree ensemble to the signal T on X_fit and returns its standardized predictions on X."""
    kind = rng.choice(["extra_trees", "gbdt"])
    seed = int(rng.integers(2**31))
    if kind == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        model = ExtraTreesRegressor(
            n_estimators=int(round(_loguniform(rng, 1, 64))),
            max_depth=int(rng.integers(2, 13)),
            min_samples_leaf=int(round(_loguniform(rng, 1, 30))),
            max_features=float(rng.uniform(0.2, 1.0)),
            n_jobs=1,
            random_state=seed,
        )
    else:
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=int(round(_loguniform(rng, 5, 200))),
            max_depth=int(rng.integers(1, 7)),
            learning_rate=_loguniform(rng, 0.03, 0.5),
            subsample=float(rng.uniform(0.5, 1.0)),
            colsample_bytree=float(rng.uniform(0.3, 1.0)),
            tree_method="hist",
            multi_strategy="multi_output_tree",
            n_jobs=1,
            random_state=seed,
        )
    model.fit(X_fit, T if T.shape[1] > 1 else T[:, 0])
    P = model.predict(X).reshape(len(X), -1)
    return (P - P.mean(0)) / (P.std(0) + 1e-9), kind


def _balanced_argmax(rng, L, target, iters=20):
    """argmax(L + b) with biases b adjusted so class frequencies approach `target`."""
    b = np.zeros(L.shape[1])
    for _ in range(iters):
        y = np.argmax(L + b, 1)
        freq = np.bincount(y, minlength=L.shape[1]) / len(y)
        b -= 0.5 * np.log((freq + 1e-3) / target)
    return np.argmax(L + b, 1)


def generate_tree_task(rng, n, d, n_classes):
    """One classification task: X (n, d) float32, y (n,) int in 0..n_classes-1."""
    if n < 2 * n_classes or d < 1 or not 2 <= n_classes <= 10:
        raise ValueError("tree task requires d>=1, 2<=n_classes<=10, n>=2*n_classes")
    # one draw of the feature distribution: the first n_fit rows train the generator trees,
    # the task's own rows are new draws from the same distribution
    fit_min = max(32, n // 2)
    n_fit = int(round(_loguniform(rng, fit_min, max(fit_min, 2 * n))))
    X_all, _ = sample_features(rng, n_fit + n, d)
    X_fit, X = X_all[:n_fit], X_all[n_fit:]
    ordinal = rng.random() < 0.15  # quantile cuts give near-equal classes: a minority of the targets
    T = sample_signal(rng, X_fit, 1 if ordinal else n_classes)
    P, _ = fit_trees(rng, X_fit, T, X)
    snr = _loguniform(rng, 1.0, 30.0)  # logit scale: low = noisy labels, high = near-deterministic
    target = rng.dirichlet(np.full(n_classes, 5.0))  # roughly balanced; imbalance is applied later
    if ordinal:
        score = P[:, 0] * snr + rng.gumbel(size=n)
        cuts = np.quantile(score, np.cumsum(target)[:-1])
        y = np.searchsorted(cuts, score)
    else:
        y = _balanced_argmax(rng, P * snr + rng.gumbel(size=P.shape), target)
    # standardized like the graph prior's output; pools are stored in float16
    # The shared wrapper rejects out-of-contract values; clipping here would conceal
    # a numerical failure instead of sending that draft through structural validation.
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return X.astype(np.float32), y.astype(np.int64)
