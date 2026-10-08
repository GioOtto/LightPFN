"""Versioned observed-feature prior and mechanism-aware rule tasks (CPU only)."""
import math

import numpy as np
import torch

# Persist the entire policy in meta.json; changing it requires a new prior version.
CONFIG = dict(
    unclipped_modes=["all", "none", "half"], unclipped_probabilities=[0.80, 0.05, 0.15],
    highcat_per_existing_cat=0.40, highcat_min=16, highcat_max=256, zipf_exponent=[0.8, 1.3],
    sink_target_probability=0.90, softmax_only_probability=0.45, trivial_oob_threshold=0.99, trivial_accept_probability=0.35,
    multiclass_alpha=[4.0, 20.0], multiclass_balanced_probability=0.58,
    rule_families=["xor", "parity", "tree", "lookup", "product", "and", "or"],
    rule_probabilities=[0.25, 0.20, 0.20, 0.10, 0.10, 0.075, 0.075],
    rule_driver_range=[2, 4], rule_quantiles=[0.3, 0.7], rule_depths=[2, 3],
    rule_noise=[0.0, 0.15], rule_mix_probability=0.25, rule_graph_weight=[0.0, 0.5],
    rule_weight=[0.75, 1.5], rule_mix_temperature=0.125, rule_min_oracle_accuracy=0.70,
    interaction_driver_policy="independent conditional marginal draws, balanced factorial cells",
    rule_multiclass_interaction="modular sum of independent uniform C-ary digits",
    rule_resampling=False, graph_logit_proxy="centered one-hot graph category",
)


def quantize_column(rng, column, k, exponent=1.0):
    """Deterministic value -> code map with Zipf bin sizes, ties never split.

    Cuts are observed values, at most k bins; permuting the labels removes ordinal meaning.
    Values remain exact float16 integers. Empty bins caused by tied quantiles are removed.
    """
    x = np.asarray(column, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or not 2 <= k <= 256:
        raise ValueError("invalid quantization input/cardinality")
    levels = np.unique(x)
    k = min(k, len(levels))
    if k < 2:
        return x.astype(np.float32)
    p = np.arange(1, k + 1, dtype=float) ** -exponent
    p /= p.sum()
    # Randomize where large/rare bins occur along the original numeric axis.
    p = p[rng.permutation(k)]
    cuts = np.unique(np.quantile(x, np.cumsum(p)[:-1], method="inverted_cdf"))
    bins = np.searchsorted(cuts, x, side="left")
    _, bins = np.unique(bins, return_inverse=True)
    codes = rng.permutation(int(bins.max()) + 1)
    if len(codes) > 2 and (np.all(np.diff(codes) > 0) or np.all(np.diff(codes) < 0)):
        codes[[0, 1]] = codes[[1, 0]]
    return codes[bins].astype(np.float32)


def preprocess_graph(X, data):
    from tabicl.prior._reg2cls import outlier_removing

    # This RNG is derived from the seeded legacy NumPy stream; no wall-clock state.
    rng = np.random.default_rng(np.random.randint(2**32))
    ncat = data.get("x_cat", torch.empty((len(X), 0))).shape[1]
    cat = np.arange(X.shape[1]) < ncat
    numerical = np.flatnonzero(~cat)
    mode = str(rng.choice(CONFIG["unclipped_modes"], p=CONFIG["unclipped_probabilities"]))
    trim = numerical if mode == "none" else np.array([], dtype=int)
    if mode == "half":
        trim = rng.choice(numerical, size=len(numerical) // 2, replace=False)
    if len(trim):
        X[:, trim] = outlier_removing(X[:, trim], threshold=4)
    # Scale in float64, including the mean/std. Avoid overflow in float32 variance
    # and preserve exact failures instead of turning +/-Inf into finite clipping.
    a = X.double().numpy()
    if np.isfinite(a).all():
        a = (a - a.mean(0)) / np.maximum(a.std(0, ddof=1 if len(a) > 1 else 0), 1e-6)
    # Allow for tied quantiles and rare levels lost at row selection.
    eligible = [j for j in numerical if len(np.unique(a[:, j])) >= CONFIG["highcat_min"]]
    target = ncat * CONFIG["highcat_per_existing_cat"]
    nnew = min(len(eligible), int(target) + int(rng.random() < target % 1))
    if nnew and np.isfinite(a).all():
        for j in rng.choice(eligible, nnew, replace=False):
            k = int(round(math.exp(rng.uniform(math.log(16), math.log(256)))))
            a[:, j] = quantize_column(rng, a[:, j], k, rng.uniform(*CONFIG["zipf_exponent"]))
            cat[j] = True
    return torch.from_numpy(a.astype(np.float32)), dict(cat_mask=cat, unclipped_mode=mode)


def filter_metrics(X, y, cfg):
    """Same 25-tree OOB/Brier/bootstrap filter as TabICL, returning its existing AUC.

    No additional fit. Multiclass AUC averages per-class OOB score rankings; unlike
    probability logloss it does not require the regressor's scores to sum to one.
    """
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.metrics import roc_auc_score

    C = int(np.max(y)) + 1
    Y = np.eye(C, dtype=np.float32)[y]
    if C == 2:
        Y = Y[:, :1]
    et = ExtraTreesRegressor(n_estimators=25, bootstrap=True, oob_score=True,
                            n_jobs=1, random_state=1, max_depth=6)
    et.fit(np.asarray(X, dtype=np.float32), Y[:, 0] if C == 2 else Y)
    pred = et.oob_prediction_.reshape(len(y), -1)
    mask = ~np.isnan(pred).any(1)
    Yv, Pv = Y[mask], pred[mask]
    if not len(Yv):
        return dict(reject=True, oob_auc=0.5, pvalue=1.0)
    base = Y.mean(0, keepdims=True)
    bmse = ((Yv - base)**2).sum(1)
    emse = ((Yv - Pv)**2).sum(1)
    imp = bmse - emse
    idx = np.random.default_rng(0).integers(0, len(imp), (200, len(imp)))
    pvalue = float(np.mean(imp[idx].mean(1) <= 0))
    aucs = [roc_auc_score(Yv[:, j], Pv[:, j]) for j in range(Y.shape[1])
            if len(np.unique(Yv[:, j])) == 2]
    reject = (cfg.filter_unpredictable_datasets and pvalue >= 0.05) or (
        cfg.remove_trivial_datasets and np.sqrt(emse.mean()) <= cfg.trivial_dataset_threshold * np.sqrt(bmse.mean()))
    return dict(reject=bool(reject), oob_auc=float(np.mean(aucs)) if aucs else 0.5, pvalue=pvalue)


def _proportions(rng, C):
    from lightpfn.prior.generate import _target_proportions
    if C == 2:
        return _target_proportions(rng, C)
    alpha = math.exp(rng.uniform(*np.log(CONFIG["multiclass_alpha"])))
    return rng.dirichlet(np.full(C, alpha))


def generate_modern_task(spec):
    from lightpfn.prior import generate as g
    n, d, C = (spec[k] for k in ("n_rows", "n_features", "n_classes"))
    if (any(not isinstance(v, (int, np.integer)) for v in (n, d, C))
            or not 1 <= d <= np.iinfo(np.int16).max or not 2 <= C <= g.MAX_CLASSES or n < 2 * C):
        raise ValueError("invalid task specification")
    if spec["prior"] == "rule":
        from lightpfn.prior.rules import generate_rule_task
        return generate_rule_task(spec)
    rng = g._seed_task(spec["seed"])
    requested = rng.random() < g.P_RESAMPLE
    n_gen = 2 * n if requested else n
    rejected = invalid = checks = trivial = balance_rejected = 0
    # Mild multiclass tasks are an explicit mixture, tested AFTER fallback. C=10
    # cannot have min>=10% unless n is divisible by 10, so use the other component.
    balanced = C > 2 and C < 10 and rng.random() < CONFIG["multiclass_balanced_probability"]
    spec = dict(spec, balanced_target=bool(balanced))
    for draft in range(1, g.MAX_TASK_DRAFTS + 1):
        X, y, info = g._graph_dataset(spec, n_gen)
        clean = g._clean_candidate(X, y, spec)
        if clean is None:
            invalid += 1
            continue
        keep = X.astype(np.float16).min(0) != X.astype(np.float16).max(0)
        cat = info["cat_mask"][keep]
        X, y = clean
        counts = np.bincount(y, minlength=C)
        if np.any(counts < 2):
            invalid += 1
            continue
        perm = rng.permutation(len(y))
        X, y = X[perm], y[perm]
        checks += 1
        cfg = info["config"]
        metric = g._unpredictable(X, y, cfg, return_metrics=True)
        easy = metric["oob_auc"] >= CONFIG["trivial_oob_threshold"]
        refuse_easy = easy and rng.random() >= CONFIG["trivial_accept_probability"]
        if metric["reject"] or refuse_easy:
            rejected += 1
            trivial += int(refuse_easy and not metric["reject"])
            if rejected > g.FILTER_TRIES:
                raise RuntimeError(f"predictability retries exhausted: {spec}")
            continue
        take = g._quotas(_proportions(rng, C), n) if requested else counts.copy()
        if balanced and requested:
            # Balanced component uses >=10% reserves, remainder allocated uniformly.
            take = g._quotas(np.ones(C), n, minimum=math.ceil(.1*n))
        if requested:
            take[np.argsort(counts, kind="stable")] = np.sort(take)
        resampled = bool(requested and np.all(take <= counts))
        if resampled:
            idx = g._select_rows(rng, y, take)
        else:
            reserved = g._select_rows(rng, y, np.full(C, 2))
            available = np.ones(len(y), dtype=bool)
            available[reserved] = False
            idx = rng.permutation(np.r_[reserved, rng.choice(np.flatnonzero(available), n-2*C, replace=False)])
        X, y = X[idx], y[idx]
        if balanced and np.bincount(y, minlength=C).min() / n < .1:
            invalid += 1
            balance_rejected += 1
            continue
        keep = X.min(0) != X.max(0)
        X, cat = X[:, keep], cat[keep]
        if not X.shape[1]:
            invalid += 1
            continue
        y = rng.permutation(C)[y]
        return dict(X=X, y=y.astype(np.uint8), n_features=d, resampled=resampled,
                    resample_requested=bool(requested), rejected=rejected, invalid=invalid,
                    drafts=draft, filter_checks=checks, cat_mask=cat,
                    unclipped_mode=info["unclipped_mode"], trivial_rejected=trivial,
                    balance_rejected=balance_rejected, oob_auc=metric["oob_auc"], balanced_target=bool(balanced))
    raise RuntimeError(f"structural retries exhausted: {spec}, invalid={invalid}, rejected={rejected}")
