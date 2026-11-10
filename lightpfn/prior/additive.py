"""Additive N1 prior: many comparable fine effects of nominal levels and numerical intervals (v2 data, R3 §2).

A task draws k drivers (1-16) among its d columns. Each driver has one unit-variance component per class: an effect
per level of a nominal column, a step function over 4-512 intervals or a smooth tanh sum of a numerical column. The
components add up with Dirichlet(3) variance shares into one logit per class, scaled by a signal-to-noise factor, and
the class biases are calibrated to the target class proportions. Centering, scaling and calibration use an independent
calibration sample of the same population, never the task's own rows. The other columns are distractors: numerical,
or nominal with many levels and no effect. Level IDs are stored exactly (int16 cat_codes); there is no learned filter.
Every random draw comes from the task's seed, so a task is fully determined by its specification.
"""

import math

import numpy as np

# Persist the entire policy in meta.json; changing it requires a new prior version.
CONFIG = dict(
    kinds=["nominal", "numeric", "mixed"], kind_probabilities=[0.4, 0.3, 0.3],
    driver_counts=[1, 2, 4, 8, 16], many_drivers_probability=0.5,  # then only counts >= 4 (when d allows)
    share_alpha=3.0,
    levels=[16, 10000], distractor_levels=[64, 10000], uniform_vocabulary_probability=0.25,
    zipf_exponent=[0.8, 1.5], level_effects=["normal", "t3"],
    intervals=[4, 512], smooth_probability=0.5, random_walk_probability=0.5,
    smooth_units=[2, 8], smooth_slope=[0.5, 5.0],
    marginals=["normal", "uniform", "lognormal", "student_t5"],
    nominal_distractor_probability=0.3,
    logit_scale=[0.5, 8.0],
    binary_minority=[0.02, 0.5], multiclass_alpha=[0.5, 5.0], min_class_share=0.02,
    calibration_rows=4096, calibration_iterations=200, label_retries=8,
)
MAX_DRAFTS = 32
MAX_LEVEL_ID = np.iinfo(np.int16).max  # stored as int16 cat_codes


def _log_uniform(rng, low, high):
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def _numeric_column(rng, size):
    kind = str(rng.choice(CONFIG["marginals"]))
    if kind == "normal":
        x = rng.standard_normal(size)
    elif kind == "uniform":
        x = rng.uniform(-1, 1, size)
    elif kind == "lognormal":
        x = np.exp(rng.normal(0, rng.uniform(.5, 1), size))
    else:
        x = rng.standard_t(5, size)
    return x, kind


def _vocabulary(rng, low, high):
    """Level probabilities of a nominal column; IDs are a random permutation of the frequency ranks."""
    L = int(round(_log_uniform(rng, low, high)))
    if rng.random() < CONFIG["uniform_vocabulary_probability"]:
        p = np.full(L, 1 / L)
    else:
        p = np.arange(1, L + 1, dtype=float) ** -rng.uniform(*CONFIG["zipf_exponent"])
        p /= p.sum()
    return p[rng.permutation(L)]


def _effects(rng, size, C):
    if rng.random() < .5:
        return rng.standard_normal((size, C))
    return rng.standard_t(3, (size, C))


def _component(rng, x_all, cal, C, levels=None):
    """(rows, C) values of one driver's component, and its description. x_all holds the task rows, then the
    calibration rows; levels: vocabulary size of a nominal column (x_all are its level IDs)."""
    if levels is not None:
        return _effects(rng, levels, C)[x_all], dict(type="levels")
    if rng.random() < CONFIG["smooth_probability"]:
        U = int(rng.integers(CONFIG["smooth_units"][0], CONFIG["smooth_units"][1] + 1))
        z = (x_all - cal.mean()) / max(cal.std(), 1e-12)
        centers = np.quantile((cal - cal.mean()) / max(cal.std(), 1e-12), rng.uniform(.05, .95, U))
        slopes = np.array([_log_uniform(rng, *CONFIG["smooth_slope"]) for _ in range(U)])
        weights = rng.standard_normal((U, C))
        return np.tanh(slopes * (z[:, None] - centers))@weights, dict(type="smooth", units=U)
    m = int(round(_log_uniform(rng, *CONFIG["intervals"])))
    cuts = np.unique(np.quantile(cal, np.sort(rng.uniform(0, 1, m - 1))))
    walk = rng.random() < CONFIG["random_walk_probability"]
    effects = _effects(rng, len(cuts) + 1, C)
    if walk:
        effects = np.cumsum(effects, 0)
    return effects[np.searchsorted(cuts, x_all, side="right")], dict(type="steps", intervals=len(cuts) + 1, walk=bool(walk))


def _proportions(rng, C):
    if C == 2:
        p = _log_uniform(rng, *CONFIG["binary_minority"])
        return np.array([p, 1 - p])
    alpha = _log_uniform(rng, *CONFIG["multiclass_alpha"])
    m = CONFIG["min_class_share"]
    return rng.dirichlet(np.full(C, alpha)) * (1 - C * m) + m


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _calibrate(logits, target):
    """Class biases that give the target mean class probabilities on the calibration rows."""
    b = np.log(target)
    for _ in range(CONFIG["calibration_iterations"]):
        b = b + np.log(target / _softmax(logits + b).mean(0))
    return b


def _draft(rng, n, d, C):
    kind = str(rng.choice(CONFIG["kinds"], p=CONFIG["kind_probabilities"]))
    counts = [k for k in CONFIG["driver_counts"] if k <= d]
    if d >= 4 and rng.random() < CONFIG["many_drivers_probability"]:
        counts = [k for k in counts if k >= 4]
    k = int(rng.choice(counts))
    if kind == "mixed" and k >= 2:
        # At least one nominal and one numerical driver, the others either with probability 1/2.
        nominal = rng.permutation(np.r_[[True, False], rng.random(k - 2) < .5])
    else:
        nominal = np.full(k, kind == "nominal" or (kind == "mixed" and rng.random() < .5))
    distractor_nominal = rng.random(d - k) < CONFIG["nominal_distractor_probability"]
    is_nominal = np.r_[nominal, distractor_nominal]
    total = n + CONFIG["calibration_rows"]
    columns, codes, info = [], [], []
    for j in range(d):
        if is_nominal[j]:
            low, high = CONFIG["levels"] if j < k else CONFIG["distractor_levels"]
            p = _vocabulary(rng, low, min(high, MAX_LEVEL_ID + 1))
            ids = rng.choice(len(p), size=total, p=p)
            columns.append(ids)
            codes.append(ids)
            info.append(dict(levels=len(p)))
        else:
            x, marginal = _numeric_column(rng, total)
            cal = x[n:]
            # Standardize on the calibration sample, then quantize to the stored float16 values first: the
            # components are functions of exactly the values the model sees.
            x = ((x - cal.mean()) / max(cal.std(), 1e-12)).astype(np.float16).astype(np.float64)
            columns.append(x)
            codes.append(None)
            info.append(dict(marginal=marginal))
    shares = rng.dirichlet(np.full(k, CONFIG["share_alpha"]))
    F = np.zeros((total, C))
    components = []
    for j in range(k):
        g, desc = _component(rng, columns[j], columns[j][n:], C, info[j]["levels"] if is_nominal[j] else None)
        cal = g[n:]
        std = cal.std(0)
        if np.any(std < 1e-6):
            return None
        F += math.sqrt(shares[j]) * (g - cal.mean(0)) / std
        components.append(dict(desc, **info[j], share=float(shares[j])))
    scale = _log_uniform(rng, *CONFIG["logit_scale"])
    target = _proportions(rng, C)
    logits = scale * F
    bias = _calibrate(logits[n:], target)
    P = _softmax(logits[:n] + bias)
    return dict(kind=kind, k=k, is_nominal=is_nominal, columns=[c[:n] for c in columns],
                codes=[None if c is None else c[:n] for c in codes], P=P, scale=scale, target=target,
                components=components)



def generate_additive_task(spec):
    from lightpfn.prior import generate as g

    n, d, C = (spec[k] for k in ("n_rows", "n_features", "n_classes"))
    if (any(not isinstance(v, (int, np.integer)) for v in (n, d, C))
            or not 1 <= d <= np.iinfo(np.int16).max or not 2 <= C <= g.MAX_CLASSES or n < 2 * C):
        raise ValueError("invalid task specification")
    rng = np.random.default_rng(np.random.SeedSequence(spec["seed"]))
    invalid = 0
    for draft in range(1, MAX_DRAFTS + 1):
        t = _draft(rng, n, d, C)
        if t is None:
            invalid += 1
            continue
        cdf = np.cumsum(t["P"], 1)
        for _ in range(CONFIG["label_retries"]):
            y = np.minimum((rng.random((n, 1)) > cdf).sum(1), C - 1)
            if np.bincount(y, minlength=C).min() >= 2:
                break
        else:
            invalid += 1
            continue
        # Random class IDs (class 0 is not always the minority); the oracle columns follow.
        relabel = rng.permutation(C)
        y = relabel[y]
        P = t["P"][:, np.argsort(relabel)]
        # Stored column order is random; drivers are columns 0..k-1 before it.
        order = rng.permutation(d)
        X = np.zeros((n, d), dtype=np.float16)
        codes = np.full((n, d), -1, dtype=np.int64)
        for new, old in enumerate(order):
            if t["is_nominal"][old]:
                ids = t["codes"][old]
                codes[:, new] = ids
                # Legacy (float) view of a nominal column: its levels ranked by ID, spread over [-1, 1].
                present, rank = np.unique(ids, return_inverse=True)
                X[:, new] = 2 * rank / max(1, len(present) - 1) - 1
            else:
                X[:, new] = t["columns"][old]
        keep = X.min(0) != X.max(0)
        drivers = [int(np.flatnonzero(order == j)[0]) for j in range(t["k"])]
        if not all(keep[j] for j in drivers) or not keep.any():
            invalid += 1  # a driver must be observable; distractor-only drops would also shift indices
            continue
        if not keep.all():
            X, codes = X[:, keep], codes[:, keep]
            remap = np.cumsum(keep) - 1
            drivers = [int(remap[j]) for j in drivers]
        meta = dict(kind=t["kind"], n_classes=int(C), drivers=drivers, scale=float(t["scale"]),
                    target=[float(t["target"][c]) for c in np.argsort(relabel)], components=t["components"])
        task = dict(X=X, y=y.astype(np.uint8), n_features=d, resampled=False, resample_requested=False,
                    rejected=0, invalid=invalid, drafts=draft, filter_checks=0,
                    cat_mask=codes[0] >= 0, cat_codes=codes, additive_meta=meta)
        if spec.get("audit"):
            task["additive_oracle"] = P.astype(np.float16)
        return task
    raise RuntimeError(f"additive drafts exhausted: {spec}, invalid={invalid}")
