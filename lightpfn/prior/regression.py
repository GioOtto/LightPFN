"""P0 graph-reg exporter and homogeneous typed regression groups.

This is a versioned extension of the existing .pt pool format, NOT the full
flat-NPY schema2 production protocol. Audit arrays never enter PoolStream.
"""

import numpy as np
import torch

from .observation import FACTORS as OBSERVATION_FACTORS, observe, moment_check

CALIBRATION_ROWS = 2048
UNIT_POLICY = dict(version=2, calibration="independent-disjoint-2048",
                   positive_scale="preserve-and-standardize-in-float64",
                   near_constant_threshold=1e-6, unresolved_relative_epsilon=32.,
                   zero_signal_reference=1., constant_task_quota="natural SCM incidence, no resampling",
                   observed_target_clipping=False)
# observation_precision (2026-11-04, first 65k-task pool on the box): a drawn target that float32 cannot store
# within the declared reference error is not storable faithfully; the whole task is redrawn (new SCM draft), the
# observation itself is never retried or clipped. Pools must keep this rejection rare (reg_heads runner: <= 0.5%).
# scm_nonfinite (2026-11-04, second attempt, ~1 in 55k tasks): an SCM operator returned a nonfinite value
# (reg_scm.structural_bound raises, never repairs). Structural, before any observation: the draft is redrawn and
# counted like scm_runtime. Pools keep it rare too (reg_heads runner: <= 0.5%).
STRUCTURAL_POLICY = dict(max_drafts=128, max_abs_X=1000.,
                         rejections=["scm_runtime", "nonfinite_base", "feature_range", "constant_features",
                                     "observation_precision", "scm_nonfinite"],
                         finite_location_tail_rejection=False, observation_retries=0)


# Recipe r2 (2026-11-05, runs/analysis_reg_prior_20261105): opt-in factors. "filter" is structural: it reads only the
# SCM location and X of the independent calibration rows, never an observed target, and redraws the SCM draft (no
# observation retry). The other factors live in observation.py. Graph v4 is allowed only with r2 (export version 3).
REG_FACTORS = ("filter", *OBSERVATION_FACTORS)
SIGNAL_FILTER = dict(target="SCM location of the independent calibration rows", features="calibration X, retained task columns",
                     estimator="ExtraTreesRegressor", n_estimators=32, min_samples_leaf=2, bootstrap=True,
                     oob_score=True, random_state=1, min_oob_r2=0.1, constant_or_unresolved="fails",
                     target_units="calibration-standardized float64 (R2 invariant)",
                     accept_failing_probability=0.05)
STRUCTURAL_POLICY_R2 = dict(STRUCTURAL_POLICY, version="r2",
                            rejections=[*STRUCTURAL_POLICY["rejections"], "no_signal"], signal_filter=SIGNAL_FILTER)


def reg_factors(spec):
    factors = spec.get("reg_factors")
    factors = () if factors is None else factors
    if not isinstance(factors, (list, tuple)) or any(not isinstance(f, str) for f in factors):
        raise ValueError("regression factors must be a list/tuple of names")
    factors = tuple(sorted(factors))
    if any(f not in REG_FACTORS for f in factors) or len(set(factors)) != len(factors):
        raise ValueError("unknown or repeated regression factor")
    return factors


def location_oob_r2(X, location):
    """Out-of-bag R2 of the calibration location from the calibration features (deterministic)."""
    from sklearn.ensemble import ExtraTreesRegressor

    f = SIGNAL_FILTER
    location = np.asarray(location, np.float64)
    units = location_units(location)
    if not units["snr_defined"]:
        return -np.inf
    # sklearn's absolute impurity tolerance otherwise turns a resolved tiny signal into a constant tree.
    target = (location - units["center"]) / units["scale"]
    et = ExtraTreesRegressor(n_estimators=f["n_estimators"], min_samples_leaf=f["min_samples_leaf"],
                             bootstrap=f["bootstrap"], oob_score=f["oob_score"], random_state=f["random_state"],
                             n_jobs=1)
    et.fit(np.asarray(X, np.float32), target)
    return float(et.oob_score_)


def location_units(calibration):
    """Choose signal units before observation; never inspect task/query targets."""
    cal = np.asarray(calibration, np.float64)
    if cal.ndim != 1 or not len(cal) or not np.isfinite(cal).all():
        raise ValueError("invalid independent location calibration")
    exact = bool(np.ptp(cal) == 0)
    center = float(cal[0] if exact else cal.mean())
    sd = 0. if exact else float(cal.std())
    resolution = float(32 * np.finfo(np.float64).eps * abs(center))
    unresolved = bool(not exact and sd <= resolution)
    regime = ("constant" if exact else "unresolved" if unresolved else
              "near_constant_resolved" if sd < 1e-6 else "regular")
    iqr = float(np.quantile(cal, .75) - np.quantile(cal, .25))
    energy = (cal - center) ** 2
    tail = energy >= np.quantile(energy, .99)
    return dict(policy=UNIT_POLICY, center=center, calibration_std=sd,
                calibration_iqr=iqr, calibration_std_over_iqr=sd / iqr if iqr else None,
                calibration_max_abs=float(np.abs(cal).max()),
                calibration_tail1pct_energy_share=float(energy[tail].sum() / energy.sum()) if energy.sum() else 0.,
                calibration_resolution=resolution, exact_constant=exact, regime=regime,
                scale=1. if exact or unresolved else sd,
                observation_scale=1., snr_defined=not (exact or unresolved))


def generate_regression_task(spec):
    from . import generate as g

    factors = reg_factors(spec)
    version = spec.get("prior_version", 3)
    if spec["prior"] != "graph" or version not in ((3, 4) if factors else (3,)):
        raise ValueError("graph-reg requires graph v3, or graph v3/v4 with the r2 factors")
    r2 = bool(factors)
    n, d = spec["n_rows"], spec["n_features"]
    if not isinstance(n, int) or not isinstance(d, int) or n < 2 or not 1 <= d <= g.MAX_LEVELS:
        raise ValueError("invalid regression geometry")
    rng = g._seed_task(spec["seed"])
    rejections = dict(scm_runtime=0, nonfinite_base=0, feature_range=0, constant_features=0,
                      observation_precision=0, scm_nonfinite=0)
    if r2:
        rejections["no_signal"] = 0
    signal = None
    # Structural failures only. Once an observation is drawn it is never filtered
    # or retried, so heavy tails are not implicitly truncated.
    for draft in range(g.MAX_TASK_DRAFTS):
        try:
            X, loc, codes, operator_counts = g._graph_dataset(dict(spec, typed=True), n + CALIBRATION_ROWS)
        except RuntimeError:
            rejections["scm_runtime"] += 1
            continue
        except FloatingPointError:  # nonfinite SCM operator, before observation: new draft, see STRUCTURAL_POLICY
            rejections["scm_nonfinite"] += 1
            continue
        if not np.isfinite(loc).all() or not np.isfinite(X).all():
            rejections["nonfinite_base"] += 1
            continue
        if np.abs(X).max() > g.MAX_ABS_X:
            rejections["feature_range"] += 1
            continue
        calibration_codes, calibration_X = codes[n:], X[n:]
        X, codes, location = X[:n], codes[:n], loc[:n]
        keep = X.min(0) != X.max(0)
        if not keep.any():
            rejections["constant_features"] += 1
            continue
        X, codes = X[:, keep].astype(np.float32), codes[:, keep].astype(np.int32)
        cat = codes[0] >= 0
        # A nominal covariate enters as equality to a calibration reference,
        # never its arbitrary ordered ID. Renaming the ID and its declared
        # reference leaves scales/probabilities invariant.
        obs_X = X.copy()
        projections = []
        for j in range(X.shape[1]):
            if cat[j]:
                reference = int(calibration_codes[:, keep][0, j])
                obs_X[:, j] = codes[:, j] == reference
                projections.append(dict(kind="nominal_equality", reference=reference))
            else:
                projections.append(dict(kind="numerical"))
        cal = loc[n:]
        units = location_units(cal)
        if "filter" in factors:
            constant = units["regime"] in ("constant", "unresolved")
            # The learner cannot use columns removed from its task, even if they vary on calibration rows.
            oob = -np.inf if constant else location_oob_r2(calibration_X[:, keep], cal)
            passed = bool(np.isfinite(oob) and oob >= SIGNAL_FILTER["min_oob_r2"])
            if not passed and rng.random() >= SIGNAL_FILTER["accept_failing_probability"]:
                rejections["no_signal"] += 1
                continue
            signal = dict(oob_r2=oob if np.isfinite(oob) else None, passed=passed)
        try:
            y, obs = observe(location, obs_X, rng, channel=spec.get("observation_channel"),
                             center=0., scale=1., location_units=units,
                             factors=[f for f in factors if f in OBSERVATION_FACTORS])
        except FloatingPointError:  # float32 cannot store this target faithfully: new task draft, see STRUCTURAL_POLICY
            rejections["observation_precision"] += 1
            continue
        from .reg_scm import POLICY
        meta = dict(family=f"graph_v{version}", task_id=list(spec["seed"]), observation=obs,
                    latent_semantics="SCM location, privileged; not E[y|observed X]",
                    node_gaussian_noise=False, calibration_rows=CALIBRATION_ROWS,
                    scm_fit="independent-draw-ensure_iid-persistent-root-sources-v1", observation_covariates=projections,
                    structural_drafts=draft + 1, structural_rejections=rejections,
                    scm_operator_policy=POLICY, scm_operator_counts=operator_counts,
                    structural_policy=STRUCTURAL_POLICY_R2 if r2 else STRUCTURAL_POLICY,
                    scm_fit_rows=n + CALIBRATION_ROWS,
                    location_units=units, export_version=3 if r2 else 2, observation_retries=0)
        if r2:
            meta.update(reg_factors=list(factors), signal=signal)
        task = dict(X=X, y_reg=y, cat_codes=codes, cat_mask=cat, n_features=d,
                    regression_meta=meta)
        if spec.get("audit"):
            task["location_audit"] = location.copy()  # float64, audit only
        return task
    raise RuntimeError("graph-reg structural drafts exhausted")


def assemble_regression_group(rng, tasks, train_frac):
    if (not tasks or not 0 < train_frac < 1 or any("y_reg" not in t or "y" in t for t in tasks)):
        raise ValueError("regression groups must be homogeneous")
    B, n = len(tasks), len(tasks[0]["y_reg"])
    m = max(t["n_features"] for t in tasks)
    X = np.zeros((B, n, m), np.float32)
    y = np.zeros((B, n), np.float32)
    mask = np.zeros((B, m), bool)
    width = max(1, max(int(t["cat_mask"].sum()) for t in tasks))
    codes = np.full((B, n, width), -1, np.int32)
    uid = np.zeros((B, n), np.int64)
    audit = []
    for i, t in enumerate(tasks):
        if len(t["y_reg"]) != n:
            raise ValueError("regression rows disagree")
        sel = rng.permutation(n)  # assigned IDs precede shuffle, no target stratification
        di = t["X"].shape[1]
        X[i, :, :di], y[i], uid[i] = t["X"][sel], t["y_reg"][sel], sel
        mask[i, :di] = t["cat_mask"]
        raw = t["cat_codes"][sel][:, t["cat_mask"]]
        codes[i, :, :raw.shape[1]] = raw
        if "location_audit" in t:
            audit.append(t["location_audit"][sel])
    group = dict(X=torch.from_numpy(X), y_reg=torch.from_numpy(y), cat_mask=torch.from_numpy(mask),
                 cat_codes=torch.from_numpy(codes), row_uid=torch.from_numpy(uid),
                 d=torch.tensor([t["X"].shape[1] for t in tasks], dtype=torch.int16),
                 n_train=min(max(round(train_frac * n), 1), n - 1),
                 regression_meta=[t["regression_meta"] for t in tasks], regression_schema=1)
    if audit:
        if len(audit) != B:
            raise ValueError("inconsistent audit flags")
        group["location_audit"] = torch.from_numpy(np.stack(audit))
    validate_regression_group(group)
    return group


def validate_regression_group(g, *, preset=None, max_features=None, group_size=None):
    if "y" in g or "n_classes" in g or g.get("regression_schema") != 1:
        raise ValueError("mixed or unknown regression schema")
    for key in ("X", "y_reg", "d", "cat_mask", "cat_codes", "row_uid"):
        if not torch.is_tensor(g.get(key)):
            raise ValueError(f"missing regression tensor: {key}")
    X, y, d = (g[k] for k in ("X", "y_reg", "d"))
    if X.ndim != 3 or X.dtype != torch.float32 or y.dtype != torch.float32 or y.shape != X.shape[:2]:
        raise ValueError("invalid regression X/y shape or dtype")
    B, n, m = X.shape
    if (min(B, n, m) < 1 or not isinstance(g.get("n_train"), int) or not 1 <= g["n_train"] < n
            or d.shape != (B,) or d.dtype != torch.int16
            or not torch.isfinite(X).all() or X.abs().max() > 1000 or not torch.isfinite(y).all()):
        raise ValueError("invalid regression dimensions/range/train budget")
    if ((preset and not preset["n_min"] <= n <= preset["n_max"])
            or (max_features is not None and m > max_features)
            or (group_size is not None and B != group_size)):
        raise ValueError("regression geometry outside preset")
    mask, codes, uid = (g[k] for k in ("cat_mask", "cat_codes", "row_uid"))
    if (mask.dtype != torch.bool or mask.shape != (B, m) or codes.dtype != torch.int32
            or codes.ndim != 3 or codes.shape[:2] != (B, n)
            or uid.dtype != torch.int64 or uid.shape != (B, n)
            or not isinstance(g.get("regression_meta"), list) or len(g["regression_meta"]) != B):
        raise ValueError("invalid typed regression fields")
    for i in range(B):
        di, k = int(d[i]), int(mask[i].sum())
        if (not 1 <= di <= m or mask[i, di:].any() or (X[i, :, di:] != 0).any()
                or (X[i, :, :di].min(0).values == X[i, :, :di].max(0).values).any()
                or k > codes.shape[2] or (codes[i, :, :k] < 0).any() or (codes[i, :, k:] != -1).any()
                or len(uid[i].unique()) != n or (uid[i] < 0).any() or (uid[i] > np.iinfo(np.uint32).max).any()):
            raise ValueError("invalid regression padding/codes/UID/constant column")
        meta = g["regression_meta"][i]
        if not isinstance(meta, dict) or not isinstance(meta.get("observation"), dict):
            raise ValueError("invalid regression metadata")
        r2 = meta.get("export_version") == 3
        if (meta.get("family") not in (("graph_v3", "graph_v4") if r2 else ("graph_v3",))
                or meta.get("node_gaussian_noise") is not False):
            raise ValueError("undeclared regression base/noise")
        if r2:
            factors = reg_factors(meta)
            observed = [f for f in factors if f in OBSERVATION_FACTORS]
            if (not factors or meta.get("reg_factors") != list(factors)
                    or (observed and (meta["observation"].get("version") != 3
                                      or meta["observation"].get("factors") != observed))
                    or (not observed and meta["observation"].get("version") != 2)
                    or ("filter" not in factors and meta.get("signal", False) is not None)):
                raise ValueError("undeclared regression r2 factors")
            if "filter" in factors:
                signal = meta.get("signal")
                if not isinstance(signal, dict) or set(signal) != {"oob_r2", "passed"}:
                    raise ValueError("invalid regression signal filter metadata")
                score, passed = signal["oob_r2"], signal["passed"]
                if (type(passed) is not bool or (score is not None and (
                        type(score) not in (int, float) or not np.isfinite(score) or score > 1))
                        or passed != (score is not None and score >= SIGNAL_FILTER["min_oob_r2"])
                        or (meta.get("location_units", {}).get("regime") in ("constant", "unresolved")
                            and score is not None)):
                    raise ValueError("inconsistent regression signal filter result")
            counts = meta.get("structural_rejections", {})
            drafts = meta.get("structural_drafts")
            if (not isinstance(counts, dict) or set(counts) != set(STRUCTURAL_POLICY_R2["rejections"])
                    or any(type(v) is not int or v < 0 for v in counts.values())
                    or type(drafts) is not int or not 1 <= drafts <= STRUCTURAL_POLICY_R2["max_drafts"]
                    or ("filter" not in factors and counts["no_signal"] != 0)):
                raise ValueError("invalid r2 rejection accounting")
            # v4 IDs rank exact preprocessed values; calibration-only levels may leave gaps in task IDs.
            if meta["family"] == "graph_v4":
                for ci, col in enumerate(torch.nonzero(mask[i]).flatten()):
                    values = X[i, :, col]
                    order = torch.argsort(values)
                    cv, vv = codes[i, order, ci], values[order]
                    if not torch.equal(cv[1:] == cv[:-1], vv[1:] == vv[:-1]) or (cv[1:] < cv[:-1]).any():
                        raise ValueError("v4 category codes disagree with distinct value ranks")
        elif "reg_factors" in meta or "signal" in meta:
            raise ValueError("r2 regression fields require export version 3")
        from .reg_scm import POLICY
        from .observation import STORAGE_POLICY, precision_plan, validate_grid
        if (meta.get("export_version") not in (2, 3) or meta.get("scm_operator_policy") != POLICY
                or meta.get("structural_policy") != (STRUCTURAL_POLICY_R2 if r2 else STRUCTURAL_POLICY)
                or meta.get("location_units", {}).get("policy") != UNIT_POLICY
                or meta["observation"].get("location_units") != meta["location_units"]
                or meta["observation"].get("storage_policy") != STORAGE_POLICY
                or meta.get("observation_retries") != 0
                or sum(meta.get("structural_rejections", {}).values()) != meta.get("structural_drafts", 0) - 1):
            raise ValueError("undeclared regression units/operators/storage/rejections")
        checked = moment_check(meta["observation"])
        plan = precision_plan(meta["observation"])
        if r2:
            s = meta["observation"]
            if (s.get("precision_plan") != plan or s.get("moment_check") != checked
                    or s["center"] != 0. or s["scale"] != 1.
                    or type(s.get("x_column")) is not int or not 0 <= s["x_column"] < di):
                raise ValueError("inconsistent r2 precision/moment declaration")
            validate_grid(y[i].numpy(), s)
    if "location_audit" in g:
        v = g["location_audit"]
        if v.shape != (B, n) or v.dtype != torch.float64 or not torch.isfinite(v).all():
            raise ValueError("invalid location audit")
