"""Versioned regression observations. No global RNG, clipping or tail rejection.

Moment guarantees are conditional on a finite location and X. Calibration moments
belong to the base generator, never to the context/query normalization in a model.
"""

import math
import numpy as np

VERSION = 2
CHANNELS = ("gaussian", "heteroscedastic", "student", "sinh_arcsinh", "positive",
            "bimodal", "hurdle", "counts", "topcoded")
QUOTAS = (.30, .20, .15, .10, .10, .05, .04, .04, .02)
RANGES = dict(sigma_ratio=[.05, 2.], hetero_ratio=[1., 10.], student_nu=[3.5, 10.],
              delta=[.5, 1.5], skew=[-.75, .75], log_noise=[.05, 1.],
              log_location_bound=3., log_rate_bound=3., mixture_gap=[.5, 3.],
              nb_dispersion=[1., 20.], top_level=[.5, 2.],
              outlier_task_probability=.10, outlier_fraction=[0., .02],
              outlier_amplitude=[3., 10.], affine_a=[.01, 100.], affine_b=[-10., 10.])
# Recipe r2 (2026-11-05, runs/analysis_reg_prior_20261105): opt-in factors, each declared before any observation is
# drawn. Without factors the draws, the spec and version 2 are exactly the reviewed P0 exporter's. The extra draws
# happen after every version-2 draw, so a factor never shifts the RNG stream of another one.
VERSION_R2 = 3
FACTORS = ("noise", "tails", "discrete")
QUOTAS_TAILS = (.20, .14, .12, .10, .22, .04, .06, .08, .04)
RANGES_R2 = dict(sigma_ratio=[.02, 1.], log_noise=[.03, .7], log_spread=[.3, 2.],
                 count_log_offset=[0., 3.], hurdle_slope=[1., 5.], hurdle_threshold=[-1., 1.], mixture_gap=[.3, 1.5],
                 discrete_probability=.35, grid_ratio=[.05, 1.],
                 discrete_channels=["gaussian", "heteroscedastic", "student", "sinh_arcsinh", "positive", "bimodal"])
STORAGE_POLICY = dict(version=2, dtype="float32", rounding="nearest, reserve declared atom by one ULP",
                      offset_reference_multiple=512., max_offset_ulp_ratio=1e-4,
                      max_rms_error_reference_ratio=1e-3, tail_retry=False)


def _loguniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _factors(factors):
    factors = () if factors is None else factors
    if not isinstance(factors, (list, tuple)) or any(not isinstance(f, str) for f in factors):
        raise ValueError("observation factors must be a list/tuple of names")
    factors = tuple(sorted(factors))
    if any(f not in FACTORS for f in factors) or len(set(factors)) != len(factors):
        raise ValueError("unknown or repeated observation factor")
    return factors


def grid_unit(spec, channel):
    """Declared spread of the target before the affine map, the unit of the discretization grid.

    With a signal (the location has calibration std 1) the grid is bounded by the signal spread: a step as coarse
    as one signal std still leaves ~6+ levels, and real discrete targets (ratings, rounded prices) lose information to
    rounding too. A positive channel's log-location has std ~log_spread. Without a signal (constant or unresolved
    location: snr_defined False) the target's spread is the noise alone (sigma_ratio, or log_noise for positive), so
    the grid is bounded by it and cannot collapse the target to one level. Never fitted to observed y.
    """
    unit = min(1., spec["log_spread"]) if channel == "positive" and "tails" in spec["factors"] else 1.
    if not (spec.get("location_units") or {}).get("snr_defined", True):
        unit = min(unit, spec["log_noise"] if channel == "positive" else spec["sigma_ratio"])
    return unit


def sample_spec(location, X, rng, channel=None, *, center=None, scale=None, location_units=None, factors=None):
    """Declare the entire mechanism before drawing observations.

    Exporters pass independent calibration mean/std. Standalone callers may use
    fixed defaults (0, 1); no threshold or mechanism is fitted to observed y.
    `factors` (recipe r2, subset of FACTORS): none keeps version 2 bit for bit.
    """
    location, X = _inputs(location, X)
    factors = _factors(factors)
    quotas = QUOTAS_TAILS if "tails" in factors else QUOTAS
    channel = str(rng.choice(CHANNELS, p=quotas)) if channel is None else channel
    if channel not in CHANNELS:
        raise ValueError("unknown observation channel")
    signal_std = float(1 if scale is None else scale)
    if not math.isfinite(signal_std) or signal_std < 0:
        raise ValueError("invalid calibration signal std")
    delta = float(rng.uniform(*RANGES["delta"]))
    nu = float(rng.uniform(max(3.5, 2 * delta + .5), 10.))
    spec = dict(version=VERSION, channel=channel, center=float(0 if center is None else center),
                scale=signal_std if signal_std > 0 else 1.,
                calibration_signal_std=signal_std, constant_location_fallback=signal_std == 0,
                sigma_ratio=_loguniform(rng, .05, 2.), hetero_ratio=_loguniform(rng, 1., 10.),
                nu=nu, delta=delta, skew=float(rng.uniform(-.75, .75)),
                log_noise=_loguniform(rng, .05, 1.), mixture_gap=float(rng.uniform(.5, 3.)),
                nb=bool(rng.integers(2)), dispersion=_loguniform(rng, 1., 20.),
                top_level=float(rng.uniform(.5, 2.)), x_column=int(rng.integers(X.shape[1])),
                x_slope=float(rng.uniform(-2., 2.)), x_bias=float(rng.uniform(-1., 1.)),
                outlier_fraction=float(rng.uniform(0., .02)) if rng.random() < .10 else 0.,
                outlier_amplitude=float(rng.uniform(3., 10.)),
                affine_a=_loguniform(rng, .01, 100.), affine_b=float(rng.uniform(-10., 10.)))
    if factors:
        # Every r2 draw comes after the version-2 draws and is made whether or not its factor is on, so the
        # streams of two factor sets differ only in the parameters the factors change.
        sigma, log_noise = _loguniform(rng, *RANGES_R2["sigma_ratio"]), _loguniform(rng, *RANGES_R2["log_noise"])
        spread = _loguniform(rng, *RANGES_R2["log_spread"])
        discrete = bool(rng.random() < RANGES_R2["discrete_probability"])
        grid = _loguniform(rng, *RANGES_R2["grid_ratio"])
        offset = float(rng.uniform(*RANGES_R2["count_log_offset"]))
        slope, threshold = _loguniform(rng, *RANGES_R2["hurdle_slope"]), float(rng.uniform(*RANGES_R2["hurdle_threshold"]))
        gap = float(rng.uniform(*RANGES_R2["mixture_gap"]))
        spec.update(version=VERSION_R2, factors=list(factors), log_spread=spread if "tails" in factors else 1.,
                    grid=None, count_log_offset=0., hurdle_slope=None, hurdle_threshold=None)
        if "noise" in factors:
            # The audit (AUDIT.md) found most remaining unlearnable tasks in three channels whose randomness ignores
            # the location: near-unit Poisson rates, hurdle zeros drawn from an unrelated column, and a mixture gap
            # of up to 3 signal std. Under "noise" the count rate is larger, hurdle zeros follow the location and
            # the mixture gap is smaller.
            spec.update(sigma_ratio=sigma, log_noise=log_noise, count_log_offset=offset, hurdle_slope=slope,
                        hurdle_threshold=threshold, mixture_gap=gap)
    if location_units is not None:
        spec["location_units"] = location_units
    if factors and "discrete" in factors and discrete and channel in RANGES_R2["discrete_channels"]:
        # Grid step in observation units before the affine map, declared here, never fitted to y. Not bounded by
        # variation_reference (review note 2026-11-05): for positive channels that is a conservative lower bound
        # (exp(-3) * log_noise) and would switch discretization off. See grid_unit.
        spec["grid"] = grid * grid_unit(spec, channel)
    # Construct admissible units, no retry based on a realized target. The bound
    # is conservative even for positive channels at their minimum log-location.
    reference = variation_reference(spec)
    spec["affine_b"] *= min(10., 512 * spec["affine_a"] * reference) / 10
    spec["storage_policy"] = STORAGE_POLICY
    spec["precision_plan"] = precision_plan(spec)
    spec["support_before_affine"] = "nonnegative_integer" if channel == "counts" else (
        "nonnegative" if channel == "hurdle" else "positive" if channel == "positive" else (
            "upper_bounded" if channel == "topcoded" else "real"))
    if spec.get("grid"):
        # Rounding to the declared grid: a positive value below half a step becomes the atom 0.
        spec["support_before_affine"] = ("nonnegative_grid" if channel == "positive" else "real_grid")
    spec["atom_before_affine"] = (0. if channel == "hurdle" else
        spec["center"] + spec["scale"] * spec["top_level"] if channel == "topcoded" else None)
    if channel == "positive" and spec.get("grid") is not None:
        spec["atom_before_affine"] = 0.  # rounding changes positive support and introduces a genuine zero atom
    spec["moment_check"] = moment_check(spec)
    return spec


def observation_location(location, spec):
    units = spec.get("location_units")
    loc = np.asarray(location, np.float64)
    return loc if units is None else (loc - units["center"]) / units["scale"]


def variation_reference(s):
    """Declared noise unit, not an estimate from realized observed targets."""
    if s["channel"] in ("positive", "hurdle"):
        return float(np.exp(-3) * s["log_noise"])
    if s["channel"] == "counts":
        return float(np.sqrt(np.exp(-3)))
    ref = s["scale"] * s["sigma_ratio"]
    if s["channel"] == "sinh_arcsinh":
        ref *= s["delta"] * np.exp(-abs(s["skew"]))
    return float(ref)


def precision_plan(s):
    reference = s["affine_a"] * variation_reference(s)
    offset = np.float32(s["affine_b"])
    ulp = float(max(abs(float(np.nextafter(offset, np.float32(np.inf))) - float(offset)),
                    abs(float(np.nextafter(offset, np.float32(-np.inf))) - float(offset))))
    ratio = ulp / reference
    if not math.isfinite(ratio) or ratio > STORAGE_POLICY["max_offset_ulp_ratio"]:
        raise FloatingPointError("affine offset ULP exceeds declared observation precision; no retry")
    if s.get("grid") is not None and not s["affine_a"] * s["grid"] > 2 * ulp:
        raise FloatingPointError("float32 offset cannot resolve the declared grid; no retry")
    return dict(reference_std=reference, offset_ulp=ulp, offset_ulp_ratio=ratio)


def quantize(raw, s):
    """Reserve the declared atom's float32 code, with a counted one-ULP rule.

    Only uncensored/nonzero values colliding with that code move one ULP into
    their support. No tail clipping or draw-dependent change of affine units.
    Shared by the diagnostic oracle and the exporter.
    """
    raw = np.asarray(raw, np.float64)
    ideal = s["affine_a"] * raw + s["affine_b"]
    if not np.isfinite(ideal).all() or np.any(np.abs(ideal) > np.finfo(np.float32).max):
        raise FloatingPointError("observation not representable as float32; no clipping/retry allowed")
    stored = ideal.astype(np.float32)
    adjusted = np.zeros(raw.shape, bool)
    if s.get("grid") is not None:
        # RMS error alone can miss merged levels in the tail. This is a storage check, not a y-quality filter.
        ulp = np.abs(np.nextafter(stored, np.where(stored >= 0, np.float32(np.inf), np.float32(-np.inf)))
                     .astype(np.float64) - stored.astype(np.float64))
        if np.any(2 * ulp >= s["affine_a"] * s["grid"]):
            raise FloatingPointError("float32 cannot resolve the declared grid levels; no retry")
    if s["channel"] in ("topcoded", "hurdle") or (s["channel"] == "positive" and s.get("grid") is not None):
        atom_raw = s["atom_before_affine"]
        atom = np.float32(s["affine_a"] * atom_raw + s["affine_b"])
        adjusted = (raw != atom_raw) & (stored == atom)
        direction = -np.inf if s["channel"] == "topcoded" else np.inf
        stored = np.where(adjusted, np.nextafter(atom, np.float32(direction)), stored)
    return stored, ideal, int(adjusted.sum())


def validate_grid(stored, s):
    """Check float32 codes against the declared affine grid, without estimating a mechanism from y."""
    if s.get("grid") is None:
        return
    values = np.asarray(stored, np.float32)
    levels = np.rint(inverse_units(values, s) / s["grid"])
    if s["channel"] == "positive" and np.any(levels < 0):
        raise ValueError("negative level on a nonnegative grid")
    expected, _, _ = quantize(s["grid"] * levels, s)
    if not np.array_equal(values, expected):
        raise ValueError("target disagrees with declared grid levels")


def moment_check(spec):
    """Analytic post-warp check, not a finite-sample claim about heavy tails."""
    numeric = ("center", "scale", "sigma_ratio", "affine_a", "affine_b", "outlier_fraction", "hetero_ratio",
               "nu", "delta", "skew", "log_noise", "dispersion", "top_level", "mixture_gap",
               "outlier_amplitude", "x_slope", "x_bias")
    if any(k not in spec or not isinstance(spec[k], (int, float)) or not math.isfinite(spec[k]) for k in numeric):
        raise ValueError("missing or nonfinite observation parameter")
    ch = spec["channel"]
    if ch not in CHANNELS or not all(math.isfinite(float(spec[k])) for k in (
            "center", "scale", "sigma_ratio", "affine_a", "affine_b")):
        raise ValueError("invalid observation specification")
    r2 = spec.get("version") == VERSION_R2
    extra = {"factors", "log_spread", "grid", "count_log_offset", "hurdle_slope", "hurdle_threshold"}
    if r2:
        if not extra <= spec.keys() or not isinstance(spec["factors"], list):
            raise ValueError("missing r2 observation fields")
        if any(type(spec[k]) not in (int, float) or not math.isfinite(spec[k])
               for k in ("log_spread", "count_log_offset", "grid", "hurdle_slope", "hurdle_threshold")
               if spec[k] is not None):
            raise ValueError("nonfinite r2 observation parameter")
        if spec["log_spread"] is None or spec["count_log_offset"] is None:
            raise ValueError("missing r2 observation parameter")
        if type(spec.get("nb")) is not bool or type(spec.get("x_column")) is not int or spec["x_column"] < 0:
            raise ValueError("invalid r2 count law/covariate")
        factors = _factors(spec["factors"])
        if not factors or spec["factors"] != list(factors):
            raise ValueError("version 3 needs canonical factors")
    elif extra & spec.keys():
        raise ValueError("r2 observation fields require version 3")
    noise = r2 and "noise" in spec["factors"]
    sigma_range = RANGES_R2["sigma_ratio"] if noise else RANGES["sigma_ratio"]
    noise_range = RANGES_R2["log_noise"] if noise else RANGES["log_noise"]
    if spec["scale"] <= 0 or spec["affine_a"] <= 0 or not sigma_range[0] <= spec["sigma_ratio"] <= sigma_range[1]:
        raise ValueError("invalid scales")
    if not 0 <= spec["outlier_fraction"] <= .02:
        raise ValueError("invalid contamination fraction")
    if not 1 <= spec["hetero_ratio"] <= 10 or not 3.5 <= spec["nu"] <= 10:
        raise ValueError("invalid heteroscedasticity/Student parameters")
    if not .5 <= spec["delta"] <= 1.5 or not abs(spec["skew"]) <= .75:
        raise ValueError("invalid sinh-arcsinh parameters")
    if ch == "sinh_arcsinh" and not spec["nu"] > 2 * spec["delta"] + .5:
        raise ValueError("post-warp second moment does not satisfy the safety margin")
    if not noise_range[0] <= spec["log_noise"] <= noise_range[1] or not 1 <= spec["dispersion"] <= 20:
        raise ValueError("invalid lognormal/NB parameters")
    if r2:
        lo, hi = RANGES_R2["log_spread"]
        if not (lo <= spec["log_spread"] <= hi if "tails" in spec["factors"] else spec["log_spread"] == 1.):
            raise ValueError("invalid log spread")
    if r2 and spec["grid"] is not None:
        unit = grid_unit(spec, ch)
        lo, hi = (v * unit for v in RANGES_R2["grid_ratio"])
        if not ("discrete" in spec["factors"] and ch in RANGES_R2["discrete_channels"]
                and lo <= spec["grid"] <= hi):
            raise ValueError("invalid discretization grid")
    if r2 and not (0 <= spec["count_log_offset"] <= RANGES_R2["count_log_offset"][1] if noise
                   else spec["count_log_offset"] == 0 and spec["hurdle_slope"] is None
                   and spec["hurdle_threshold"] is None):
        raise ValueError("invalid r2 count/hurdle parameters")
    if noise and not (spec["hurdle_slope"] is not None and spec["hurdle_threshold"] is not None
                      and RANGES_R2["hurdle_slope"][0] <= spec["hurdle_slope"] <= RANGES_R2["hurdle_slope"][1]
                      and abs(spec["hurdle_threshold"]) <= 1):
        raise ValueError("invalid r2 hurdle parameters")
    gap_range = RANGES_R2["mixture_gap"] if noise else RANGES["mixture_gap"]
    if (spec.get("version") not in (VERSION, VERSION_R2) or not .5 <= spec["top_level"] <= 2
            or not gap_range[0] <= spec["mixture_gap"] <= gap_range[1] or not 3 <= spec["outlier_amplitude"] <= 10
            or not -2 <= spec["x_slope"] <= 2 or not -1 <= spec["x_bias"] <= 1):
        raise ValueError("unknown version or invalid observation parameters")
    if r2:
        support = ("nonnegative_grid" if ch == "positive" else "real_grid") if spec["grid"] is not None else (
            "nonnegative_integer" if ch == "counts" else "nonnegative" if ch == "hurdle" else
            "positive" if ch == "positive" else "upper_bounded" if ch == "topcoded" else "real")
        atom = (0. if ch == "hurdle" or (ch == "positive" and spec["grid"] is not None) else
                spec["center"] + spec["scale"] * spec["top_level"] if ch == "topcoded" else None)
        if spec.get("support_before_affine") != support or spec.get("atom_before_affine") != atom:
            raise ValueError("inconsistent r2 support/atom")
    # sinh(delta*asinh(t)+skew) grows like |t|**delta, so E[Y^2]
    # requires nu>2delta. Positive/hurdle exponentiate ONLY bounded location
    # plus Gaussian noise, never Student noise. Counts have bounded rate.
    return dict(finite_mean=True, finite_variance=True, conditional_on="location,X",
                post_warp=True, reason=("nu>2delta+0.5" if ch == "sinh_arcsinh" else
                    "bounded log-location plus Gaussian" if ch in ("positive", "hurdle") else
                    "bounded Poisson/NB rate" if ch == "counts" else "finite second moment"))


def _inputs(location, X):
    loc, X = np.asarray(location, dtype=np.float64), np.asarray(X, dtype=np.float64)
    if (loc.ndim != 1 or not len(loc) or X.ndim != 2 or X.shape[0] != len(loc)
            or X.shape[1] < 1 or not np.isfinite(loc).all() or not np.isfinite(X).all()):
        raise ValueError("observations need finite location and X of matching length")
    return loc, X


def _x_score(X, s):
    # Bounded functions of X, independent of sample ranks or nominal ID ordering.
    return np.tanh(X[:, s["x_column"]] * s["x_slope"] + s["x_bias"])


def gaussian(location, X, rng, s):
    return location + s["scale"] * s["sigma_ratio"] * rng.normal(size=len(location))


def heteroscedastic(location, X, rng, s):
    scale = np.exp(np.log(s["hetero_ratio"]) * (_x_score(X, s) + 1) / 2)
    return location + s["scale"] * s["sigma_ratio"] * scale * rng.normal(size=len(location))


def student(location, X, rng, s):
    return location + s["scale"] * s["sigma_ratio"] * rng.standard_t(s["nu"], len(location))


def sinh_arcsinh(location, X, rng, s):
    noise = rng.standard_t(s["nu"], len(location))
    return location + s["scale"] * s["sigma_ratio"] * np.sinh(s["delta"] * np.arcsinh(noise) + s["skew"])


def positive(location, X, rng, s):
    logloc = 3 * np.tanh(s.get("log_spread", 1.) * (location - s["center"]) / (3 * s["scale"]))
    return np.exp(logloc + s["log_noise"] * rng.normal(size=len(location)))


def bimodal(location, X, rng, s):
    p = .1 + .8 * (_x_score(X, s) + 1) / 2
    mode = np.where(rng.random(len(location)) < p, 1., -1.)
    return gaussian(location, X, rng, s) + s["scale"] * s["mixture_gap"] * mode


def hurdle(location, X, rng, s):
    y = positive(location, X, rng, s)
    if s.get("hurdle_slope") is not None:
        # r2 noise: zeros where the (calibration-standardized) location is low, a logistic threshold.
        from scipy.special import expit
        p = expit(s["hurdle_slope"] * (s["hurdle_threshold"] - (location - s["center"]) / s["scale"]))
    else:
        p = .1 + .8 * (_x_score(X, s) + 1) / 2
    y[rng.random(len(location)) < p] = 0
    return y


def counts(location, X, rng, s):
    rate = np.exp(s.get("count_log_offset", 0.)
                  + 3 * np.tanh(s.get("log_spread", 1.) * (location - s["center"]) / (3 * s["scale"])))
    if s["nb"]:
        return rng.negative_binomial(s["dispersion"], s["dispersion"] / (s["dispersion"] + rate)).astype(float)
    return rng.poisson(rate).astype(float)


def topcoded(location, X, rng, s):
    # A modelled observation threshold, fixed in calibration/population units.
    return np.minimum(gaussian(location, X, rng, s), s["atom_before_affine"])


def observe(location, X, rng, spec=None, *, channel=None, center=None, scale=None, location_units=None,
            factors=None):
    """Return original-unit float32 y and a JSON-only spec. Extreme draws fail loudly.

    Contamination is bounded in amplitude and respects support, zero atoms,
    integer counts and the censoring threshold. Affine units apply last.
    """
    location, X = _inputs(location, X)
    s = sample_spec(location, X, rng, channel, center=center, scale=scale,
                    location_units=location_units, factors=factors) if spec is None else dict(spec)
    moment_check(s)
    plan = precision_plan(s)  # before consuming any observation RNG
    location = observation_location(location, s)
    if not isinstance(s["x_column"], int) or not 0 <= s["x_column"] < X.shape[1]:
        raise ValueError("invalid observation covariate")
    # Exporters pass numerical values or declared nominal equality indicators,
    # never an arbitrary ordering of nominal codes.
    y = globals()[s["channel"]](location, X, rng, s)
    out = np.zeros(len(y), dtype=bool)
    # Bound the realized fraction too, including on tiny tasks.
    out[rng.choice(len(y), int(len(y) * s["outlier_fraction"]), replace=False)] = True
    ch, amplitude = s["channel"], s["outlier_amplitude"]
    if ch in ("positive", "hurdle", "counts"):
        y[out] *= amplitude
        if ch == "counts":
            y[out] = np.rint(y[out])
    else:
        y[out] += s["scale"] * amplitude * rng.choice([-1., 1.], out.sum())
        if ch == "topcoded":
            y = np.minimum(y, s["atom_before_affine"])
    if s.get("grid"):
        # Deterministic rounding to the declared grid (no RNG), outliers included: a discrete observed target.
        y = s["grid"] * np.rint(y / s["grid"])
    stored, ideal, adjusted = quantize(y, s)
    error = float(np.sqrt(np.mean((stored.astype(np.float64) - ideal) ** 2)))
    ratio = error / plan["reference_std"]
    if not math.isfinite(ratio) or ratio > STORAGE_POLICY["max_rms_error_reference_ratio"]:
        raise FloatingPointError("float32 precision loss exceeds declared reference; no clipping/retry allowed")
    s["precision_realized"] = dict(**plan, rms_error=error, rms_error_reference_ratio=ratio,
                                    atom_code_adjustments=adjusted)
    s["outliers_realized"] = int(out.sum())
    return stored, s


def inverse_units(y, spec):
    return (np.asarray(y, dtype=np.float64) - spec["affine_b"]) / spec["affine_a"]
