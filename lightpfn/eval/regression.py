"""Experimental D1-reg P0: frozen graph-only dev/confirmation seeds and metrics.

Not the complete multi-family D1-reg gate. No selection or quality claims.
Quantile convention: tau=k/1000, constant tails to 0/1, trapezoidal CRPS.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

TAUS = np.arange(1, 1000, dtype=np.float64) / 1000
D1_REG = dict(version=2, name="D1-reg-P1fix-graph", seed_namespace=0xD1AE6101,
              n_rows=128, n_features=6, n_context=80, tasks_per_channel=2,
              splits={"dev": 0, "confirmation": 1},
              channels=["gaussian", "heteroscedastic", "student", "sinh_arcsinh", "positive",
                        "bimodal", "hurdle", "counts", "topcoded"])


def regression_metrics(y_context, y_query, quantiles, *, seed=0):
    """Original-unit metrics. RMSE uses integrated mean, MAE uses median.

    Std is population std (ddof=0) of the context in float64, floored at 1e-6.
    PIT uses randomized CDF intervals at atoms of the predicted quantile curve.
    Return row-level PIT for diagnostics; aggregate scores by task, then family.
    """
    ctx, y, q = (np.asarray(v, np.float64) for v in (y_context, y_query, quantiles))
    if (ctx.ndim != 1 or not len(ctx) or y.ndim != 1 or not len(y) or q.shape != (len(y), 999)
            or not all(np.isfinite(v).all() for v in (ctx, y, q)) or np.any(np.diff(q, axis=1) < 0)):
        raise ValueError("finite ordered original-unit quantiles and nonempty targets required")
    raw_std = float(ctx.std())
    scale = max(raw_std, 1e-6)
    levels = np.r_[0., TAUS, 1.]
    extended = np.column_stack((q[:, 0], q, q[:, -1]))
    mean = np.trapezoid(extended, levels, axis=1)
    error = y[:, None] - extended
    pinball = np.maximum(levels * error, (levels - 1) * error)
    crps = float((2 * np.trapezoid(pinball, levels, axis=1)).mean())
    rmse = float(np.sqrt(np.mean((y - mean) ** 2)))
    result = dict(crps=crps, crps_std=crps / scale, rmse=rmse, rmse_std=rmse / scale,
                  mae=float(np.abs(y - q[:, 499]).mean()), context_std=raw_std,
                  normalization_scale=scale, near_constant=raw_std < 1e-6)
    for mass in (50, 80, 90):
        lo, hi = (100 - mass) // 2, (100 + mass) // 2
        low, high = q[:, lo * 10 - 1], q[:, hi * 10 - 1]
        result[f"coverage_{mass}"] = float(((y >= low) & (y <= high)).mean())
    rng, pit = np.random.default_rng(seed), []
    for row, value in zip(extended, y):
        left, right = np.searchsorted(row, value, side="left"), np.searchsorted(row, value, side="right")
        if left < right:  # plateau in Q(tau), a point mass in F
            low, high = levels[left], levels[right - 1]
            pit.append(float(rng.uniform(low, high)))
        else:
            pit.append(float(np.interp(value, row, levels)))
    result["pit"] = pit
    return result


def build_d1_reg(out, split="dev"):
    """Freeze config, source hashes, dependencies and group digests, manifest last.

    This deliberately small set exercises all observation channels at fixed
    geometry. It cannot certify unseen mechanisms or long-table generalization.
    """
    from lightpfn.prior import generate as gen
    from lightpfn.prior.regression import CALIBRATION_ROWS, UNIT_POLICY, STRUCTURAL_POLICY
    from lightpfn.prior.observation import VERSION, STORAGE_POLICY
    from lightpfn.prior.reg_scm import POLICY
    if split not in D1_REG["splits"]:
        raise ValueError("unknown D1-reg split")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("D1-reg output must be empty; frozen sets are never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for channel_id, channel in enumerate(D1_REG["channels"]):
        for task in range(D1_REG["tasks_per_channel"]):
            seed = [D1_REG["seed_namespace"], D1_REG["splits"][split], channel_id, task]
            spec = dict(seed=seed, n_rows=D1_REG["n_rows"], n_features=D1_REG["n_features"],
                        n_classes=2, prior="graph", prior_version=3, task="regression", audit=True,
                        observation_channel=channel)
            group = gen.assemble_group(np.random.default_rng([*seed, 0x53504C54]),
                                       [gen.generate_task(spec)], D1_REG["n_context"] / D1_REG["n_rows"])
            filename = f"task_{channel_id:02d}_{task:02d}.pt"
            gen.atomic_write(out / filename, lambda f: torch.save(group, f))
            records.append(dict(file=filename, seed=seed, channel=channel,
                                sha256=hashlib.sha256((out / filename).read_bytes()).hexdigest()))
    from importlib.metadata import version
    source = Path(gen.__file__).parent
    manifest = dict(config=D1_REG, split=split, calibration_rows=CALIBRATION_ROWS, records=records,
                    observation_version=VERSION, location_unit_policy=UNIT_POLICY,
                    structural_policy=STRUCTURAL_POLICY, scm_operator_policy=POLICY,
                    observation_storage_policy=STORAGE_POLICY,
                    versions={k: version(k) for k in ("numpy", "torch", "tabicl")},
                    source_sha256={name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                   for name in ("generate.py", "regression.py", "observation.py", "reg_scm.py")})
    gen.atomic_write(out / "manifest.json", lambda f: f.write(json.dumps(manifest, indent=2).encode()))
    return manifest


def evaluate_d1_reg(model, root, *, device="cpu"):
    from lightpfn.model_v2.estimator import V2Regressor
    from lightpfn.prior.generate import validate_group
    from lightpfn.train import typed_codes
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["config"] != D1_REG:
        raise ValueError("unknown frozen D1-reg configuration")
    results = []
    for record in manifest["records"]:
        path = root / record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("D1-reg hash mismatch")
        group = torch.load(path, weights_only=True)
        validate_group(group)
        ntr, di = group["n_train"], int(group["d"][0])
        X = group["X"][0, :, :di].double().numpy().copy()
        cat = group["cat_mask"][0, :di].numpy()
        codes = typed_codes(group)[0, :, :di]
        X[:, cat] = codes[:, cat]
        y = group["y_reg"][0].numpy()
        estimator = V2Regressor(model, device=device).fit(X[:ntr], y[:ntr], cat)
        q = estimator.predict_quantiles(X[ntr:])
        metrics = regression_metrics(y[:ntr], y[ntr:], q, seed=record["seed"])
        results.append(dict(file=record["file"], family="graph_v3", channel=record["channel"], **metrics))
    # Near-constants are explicit, not silently averaged into std-normalized metrics.
    regular = [r for r in results if not r["near_constant"]]
    summary = {k: float(np.mean([r[k] for r in regular])) if regular else None
               for k in ("crps_std", "rmse_std", "mae", "coverage_50", "coverage_80", "coverage_90")}
    return dict(name=D1_REG["name"], split=manifest["split"], tasks=results, macro=summary,
                near_constant_tasks=sum(r["near_constant"] for r in results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--split", choices=list(D1_REG["splits"]), default="dev")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--root", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "build":
        build_d1_reg(args.out, args.split)
    else:
        from lightpfn.model_v2.estimator import load_v2
        model = load_v2(args.checkpoint, args.device)
        if model is None or model.regression_head is None:
            parser.error("a v2 regression checkpoint is required")
        args.out.write_text(json.dumps(evaluate_d1_reg(model, args.root, device=args.device), indent=2))


if __name__ == "__main__":
    main()
