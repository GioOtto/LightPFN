"""Held-out synthetic D1 and known-mechanism D2 evaluation.

Build: python -m lightpfn.eval.dev build --n-jobs 8
Eval:  python -m lightpfn.eval.dev evaluate --checkpoint runs/train/r2/ema_step015000.pt --run r2
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score

from lightpfn.prior.generate import check_resume, validate_group
from lightpfn.eval.probes import N_TRAIN, PROBES, probe

ROOT = Path(__file__).resolve().parents[2]
D1 = ROOT / "data/dev/D1"
DEV_SEED = 0xD1DE0001
FAMILIES = {"graph_v3": ("graph", 3, .55), "tree_v4": ("tree", 4, .55),
            "graph_v4": ("graph", 4, .78), "rule": ("rule", 1, .78)}


def build_dev(out=D1, tasks_per_family=384, seed=DEV_SEED, n_jobs=8):
    """Build four versioned pools, rejecting seed reuse with training/pilot pools."""
    if tasks_per_family < 8 or tasks_per_family % 8 or not 1 <= n_jobs <= 12:
        raise ValueError("dev task count must be a multiple of 8; jobs must be 1..12")
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("dev seed must fit uint32")
    out = Path(out)
    for base in (ROOT / "data/prior",):
        for path in base.rglob("meta.json"):
            meta = json.loads(path.read_text())
            if meta.get("seed") == seed:
                raise ValueError(f"dev seed already used in a non-dev pool: {path}")
    for family, (prior, version, pbin) in FAMILIES.items():
        command = [sys.executable, "-m", "lightpfn.prior.generate", "--preset", "s1b",
                   "--prior", prior, "--prior-version", str(version), "--geometry", "ratio",
                   "--p-binary", str(pbin), "--out", str(out/family), "--n-tasks", str(tasks_per_family),
                   "--n-jobs", str(n_jobs), "--groups-per-shard", "8", "--seed", str(seed),
                   "--min-free-gb", "0"]
        subprocess.run(command, cwd=ROOT, check=True)
    return out


def dev_tasks(out=D1, limit=None):
    """Stable task IDs; no missingness/augmentation in validation."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    for family in FAMILIES:
        folder = Path(out)/family
        meta = json.loads((folder/"meta.json").read_text())
        prior, version, pbin = FAMILIES[family]
        if (meta["prior"] != prior or meta["prior_version"] != version or meta["geometry"] != "ratio"
                or meta["preset"] != "s1b" or meta["p_binary"] != pbin):
            raise ValueError(f"incorrect dev family configuration: {family}")
        lengths = check_resume(folder, meta)
        if sum(lengths) * meta["preset_cfg"]["group_size"] != meta["n_tasks"]:
            raise ValueError(f"incomplete dev family: {family}")
        count = 0
        for f in sorted(folder.glob("shard_*.pt")):
            for gi, group in enumerate(torch.load(f, map_location="cpu", weights_only=True)):
                validate_group(group)
                for i, d in enumerate(group["d"]):
                    yield family, f"{f.stem}:{gi}:{i}", group["X"][i, :, :int(d)].float().numpy(), group["y"][i].numpy(), group["n_train"]
                    count += 1
                    if limit is not None and count >= limit:
                        break
                if limit is not None and count >= limit:
                    break
            if limit is not None and count >= limit:
                break
        if count < min(meta["n_tasks"], limit or meta["n_tasks"]):
            raise ValueError(f"incomplete dev family: {family}")


def metrics(y, p):
    p = np.asarray(p, dtype=np.float64)
    if (p.ndim != 2 or p.shape[0] != len(y) or p.shape[1] < 2 or not np.isfinite(p).all()
            or np.any(p < 0) or not np.allclose(p.sum(1), 1, rtol=1e-4, atol=1e-8)):
        raise ValueError("invalid class probabilities")
    # GPU/CPU float32 softmax can differ from unity by a few ulps. Normalize in
    # float64 before sklearn's metrics, without concealing invalid predictions.
    p = p / p.sum(1, keepdims=True)
    C = p.shape[1]
    auc = roc_auc_score(y, p[:, 1]) if C == 2 else roc_auc_score(y, p, multi_class="ovr", labels=np.arange(C))
    return dict(auc=float(auc), logloss=float(log_loss(y, p, labels=np.arange(C))))


@contextmanager
def _evaluation_threads(threads):
    from threadpoolctl import threadpool_limits
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        with threadpool_limits(limits=threads):
            yield
    finally:
        torch.set_num_threads(previous)


def evaluate_checkpoint(checkpoint, run, *, dev_dir=D1, device="auto", threads=4,
                        limit=None, probe_limit=None, probe_seeds=5, classifier=None):
    """CSV per task, D1 family means, and D2 probe means; exactly one estimator."""
    if not 1 <= threads <= 12 or probe_seeds < 1 or (probe_limit is not None and probe_limit < 0):
        raise ValueError("invalid evaluation limits")
    if not run or Path(run).name != run or run in (".", ".."):
        raise ValueError("run must be a single directory name")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    from lightpfn.sklearn import LightPFNClassifier
    out = ROOT / "runs/dev" / run
    out.mkdir(parents=True, exist_ok=True)
    rows, probes = [], []
    with _evaluation_threads(threads):
        clf = classifier or LightPFNClassifier(checkpoint=checkpoint, device=device, n_estimators=1, n_threads=threads, seed=DEV_SEED)
        for family, task_id, X, y, ntr in dev_tasks(dev_dir, limit):
            t0 = time.perf_counter()
            clf.fit(X[:ntr], y[:ntr])
            result = metrics(y[ntr:], clf.predict_proba(X[ntr:]))
            rows.append(dict(family=family, task_id=task_id, **result, seconds=time.perf_counter()-t0))
        for name, d in PROBES[:probe_limit]:
            for s in range(probe_seeds):
                X, y = probe(name, d, np.random.default_rng([DEV_SEED, s, d, len(name)]))
                t0 = time.perf_counter()
                clf.fit(X[:N_TRAIN], y[:N_TRAIN])
                result = metrics(y[N_TRAIN:], clf.predict_proba(X[N_TRAIN:]))
                probes.append(dict(probe=name, d=d, seed=s, **result, seconds=time.perf_counter()-t0))
    df = pd.DataFrame(rows)
    df.to_csv(out/"D1_tasks.csv", index=False)
    df.groupby("family").agg(tasks=("auc", "size"), auc=("auc", "mean"), logloss=("logloss", "mean")).to_csv(out/"D1.csv")
    pd.DataFrame(probes, columns=["probe", "d", "seed", "auc", "logloss", "seconds"]).to_csv(out/"D2_tasks.csv", index=False)
    if probes:
        pd.DataFrame(probes).groupby(["probe", "d"])[["auc", "logloss"]].mean().to_csv(out/"D2.csv")
    else:
        pd.DataFrame(columns=["probe", "d", "auc", "logloss"]).to_csv(out/"D2.csv", index=False)
    (out/"meta.json").write_text(json.dumps(dict(checkpoint=str(checkpoint), dev_dir=str(dev_dir), device=device,
        n_estimators=1, threads=threads, limit=limit, probe_limit=probe_limit, probe_seeds=probe_seeds, seed=DEV_SEED), indent=2))
    return df, pd.DataFrame(probes)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build")
    build.add_argument("--out", type=Path, default=D1)
    build.add_argument("--tasks-per-family", type=int, default=384)
    build.add_argument("--seed", type=lambda s: int(s, 0), default=DEV_SEED)
    build.add_argument("--n-jobs", type=int, default=8)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--checkpoint", type=Path, required=True)
    ev.add_argument("--run", required=True)
    ev.add_argument("--dev-dir", type=Path, default=D1)
    ev.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ev.add_argument("--threads", type=int, default=4)
    ev.add_argument("--limit", type=int)
    ev.add_argument("--probe-limit", type=int)
    ev.add_argument("--probe-seeds", type=int, default=5)
    args = vars(ap.parse_args())
    if args.pop("action") == "build":
        build_dev(**args)
    else:
        checkpoint, run = args.pop("checkpoint"), args.pop("run")
        df, _ = evaluate_checkpoint(checkpoint, run, **args)
        print(df.groupby("family")[["auc", "logloss"]].mean().to_string())


if __name__ == "__main__":
    main()
