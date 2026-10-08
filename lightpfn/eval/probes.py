"""Diagnostic probes: small synthetic tasks, each isolating one structure the prior may miss.

Each probe has 800 train / 1000 test rows, relevant columns at random positions among noise
columns, several seeds. Models: LightPFN checkpoints (1 or 4 estimators), CatBoost default and,
for diagnosis only, TabICLv2 (it tells whether a 3-stage column -> row -> ICL model can solve
the probe at all). CPU only. Output: runs/study/probes.csv by default (one row per probe/seed/model).

    python -m lightpfn.eval.probes --threads 16 --models r2 r2x4 r3 catboost tabicl
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

N_TRAIN, N_TEST = 800, 1000


def _place(rng, cols, d):
    """Relevant columns `cols` (n, k) placed at random positions among d - k Gaussian noise columns."""
    n, k = cols.shape
    X = rng.normal(size=(n, d))
    pos = rng.choice(d, k, replace=False)
    X[:, pos] = cols
    return X


def probe(name, d, rng):
    n = N_TRAIN + N_TEST
    z = rng.normal(size=(n, 3))
    if name == "xor_decoys":
        y = (z[:, 0] > 0) ^ (z[:, 1] > 0)
        cols = np.c_[z[:, :2], z[:, :2] + rng.normal(0, .05, (n, 2))]
        X = _place(rng, cols, d)
    elif name == "xor2":
        y = (z[:, 0] > 0) ^ (z[:, 1] > 0)
        X = _place(rng, z[:, :2], d)
    elif name == "parity3":
        y = (z[:, 0] > 0) ^ (z[:, 1] > 0) ^ (z[:, 2] > 0)
        X = _place(rng, z, d)
    elif name == "tree_rule":  # depth-2 rule with uneven thresholds
        y = ((z[:, 0] > 0.5) & (z[:, 1] < 0.3)) | (z[:, 2] > 1.0)
        X = _place(rng, z, d)
    elif name == "sparse_linear":  # control: additive signal
        y = z @ np.array([1.0, -0.8, 0.6]) + 0.3 * rng.normal(size=n) > 0
        X = _place(rng, z, d)
    elif name in ("lookup16", "highcard64"):
        K = 16 if name == "lookup16" else 64
        p = np.ones(K) / K if K == 16 else (1 / np.arange(1, K + 1)) / (1 / np.arange(1, K + 1)).sum()
        c = rng.choice(K, n, p=p)
        table = rng.permutation(np.r_[np.zeros(K // 2), np.ones(K - K // 2)])
        y = table[c].astype(bool)
        if K == 64:
            flip = rng.random(n) < 0.1
            y = y ^ flip
        X = _place(rng, rng.permutation(K)[c][:, None].astype(float), d)
    elif name == "heavy_tail":  # zero-inflated log-normal, threshold in the far tail
        v = np.where(rng.random(n) < 0.7, 0.0, np.exp(2.0 * rng.normal(size=n)))
        y = v > np.quantile(v, 0.9)
        X = _place(rng, v[:, None], d)
    elif name == "ratio":  # multiplicative: x0 / x1 > 1 for positive log-normal columns
        a, b = np.exp(rng.normal(size=(2, n)))
        y = a / b > 1
        X = _place(rng, np.c_[a, b], d)
    else:
        raise ValueError(name)
    return X.astype(np.float32), y.astype(np.int64)


PROBES = [("xor2", d) for d in (2, 10, 30, 100)] + [("parity3", d) for d in (3, 10, 30)] \
    + [("tree_rule", d) for d in (3, 30, 100)] + [("sparse_linear", d) for d in (3, 100)] \
    + [("lookup16", d) for d in (1, 30, 100)] + [("highcard64", d) for d in (1, 30)] \
    + [("heavy_tail", d) for d in (1, 30)] + [("ratio", d) for d in (2, 30)]
PROBES += [("xor2", 50), ("parity3", 100), ("highcard64", 100), ("xor_decoys", 30)]


def make(model, threads, seed):
    if model.startswith(("r1", "r2", "r3")):
        from lightpfn.sklearn import LightPFNClassifier, load_model
        run, n_est = (model.split("x") + ["1"])[:2]
        ckpt = ROOT / "runs/train" / run / "ema_step015000.pt"
        return LightPFNClassifier(model=load_model(ckpt, "cpu"), device="cpu", n_estimators=int(n_est), n_threads=threads, seed=seed)
    if model == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(thread_count=threads, verbose=0, allow_writing_files=False, random_seed=seed)
    if model == "tabicl":
        from lightpfn.eval.models import TabICL
        return TabICL(threads, seed)
    raise ValueError(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["r2", "r2x4", "r3", "catboost", "tabicl"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default=str(ROOT / "runs/study/probes.csv"))
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    rows = []
    for model in args.models:
        clf_cache = None
        for name, d in PROBES:
            for s in range(args.seeds):
                X, y = probe(name, d, np.random.default_rng([s, d, len(name)]))
                Xtr, ytr, Xte, yte = X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]
                clf = make(model, args.threads, s) if clf_cache is None or not model.startswith("r") else clf_cache
                clf_cache = clf
                t0 = time.perf_counter()
                clf.fit(Xtr, ytr, None) if model == "tabicl" else clf.fit(Xtr, ytr)
                p = clf.predict_proba(Xte)[:, 1]
                rows.append(dict(model=model, probe=name, d=d, seed=s, auc=roc_auc_score(yte, p),
                                 seconds=time.perf_counter() - t0))
            r = [x["auc"] for x in rows if x["model"] == model and x["probe"] == name and x["d"] == d]
            print(f"{model:9s} {name:14s} d={d:3d} auc={np.mean(r):.4f}", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
    df = pd.DataFrame(rows)
    piv = df.groupby(["probe", "d", "model"]).auc.mean().unstack("model")
    print(piv.round(4).to_string())


if __name__ == "__main__":
    main()
