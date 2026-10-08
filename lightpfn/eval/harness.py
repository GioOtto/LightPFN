"""Evaluates classifiers on the 38 TabArena-v0.1 classification tasks.

Two modes:
  lite  every task subsampled to <= 1000 rows (stratified, >= 5 rows per class when possible) and
        <= 100 random features, 5-fold stratified CV. Cheap enough to run during training.
  full  the official OpenML splits (3 folds x up to --max-repeats repeats) on the full data.
Scores per split: ROC AUC (macro one-vs-rest for multiclass), log loss, accuracy, fit and predict
wall time. Every split has a split_id (hash of task, rows, columns and model seed). Results are
appended to runs/eval/<mode>/<model>.csv as jobs finish; a rerun skips the splits that already
succeeded with the same split_id and reruns failed or changed ones. A failed split is recorded
with the error and scored as a uniform prediction (explicit penalty).
Summaries: `python -m lightpfn.eval.report --mode lite`.

Usage (from the project root):
    python -m lightpfn.eval.harness --models rf lightgbm xgboost catboost --mode lite
    python -m lightpfn.eval.harness --models catboost --mode full --max-repeats 1 --n-threads 8 --n-workers 2
    python -m lightpfn.eval.harness --checkpoint model.safetensors --name lightpfn_n4 --n-estimators 4 --mode full
"""

import argparse
import time
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from lightpfn.eval.data import ROOT, load_task, load_tasks
from lightpfn.sklearn import stratified_subsample

RUNS_DIR = ROOT / "runs" / "eval"
LOGLOSS_EPS = 1e-7


@dataclass
class Job:
    task_id: int
    repeat: int
    fold: int
    seed: int
    train: np.ndarray | None = None  # explicit rows (lite); None = official split (full)
    test: np.ndarray | None = None
    cols: np.ndarray | None = None  # feature subset (lite), None = all features
    cost: float = 0.0
    split_id: str = ""  # hash of task, rows, columns and model seed: identifies the split in results


def split_hash(task_id, train, test, cols, seed):
    import hashlib

    h = hashlib.blake2b(digest_size=8)
    for a in (np.array([task_id, seed]), np.sort(train), np.sort(test), np.array([-1] if cols is None else cols)):
        h.update(np.ascontiguousarray(a, dtype=np.int64).tobytes())
    return h.hexdigest()


def lite_jobs(tasks, seed=0, max_rows=1000, max_features=100, n_folds=5, n_repeats=1):
    from sklearn.model_selection import StratifiedKFold

    jobs = []
    for task in tasks:
        n, m = task.X.shape
        for r in range(n_repeats):
            rng = np.random.default_rng([seed, task.task_id, r])
            rows = stratified_subsample(rng, task.y, max_rows) if n > max_rows else np.arange(n)
            cols = np.sort(rng.choice(m, max_features, replace=False)) if m > max_features else None
            skf = StratifiedKFold(n_folds, shuffle=True, random_state=int(rng.integers(2**31)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # classes with fewer rows than folds
                splits = list(skf.split(rows, task.y[rows]))
            for f, (tr, te) in enumerate(splits):
                cost = len(tr) * min(m, max_features)
                s = seed * 1000 + r * 10 + f
                sid = split_hash(task.task_id, rows[tr], rows[te], cols, s)
                jobs.append(Job(task.task_id, r, f, s, rows[tr], rows[te], cols, cost, sid))
    return jobs


def full_jobs(tasks, seed=0, max_repeats=1):
    jobs = []
    for task in tasks:
        for r in range(min(max_repeats, task.n_repeats)):
            for f in range(task.n_folds):
                cost = len(task.y) * task.X.shape[1]
                s = seed * 1000 + r * 10 + f
                sid = split_hash(task.task_id, *task.split(r, f), None, s)
                jobs.append(Job(task.task_id, r, f, s, cost=cost, split_id=sid))
    return jobs


def auc_ovr(y, P):
    from sklearn.metrics import roc_auc_score

    if P.shape[1] == 2:
        return roc_auc_score(y, P[:, 1]) if 0 < y.sum() < len(y) else np.nan
    scores = [roc_auc_score(y == c, P[:, c]) for c in range(P.shape[1]) if 0 < (y == c).sum() < len(y)]
    return float(np.mean(scores)) if scores else np.nan


def log_loss(y, P):
    P = np.clip(P, LOGLOSS_EPS, 1.0)
    P = P / P.sum(1, keepdims=True)
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))


@lru_cache(maxsize=8)
def _task(task_id):
    return load_task(task_id)


def run_job(model, job, n_threads):
    """Fits and scores one split. `model` is a registry name or a callable (n_threads, seed)."""
    from threadpoolctl import threadpool_limits

    from lightpfn.eval.models import make_model

    task = _task(job.task_id)
    if job.train is None:
        train, test = task.split(job.repeat, job.fold)
    else:
        train, test = job.train, job.test
    X = task.X if job.cols is None else task.X[:, job.cols]
    cat = task.cat if job.cols is None else task.cat[job.cols]
    Xtr, ytr, Xte, yte = X[train], task.y[train], X[test], task.y[test]
    C = task.n_classes
    classes = np.unique(ytr)

    res = dict(task_id=job.task_id, name=task.name, repeat=job.repeat, split_id=job.split_id, fold=job.fold,
               n_train=len(train), n_test=len(test), n_features=X.shape[1], n_classes=C, fit_s=np.nan,
               predict_s=np.nan, error="")
    P = np.zeros((len(test), C))
    try:
        if len(classes) == 1:
            P[:, classes[0]] = 1.0
        else:
            with threadpool_limits(n_threads), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est = make_model(model, n_threads, job.seed) if isinstance(model, str) else model(n_threads, job.seed)
                t0 = time.perf_counter()
                est.fit(Xtr, np.searchsorted(classes, ytr), cat)
                res["fit_s"] = time.perf_counter() - t0
                t0 = time.perf_counter()
                P[:, classes] = est.predict_proba(Xte)
                res["predict_s"] = time.perf_counter() - t0
    except Exception as e:  # a failed split is recorded, not fatal
        res["error"] = f"{type(e).__name__}: {e}"[:300]
        P[:] = 1.0 / C
    res.update(auc=auc_ovr(yte, P), logloss=log_loss(yte, P), acc=float(np.mean(P.argmax(1) == yte)))
    return res


def evaluate(model, jobs, n_threads=4, n_workers=1, out_csv=None, verbose=True):
    """Runs `jobs` and returns their results as a DataFrame. With out_csv, appends each result as it
    arrives, skips splits that already succeeded (same split_id) and reruns failed ones."""
    done, columns = set(), None
    if out_csv is not None and Path(out_csv).exists():
        prev = pd.read_csv(out_csv)
        if "split_id" not in prev.columns:
            raise SystemExit(f"{out_csv} has no split_id column: results from an older harness")
        failed = prev["error"].fillna("") != ""
        if failed.any():  # failed splits are dropped from the file and run again
            prev = prev[~failed]
            prev.to_csv(out_csv, index=False)
        done, columns = set(prev.split_id), list(prev.columns)
    todo = sorted([j for j in jobs if j.split_id not in done], key=lambda j: -j.cost)
    if verbose:
        print(f"{len(jobs) - len(todo)} splits done, {len(todo)} to run", flush=True)

    if n_workers > 1:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_workers, return_as="generator_unordered")(
            delayed(run_job)(model, j, n_threads) for j in todo
        )
    else:
        results = (run_job(model, j, n_threads) for j in todo)

    rows = []
    t0 = time.time()
    for i, res in enumerate(results):
        rows.append(res)
        if out_csv is not None:
            row = pd.DataFrame([res])
            if columns is None:
                columns = list(row.columns)
                row.to_csv(out_csv, index=False)
            else:
                row[columns].to_csv(out_csv, mode="a", header=False, index=False)
        if verbose:
            err = f" ERROR {res['error']}" if res["error"] else ""
            print(f"[{i + 1}/{len(todo)} {time.time() - t0:.0f}s] {res['name']} r{res['repeat']}f{res['fold']} "
                  f"auc={res['auc']:.4f} ll={res['logloss']:.4f} fit={res['fit_s']:.2f}s{err}", flush=True)
    df = pd.DataFrame(rows)
    if out_csv is not None:
        df = pd.read_csv(out_csv)
    return df[df.split_id.isin({j.split_id for j in jobs})] if len(df) else df


def make_jobs(mode, tasks=None, seed=0, max_repeats=1):
    tasks = load_tasks() if tasks is None else tasks
    return lite_jobs(tasks, seed) if mode == "lite" else full_jobs(tasks, seed, max_repeats)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=[], help="baselines: rf lightgbm xgboost catboost tabicl tabicl_gpu")
    p.add_argument("--checkpoint", default=None, help="a LightPFN checkpoint to evaluate as --name")
    p.add_argument("--name", default=None)
    p.add_argument("--n-estimators", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--mode", choices=["lite", "full"], default="lite")
    p.add_argument("--tasks", nargs="*", default=None, help="dataset names or task ids (default: all 38)")
    p.add_argument("--max-repeats", type=int, default=1, help="full mode: official repeats to run (TabArena uses 3 or 10)")
    p.add_argument("--n-threads", type=int, default=4, help="threads per fit")
    p.add_argument("--n-workers", type=int, default=5, help="fits in parallel")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if not args.models and not args.checkpoint:
        p.error("provide --models or --checkpoint with --name")
    if args.checkpoint and not args.name:
        p.error("--checkpoint needs --name")
    if args.name and not args.checkpoint:
        p.error("--name needs --checkpoint")
    if min(args.n_estimators, args.n_threads, args.n_workers, args.max_repeats) < 1:
        p.error("--n-estimators, --n-threads, --n-workers and --max-repeats must be positive")

    # use the imported module, not __main__: loky workers must be able to unpickle run_job
    from lightpfn.eval import harness

    tasks = load_tasks(args.tasks)
    jobs = harness.make_jobs(args.mode, tasks, args.seed, args.max_repeats)
    out_dir = RUNS_DIR / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [(name, name) for name in args.models]
    if args.checkpoint:
        from lightpfn.sklearn import LightPFNClassifier, load_model

        network = load_model(args.checkpoint, "cpu")

        def factory(n_threads, seed):
            return LightPFNClassifier(model=network, device=args.device, seed=seed, n_estimators=args.n_estimators,
                                      n_threads=n_threads if args.device == "cpu" else None)

        runs.append((args.name, factory))
    for name, model in runs:
        print(f"=== {name} ({args.mode}, {len(jobs)} splits, {args.n_workers}x{args.n_threads} threads)", flush=True)
        harness.evaluate(model, jobs, args.n_threads, args.n_workers, out_dir / f"{name}.csv")


if __name__ == "__main__":
    main()
