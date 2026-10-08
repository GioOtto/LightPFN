"""Learning curves on the large D3 tasks: AUC against the number of training rows. D3 caps every task
at 1000 rows, so it says nothing about large tables; this does.

Tasks: the kept D3 tasks with >= 5000 rows. Per task and repeat: a stratified test set of min(2000, n/5)
rows, then stratified training sets of 1k, 2k, 4k, 8k, 16k and 32k rows and one with all the remaining
rows, drawn from the rest; <= 100 random features as in D3. LightPFN runs as three variants: the whole
training set as context (`full`), a 2048-row stratified subsample (`c2k`, the largest context seen in
pretraining) and the mean of 8 such subsamples (`c2k_x8`). Results go to runs/eval/scale/<model>.csv,
with the harness's columns and resume rules.

    python -m lightpfn.eval.scale run --models catboost --n-threads 10
    python -m lightpfn.eval.scale run --checkpoint runs/train/final/ema_step030000.pt --name final_s30000
    python -m lightpfn.eval.scale report --models final_s30000_full final_s30000_c2k final_s30000_c2k_x8 catboost
"""

import argparse

import numpy as np
import pandas as pd

from lightpfn.eval.data import ROOT
from lightpfn.sklearn import stratified_subsample

RESULTS = ROOT / "runs" / "eval" / "scale"
SIZES = [1000, 2000, 4000, 8000, 16000, 32000]
MIN_ROWS = 5000
VARIANTS = {"full": dict(max_context=10**9, n_estimators=1),
            "c2k": dict(max_context=2048, n_estimators=1),
            "c2k_x8": dict(max_context=2048, n_estimators=8)}


def scale_jobs(tasks, seed=0, n_repeats=2, max_features=100, sizes=SIZES, min_rows=MIN_ROWS, test_rows=2000,
               all_rows=True):
    """Per task and repeat: a stratified test set of min(test_rows, n/5) rows, then stratified training sets
    of each size below the remaining rows, plus all of them if all_rows."""
    from lightpfn.eval.harness import Job, split_hash

    jobs = []
    for task in tasks:
        n, m = task.X.shape
        if n < min_rows:
            continue
        for r in range(n_repeats):
            rng = np.random.default_rng([seed, task.task_id, r, 7])
            test = stratified_subsample(rng, task.y, min(test_rows, n // 5))
            pool = np.setdiff1d(np.arange(n), test)
            cols = np.sort(rng.choice(m, max_features, replace=False)) if m > max_features else None
            sizes_t = [s for s in sizes if s < len(pool)] + ([len(pool)] if all_rows else [])
            for i, size in enumerate(sizes_t):
                train = pool[stratified_subsample(rng, task.y[pool], size)] if size < len(pool) else pool
                s = seed * 1000 + r * 10 + i
                jobs.append(Job(task.task_id, r, i, s, train, test, cols, cost=size * min(m, max_features),
                                split_id=split_hash(task.task_id, train, test, cols, s)))
    return jobs


def run(model_names=(), checkpoint=None, name=None, variants=tuple(VARIANTS), n_threads=8):
    from lightpfn.eval.external import _use_external_tasks, kept_tasks

    _use_external_tasks()
    evaluate_all(scale_jobs(kept_tasks()), RESULTS, model_names, checkpoint, name, variants, n_threads)


def evaluate_all(jobs, results, model_names=(), checkpoint=None, name=None, variants=tuple(VARIANTS), n_threads=8):
    """The tree ensembles by name and the LightPFN variants of a checkpoint on jobs, into results/<model>.csv."""
    from lightpfn.eval.harness import evaluate

    results.mkdir(parents=True, exist_ok=True)
    for m in model_names:
        print(f"== {m} ({len(jobs)} splits)", flush=True)
        evaluate(m, jobs, n_threads=n_threads, out_csv=results / f"{m}.csv", verbose=True)
    if checkpoint:
        from lightpfn.sklearn import LightPFNClassifier, load_model

        model = load_model(checkpoint, "cpu")
        for v in variants:
            def factory(n_threads, seed, kw=VARIANTS[v]):
                return LightPFNClassifier(model=model, device="cpu", seed=seed, n_threads=n_threads, **kw)

            print(f"== {name}_{v} ({len(jobs)} splits)", flush=True)
            evaluate(factory, jobs, n_threads=n_threads, out_csv=results / f"{name}_{v}.csv", verbose=True)


def size_label(n_train, sizes=SIZES):
    return f"{n_train // 1000}k" if n_train in sizes else "all"


def report(models, ref="catboost", n_boot=20000, results=RESULTS, sizes=SIZES):
    frames = [pd.read_csv(results / f"{m}.csv").assign(model=m) for m in models]
    df = pd.concat(frames, ignore_index=True)
    df["size"] = df.n_train.map(lambda n: size_label(n, sizes))
    df["seconds"] = df.fit_s + df.predict_s
    order = [o for o in [f"{s // 1000}k" for s in sizes] + ["all"] if o in set(df["size"])]
    # per task and size: mean over repeats (only splits every model has)
    t = df.pivot_table(index=["size", "task_id"], columns="model", values="auc", aggfunc="mean").dropna()
    pd.set_option("display.width", 200)
    print("mean AUC over tasks")
    print(t.groupby("size").mean().reindex(order)[models].round(4).assign(tasks=t.groupby("size").size()).to_string())
    print(f"\n{'model'} - {ref}, AUC points: mean [95% bootstrap over tasks], tasks where higher")
    rng = np.random.default_rng(0)
    for m in models:
        if m == ref:
            continue
        cells = []
        for size in order:
            if size not in t.index.get_level_values(0):
                continue
            d = 100 * (t.loc[size, m] - t.loc[size, ref]).to_numpy()
            boot = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
            cells.append(f"{size}: {d.mean():+.2f} [{np.quantile(boot, .025):+.2f}, {np.quantile(boot, .975):+.2f}] "
                         f"{(d > 0).sum()}/{len(d)}")
        print(f"{m}\n  " + "\n  ".join(cells))
    print("\nmedian seconds per split (fit + predict)")
    print(df.pivot_table(index="size", columns="model", values="seconds", aggfunc="median")
          .reindex(order)[models].round(1).to_string())
    errors = df[df.error.fillna("") != ""]
    if len(errors):
        print(f"\n{len(errors)} failed splits (scored as uniform predictions):")
        print(errors[["model", "name", "n_train", "error"]].to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--models", nargs="*", default=[])
    r.add_argument("--checkpoint")
    r.add_argument("--name")
    r.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    r.add_argument("--n-threads", type=int, default=8)
    s = sub.add_parser("report")
    s.add_argument("--models", nargs="+", required=True)
    s.add_argument("--ref", default="catboost")
    args = p.parse_args()
    if args.cmd == "run":
        run(args.models, args.checkpoint, args.name, args.variants, args.n_threads)
    else:
        report(args.models, args.ref)


if __name__ == "__main__":
    main()
