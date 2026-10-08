"""Summarizes the eval harness results of all models run in one mode.

Per task, the split scores are averaged. The ranking metric follows TabArena: 1 - AUC for binary
tasks, log loss for multiclass tasks. Reported per model:
  auc         mean ROC AUC over tasks
  rank        mean rank over tasks (1 = best)
  norm_err    ranking metric rescaled per task to [0, 1] between the best and worst model
  win_<m>     fraction of tasks where the model beats model m (ties count 1/2)
  fit_s       mean fit time per split, pred_s_1k predict time per 1000 test rows
and mean AUC / rank per task group (binary, multiclass, imbalanced, small/medium/large, ...).

Usage (from the project root):
    python -m lightpfn.eval.report --mode lite [--models rf catboost ...]
"""

import argparse

import numpy as np
import pandas as pd

from lightpfn.eval.data import DATA_DIR
from lightpfn.eval.harness import RUNS_DIR


def task_groups():
    info = pd.read_json(DATA_DIR / "info.json")
    g = pd.DataFrame(index=info.task_id)
    g["binary"] = (info.n_classes == 2).values
    g["multiclass"] = (info.n_classes > 2).values
    g["imbalanced"] = (info.minority_frac < 0.1).values
    g["small<2.5k"] = (info.n_rows < 2500).values
    g["medium"] = ((info.n_rows >= 2500) & (info.n_rows < 10000)).values
    g["large>=10k"] = (info.n_rows >= 10000).values
    g["features>100"] = (info.n_features > 100).values
    g["cat>50%"] = (info.cat_frac > 0.5).values
    g["missing"] = (info.nan_frac > 0).values
    return g


def load_results(mode, models=None, max_repeats=1):
    """Results of the splits in the current benchmark definition (rows of older or changed splits
    in the files are ignored)."""
    from lightpfn.eval.harness import make_jobs

    current = {j.split_id for j in make_jobs(mode, max_repeats=max_repeats)}
    frames = []
    for f in sorted((RUNS_DIR / mode).glob("*.csv")):
        if models is None or f.stem in models:
            df = pd.read_csv(f)
            if "split_id" not in df.columns:
                print(f"skipping {f}: no split_id column")
                continue
            frames.append(df[df.split_id.isin(current)].assign(model=f.stem))
    df = pd.concat(frames, ignore_index=True)
    df["error"] = df["error"].fillna("")
    return df


def per_task(df):
    df = df.assign(pred_s_1k=df.predict_s / df.n_test * 1000, failed=df.error != "")
    t = df.groupby(["model", "task_id"]).agg(
        name=("name", "first"), n_classes=("n_classes", "first"), auc=("auc", "mean"), logloss=("logloss", "mean"),
        fit_s=("fit_s", "mean"), pred_s_1k=("pred_s_1k", "mean"), failed=("failed", "sum"),
        splits=("split_id", frozenset),
    ).reset_index()
    t["err"] = np.where(t.n_classes == 2, 1 - t.auc, t.logloss)
    return t


def summarize(t, groups):
    # tasks are compared only when every model has exactly the same set of splits; failed splits
    # keep their uniform-prediction scores (penalty) but are excluded from the timings
    n_models = t.model.nunique()
    same = t.groupby("task_id").agg(n=("model", "size"), k=("splits", lambda s: len(set(s))))
    keep = same.index[(same.n == n_models) & (same.k == 1)]
    t = t[t.task_id.isin(keep)].copy()
    no_failures = t.groupby("task_id").failed.sum().loc[lambda s: s == 0].index
    timed = t[t.task_id.isin(no_failures)]
    err = t.pivot(index="task_id", columns="model", values="err")
    auc = t.pivot(index="task_id", columns="model", values="auc")
    rank = err.rank(axis=1)
    span = (err.max(1) - err.min(1)).replace(0, np.nan)
    norm = err.sub(err.min(1), axis=0).div(span, axis=0).fillna(0)

    s = pd.DataFrame({"auc": auc.mean(), "rank": rank.mean(), "norm_err": norm.mean()})
    for m in err.columns:
        wins = (err.lt(err[m], axis=0).astype(float) + 0.5 * err.eq(err[m], axis=0)).mean()
        s[f"win_{m}"] = wins
    s["fit_s"] = timed.groupby("model").fit_s.mean()
    s["pred_s_1k"] = timed.groupby("model").pred_s_1k.mean()
    s["failed"] = t.groupby("model").failed.sum()
    s = s.sort_values("rank")

    g_rank, g_auc = {}, {}
    for gname in groups.columns:
        ids = [i for i in groups.index[groups[gname]] if i in rank.index]
        g_rank[f"{gname} ({len(ids)})"] = rank.loc[ids].mean()
        g_auc[f"{gname} ({len(ids)})"] = auc.loc[ids].mean()
    return s, pd.DataFrame(g_rank).loc[s.index], pd.DataFrame(g_auc).loc[s.index], auc, len(keep)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["lite", "full"], default="lite")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--max-repeats", type=int, default=1, help="full mode: repeats in the benchmark definition")
    p.add_argument("--per-task", action="store_true", help="also print the AUC of every task")
    args = p.parse_args()

    t = per_task(load_results(args.mode, args.models, args.max_repeats))
    s, g_rank, g_auc, auc, n_tasks = summarize(t, task_groups())
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    missing = sorted(set(t.name) - set(t[t.task_id.isin(auc.index)].name))
    print(f"=== {args.mode}: {n_tasks} tasks with the same splits for every model ===")
    if missing:
        print(f"excluded (splits differ or missing): {', '.join(missing)}")
    print(s.round(4).to_string())
    print("\n=== mean rank per task group (1 = best) ===")
    print(g_rank.round(2).to_string())
    print("\n=== mean AUC per task group ===")
    print(g_auc.round(4).to_string())
    if args.per_task:
        names = t.drop_duplicates("task_id").set_index("task_id").name
        print("\n=== AUC per task ===")
        print(auc.rename(index=names).round(4).to_string())
    s.to_csv(RUNS_DIR / f"{args.mode}_summary.csv")


if __name__ == "__main__":
    main()
