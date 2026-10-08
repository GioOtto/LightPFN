"""External real validation set D3: OpenML-CC18 classification tasks that are NOT in TabArena.

Used to choose between model variants without turning TabArena into a selection set: TabArena is
only looked at for the finalists. Never used for training. Same file format as lightpfn.eval.data
(data/external/cc18/<task_id>.npz) and the same lite protocol (<= 1000 rows, <= 100 random
features, 5-fold stratified CV, seed 0).

Exclusions: datasets of TabArena (by normalized name, or by the same row count and class counts),
more than 10 classes (pretraining covers 2-10), image/pixel datasets.

    python -m lightpfn.eval.external build                 # download once
    python -m lightpfn.eval.external run --models catboost lightgbm xgboost rf
    python -m lightpfn.eval.external run --checkpoint runs/train/r2/ema_step015000.pt --name r2 [--n-estimators 4]
    python -m lightpfn.eval.external report
"""

import argparse
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from lightpfn.eval.data import DATA_DIR, OPENML_CACHE, ROOT, build_task, load_task

EXT_DIR = ROOT / "data" / "external" / "cc18"
RESULTS = ROOT / "runs" / "eval" / "external"
CC18_SUITE = 99
IMAGE_DATASETS = {"mnist_784", "fashion-mnist", "cifar_10", "devnagari-script"}


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _fingerprints():
    """(rows, sorted class counts) and normalized names of the TabArena classification tasks."""
    prints, names = set(), set()
    for f in DATA_DIR.glob("*.npz"):
        with np.load(f, allow_pickle=True) as z:
            prints.add((len(z["y"]), tuple(sorted(np.bincount(z["y"]).tolist()))))
            names.add(_norm(str(z["name"])))
    return prints, names


def build(n_threads=8):
    import openml
    from concurrent.futures import ThreadPoolExecutor

    openml.config.set_root_cache_directory(str(OPENML_CACHE))
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    task_ids = openml.study.get_suite(CC18_SUITE).tasks
    with ThreadPoolExecutor(n_threads) as ex:
        for msg in ex.map(lambda t: build_task(t, 1, EXT_DIR), task_ids):
            print(msg, flush=True)
    prints, names = _fingerprints()
    rows = []
    for f in sorted(EXT_DIR.glob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        t = load_task(int(f.stem), EXT_DIR)
        counts = np.bincount(t.y)
        reason = ""
        if _norm(t.name) in names:
            reason = "tabarena name"
        elif (len(t.y), tuple(sorted(counts.tolist()))) in prints:
            reason = "tabarena fingerprint"
        elif t.n_classes > 10:
            reason = ">10 classes"
        elif t.name.lower() in IMAGE_DATASETS:
            reason = "image"
        rows.append(dict(task_id=t.task_id, name=t.name, n_rows=len(t.y), n_features=t.X.shape[1],
                         n_classes=t.n_classes, cat_frac=float(t.cat.mean()),
                         nan_frac=float(np.isnan(t.X).mean()), excluded=reason))
    table = pd.DataFrame(rows)
    table.to_csv(EXT_DIR / "tasks.csv", index=False)
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print(table.to_string(index=False))
    print(f"kept {int((table.excluded == '').sum())} of {len(table)}")


def kept_tasks():
    table = pd.read_csv(EXT_DIR / "tasks.csv").fillna({"excluded": ""})
    return [load_task(int(t), EXT_DIR) for t in table[table.excluded == ""].task_id]


def _use_external_tasks():
    """The harness loads tasks by id from data/tabarena; point it at the external folder."""
    from lightpfn.eval import harness

    harness._task = lru_cache(maxsize=8)(lambda task_id: load_task(task_id, EXT_DIR))


def run(model_names=(), checkpoint=None, name=None, n_estimators=1, n_threads=8, device="cpu"):
    from lightpfn.eval.harness import evaluate, lite_jobs

    _use_external_tasks()
    jobs = lite_jobs(kept_tasks(), seed=0)
    RESULTS.mkdir(parents=True, exist_ok=True)
    for m in model_names:
        print(f"== {m}", flush=True)
        evaluate(m, jobs, n_threads=n_threads, out_csv=RESULTS / f"{m}.csv", verbose=False)
    if checkpoint:
        from lightpfn.sklearn import LightPFNClassifier, load_model

        model = load_model(checkpoint, device)

        def factory(n_threads, seed):
            return LightPFNClassifier(model=model, device=device, seed=seed, n_estimators=n_estimators,
                                    n_threads=n_threads if device == "cpu" else None)

        print(f"== {name}", flush=True)
        evaluate(factory, jobs, n_threads=n_threads, out_csv=RESULTS / f"{name}.csv", verbose=False)


def report(models=None):
    from lightpfn.eval.report import per_task, summarize

    frames = []
    for f in sorted(RESULTS.glob("*.csv")):
        if models is None or f.stem in models:
            frames.append(pd.read_csv(f).assign(model=f.stem))
    t = per_task(pd.concat(frames, ignore_index=True).fillna({"error": ""}))
    classes = t.drop_duplicates("task_id").set_index("task_id").n_classes
    groups = pd.DataFrame({"binary": classes == 2, "multiclass": classes > 2})
    s, g_rank, g_auc, auc, n_tasks = summarize(t, groups)
    pd.set_option("display.width", 250, "display.max_columns", 50)
    print(f"=== D3 external: {n_tasks} tasks with the same splits for every model ===")
    print(s.round(4).to_string())
    print("\n=== mean rank per task group (1 = best) ===")
    print(g_rank.round(2).to_string())
    print("\n=== mean AUC per task group ===")
    print(g_auc.round(4).to_string())
    s.to_csv(RESULTS.parent / "external_summary.csv")
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["build", "run", "report"])
    p.add_argument("--models", nargs="*", default=[])
    p.add_argument("--checkpoint")
    p.add_argument("--name")
    p.add_argument("--n-estimators", type=int, default=1)
    p.add_argument("--n-threads", type=int, default=8)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.cmd == "build":
        build(args.n_threads)
    elif args.cmd == "run":
        if args.checkpoint and not args.name:
            p.error("--checkpoint needs --name")
        run(args.models, args.checkpoint, args.name, args.n_estimators, args.n_threads, args.device)
    else:
        report(args.models or None)


if __name__ == "__main__":
    main()
