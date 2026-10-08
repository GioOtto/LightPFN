"""Large-table set D4: OpenML classification datasets with at least 50,000 rows that are in neither TabArena nor
D3 (OpenML-CC18), to compare LightPFN with default tree ensembles at 10k-100k training rows. The scale test
(lightpfn.eval.scale) reaches 32k rows on 6 D3 datasets only and guided the long-context stage; D4 is a fresh check.
Never used for training.

Candidates: the classification tasks of the AutoML benchmark (OpenML suite 271) and of the tabular benchmark of
Grinsztajn et al. (suites 334 and 337) with 50,000+ rows, 2-10 classes and <= 1,000 columns, minus TabArena (by name, or by row and
class counts, as D3), D3 by name, image datasets and repeated names. Protocol per dataset, one repeat: a
stratified test set of 5,000 rows, stratified training sets of 10k, 25k, 50k and 100k rows from the rest (those
that fit), <= 100 random features (lightpfn.eval.scale.scale_jobs). Data in data/external/large/<task_id>.npz,
results in runs/eval/large/<model>.csv.

    python -m lightpfn.eval.large build
    python -m lightpfn.eval.large run --models catboost lightgbm xgboost --n-threads 24
    python -m lightpfn.eval.large run --checkpoint runs/train/final_long/ema_step022000.pt --name final_long
    python -m lightpfn.eval.large report --models final_long_full catboost lightgbm xgboost
"""

import argparse
from functools import lru_cache

import pandas as pd

from lightpfn.eval.data import OPENML_CACHE, ROOT, build_task, load_task
from lightpfn.eval.external import EXT_DIR, IMAGE_DATASETS, _fingerprints, _norm
from lightpfn.eval.scale import evaluate_all, report, scale_jobs

LARGE_DIR = ROOT / "data" / "external" / "large"
RESULTS = ROOT / "runs" / "eval" / "large"
SUITES = [271, 334, 337]
MIN_ROWS = 50_000
SIZES = [10_000, 25_000, 50_000, 100_000]
TEST_ROWS = 5_000
MAX_COLUMNS = 1_000  # wider sets (KDDCup09-Upselling, 14,892 sparse columns) would be dense arrays of GBs


def candidates():
    """Classification tasks of the suites with MIN_ROWS+ rows, 2-10 classes and <= MAX_COLUMNS columns, one per name."""
    import openml

    openml.config.set_root_cache_directory(str(OPENML_CACHE))
    ids = sorted({t for s in SUITES for t in openml.study.get_suite(s).tasks})
    df = openml.tasks.list_tasks(task_id=ids, output_format="dataframe")
    df = df[(df.task_type == "Supervised Classification") & (df.NumberOfInstances >= MIN_ROWS)
            & df.NumberOfClasses.between(2, 10) & (df.NumberOfFeatures <= MAX_COLUMNS)].copy()
    df["norm"] = df.name.map(_norm)
    return df.sort_values("tid").drop_duplicates("norm")[["tid", "name", "NumberOfInstances", "NumberOfFeatures",
                                                          "NumberOfClasses"]]


def build():
    cand = candidates()
    d3 = set(pd.read_csv(EXT_DIR / "tasks.csv").name.map(_norm))
    prints, names = _fingerprints()
    LARGE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in cand.itertuples():
        reason = ""
        if _norm(c.name) in names:
            reason = "tabarena name"
        elif _norm(c.name) in d3:
            reason = "D3"
        elif c.name.lower() in IMAGE_DATASETS:
            reason = "image"
        if not reason:
            try:
                print(build_task(int(c.tid), 1, LARGE_DIR), flush=True)
                t = load_task(int(c.tid), LARGE_DIR)
                counts = tuple(sorted(pd.Series(t.y).value_counts().tolist()))
                if (len(t.y), counts) in prints:
                    reason = "tabarena fingerprint"
            except Exception as e:  # a dataset that does not download or parse is left out, with the reason
                reason = f"error: {type(e).__name__}: {e}"[:200]
        rows.append(dict(task_id=int(c.tid), name=c.name, n_rows=int(c.NumberOfInstances),
                         n_features=int(c.NumberOfFeatures) - 1, n_classes=int(c.NumberOfClasses), excluded=reason))
    table = pd.DataFrame(rows)
    table.to_csv(LARGE_DIR / "tasks.csv", index=False)
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print(table.to_string(index=False))
    print(f"kept {int((table.excluded == '').sum())} of {len(table)}")


def kept_tasks():
    table = pd.read_csv(LARGE_DIR / "tasks.csv").fillna({"excluded": ""})
    return [load_task(int(t), LARGE_DIR) for t in table[table.excluded == ""].task_id]


def jobs():
    from lightpfn.eval import harness

    harness._task = lru_cache(maxsize=4)(lambda task_id: load_task(task_id, LARGE_DIR))
    return scale_jobs(kept_tasks(), n_repeats=1, sizes=SIZES, min_rows=MIN_ROWS, test_rows=TEST_ROWS, all_rows=False)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    r = sub.add_parser("run")
    r.add_argument("--models", nargs="*", default=[])
    r.add_argument("--checkpoint")
    r.add_argument("--name")
    r.add_argument("--n-threads", type=int, default=24)
    s = sub.add_parser("report")
    s.add_argument("--models", nargs="+", required=True)
    s.add_argument("--ref", default="catboost")
    args = p.parse_args()
    if args.cmd == "build":
        build()
    elif args.cmd == "run":
        if args.checkpoint and not args.name:
            p.error("--checkpoint needs --name")
        evaluate_all(jobs(), RESULTS, args.models, args.checkpoint, args.name, ["full"], args.n_threads)
    else:
        report(args.models, args.ref, results=RESULTS, sizes=SIZES)


if __name__ == "__main__":
    main()
