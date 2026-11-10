"""External real regression set D3-reg: OpenML-CTR23 regression tasks that are NOT in TabArena.

Regression counterpart of lightpfn.eval.external (D3). Used to choose between v2 regression variants without
turning TabArena into a selection set; never used for training. Protocol as D3 lite: at most 1000 rows
(uniform subsample) and 100 random features, 5-fold KFold, seed 0. The subsample, the columns and the folds are
drawn ONCE at build time and saved in the task file, so every model sees exactly the same splits.

File format, data/external/ctr23/<task_id>.npz:
  X      float32 (n, m): numerical values, or ordinal codes for categorical columns; NaN = missing (all rows)
  y      float64 (n,): target in original units
  cat    bool (m,)
  rows   int64 (r,): the kept subsample (r <= 1000), in X/y coordinates
  cols   int64 (k,): the kept columns (k <= 100)
  folds  int8 (r,): fold of each kept row, 0..4
  name   str

Exclusions (tasks.csv, column excluded; empty = kept):
  "tabarena name"         normalized name equal to a TabArena regression dataset, or listed in TABARENA_ALIASES
                          (same origin under another name, e.g. the red/white parts of wine_quality)
  "tabarena fingerprint"  same row count and the same rounded target mean/std as a TabArena regression dataset
  "constant target"       context std < 1e-6 on the kept rows, or fewer than 20 distinct target values
  "too small"             fewer than 100 rows with a finite target
  "nonfinite target"      more than 1% of the rows without a finite target (others are dropped)
  "build failed: ..."     download/encoding error
The automatic checks cannot see every same-origin duplicate: read tasks.csv after a build (see the PROTOCOL of
runs/v2_arch/reg_heads_20261104) and add aliases here when needed.

    python -m lightpfn.eval.external_reg build                      # download once (OpenML cache data/openml)
    python -m lightpfn.eval.external_reg run --models context catboost lightgbm xgboost
    python -m lightpfn.eval.external_reg run --checkpoint runs/train/X/ema_step002000.pt --name X [--device cuda]
    python -m lightpfn.eval.external_reg report

Metrics per split as lightpfn.eval.regression.regression_metrics (original units; CRPS/std and RMSE/std with the
context std; coverages), plus RMSE and R2 (1 - SSE/SST around the CONTEXT mean). GBDT quantiles come from
held-out residuals (a fit on the first 80% of the context), as runs/v2_data/reg_pilot/d1reg_compare.py.
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lightpfn.eval.data import OPENML_CACHE, ROOT, _encode

EXT_DIR = ROOT / "data" / "external" / "ctr23"
RESULTS = ROOT / "runs" / "eval" / "external_reg"
CTR23_SUITE = 353  # OpenML-CTR23 (Fischer et al. 2023), 35 curated regression tasks
TABARENA_METADATA = Path(__file__).with_name("tabarena_metadata.csv")
MAX_ROWS, MAX_FEATURES, N_FOLDS, SEED = 1000, 100, 5, 0
# Same origin as a TabArena regression dataset under another name (normalized). Checked by hand after the build.
TABARENA_ALIASES = {"whitewine", "redwine", "winequalitywhite", "winequalityred", "winequality",
                    "californiahousing", "house", "insurance", "medicalcharges", "fiat500", "usedfiat500",
                    "proteintertiarystructure", "casp", "concrete", "airfoil", "fishtoxicity", "qsarfishtoxicity",
                    "superconduct", "superconductivty"}
KEYS = ("crps_std", "rmse_std", "coverage_50", "coverage_80", "coverage_90", "r2")


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def tabarena_regression():
    """Normalized names and (rows, mean, std) fingerprints of the TabArena regression datasets."""
    meta = pd.read_csv(TABARENA_METADATA)
    reg = meta[~meta.is_classification]
    names = {_norm(n) for col in ("dataset_name", "openml_dataset_name") for n in reg[col].dropna()}
    return names | TABARENA_ALIASES, reg


def _fingerprint(y):
    return (len(y), round(float(np.mean(y)), 3), round(float(np.std(y)), 3))


def tabarena_fingerprints(reg):
    """Fingerprints need the TabArena targets: read from the OpenML cache when present, never downloaded here."""
    import openml
    prints = set()
    for row in reg.itertuples():
        try:
            d = openml.datasets.get_dataset(int(row.dataset_id), download_data=True, download_qualities=False,
                                            download_features_meta_data=False)
            _, y, _, _ = d.get_data(target=row.target_feature)
            y = pd.to_numeric(y, errors="coerce").to_numpy(np.float64)
            prints.add(_fingerprint(y[np.isfinite(y)]))
        except Exception as e:  # noqa: BLE001  (recorded, the name check still applies)
            print(f"fingerprint of TabArena {row.dataset_name} unavailable: {e!r}", flush=True)
    return prints


def build_task(task_id, out_dir=None):
    out_dir = EXT_DIR if out_dir is None else out_dir  # resolved at call time (tests patch EXT_DIR)
    import openml

    out = Path(out_dir) / f"{task_id}.npz"
    if out.exists():
        return f"{task_id}: cached"
    task = openml.tasks.get_task(int(task_id), download_splits=False, download_data=True, download_qualities=False)
    dataset = task.get_dataset()
    X_df, y_s, _, _ = dataset.get_data(target=task.target_name)
    X, cat = _encode(X_df)
    y = pd.to_numeric(y_s, errors="coerce").to_numpy(np.float64)
    rng = np.random.default_rng([SEED, int(task_id)])
    finite = np.flatnonzero(np.isfinite(y))
    rows = np.sort(rng.choice(finite, MAX_ROWS, replace=False)) if len(finite) > MAX_ROWS else finite
    m = X.shape[1]
    cols = np.sort(rng.choice(m, MAX_FEATURES, replace=False)) if m > MAX_FEATURES else np.arange(m)
    folds = np.empty(len(rows), np.int8)
    perm = rng.permutation(len(rows))
    folds[perm] = np.arange(len(rows)) % N_FOLDS
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, X=X, y=y, cat=cat, rows=rows.astype(np.int64), cols=cols.astype(np.int64), folds=folds,
             name=np.array(dataset.name), nonfinite_target=np.array(1 - len(finite) / max(len(y), 1)))
    tmp.replace(out)
    return f"{task_id}: {dataset.name} X={X.shape} kept rows={len(rows)} cols={len(cols)} cat={int(cat.sum())}"


def load(task_id, data_dir=None):
    data_dir = EXT_DIR if data_dir is None else data_dir  # resolved at call time (tests patch EXT_DIR)
    with np.load(Path(data_dir) / f"{task_id}.npz") as f:
        return {k: f[k] for k in f.files}


def exclusion(t, names, prints):
    y_all = t["y"]
    finite = y_all[np.isfinite(y_all)]
    y = y_all[t["rows"]]
    if _norm(t["name"]) in names:
        return "tabarena name"
    if len(finite) and _fingerprint(finite) in prints:
        return "tabarena fingerprint"
    if float(t["nonfinite_target"]) > .01:
        return "nonfinite target"
    if len(finite) < 100:
        return "too small"
    if np.std(y) < 1e-6 or len(np.unique(y)) < 20:
        return "constant target"
    return ""


def build(n_threads=4, fingerprints=True):
    import openml
    from concurrent.futures import ThreadPoolExecutor

    openml.config.set_root_cache_directory(str(OPENML_CACHE))
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    task_ids = list(openml.study.get_suite(CTR23_SUITE).tasks)
    failed = {}

    def one(t):
        try:
            return build_task(t)
        except Exception as e:  # noqa: BLE001
            failed[t] = repr(e)[:120]
            return f"{t}: FAILED {e!r}"

    with ThreadPoolExecutor(n_threads) as ex:
        for msg in ex.map(one, task_ids):
            print(msg, flush=True)
    names, reg = tabarena_regression()
    prints = tabarena_fingerprints(reg) if fingerprints else set()
    rows = []
    for t in task_ids:
        if t in failed:
            rows.append(dict(task_id=t, name="", excluded=f"build failed: {failed[t]}"))
            continue
        d = load(t)
        y = d["y"][d["rows"]]
        rows.append(dict(task_id=t, name=str(d["name"]), n_rows=len(d["y"]), kept_rows=len(d["rows"]),
                         n_features=d["X"].shape[1], kept_features=len(d["cols"]), cat_frac=float(d["cat"].mean()),
                         nan_frac=float(np.isnan(d["X"][np.ix_(d["rows"], d["cols"])]).mean()),
                         y_std=float(np.std(y)), y_skew=float(pd.Series(y).skew()),
                         y_distinct=int(len(np.unique(y))), excluded=exclusion(d, names, prints)))
    table = pd.DataFrame(rows)
    table.to_csv(EXT_DIR / "tasks.csv", index=False)
    pd.set_option("display.width", 220, "display.max_rows", 200)
    print(table.to_string(index=False))
    print(f"kept {int((table.excluded.fillna('') == '').sum())} of {len(table)}; fingerprints={len(prints)}")


def kept_tasks():
    table = pd.read_csv(EXT_DIR / "tasks.csv").fillna({"excluded": ""})
    return [int(t) for t in table[table.excluded == ""].task_id]


def splits(task_id):
    """(fold, Xtr, ytr, Xte, yte, cat) in the saved protocol; context rows first, in their saved order."""
    d = load(task_id)
    X = d["X"][np.ix_(d["rows"], d["cols"])].astype(np.float64)
    y, cat = d["y"][d["rows"]], d["cat"][d["cols"]]
    for f in range(N_FOLDS):
        te = d["folds"] == f
        yield f, X[~te], y[~te], X[te], y[te], cat


# ---- models -------------------------------------------------------------------------------------------------

def _frames(Xs, cat):
    out = []
    for X in Xs:
        df = pd.DataFrame(X)
        for j in np.flatnonzero(cat):
            df[j] = pd.Series(np.where(np.isnan(X[:, j]), -1, X[:, j]).astype(np.int64)).astype(str)
        out.append(df)
    return out


def point(kind, Xtr, ytr, Xq, cat, n_threads):
    """Default regressors, as people use them: no tuning, native categorical handling where offered."""
    if kind == "catboost":
        from catboost import CatBoostRegressor
        a, b = _frames((Xtr, Xq), cat)
        m = CatBoostRegressor(thread_count=n_threads, verbose=False, allow_writing_files=False, random_seed=0)
        return m.fit(a, ytr, cat_features=np.flatnonzero(cat).tolist()).predict(b)
    if kind == "lightgbm":
        from lightgbm import LGBMRegressor
        a, b = (pd.DataFrame(X) for X in (Xtr, Xq))
        for j in np.flatnonzero(cat):
            levels = sorted(set(np.unique(Xtr[:, j][np.isfinite(Xtr[:, j])])) | set(np.unique(Xq[:, j][np.isfinite(Xq[:, j])])))
            a[j], b[j] = (pd.Categorical(df[j], categories=levels) for df in (a, b))
        return LGBMRegressor(n_jobs=n_threads, verbose=-1, random_state=0).fit(a, ytr).predict(b)
    if kind == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_jobs=n_threads, random_state=0).fit(Xtr, ytr).predict(Xq)
    raise ValueError(kind)


def residual_quantiles(kind, Xtr, ytr, Xq, cat, n_threads):
    """Point fit on the whole context; quantile offsets from the residuals of a fit on its first 80%."""
    from lightpfn.eval.regression import TAUS
    mu, norm = ytr.mean(), max(ytr.std(), 1e-6)
    pred = point(kind, Xtr, (ytr - mu) / norm, Xq, cat, n_threads)
    nfit = int(.8 * len(ytr))
    mu_a, std_a = ytr[:nfit].mean(), max(ytr[:nfit].std(), 1e-6)
    cal = point(kind, Xtr[:nfit], (ytr[:nfit] - mu_a) / std_a, Xtr[nfit:], cat, n_threads)
    offsets = np.quantile(((ytr[nfit:] - mu_a) / std_a - cal) * std_a / norm, TAUS)
    return mu + norm * (pred[:, None] + offsets)


def score(ytr, yte, q, seed):
    from lightpfn.eval.regression import regression_metrics
    m = regression_metrics(ytr, yte, np.asarray(q, np.float64), seed=seed)
    m.pop("pit")
    levels = np.r_[0., np.arange(1, 1000) / 1000, 1.]
    mean = np.trapezoid(np.column_stack((q[:, 0], q, q[:, -1])), levels, axis=1)
    sst = float(((yte - ytr.mean()) ** 2).sum())
    m["r2"] = float(1 - ((yte - mean) ** 2).sum() / sst) if sst > 0 else float("nan")
    return m


def run(model_names=(), checkpoint=None, name=None, n_threads=4, device="cpu", tasks=None, out_name=None):
    from lightpfn.eval.regression import TAUS
    RESULTS.mkdir(parents=True, exist_ok=True)
    ids = tasks or kept_tasks()
    estimator = None
    if checkpoint:
        from lightpfn.model_v2.estimator import V2Regressor, load_v2
        model = load_v2(checkpoint, device)
        if model is not None and model.regression_head is not None:
            estimator = lambda: V2Regressor(model, device=device)  # noqa: E731
        else:
            from lightpfn.regressor import V15Regressor, load_regression_model
            model = None if model is not None else load_regression_model(checkpoint, device)
            if model is None:
                raise ValueError("a regression checkpoint is required (v2, or v1.5 with head=quantiles)")
            estimator = lambda: V15Regressor(model, device=device)  # noqa: E731
        model_names = [name]
    for m in model_names:
        rows = []
        for t in ids:
            for f, Xtr, ytr, Xte, yte, cat in splits(t):
                start, error = time.time(), ""
                try:
                    if estimator is not None:
                        q = estimator().fit(Xtr, ytr, cat).predict_quantiles(Xte)
                    elif m == "context":
                        q = np.broadcast_to(np.quantile(ytr, TAUS), (len(yte), len(TAUS)))
                    else:
                        q = residual_quantiles(m, Xtr, ytr, Xte, cat, n_threads)
                    metrics = score(ytr, yte, q, seed=[t, f])
                except Exception as e:  # noqa: BLE001  (an error is a row, as in the classification harness)
                    metrics, error = {}, repr(e)[:200]
                rows.append(dict(task_id=t, fold=f, n_train=len(ytr), n_test=len(yte), n_features=Xtr.shape[1],
                                 seconds=time.time() - start, error=error, **metrics))
            print(f"{m} {t} done", flush=True)
        pd.DataFrame(rows).to_csv(RESULTS / f"{out_name or m}.csv", index=False)


def report(models=None):
    frames = [pd.read_csv(f).assign(model=f.stem) for f in sorted(RESULTS.glob("*.csv"))
              if models is None or f.stem in models]
    df = pd.concat(frames, ignore_index=True).fillna({"error": ""})
    bad = df[df.error != ""]
    if len(bad):
        print(f"{len(bad)} splits with errors:\n{bad[['model', 'task_id', 'fold', 'error']].to_string(index=False)}")
    ok = df[df.error == ""].copy()
    if "near_constant" in ok:
        ok = ok[~ok.near_constant.astype(bool)]
    # Only tasks complete for every model; macro over tasks of the fold means.
    full = ok.groupby(["model", "task_id"]).fold.count().unstack(0)
    common = full.index[(full == N_FOLDS).all(1)]
    per_task = ok[ok.task_id.isin(common)].groupby(["model", "task_id"])[[*KEYS, "rmse"]].mean()
    s = per_task.groupby(level="model").mean()
    print(f"=== D3-reg: {len(common)} tasks complete for every model ===")
    print(s.round(4).to_string())
    s.to_csv(RESULTS.parent / "external_reg_summary.csv")
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["build", "run", "report"])
    p.add_argument("--models", nargs="*", default=[])
    p.add_argument("--checkpoint")
    p.add_argument("--name")
    p.add_argument("--tasks", nargs="*", type=int, help="subset of kept task ids (smoke tests)")
    p.add_argument("--out-name", help="CSV stem (default: model name)")
    p.add_argument("--n-threads", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-fingerprints", action="store_true", help="build: skip the TabArena target fingerprints")
    args = p.parse_args()
    if args.cmd == "build":
        build(args.n_threads, not args.no_fingerprints)
    elif args.cmd == "run":
        if args.checkpoint and not args.name:
            p.error("--checkpoint needs --name")
        run(args.models, args.checkpoint, args.name, args.n_threads, args.device, args.tasks, args.out_name)
    else:
        report(args.models or None)


if __name__ == "__main__":
    main()
