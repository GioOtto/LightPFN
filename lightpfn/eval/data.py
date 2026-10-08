"""TabArena-v0.1 classification tasks as compact numpy arrays.

`python -m lightpfn.eval.data` downloads the 38 classification tasks of TabArena-v0.1 from OpenML
once (OpenML cache in data/openml) and writes one file per task, data/tabarena/<task_id>.npz:
  X      float32 (n, m): numerical values, or ordinal codes for categorical columns; NaN = missing
  y      int64 (n,): labels 0..C-1 (classes sorted by their original value)
  cat    bool (m,): categorical columns
  folds  int8 (R, n): official OpenML split, row i is in the test set of fold folds[r, i] in repeat r
The task list and repeat counts come from TabArena's curated metadata (tabarena_metadata.csv).
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "tabarena"
OPENML_CACHE = ROOT / "data" / "openml"
METADATA = Path(__file__).with_name("tabarena_metadata.csv")


def task_table():
    df = pd.read_csv(METADATA)
    df = df[df.is_classification].sort_values("num_instances").reset_index(drop=True)
    return df[["task_id", "dataset_name", "problem_type", "num_instances", "num_features", "num_classes", "tabarena_num_repeats"]]


@dataclass
class Task:
    task_id: int
    name: str
    X: np.ndarray
    y: np.ndarray
    cat: np.ndarray
    folds: np.ndarray

    @property
    def n_classes(self):
        return int(self.y.max()) + 1

    @property
    def n_repeats(self):
        return self.folds.shape[0]

    @property
    def n_folds(self):
        return int(self.folds.max()) + 1

    def split(self, repeat, fold):
        test = self.folds[repeat] == fold
        return np.flatnonzero(~test), np.flatnonzero(test)

    def info(self):
        counts = np.bincount(self.y)
        return dict(
            task_id=self.task_id,
            name=self.name,
            n_rows=len(self.y),
            n_features=self.X.shape[1],
            n_classes=self.n_classes,
            minority_frac=counts.min() / len(self.y),
            cat_frac=float(self.cat.mean()),
            nan_frac=float(np.isnan(self.X).mean()),
        )


def load_task(task_id, data_dir=DATA_DIR):
    with np.load(Path(data_dir) / f"{task_id}.npz") as f:
        return Task(int(task_id), str(f["name"]), f["X"], f["y"], f["cat"], f["folds"])


def load_tasks(names=None, data_dir=DATA_DIR):
    """Loads all tasks (smallest first), or only those whose name or task id is in `names`."""
    tasks = []
    for row in task_table().itertuples():
        if names is None or row.dataset_name in names or str(row.task_id) in names:
            tasks.append(load_task(row.task_id, data_dir))
    return tasks


def _encode(X_df):
    cols, cat = [], []
    for c in X_df.columns:
        s = X_df[c]
        if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object or s.dtype == bool or pd.api.types.is_string_dtype(s):
            codes = s.astype("category").cat.codes.to_numpy().astype(np.float32)
            codes[codes < 0] = np.nan
            cols.append(codes)
            cat.append(True)
        else:
            cols.append(pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float32, na_value=np.nan))
            cat.append(False)
    return np.stack(cols, axis=1), np.array(cat)


def build_task(task_id, n_repeats, out_dir=DATA_DIR):
    import openml

    out = Path(out_dir) / f"{task_id}.npz"
    if out.exists():
        return f"{task_id}: cached"
    task = openml.tasks.get_task(int(task_id), download_splits=True, download_data=True, download_qualities=False)
    dataset = task.get_dataset()
    X_df, y_s, _, _ = dataset.get_data(target=task.target_name)
    X, cat = _encode(X_df)
    _, y = np.unique(y_s.astype(str).to_numpy(), return_inverse=True)

    task_repeats, n_folds, _ = task.get_split_dimensions()
    assert task_repeats >= n_repeats, (task_id, task_repeats, n_repeats)
    folds = np.full((n_repeats, len(y)), -1, dtype=np.int8)
    for r in range(n_repeats):
        for f in range(n_folds):
            train_idx, test_idx = task.get_train_test_split_indices(fold=f, repeat=r)
            assert (folds[r, test_idx] == -1).all(), "test sets of a repeat overlap"
            folds[r, test_idx] = f
            assert len(train_idx) + len(test_idx) == len(y)
    assert (folds >= 0).all(), "test sets of a repeat do not cover all rows"

    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, X=X, y=y.astype(np.int64), cat=cat, folds=folds, name=np.array(dataset.name))
    tmp.replace(out)
    return f"{task_id}: {dataset.name} X={X.shape} classes={y.max() + 1} cat={cat.sum()} folds={n_folds}x{n_repeats}"


def main():
    import openml

    p = argparse.ArgumentParser()
    p.add_argument("--n-threads", type=int, default=8)
    args = p.parse_args()

    OPENML_CACHE.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    openml.config.set_root_cache_directory(str(OPENML_CACHE))
    table = task_table()
    with ThreadPoolExecutor(args.n_threads) as ex:
        jobs = [ex.submit(build_task, r.task_id, r.tabarena_num_repeats) for r in table.itertuples()]
        for j in jobs:
            print(j.result(), flush=True)

    info = pd.DataFrame([load_task(t).info() for t in table.task_id])
    pd.set_option("display.width", 200)
    print(info.to_string(index=False))
    (DATA_DIR / "info.json").write_text(json.dumps(info.to_dict(orient="records"), indent=1))


if __name__ == "__main__":
    main()
