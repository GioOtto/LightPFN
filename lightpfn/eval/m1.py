"""Frozen M1 additive probes, a standalone D2 extension (protocol M1-v1).

M1 is close in form to the N1 additive training prior, hence near in-distribution
for the m70 runs. It diagnoses additive learning, not generalization to unseen
mechanisms, and does not replace the other D1/D2/D3 guards in R3.

The generator follows runs/v2_arch/R1/OPUS/m1_probe.py: 16 columns, 1 or 8
drivers, nominal random effects with Zipf(1.1) frequencies and permuted IDs
(re), or independent step heights at uniform population quantiles of Gaussian
columns (step). Effects have variance 3/s. Half the remaining re columns are
8-level categorical distractors; columns are shuffled. Labels are Bernoulli
draws from the sigmoid of the summed effects, without balancing or filtering.
L=16/128/1024 means levels for re and change points (L+1 heights) for step.

Changes from the prototype: contexts 2048/8192/16384, 2000 test rows instead of
4000, and a separate evaluation seed namespace. Each of the 12 family/s/size
cells has six tasks: two independent seeds at each L, 72 tasks total (24 at
16k). Seeds are uint32 values from SeedSequence([0x4D315F01, family_index,
s, L, n_train, replicate]), family_index=0 for re, 1 for step. This namespace
is independent of training seeds and frozen here; do not use M1 to train.
Sizes are independent tasks, not nested contexts. Generation retains float64
and the prototype's draw order; ndtr equals norm.cdf, expit is a stable sigmoid.

Budget estimate, NOT a GPU measurement: on an idle RTX 5090, about 0.5-2 min
for v1, 2-5 min for C2v2, for all 72 tasks with one estimator and whole context;
the 24 largest tasks should take roughly half. Prototype v1 used ~0.7 s per
16k task with 4000 queries. C2v2 is an unmeasured allowance. CatBoost retains
CPU library defaults (only threads, seed, verbosity and file writing are set),
even with --device cuda; budget 1-4 CPU minutes. Oracle is normally <5 s.
Check these estimates on the target machine using the recorded task timings.
Seconds include fit and predict, plus synchronization on CUDA, excluding task
generation and metrics. V1's first task also includes its lazy checkpoint load;
v2 loads once before timing. --limit takes a deterministic prefix, interleaving
re and step, and is only a smoke test, never a full M1 score.

Examples::

    python -m lightpfn.eval.m1 run --checkpoint runs/train/v2_A3/ema_step008000.pt --name A3_m1 --device cuda
    python -m lightpfn.eval.m1 run --checkpoint PATH --name m70_m1 --device cuda --categorical
    python -m lightpfn.eval.m1 run --model catboost --name catboost_m1
    python -m lightpfn.eval.m1 run --model oracle --name oracle_m1
    python -m lightpfn.eval.m1 report --runs A3_m1 m70_m1 catboost_m1 oracle_m1

--categorical enables the v2 typed path: the true mask reaches fit(X, y, cat),
and nominal values remain exact integer level IDs, without ordinal scaling or
vocabulary fitting here. With the flag off, v2 treats those same values as
numbers, matching legacy-pool training. V1 always reads them as numeric codes.
No prior or dev module is imported, including on Windows.
"""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.special import expit, ndtr
from sklearn.metrics import log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
DEV_ROOT = ROOT / "runs" / "dev"
PROTOCOL = "M1-v1"
SEED_NAMESPACE = 0x4D315F01
N_TEST = 2000
M = 16
GROUP = ["family", "s", "n_train"]
TASK_KEY = GROUP + ["L", "seed"]


@dataclass(frozen=True)
class TaskSpec:
    family: str
    s: int
    L: int
    n_train: int
    seed: int


def task_set():
    """The immutable, ordered 72-task manifest. No data-dependent selection."""
    return tuple(
        TaskSpec(family, s, L, n_train, int(np.random.SeedSequence(
            [SEED_NAMESPACE, f, s, L, n_train, replicate]
        ).generate_state(1)[0]))
        for n_train in (2048, 8192, 16384)
        for replicate in (0, 1)
        for L in (16, 128, 1024)
        for s in (1, 8)
        for f, family in enumerate(("re", "step"))
    )


def make_task(spec):
    """Return X_train, y_train, X_test, y_test, nominal mask, true test logit."""
    if (spec.family not in ("re", "step") or not 1 <= spec.s <= M
            or spec.L < 1 or spec.n_train < 1):
        raise ValueError("Invalid M1 task specification")
    rng = np.random.default_rng(spec.seed)
    n = spec.n_train + N_TEST
    X = rng.standard_normal((n, M))
    cat = np.zeros(M, dtype=bool)
    logit = np.zeros(n)
    tau = np.sqrt(3.0 / spec.s)
    for j in range(spec.s):
        if spec.family == "re":
            freq = 1.0 / np.arange(1, spec.L + 1) ** 1.1
            levels = rng.choice(spec.L, size=n, p=freq / freq.sum())
            effects = rng.normal(0, tau, spec.L)
            logit += effects[levels]
            X[:, j] = rng.permutation(spec.L)[levels]
            cat[j] = True
        else:
            cuts = np.sort(rng.uniform(0, 1, spec.L))
            heights = rng.normal(0, tau, spec.L + 1)
            logit += heights[np.searchsorted(cuts, ndtr(X[:, j]))]
    if spec.family == "re":
        for j in range(spec.s, M, 2):
            X[:, j] = rng.integers(0, 8, n)
            cat[j] = True
    perm = rng.permutation(M)
    X, cat = X[:, perm], cat[perm]
    y = (rng.uniform(size=n) < expit(logit)).astype(np.int64)
    return X[:spec.n_train], y[:spec.n_train], X[spec.n_train:], y[spec.n_train:], cat, logit[spec.n_train:]


def _checkpoint_classifier(checkpoint, device, categorical, threads):
    # Lazy imports keep task generation/reporting independent of torch/prior.
    from lightpfn.model_v2.estimator import V2Classifier, load_v2
    from lightpfn.sklearn import LightPFNClassifier

    model = load_v2(checkpoint, device)
    if model is not None:
        return V2Classifier(model, device=device, n_estimators=1, categorical=categorical,
                            max_context=16384, seed=0), "v2"
    if categorical:
        raise ValueError("--categorical requires a C2v2 checkpoint; v1 uses numeric codes")
    return LightPFNClassifier(checkpoint=checkpoint, device=device, n_estimators=1,
                              max_context=16384, n_threads=threads, seed=0), "v1"


def _sync(device):
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)


def run(*, name, checkpoint=None, model=None, device="cpu", categorical=False,
        limit=None, threads=8):
    """Evaluate the frozen prefix, writing per-task results and unweighted cell means.

    A name is a single directory component below runs/dev. Existing result files
    are refused to protect completed evaluations. Metadata distinguishes protocol,
    model, typed path, checkpoint and device. A failed run leaves its partial task
    CSV for diagnosis, but no final M1.csv.
    """
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("name must be one directory component")
    if (checkpoint is None) == (model is None):
        raise ValueError("Specify exactly one of checkpoint or model")
    if model is not None and model not in ("catboost", "oracle"):
        raise ValueError("model must be catboost or oracle")
    if categorical and checkpoint is None:
        raise ValueError("--categorical requires a C2v2 checkpoint")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if threads < 1:
        raise ValueError("threads must be positive")
    out = DEV_ROOT / name
    if any((out / filename).exists() for filename in ("M1_tasks.csv", "M1.csv")):
        raise FileExistsError(f"M1 results already exist in {out}; use a new --name")

    clf = None
    kind = model
    actual_device = device if checkpoint is not None else "cpu"
    if checkpoint is not None:
        import torch

        torch.set_num_threads(threads)
        t0 = time.perf_counter()
        clf, kind = _checkpoint_classifier(checkpoint, device, categorical, threads)
        _sync(actual_device)
        print(f"{kind} setup: {time.perf_counter() - t0:.3f}s (v1 loads lazily)", flush=True)

    specs = task_set()[:limit]
    rows = []
    out.mkdir(parents=True, exist_ok=True)
    # Exclusive create: never silently replace an earlier model's results.
    with (out / "M1_tasks.csv").open("x", newline="", encoding="utf-8") as stream:
        for i, spec in enumerate(specs, 1):
            Xtr, ytr, Xte, yte, cat, true_logit = make_task(spec)
            if model == "catboost":
                from lightpfn.eval.models import CatBoost

                # Fixed algorithm seed across tasks/models; task seed only drives data.
                clf = CatBoost(threads, 0)
            _sync(actual_device)
            t0 = time.perf_counter()
            if model == "oracle":
                p = expit(true_logit)
            else:
                if kind == "v1":
                    clf.fit(Xtr, ytr)
                else:
                    clf.fit(Xtr, ytr, cat)
                p = clf.predict_proba(Xte)[:, 1]
            _sync(actual_device)
            seconds = time.perf_counter() - t0
            if p.shape != (N_TEST,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise ValueError(f"Invalid probabilities for {spec}")
            row = dict(asdict(spec), auc=roc_auc_score(yte, true_logit if model == "oracle" else p),
                       logloss=log_loss(yte, p, labels=[0, 1]), seconds=seconds,
                       protocol=PROTOCOL, n_test=N_TEST, model=kind, categorical=categorical,
                       checkpoint=str(checkpoint) if checkpoint is not None else "",
                       device=actual_device, threads=threads)
            rows.append(row)
            pd.DataFrame([row]).to_csv(stream, index=False, header=i == 1)
            stream.flush()
            print(f"{i:2d}/{len(specs)} {spec.family} s={spec.s} n={spec.n_train} L={spec.L} "
                  f"seed={spec.seed} auc={row['auc']:.5f} logloss={row['logloss']:.5f} "
                  f"seconds={seconds:.3f}", flush=True)
    tasks = pd.DataFrame(rows)
    means = tasks.groupby(GROUP, as_index=False).agg(
        tasks=("auc", "size"), auc=("auc", "mean"), logloss=("logloss", "mean"),
        seconds=("seconds", "mean"), total_seconds=("seconds", "sum"))
    means["protocol"] = PROTOCOL
    means["model"] = kind
    means["categorical"] = categorical
    means.to_csv(out / "M1.csv", index=False, mode="x")
    print(f"Saved {len(tasks)}/{len(task_set())} tasks to {out}; "
          f"fit+predict total {tasks.seconds.sum():.3f}s", flush=True)
    return tasks


def _read_run(name):
    path = DEV_ROOT / name / "M1_tasks.csv"
    df = pd.read_csv(path)
    if df.empty or set(df.protocol) != {PROTOCOL} or set(df.n_test) != {N_TEST}:
        raise ValueError(f"Incompatible or empty M1 protocol in {path}")
    if df.duplicated(TASK_KEY).any():
        raise ValueError(f"Duplicate M1 tasks in {path}")
    manifest = {tuple(asdict(spec)[key] for key in TASK_KEY) for spec in task_set()}
    if not set(df[TASK_KEY].itertuples(index=False, name=None)) <= manifest:
        raise ValueError(f"Unknown M1 tasks in {path}")
    return df


def report(runs, catboost="catboost_m1"):
    """Print cell tables; gaps use only identical tasks paired with CatBoost.

    Partial runs remain visible with paired/task counts; missing baselines yield
    NaN gaps, never a subtraction of means from different task sets.
    """
    frames = {name: _read_run(name) for name in runs}
    baselines = [name for name, df in frames.items() if set(df.model) == {"catboost"}]
    if len(baselines) > 1:
        raise ValueError("More than one CatBoost run requested; keep one baseline in --runs")
    if baselines:
        baseline = frames[baselines[0]]
    elif (DEV_ROOT / catboost / "M1_tasks.csv").exists():
        baseline = _read_run(catboost)
        if set(baseline.model) != {"catboost"}:
            raise ValueError(f"{catboost} is not a CatBoost run")
    else:
        baseline = pd.DataFrame(columns=TASK_KEY + ["auc"])
        print(f"CatBoost missing: run --model catboost --name {catboost} to obtain gaps.")
    rows = []
    for name, df in frames.items():
        paired = df.merge(baseline[TASK_KEY + ["auc"]], on=TASK_KEY, suffixes=("", "_cb"),
                          how="left", validate="one_to_one")
        paired["gap_pp"] = 100 * (paired.auc - paired.auc_cb)
        cell = paired.groupby(GROUP, as_index=False).agg(
            tasks=("auc", "size"), paired=("auc_cb", "count"), auc=("auc", "mean"),
            logloss=("logloss", "mean"), seconds=("seconds", "mean"), gap_pp=("gap_pp", "mean"))
        cell["run"] = name
        rows.append(cell)
    result = pd.concat(rows, ignore_index=True)
    print("gap_pp = paired AUC(model) - AUC(CatBoost), percentage points; "
          "full cell = 6 tasks. Partial cells are smoke results.")
    for (family, s, n_train), cell in result.groupby(GROUP):
        print(f"\n{family}  s={s}  n_train={n_train}")
        print(cell[["run", "tasks", "paired", "auc", "logloss", "seconds", "gap_pp"]].to_string(
            index=False, float_format=lambda value: f"{value:.5f}"))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="Evaluate the frozen M1 task set")
    source = run_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--model", choices=("catboost", "oracle"))
    run_parser.add_argument("--name", required=True)
    run_parser.add_argument("--device", default="cpu", help="Checkpoint device, e.g. cpu or cuda:0 (default cpu)")
    run_parser.add_argument("--categorical", action="store_true", help="Enable native nominal IDs for C2v2")
    run_parser.add_argument("--limit", type=int, help="Smoke only: first N tasks of the fixed manifest")
    run_parser.add_argument("--threads", type=int, default=8)
    report_parser = commands.add_parser("report", help="Print cell scores and paired gaps to CatBoost")
    report_parser.add_argument("--runs", nargs="+", required=True)
    report_parser.add_argument("--catboost", default="catboost_m1", help="Baseline name if absent from --runs")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    return run(**args) if command == "run" else report(**args)


if __name__ == "__main__":
    main()
