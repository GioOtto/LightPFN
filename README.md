<div align="center">
  <img src="https://raw.githubusercontent.com/GioOtto/LightPFN/main/docs/assets/logo.svg" width="112" height="112" alt="LightPFN logo: a small table with one highlighted cell" />
  <h1>LightPFN</h1>
  <p><strong>A 4.6M-parameter tabular foundation model for classification.</strong></p>
  <p>Pretrained only on synthetic data. A scikit-learn classifier that runs on CPU, CUDA, ROCm and any Vulkan GPU.</p>

  <p>
    <a href="https://pypi.org/project/LightPFN/"><img alt="PyPI" src="https://img.shields.io/pypi/v/lightpfn?style=for-the-badge&color=111111&label=PyPI" /></a>
    <a href="https://huggingface.co/ueuegio/LightPFN"><img alt="Weights on Hugging Face" src="https://img.shields.io/badge/Weights-Hugging%20Face-111111?style=for-the-badge&logo=huggingface&logoColor=white" /></a>
    <a href="https://github.com/GioOtto/LightPFN/blob/main/paper/LightPFN_report.pdf"><img alt="Technical report (PDF)" src="https://img.shields.io/badge/Technical%20report-PDF-8B1A1A?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" /></a>
  </p>

  <p>
    <a href="https://github.com/GioOtto/LightPFN/blob/main/docs/en/GUIDE.md">User guide</a> &nbsp;&nbsp;
    <a href="https://github.com/GioOtto/LightPFN/blob/main/docs/en/RESULTS.md">Results</a> &nbsp;&nbsp;
    <a href="https://github.com/GioOtto/LightPFN/blob/main/docs/en/VULKAN.md">Vulkan backend</a> &nbsp;&nbsp;
    <a href="https://github.com/GioOtto/LightPFN/blob/main/docs/en/TRAINING.md">Training</a> &nbsp;&nbsp;
    <a href="https://github.com/GioOtto/LightPFN/issues">Issues</a>
  </p>
  <p><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-181818" /> <img alt="4.6M parameters" src="https://img.shields.io/badge/parameters-4.6M-181818" /> <img alt="Code and weights Apache 2.0" src="https://img.shields.io/badge/code%20%2B%20weights-Apache%202.0-181818" /> <img alt="Trained only on synthetic data" src="https://img.shields.io/badge/training%20data-synthetic%20only-181818" /></p>
  <p><strong>English</strong> &nbsp; <a href="https://github.com/GioOtto/LightPFN/blob/main/README.it.md">Italiano</a></p>
</div>

## What it is

LightPFN is a prior-data fitted network: a transformer pretrained once on millions of synthetic
classification tasks. At `fit` it stores your training set as context; at `predict_proba` it reads that
context and the test rows in one forward pass. It does not train on your data and requires no tuning.

```python
from lightpfn import LightPFNClassifier

clf = LightPFNClassifier().fit(X_train, y_train)
proba = clf.predict_proba(X_test)
```

- **Small.** 4,603,088 parameters, 18 MB of weights, designed for a commodity CPU. TabICLv2 has six
  times as many.
- **Accurate without tuning.** On the official TabArena leaderboard (38 classification datasets) it ranked
  21st of 92 methods as of 11 October 2026, above every tree-based method, tuned and ensembled LightGBM,
  CatBoost and XGBoost included; the intervals of the closest ones overlap with its own.
- **Any GPU.** CUDA and ROCm through PyTorch, and a Vulkan backend with its own WGSL compute kernels for
  AMD, Intel and NVIDIA GPUs on Linux and Windows, with no CUDA or ROCm install. On an RX 7900 XT it is
  6 to 10 times faster than a 16-thread CPU.
- **scikit-learn API.** `Pipeline`, `GridSearchCV`, `clone`, `cross_val_score`, and pandas DataFrames
  with categorical, string, boolean and missing values.
- **Open.** Code, weights and the full training pipeline under Apache 2.0. Trained from scratch: no
  distillation, no weights or outputs of other tabular foundation models.

## Benchmarks

**[TabArena](https://huggingface.co/spaces/TabArena/leaderboard), official leaderboard, 38 classification
datasets.** The TabArena maintainers ran LightPFN in its default configuration (four estimators) and verified
the result. Leaderboard of 11 October 2026, imputed models excluded:

| View | Position | Elo (95% CI) |
|---|---:|---:|
| All splits | **21st of 92** | **1441** (+69 / -79) |
| TabArena-Lite (first split only) | 20th of 92 | 1426 (+71 / -70) |

<p align="center"><img src="https://raw.githubusercontent.com/GioOtto/LightPFN/main/docs/assets/tabarena_elo.png" width="100%" alt="TabArena, 38 classification datasets, all splits: Elo with 95% intervals for selected methods; LightPFN is 21st, above every tree-based method" /></p>

- **Above every tree-based method**, tuned and ensembled ones included. The intervals overlap with those of
  default CatBoost and of the tuned gradient-boosting entries, so against them this is not a clear win; default
  XGBoost and LightGBM are more than 200 Elo points lower.
- **Behind larger foundation models.** The first 18 places are all foundation models; TabICLv2 leads by 134 points.
- **By size:** 19th on the 28 small datasets, above every tree-based method; 44th of 93 on the 10 medium ones,
  where default CatBoost is 17th.

<p align="center"><img src="https://raw.githubusercontent.com/GioOtto/LightPFN/main/docs/assets/tabarena_pareto.png" width="100%" alt="TabArena: Elo against median training time for 92 methods; LightPFN is on the Pareto front" /></p>

LightPFN is on the Pareto front of Elo against time. At 0.26 s per 1,000 rows only KNN trains faster; at
0.029 s per 1,000 rows only EBM and default CatBoost predict faster, both with a lower Elo. Timings are
TabArena's, with LightPFN on a GPU.

On the leaderboard page, select **Classification** and untick **Include imputed models** to get this view. The
default view also covers the 13 regression datasets, which LightPFN does not support, so they are imputed.
[Protocol, CSVs and the earlier author-run evaluation](https://github.com/GioOtto/LightPFN/blob/main/docs/en/RESULTS.md#tabarena-official-leaderboard).

## Install

```bash
pip install LightPFN             # CPU, CUDA or ROCm, through your PyTorch install
pip install "LightPFN[vulkan]"   # adds the Vulkan GPU backend (wgpu)
```

Python 3.10 or newer. The first `fit` downloads the weights (18 MB) from
[Hugging Face](https://huggingface.co/ueuegio/LightPFN) at a pinned commit and caches them. On a CPU-only
machine, install PyTorch from its CPU index first to skip the CUDA libraries:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Quick start

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from lightpfn import LightPFNClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

clf = LightPFNClassifier(random_state=0)
clf.fit(X_train, y_train)
print(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
```

A pandas DataFrame with categorical and missing values goes in as it is:

```python
import pandas as pd

df = pd.DataFrame({
    "age": [34, 51, None, 28, 45, 39],
    "plan": ["basic", "pro", "pro", None, "basic", "enterprise"],
    "region": pd.Categorical(["north", "south", "south", "east", "north", "east"]),
    "active": [True, False, True, True, False, True],
})
y = [0, 1, 1, 0, 1, 0]
clf = LightPFNClassifier().fit(df, y)
clf.predict_proba(df.head(2))
```

Columns of category, string, object or bool dtype become ordinal codes of the categories seen in `fit`.
A category dtype keeps its declared levels, including unused ones. Missing values and values outside
that vocabulary become NaN, which the model handles natively. More in
[examples/](https://github.com/GioOtto/LightPFN/tree/main/examples/) and in the [user guide](https://github.com/GioOtto/LightPFN/blob/main/docs/en/GUIDE.md).

## Devices

| `device=` | Runs on |
|---|---|
| `"auto"` (default) | a CUDA or ROCm GPU through PyTorch, else a Vulkan GPU, else the CPU |
| `"cpu"` | the CPU, through PyTorch |
| `"cuda"`, `"cuda:1"` | an NVIDIA (CUDA) or AMD (ROCm) GPU through PyTorch |
| `"vulkan"`, `"vulkan:1"` | any GPU with a Vulkan driver, through `lightpfn.vulkan` (needs `lightpfn[vulkan]`) |

The environment variable `LIGHTPFN_DEVICE` replaces `"auto"`. Every device returns the same probabilities
up to floating-point rounding (within 3e-5). See the [Vulkan backend](https://github.com/GioOtto/LightPFN/blob/main/docs/en/VULKAN.md).

## When to use it

LightPFN is a good default for classification tables with 2 to 10 classes and up to tens of thousands of
rows, when you want a strong model without tuning. Know its limits:

- **Classification only**, 2 to 10 classes. Regression comes with version 2.
- **Large tables.** Above `max_context` rows (default 20,000) each estimator reads a stratified subsample.
  In the report, with the whole training set as context, its mean AUC is level with default CatBoost
  up to 100,000 rows. Larger contexts cost more; sizes beyond 100,000 were not evaluated.
- **Categorical columns** are read as ordinal codes. On tables dominated by high-cardinality categorical
  columns, CatBoost is ahead (by up to 4.5 AUC points on two TabArena tasks).
- **CPU time grows with the context.** On TabArena, four estimators take a median of 1 s per split on small
  tables, 15 s on medium and 54 s on large ones, against 3, 4 and 7 s for CatBoost. When time matters, use a
  GPU or `n_estimators=1` (about four times faster, slightly less accurate).

## Results

Baselines at default settings. Differences in AUC points (100 times the AUC difference) with 95% paired
bootstrap intervals. Full tables, per-task results and timings: [docs/en/RESULTS.md](https://github.com/GioOtto/LightPFN/blob/main/docs/en/RESULTS.md).

**TabArena, 38 classification tasks** (official splits, first repeat, run with our own harness, which is
not the official leaderboard protocol):

| Model | Mean AUC | Mean rank of 7 | Lower error than CatBoost |
|---|---:|---:|---:|
| TabICLv2 (28M parameters, GPU) | 0.864 | 1.58 | 87% |
| **LightPFN, 4 estimators** | **0.858** | **2.50** | **76%** |
| LightPFN, 1 estimator | 0.857 | 3.37 | 68% |
| CatBoost | 0.855 | 3.50 | |
| LightGBM | 0.844 | 5.34 | 5% |
| Random forest | 0.838 | 5.89 | 3% |
| XGBoost | 0.833 | 5.82 | 11% |

**Development set: 55 OpenML-CC18 datasets outside TabArena** (used for design decisions, so the estimate is
optimistic; at most 1,000 rows and 100 features, five-fold cross-validation, one estimator):

| Model | Mean AUC | LightPFN minus model |
|---|---:|---:|
| **LightPFN** | **0.911** | |
| CatBoost | 0.902 | +0.86 [0.42, 1.40] |
| Random forest | 0.893 | +1.76 |
| LightGBM | 0.891 | +2.02 |
| XGBoost (54 datasets: its wrapper fails on one) | 0.889 | +2.06 [1.32, 2.94] |

## How it works

<p align="center"><img src="https://raw.githubusercontent.com/GioOtto/LightPFN/main/docs/assets/architecture.png" width="100%" alt="LightPFN architecture: cell embedding, two column stages, row refinement and compression, in-context learning, retrieval decoder" /></p>

Each cell is embedded from its value, its rank in the column and a missing-value flag. Two column stages
with induced attention read the labelled context of each column. A row stage with four summary tokens mixes
the features of each row and compresses it into one vector. Seven in-context blocks let every test row
attend to the training rows, and a retrieval decoder turns the attention into a vote over the training
labels.

Pretraining used 7.68 million task draws from 4.03 million distinct synthetic tasks: 90% from a structural causal graph prior and 10% from a rule
prior (XOR, parity, lookup tables, trees) that teaches feature interactions. A second stage trained on
tables of up to 60,000 rows. The [technical report](https://github.com/GioOtto/LightPFN/blob/main/paper/LightPFN_report.pdf) describes the model, the
priors and the selection protocol; [docs/en/TRAINING.md](https://github.com/GioOtto/LightPFN/blob/main/docs/en/TRAINING.md) explains how to reproduce
the training.

## Roadmap

Version 2 will add:

- **Native categorical features.** The model will see the categorical mask from the start of pretraining,
  and the prior will contain high-cardinality categorical columns whose many rare levels each carry a small
  effect. An adapter trained on top of the frozen v1 weights did not help, because the v1 prior has no such
  columns to learn from (report, Section 12).
- **Regression.**

## Repository layout

| Path | Content |
|---|---|
| `lightpfn/` | the package: model, scikit-learn wrapper, devices, Vulkan backend (`vulkan/`), and the training code: priors (`prior/`), trainer (`train.py`), evaluation harness (`eval/`) |
| `tests/` | unit and equivalence tests (CPU; the Vulkan tests also run on a CPU driver) |
| `examples/` | runnable examples |
| `docs/` | user guide, results, Vulkan backend and training in English (`en/`) and Italian (`it/`), changelog, third-party licenses |
| `paper/` | technical report: PDF, LaTeX source, figures and plot data |

The published wheel contains only the inference code; training needs a source checkout
(`pip install -e ".[train,eval]"`).

## Contributing and support

Bug reports and focused pull requests are welcome: see [CONTRIBUTING](https://github.com/GioOtto/LightPFN/blob/main/.github/CONTRIBUTING.md). Security
issues go through [SECURITY](https://github.com/GioOtto/LightPFN/blob/main/.github/SECURITY.md). Changes between versions are in
[CHANGELOG](https://github.com/GioOtto/LightPFN/blob/main/docs/CHANGELOG.md).

## Citation

```bibtex
@techreport{ottoboni2026lightpfn,
  title  = {A Sling Against Giants: {LightPFN}, a 4.6M-parameter tabular in-context classifier designed to stay small},
  author = {Ottoboni, Giorgio},
  year   = {2026},
  url    = {https://github.com/GioOtto/LightPFN}
}
```

GitHub's "Cite this repository" button reads [CITATION.cff](https://github.com/GioOtto/LightPFN/blob/main/CITATION.cff).

## License

Code and weights: [Apache License 2.0](https://github.com/GioOtto/LightPFN/blob/main/LICENSE), with the attribution notice in [NOTICE](https://github.com/GioOtto/LightPFN/blob/main/NOTICE).
Dependencies keep their own licenses, listed in [THIRD_PARTY_LICENSES](https://github.com/GioOtto/LightPFN/blob/main/docs/THIRD_PARTY_LICENSES.md).
