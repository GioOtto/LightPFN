<div align="center">
  <img src="https://raw.githubusercontent.com/GioOtto/LightPFN/main/docs/assets/logo.svg" width="112" height="112" alt="LightPFN logo: a small table with one highlighted cell" />
  <h1>LightPFN</h1>
  <p><strong>Compact ~4.7M-parameter tabular foundation models for classification and regression.</strong></p>
  <p>Pretrained only on synthetic data. scikit-learn estimators that run on CPU, CUDA, ROCm and Vulkan GPUs.</p>

  <p>
    <a href="https://pypi.org/project/LightPFN/"><img alt="PyPI" src="https://img.shields.io/pypi/v/lightpfn?style=for-the-badge&color=111111&label=PyPI" /></a>
    <a href="https://huggingface.co/ueuegio/LightPFN"><img alt="Weights on Hugging Face" src="https://img.shields.io/badge/Weights-Hugging%20Face-111111?style=for-the-badge&logo=huggingface&logoColor=white" /></a>
    <a href="https://github.com/GioOtto/LightPFN/blob/main/paper/LightPFN15_report.pdf"><img alt="1.5 technical report (PDF)" src="https://img.shields.io/badge/1.5%20Report-PDF-8B1A1A?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" /></a>
    <a href="https://github.com/GioOtto/LightPFN/blob/main/paper/LightPFN_report.pdf"><img alt="Report of 1.1 (PDF)" src="https://img.shields.io/badge/1.1%20Report-PDF-444444?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" /></a>
  </p>

  <p>
    <a href="docs/en/GUIDE.md">User guide</a> &nbsp;&nbsp;
    <a href="docs/en/RESULTS.md">Results</a> &nbsp;&nbsp;
    <a href="docs/en/VULKAN.md">Vulkan backend</a> &nbsp;&nbsp;
    <a href="docs/en/TRAINING.md">Training</a> &nbsp;&nbsp;
    <a href="https://github.com/GioOtto/LightPFN/issues">Issues</a>
  </p>
  <p><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-181818" /> <img alt="4.6M / 4.8M parameters" src="https://img.shields.io/badge/parameters-4.6M%20%7C%204.8M-181818" /> <img alt="Code and weights Apache 2.0" src="https://img.shields.io/badge/code%20%2B%20weights-Apache%202.0-181818" /> <img alt="Trained only on synthetic data" src="https://img.shields.io/badge/training%20data-synthetic%20only-181818" /></p>
  <p><strong>English</strong> &nbsp; <a href="README.it.md">Italiano</a></p>
</div>

## What it is

LightPFN is a family of prior-data fitted networks: transformers pretrained once on millions of synthetic
tabular tasks. At `fit` they store and encode your training set as context; at `predict_proba`, `predict`
or `predict_quantiles` they read that context and the test rows in one forward pass. They do not optimize
weights on your data and require no hyperparameter tuning.

Version 1.5 provides separate checkpoints for classification (2 to 10 classes) and regression:

```python
from lightpfn import LightPFNClassifier, LightPFNRegressor

clf = LightPFNClassifier().fit(X_train, y_train_cls)
proba = clf.predict_proba(X_test)

reg = LightPFNRegressor().fit(X_train, y_train_reg)
pred = reg.predict(X_test)
quantiles = reg.predict_quantiles(X_test, quantiles=[0.1, 0.5, 0.9])
```

- **Small.** 4,609,472 parameters for the 1.5 classifier and 4,772,519 for the 1.5 regressor (about 19 MB
  of weights each; 4,603,088 parameters, 18 MB in 1.1), designed to run on a commodity CPU as well as a GPU.
  TabICLv2 has six times as many parameters.
- **Accurate without tuning.** On the official TabArena leaderboard (38 classification datasets), LightPFN 1.1
  ranked 21st of 92 methods as of 11 October 2026, above every tree-based method, tuned and ensembled LightGBM,
  CatBoost and XGBoost included; the intervals of the closest ones overlap with its own. LightPFN 1.5 has no
  official leaderboard entry yet.
- **Any GPU.** CUDA and ROCm through PyTorch (with automatic fp16 inference in 1.5), and a Vulkan backend with
  its own WGSL compute kernels for AMD, Intel and NVIDIA GPUs on Linux and Windows, with no CUDA or ROCm
  install. Vulkan supports the categorical adapter and regression. On an RX 7900 XT, the measured 1.5
  workloads with larger contexts run 3 to 7 times faster than CPU; small tables can be slower. Intel and
  NVIDIA Vulkan drivers use the same code but have not been benchmarked. See
  [Devices and precision](#devices-and-precision).
- **scikit-learn API.** `Pipeline`, `GridSearchCV`, `clone`, `cross_val_score`, and pandas DataFrames with
  categorical, string, boolean and missing values, plus `cat="auto"` categorical detection.
- **Open.** Code, weights and the full training pipeline under Apache 2.0. Trained from scratch on synthetic
  data only: no distillation, no weights or outputs of other tabular foundation models.

## New in 1.5

- Categorical adapter trained from scratch and `cat="auto"` detection (pandas categorical, string, object and
  bool dtypes, plus integer-valued numeric columns with 2 to 20 distinct levels).
- `LightPFNRegressor`: predictive mean and 999 quantiles (`.001` through `.999`), two members and up to 50,000
  context rows by default.
- `precision="auto"`: fp16 on supported CUDA/ROCm GPUs above the cell threshold (2,048 context cells per
  member), fp32 on CPU. Non-finite fp16 blocks automatically retry in fp32.
- Long-context training stage with tables up to 60,000 rows for both classification and regression.
- Classifier fine-tuned on synthetic tables with 64 to 500 features (pretraining stopped at 100): better on
  wide tables, unchanged elsewhere within noise.
- Separate `classifier/` and `regressor/` folders (`config.json` and `model.safetensors`) on Hugging Face
  (`ueuegio/LightPFN`), pinned by commit.

## Benchmarks (LightPFN 1.1)

**[TabArena](https://huggingface.co/spaces/TabArena/leaderboard), official leaderboard, 38 classification
datasets (LightPFN 1.1).** The TabArena maintainers ran LightPFN 1.1 in its default configuration (four
estimators) and verified the result. LightPFN 1.5 has not been evaluated on the official leaderboard yet.
Leaderboard of 11 October 2026, imputed models excluded:

| View (LightPFN 1.1) | Position | Elo (95% CI) |
|---|---:|---:|
| All splits | **21st of 92** | **1441** (+69 / -79) |
| TabArena-Lite (first split only) | 20th of 92 | 1426 (+71 / -70) |

<p align="center"><img src="https://raw.githubusercontent.com/GioOtto/LightPFN/main/docs/assets/tabarena_elo.png" width="100%" alt="TabArena, 38 classification datasets, all splits: Elo with 95% intervals for selected methods; LightPFN 1.1 is 21st, above every tree-based method" /></p>

- **Above every tree-based method**, tuned and ensembled ones included. The intervals overlap with those of
  default CatBoost and of the tuned gradient-boosting entries, so against them this is not a clear win; default
  XGBoost and LightGBM are more than 200 Elo points lower.
- **Behind larger foundation models.** The first 18 places are all foundation models; TabICLv2 leads by 134 points.
- **By size:** 19th on the 28 small datasets, above every tree-based method; 44th of 93 on the 10 medium ones,
  where default CatBoost is 17th.

<p align="center"><img src="https://raw.githubusercontent.com/GioOtto/LightPFN/main/docs/assets/tabarena_pareto.png" width="100%" alt="TabArena: Elo against median training time for 92 methods; LightPFN 1.1 is on the Pareto front" /></p>

LightPFN 1.1 is on the Pareto front of Elo against time. At 0.26 s per 1,000 rows only KNN trains faster; at
0.029 s per 1,000 rows only EBM and default CatBoost predict faster, both with a lower Elo. Timings are
TabArena's, with LightPFN 1.1 on a GPU.

On the leaderboard page, select **Classification** and untick **Include imputed models** to get this view for the
1.1 entry. The default view of the page also covers the 13 regression datasets, which LightPFN 1.1 did not
support, so they are imputed there.
[Protocol, CSVs and the earlier author-run evaluation](docs/en/RESULTS.md#tabarena-official-leaderboard).

## Install

```bash
pip install LightPFN             # CPU, CUDA or ROCm, through your PyTorch install (lightpfn==1.5.0)
pip install "LightPFN[vulkan]"   # adds the Vulkan GPU backend (wgpu)
```

Python 3.10 or newer. The first `fit` downloads only the needed checkpoint (about 19 MB) from
[Hugging Face](https://huggingface.co/ueuegio/LightPFN) at the package's pinned commit and caches it. The
`classifier/` and `regressor/` folders each contain `config.json` and `model.safetensors`; use `checkpoint=`
for a local folder. On a CPU-only machine, install PyTorch from its CPU index first to skip the CUDA libraries:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Quick start

### Classification

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

A pandas DataFrame with categorical and missing values goes in as it is (pandas is optional and only needed for
DataFrame input):

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

Columns of category, string, object or bool dtype become ordinal codes of the categories seen in `fit`. A
category dtype keeps its declared levels, including unused ones. Missing values and values outside that
vocabulary become NaN, which the model handles natively. With the default `cat="auto"`, `fit` also marks
integer-valued numeric columns that have 2 to 20 distinct levels in the training set so the categorical adapter
processes them.

### Regression

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from lightpfn import LightPFNRegressor

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

reg = LightPFNRegressor(random_state=0).fit(X_train, y_train)
pred = reg.predict(X_test)
interval = reg.predict_quantiles(X_test, quantiles=[0.1, 0.5, 0.9])
```

`fit` encodes the context without optimizing weights. Classification defaults to four members and at most 20,000
context rows per member; regression defaults to two members and at most 50,000 rows. More in
[examples/](examples/) and in the [user guide](docs/en/GUIDE.md).

## Devices and precision

| `device=` | Classifier runs on | Regressor runs on |
|---|---|---|
| `"auto"` (default) | a CUDA or ROCm GPU through PyTorch, else a Vulkan GPU, else the CPU | a CUDA or ROCm GPU through PyTorch, else a Vulkan GPU, else the CPU |
| `"cpu"` | the CPU, through PyTorch | the CPU, through PyTorch |
| `"cuda"`, `"cuda:1"` | an NVIDIA (CUDA) or AMD (ROCm) GPU through PyTorch | an NVIDIA (CUDA) or AMD (ROCm) GPU through PyTorch |
| `"vulkan"`, `"vulkan:1"` | any GPU with a Vulkan driver, through `lightpfn.vulkan`, including marked categorical columns | the same Vulkan backend, including marked categorical columns |

The environment variable `LIGHTPFN_DEVICE` replaces `"auto"`.

- **Vulkan in 1.5.** Both estimators support numerical and marked categorical columns in fp32. Cell
  embeddings, categorical-statistics projections, column and row stages, ICL and decoder attention run on
  the GPU. Column/category statistics and rank lookup stay on the host; regression target normalization and
  the final quantile head readout also run on the host. On an RX 7900 XT, classification probabilities have a maximum absolute
  difference of about 2.6e-5 from torch CPU, with identical argmax, tested up to 12,000 context rows with
  high-cardinality categoricals. Regression probes at quantiles 0.1/0.5/0.9, up to 40,000 context rows,
  give max absolute difference / max absolute CPU quantile of about 5.5e-5; this is not a pointwise bound
  over all quantiles, and these probes used the preceding second-member rule. If a fit
  exceeds GPU buffer limits, `device="auto"` falls back to CPU with a warning; explicit `device="vulkan"`
  raises `MemoryError`. `to()` moves only between torch devices; refit to switch to Vulkan. Install the
  optional backend with `pip install "LightPFN[vulkan]"`. See the [Vulkan backend](docs/en/VULKAN.md).
- **Precision.** `precision="auto"` (default) uses fp16 on supported CUDA/ROCm GPUs above the cell threshold
  (2,048 context cells per member) and fp32 on CPU and Vulkan. Non-finite fp16 blocks automatically retry in fp32.
  `precision="fp32"` forces float32, with close agreement on the tested device configurations; the measured
  Vulkan classification difference above is not a universal error bound. `precision="fp16"` requires CUDA/ROCm. fp16 predictions can differ from
  fp32.

## When to use it

LightPFN is a strong default for classification tables with 2 to 10 classes and single-target regression tables
with up to tens of thousands of rows, when you want a good model without hyperparameter tuning. Know its limits:

- **Task types.** Classification supports 2 to 10 classes; `LightPFNRegressor` supports single-target regression
  (predictive mean and quantiles). Multioutput regression is not supported.
- **Large tables.** Above `max_context` rows (default 20,000 per member with stratified subsampling in
  classification; 50,000 with uniform subsampling in regression), the training context is subsampled. Both 1.5
  models include a long-context training stage up to 60,000 rows. In the 1.1 report, with the whole training set
  as context, mean classification AUC is level with default CatBoost up to 100,000 rows. Larger contexts cost
  more in time and memory; sizes beyond 100,000 rows were not evaluated.
- **Categorical columns and regression.** Version 1.5 trains a categorical adapter from scratch and enables
  `cat="auto"`. Even so, difficult tables dominated by high-cardinality categorical columns and regression tasks
  still trail larger foundation models and, on some tasks, CatBoost.
- **CPU time grows with the context.** On TabArena (1.1), four classification estimators take a median of 1 s per
  split on small tables, 15 s on medium and 54 s on large ones, against 3, 4 and 7 s for CatBoost. When time
  matters, use a GPU or `n_estimators=1` (about four times faster in classification, slightly less accurate).

## Results

Full tables, per-task results, protocols and timings: [docs/en/RESULTS.md](docs/en/RESULTS.md).

### 1.5 local results

The 1.5 measurements are local runs, not an official leaderboard.

- **TabArena-Lite (author runs, 9 November 2026):** 38 classification tasks, Elo **1436** (+74 / -65;
  1.1: 1423) for the final classifier on an RTX 5090. On the same GPU the 1.5 classifier before the wide-table
  fine-tuning scored 1416 (1415 on the RX 7900 XT); the final one has lower error on 24 tasks and higher on 14.
  13 regression tasks, Elo **1450** (+107 / -103; default CatBoost: **1416**), on the RX 7900 XT. These are
  first-split results against a hosted baseline snapshot, not verified maintainer results. The Elo intervals
  of the final classifier, of the earlier one and of 1.1 overlap. The regressor uses the final two-member rule: `log(y)` only for positive targets
  with adjusted sample skewness at least 2 (`log_skew=2.0`), otherwise `-y`. On CPU fp32 this rule reduces
  per-task RMSE by 0.70% on D3-reg and 1.78% on D3-reg-L relative to the previous rule on the same CPU.
  The threshold was selected on these development sets, so those gains are optimistic.
- **GPU inference speed (local RTX 5090 benchmark):** on eight classification tasks with the 8-fold x 4-member
  protocol, the measured `precision="auto"` path gives geometric-mean speedups of **1.89x** for training and **2.04x**
  for prediction relative to 1.1.0 on the same GPU (measured with the earlier 8,192-cell threshold; scaling the
  official 1.1.0 leaderboard timings suggests ~0.14 s train and ~0.014 s predict per 1,000 rows; these are local ratios, not official 1.5 leaderboard timings).
- **Wide-table fine-tuning (final classifier):** pretraining never went past 100 features, and tables with more
  than 60 to 100 features were the one repeated weakness of the earlier 1.5 classifier. The final classifier
  continues it for 8,500 steps with half of the batches drawn from synthetic tables with 64 to 500 features,
  the rest from the earlier data mix. Paired changes against the earlier 1.5 classifier, four members, 95%
  bootstrap intervals: on the nine D3 datasets with more than 100 features, AUC +1.93 points [-0.06, +5.82]
  and log loss -0.048 [-0.108, -0.004]; D3 +0.01 points [-0.06, +0.09]; Dcat 10k +0.07 [+0.00, +0.16];
  D4 +0.12 [+0.05, +0.21]; TabArena first-split relative error -0.85% [-1.83, -0.04], -3.3% on the nine tasks
  with more than 60 features. The TabArena first split was monitored during the fine-tuning, so it is not an
  independent test. With one estimator the final classifier has D3 AUC 0.9102 and Dcat 10k 0.7082.
- **Classification development sets (1.5 classifier before the wide-table fine-tuning, one estimator):** D3 (55 OpenML-CC18 datasets not
  in TabArena) mean AUC 0.9103 versus 0.9109 for 1.1.0 and 0.9023 for default CatBoost; the six high-cardinality
  categorical datasets (Dcat) 0.7075 at 10,000 rows and 0.7134 at 25,000 rows, versus 0.7047 / 0.7098 for 1.1.0
  and 0.7031 / 0.7118 for default CatBoost; large tables (D4) 0.8200 versus 0.8188 (1.1.0) and 0.8189
  (CatBoost). On ordinary tables 1.5 matches 1.1.0; the gain is on categorical columns.

### 1.1 results

Baselines at default settings. Differences in AUC points (100 times the AUC difference) with 95% paired
bootstrap intervals. The tables below are for the **LightPFN 1.1** weights.

**TabArena, 38 classification tasks (LightPFN 1.1, our harness)** (official splits, first repeat, run with our
own harness, which is not the official leaderboard protocol):

| Model | Mean AUC | Mean rank of 7 | Lower error than CatBoost |
|---|---:|---:|---:|
| TabICLv2 (28M parameters, GPU) | 0.864 | 1.58 | 87% |
| **LightPFN 1.1, 4 estimators** | **0.858** | **2.50** | **76%** |
| LightPFN 1.1, 1 estimator | 0.857 | 3.37 | 68% |
| CatBoost | 0.855 | 3.50 | |
| LightGBM | 0.844 | 5.34 | 5% |
| Random forest | 0.838 | 5.89 | 3% |
| XGBoost | 0.833 | 5.82 | 11% |

**Development set: 55 OpenML-CC18 datasets outside TabArena (LightPFN 1.1)** (used for design decisions, so the
estimate is optimistic; at most 1,000 rows and 100 features, five-fold cross-validation, one estimator):

| Model | Mean AUC | LightPFN 1.1 minus model |
|---|---:|---:|
| **LightPFN 1.1** | **0.911** | |
| CatBoost | 0.902 | +0.86 [0.42, 1.40] |
| Random forest | 0.893 | +1.76 |
| LightGBM | 0.891 | +2.02 |
| XGBoost (54 datasets: its wrapper fails on one) | 0.889 | +2.06 [1.32, 2.94] |

## From 1.1 to 1.5

Version 1.5 changes the default classification weights, enables the categorical adapter with `cat="auto"`, uses
`precision="auto"` on CUDA/ROCm GPUs, and adds `LightPFNRegressor`. Predictions therefore change from 1.1. To
keep the exact earlier package:

```bash
pip install lightpfn==1.1.0
```

Or load the 1.1 weights in the 1.5 classifier (pass `precision="fp32"` to avoid precision changes):

```python
clf = LightPFNClassifier(
    revision="bd389ab59a89dd0e05c9ecb7c642c08ee52e9637", precision="fp32"
)
```

The 1.1 revision keeps its root files on Hugging Face and remains available; it has no regression model.

## How it works

<p align="center"><img src="https://raw.githubusercontent.com/GioOtto/LightPFN/main/docs/assets/architecture.png" width="100%" alt="LightPFN architecture: cell embedding, two column stages, row refinement and compression, in-context learning, retrieval decoder" /></p>

Each cell is embedded from its value, its rank in the column and a missing-value flag. Two column stages with
induced attention read the labelled context of each column. In 1.5, a categorical adapter adds level-aware
embeddings and ordered target statistics for columns marked as categorical. A row stage with four summary tokens
mixes the features of each row and compresses it into one vector. Seven in-context blocks let every test row
attend to the training rows. In the classifier, a retrieval decoder turns the attention into a vote over the
training labels; in the regressor, the decoder retrieves context target moments and a quantile head predicts 999
sorted quantiles from `.001` to `.999`.

Pretraining uses only synthetic tasks:

- **1.1 classifier:** 7.68 million task draws from 4.03 million distinct synthetic tasks (90% from the `graph`
  v3 structural causal model prior and 10% from the `rule` v1 prior of XOR, parity, lookup tables and shallow
  trees), followed by a long-context stage on tables of up to 60,000 rows.
- **1.5 classifier:** 120,000 steps of 64 tasks (7.68 million task draws: 90% `graph` v4 with heavy tails and
  quantized high-cardinality categories, 10% `rule` v1, both at `p_binary=0.55`, with `--cat-mask-drop 0.25`),
  followed by a 15,000-step long-context stage on tables of up to 60,000 rows.
- **1.5 regressor:** fine-tuned from the classification trunk on the `r2` regression prior (`graph` v3 with
  learnability filtering, observation noise, heavy tails and discrete targets), followed by a 5,000-step
  long-context stage up to 60,000 rows.

The [LightPFN 1.5 technical report](https://github.com/GioOtto/LightPFN/blob/main/paper/LightPFN15_report.pdf)
and the [report of 1.1](https://github.com/GioOtto/LightPFN/blob/main/paper/LightPFN_report.pdf) describe the
models, priors and selection protocols; [docs/en/TRAINING.md](docs/en/TRAINING.md) explains how to reproduce the
training.

## Roadmap

Version 1.5 delivered the two items on the 1.1 roadmap:

- **Native categorical features:** the categorical adapter is trained from scratch with the categorical mask
  active during pretraining (`--cat-mask-drop 0.25`) on a prior mix containing high-cardinality categorical
  columns (`graph` v4), and `cat="auto"` detects categorical columns at inference. (In 1.1, an adapter trained
  on top of the frozen v1 weights did not help in the tested 1.1 setup; 1.1 report, Section 12.)
- **Regression:** `LightPFNRegressor` predicts the mean and 999 quantiles from a dedicated 4.77M-parameter
  checkpoint.

Vulkan also supports the categorical adapter and regression in 1.5. The next step is running the official
TabArena benchmark for 1.5.

## Repository layout

| Path | Content |
|---|---|
| `lightpfn/` | the package: model (`model/`), scikit-learn classifier (`sklearn.py`) and regressor (`regressor.py`), precision helpers (`precision.py`), devices (`device.py`), Vulkan backend (`vulkan/`), and the training code: priors (`prior/`), trainer (`train.py`), evaluation harness (`eval/`) |
| `tests/` | unit and equivalence tests (CPU; the Vulkan tests also run on a CPU driver) |
| `examples/` | runnable classification, regression, pandas, pipeline and device examples |
| `docs/` | user guide, results, Vulkan backend and training in English (`en/`) and Italian (`it/`), changelog, third-party licenses |
| `paper/` | technical reports (`LightPFN15_report.pdf` for 1.5 and `LightPFN_report.pdf` for 1.1), LaTeX sources, figures and plot data |
| `CHANGELOG.md` | release history at the root of the repository |

The published wheel contains only the inference code; training needs a source checkout
(`pip install -e ".[train,eval]"`).

## Contributing and support

Bug reports and focused pull requests are welcome: see
[CONTRIBUTING](.github/CONTRIBUTING.md). Security issues go through
[SECURITY](.github/SECURITY.md). Changes between versions are in
[CHANGELOG](CHANGELOG.md).

## Citation

```bibtex
@techreport{ottoboni2026lightpfn15,
  title  = {{LightPFN} 1.5: Native categoricals, quantile regression, and fast portable inference under 5 million parameters},
  author = {Ottoboni, Giorgio},
  year   = {2026},
  url    = {https://github.com/GioOtto/LightPFN/blob/main/paper/LightPFN15_report.pdf}
}

@techreport{ottoboni2026lightpfn,
  title  = {A Sling Against Giants: {LightPFN}, a 4.6M-parameter tabular in-context classifier designed to stay small},
  author = {Ottoboni, Giorgio},
  year   = {2026},
  url    = {https://github.com/GioOtto/LightPFN/blob/main/paper/LightPFN_report.pdf}
}
```

The first entry is the 1.5 report, the second the report of 1.1. See the [LightPFN 1.5 technical report](https://github.com/GioOtto/LightPFN/blob/main/paper/LightPFN15_report.pdf)
and the [report of 1.1](https://github.com/GioOtto/LightPFN/blob/main/paper/LightPFN_report.pdf). GitHub's "Cite
this repository" button reads [CITATION.cff](CITATION.cff).

## License

Code and weights: [Apache License 2.0](LICENSE), with the attribution notice in [NOTICE](NOTICE).
Dependencies keep their own licenses, listed in [THIRD_PARTY_LICENSES](docs/THIRD_PARTY_LICENSES.md).
