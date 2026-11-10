# User guide

**English** · [Italiano](../it/GUIDA.md)

See the [README](../../README.md) for classification and regression quickstarts. Both `LightPFNClassifier` and
`LightPFNRegressor` expose `fit`, `predict`, `score`, `get_params` and `set_params`, and support `clone`,
`Pipeline` and `GridSearchCV`. The classifier adds `predict_proba`; the regressor adds
`predict_quantiles(X, quantiles=None)`.

- [How a fit works](#how-a-fit-works)
- [Regression](#regression)
- [Parameters](#parameters)
- [Fitted attributes](#fitted-attributes)
- [Input data](#input-data)
- [Categories](#categories)
- [Large tables](#large-tables)
- [Devices](#devices)
- [Precision](#precision)
- [Speed](#speed)
- [Weights, offline use and custom checkpoints](#weights-offline-use-and-custom-checkpoints)
- [Reproducibility](#reproducibility)
- [Low-level API](#low-level-api)

## How a fit works

`fit(X, y, cat="auto")` does not train anything. It validates the data, loads the pretrained network (once per
estimator instance), and encodes the training set: column statistics, categorical statistics when the categorical
adapter is active, the inducing-point states of the two column stages, and the keys and values of the in-context
blocks. `predict_proba(X)` (or `predict` / `predict_quantiles` in regression) runs only the test rows through
the network, in chunks, and attends to that cached context. A test row's prediction never depends on the other
test rows.

In `LightPFNClassifier`, this describes the default `cache_context=True`. With `cache_context=False`, `fit`
stores owned input arrays; each prediction builds the same context and releases it afterwards. Predictions are
unchanged.

In `LightPFNClassifier` with `n_estimators > 1`, the first estimator uses the original feature order and class
slots and every further one a random feature permutation and a random assignment of classes to the model's label
slots; the probabilities are averaged. On TabArena (1.1), four estimators raise the mean AUC by about 0.13 points
over one and improve the mean rank among seven models from 3.37 to 2.50, at four times the cost on CPU.

## Regression

`LightPFNRegressor` is a pretrained in-context quantile regressor with 4,772,519 parameters. Regression targets
`y` must be a finite 1-D vector of shape `(n_rows,)`; multioutput target matrices are not supported.

- `predict_quantiles(X, quantiles=None)` returns a `(n_rows, 999)` array for the 999 fixed quantile levels
  `.001` through `.999` when `quantiles=None`, or interpolates the requested levels in `[0, 1]`.
- `predict(X)` returns the predictive mean by integrating the 999-quantile curve with the trapezoidal rule.
- With the default `n_estimators=2` (`1` or `2` supported), the first member encodes `y` in original units.
  Controlled by `log_skew=2.0`, the second member encodes `log(y)` only when every subsampled context target is
  strictly positive and the adjusted Fisher-Pearson sample skewness (as in `pandas.Series.skew()`) reaches
  `log_skew`; otherwise the second member encodes `-y` (`"flip"`). Setting `log_skew=None` disables the log
  transform and always uses `-y` for the second member. The fitted attribute `second_` records `"log"`,
  `"flip"`, or `None` when `n_estimators=1`.
- Quantiles from a `"log"` member are mapped back to original units with `exp` and clipped to the context target
  range widened by one range on each side, with a lower bound of zero. Quantiles from a `"flip"` member are
  negated and reversed. The two members are averaged at each quantile level and sorted.
- `query_rows=4096` bounds the number of test rows per regression prediction chunk; `chunk_cells="auto"` bounds
  frontend cell-stage work.

## Parameters

### Overview

| Parameter | `LightPFNClassifier` default | `LightPFNRegressor` default |
|---|---|---|
| `n_estimators` | `4` (feature and label-slot permutations) | `2` (`1` or `2`: `y` plus `log(y)` or `-y`) |
| `max_context` | `20000` (stratified subsample per member) | `50000` (uniform subsample) |
| `random_state` | `None` (`seed=0` alias) | `0` |
| `device` | `"auto"` (`cuda`, `vulkan`, `cpu`) | `"auto"` (`cuda`, `vulkan`, `cpu`) |
| `precision` | `"auto"` (`"auto"`, `"fp32"`, `"fp16"`) | `"auto"` (`"auto"`, `"fp32"`, `"fp16"`) |
| `checkpoint` | `None` (local `.pt` or safetensors folder) | `None` (local `.pt` or safetensors folder) |
| `repo_id`, `revision`, `cache_dir`, `local_files_only` | Hugging Face options | Hugging Face options |
| `n_threads` | `None` (process-wide PyTorch CPU threads) | `None` (process-wide PyTorch CPU threads) |

### Classifier parameters

```python
LightPFNClassifier(model=None, checkpoint=None, device="auto", n_estimators=4, max_context=20000,
                   chunk_rows=2048, n_threads=None, seed=0, *, chunk_cells="auto", batch_cells="auto",
                   fold=True, random_state=None, repo_id=None, revision=None, cache_dir=None,
                   local_files_only=False, cache_context=True, precision="auto")
```

| Parameter | Default | Meaning |
|---|---|---|
| `n_estimators` | `4` | estimators averaged over feature and label-slot permutations; 1 is about four times faster on CPU |
| `device` | `"auto"` | `"auto"`, `"cpu"`, `"cuda[:i]"`, `"vulkan[:i]"`; see [Devices](#devices) |
| `precision` | `"auto"` | `"auto"` (fp16 on supported CUDA/ROCm GPUs from 2,048 context cells, fp32 on CPU and Vulkan), `"fp32"`, or `"fp16"`; see [Precision](#precision) |
| `max_context` | `20000` | above this many training rows, each estimator reads a stratified subsample of this size |
| `random_state` | `None` | seed of the permutations and subsamples (`seed` is the older alias; set one of the two) |
| `n_threads` | `None` | PyTorch CPU threads (process-wide `torch.set_num_threads`); `None` keeps PyTorch's setting |
| `chunk_rows` | `2048` | test rows per prediction chunk (memory bound) |
| `checkpoint` | `None` | a local `model.safetensors` (with `config.json` next to it), its folder, or a `.pt` file |
| `model` | `None` | a `LightPFN` torch module to use instead of loading weights (copied at fit) |
| `repo_id`, `revision` | `None` | another Hugging Face repository or revision; a custom repository needs an explicit revision |
| `cache_dir` | `None` | Hugging Face cache folder |
| `local_files_only` | `False` | never touch the network; the weights must already be cached |
| `cache_context` | `True` | keep encoded training contexts between predictions; `False` stores owned input arrays and rebuilds then releases contexts at each prediction, useful for fold models predicted only once |
| `chunk_cells`, `batch_cells`, `fold` | `"auto"`, `"auto"`, `True` | inference implementation: cache blocking, GPU estimator batching and the folded network. They change speed and memory, not the predictions (up to rounding). `chunk_cells=None, batch_cells=0, fold=False` is the plain path |

### Regressor parameters

```python
LightPFNRegressor(model=None, checkpoint=None, device="auto", precision="auto", n_estimators=2,
                  max_context=50000, random_state=0, log_skew=2.0, query_rows=4096, chunk_cells="auto",
                  n_threads=None, repo_id=None, revision=None, cache_dir=None, local_files_only=False)
```

| Parameter | Default | Meaning |
|---|---|---|
| `n_estimators` | `2` | `1` or `2`; when `2`, averages `y` with `log(y)` or `-y` in quantile space |
| `device` | `"auto"` | `"auto"` (CUDA/ROCm if available, else CPU), `"cpu"`, `"cuda[:i]"` |
| `precision` | `"auto"` | `"auto"`, `"fp32"`, or `"fp16"`; see [Precision](#precision) |
| `max_context` | `50000` | above this many training rows, subsamples `max_context` rows uniformly without replacement |
| `random_state` | `0` | seed for context subsampling |
| `log_skew` | `2.0` | skewness threshold for using `log(y)` as the second member on positive targets; `None` always uses `-y` |
| `query_rows` | `4096` | test rows per prediction chunk |
| `chunk_cells` | `"auto"` | cells per frontend piece |
| `n_threads` | `None` | PyTorch CPU threads (process-wide `torch.set_num_threads`); `None` keeps PyTorch's setting |
| `checkpoint`, `model`, `repo_id`, `revision`, `cache_dir`, `local_files_only` | `None` / `False` | local checkpoint, custom `LightPFN` module, or Hugging Face download options |

Both constructors store their parameters and do nothing else, so `clone`, `get_params` and `set_params` work as
in any scikit-learn estimator and never download weights or touch a GPU.

## Fitted attributes

### Classifier fitted attributes

| Attribute | Content |
|---|---|
| `classes_` | the class labels, in the column order of `predict_proba` |
| `n_features_in_`, `feature_names_in_` | number and names (for DataFrames with string column names) of the features |
| `is_categorical_` | boolean mask of the columns encoded as categories from a pandas DataFrame |
| `cat_` | boolean mask passed to the categorical adapter (`None` when no column is marked or the loaded model has no adapter) |
| `categories_` | dict from column position to the fitted category vocabulary; a category dtype keeps all its declared levels, including unused ones |
| `device_` | the selected backend device, updated by `to()` |
| `fit_device_` | the device that actually runs inference, including any automatic CPU fallback |
| `precision_` | `"fp16"` or `"fp32"`, the precision policy selected during `fit` |
| `n_precision_fallbacks_` | cumulative count of non-finite fp16 blocks retried in fp32 since `fit` |
| `model_` | the loaded `LightPFN` network |

### Regressor fitted attributes

| Attribute | Content |
|---|---|
| `n_features_in_`, `feature_names_in_` | number and names (for DataFrames with string column names) of the features |
| `device_` | the resolved torch device |
| `second_` | `"log"`, `"flip"`, or `None` (when `n_estimators=1`), recording the second-member transform |
| `precision_` | `"fp16"` or `"fp32"`, the precision policy selected during `fit` |
| `n_precision_fallbacks_` | cumulative count of non-finite fp16 blocks retried in fp32 since `fit` |
| `regressor_` | the underlying fitted `V15Regressor` engine (holds `cat_`, `categories_`, `skew_`, `y_range_`, `ctx_`) |

## Input data

- **Arrays**: any numeric array-like of shape `(n_rows, n_features)`. NaN marks a missing value; infinite
  values and sparse matrices are rejected.
- **pandas DataFrames**: columns of category, string, object or bool dtype are encoded as ordinal codes of
  the categories seen in `fit` (in pandas order: sorted values for strings, the dtype's own order for a
  category dtype, including unused declared levels). Missing values and values outside that vocabulary become
  NaN. Numeric columns, including nullable `Int64` and `Float64`, are converted to float with `pd.NA` as NaN.
  At prediction the columns must have the same names and order as in `fit`.
- **Classification labels**: any labels scikit-learn accepts (integers, strings). The classifier was trained on
  2 to 10 classes; more than 10 raise an error. With a single class, `predict_proba` returns 1 for that class.
- **Regression targets**: a finite 1-D numeric array-like of shape `(n_rows,)`. NaN or infinite targets and 2-D
  multioutput arrays raise a `ValueError`.
- **Feature count**: the models were trained on up to 100 features and run on more; wide tables cost more
  (the cell stages are linear in the number of features).
- **Scaling**: not needed. Each column is standardized and ranked on the training rows inside the model.

## Categories

Both `LightPFNClassifier.fit(X, y, cat="auto")` and `LightPFNRegressor.fit(X, y, cat="auto")` accept a `cat`
argument:

- `cat="auto"` (default): when the loaded network has a categorical adapter (`Config.cat_adapter=True`, enabled
  in the 1.5 default weights), `auto_categorical(X)` marks pandas `category`, string, `object` and `bool`
  columns plus numeric columns whose non-missing training values are all integers with 2 to 20 distinct levels.
  Detection inspects the training rows only.
- Explicit boolean mask: a 1-D boolean array-like of length `n_features` marks the categorical columns directly.
- `cat=None`: in `LightPFNClassifier`, retains pandas dtype-based categorical marking (`is_categorical_`) without
  inspecting integer-valued numeric columns; in `LightPFNRegressor`, disables the categorical mask.

Columns marked as categorical are processed by the 1.5 categorical adapter (separate Fourier weights and
ordered target statistics). In the 1.1 weights (`Config.cat_adapter=False`), categorical columns are read only as
ordinal codes; passing an explicit `cat` mask to a model without an adapter emits a `FutureWarning`. For tables
dominated by columns with thousands of rare levels, dedicated target-statistics tree models such as CatBoost can
still be stronger.

## Large tables

The in-context attention grows with the square of the context length, so time and memory grow quickly with the
number of training rows.

- **Classification (`max_context=20000`):** up to `max_context` rows, every estimator reads the whole training
  set. Above it, each estimator reads its own stratified subsample of `max_context` rows that keeps every class
  (at least five rows of each class when the class counts and context budget permit). More estimators then also
  cover more of the data.
- **Regression (`max_context=50000`):** above `max_context` rows, `fit` draws a uniform subsample of
  `max_context` rows without replacement using `random_state`.
- `max_context` can be raised. Both 1.5 checkpoints were trained with a long-context stage up to 60,000 rows. In
  the 1.1 report, contexts of up to 100,000 rows were evaluated for classification, where the model is level
  with default CatBoost on average; the cost on CPU is high (about 75 s for 94,000 rows with one estimator on 24
  threads).

## Devices

<a id="precision-and-devices"></a>

Both `LightPFNClassifier` and `LightPFNRegressor` use the same device policy: `device="auto"` tries a CUDA or
ROCm GPU through PyTorch (`torch.cuda.is_available()`), a GPU with a Vulkan driver if `wgpu` is installed,
then the CPU. The environment variable `LIGHTPFN_DEVICE` replaces `"auto"` (for example
`LIGHTPFN_DEVICE=cpu`). An explicit device is used as given, with an error if it is unavailable.

**Vulkan support and limits in 1.5:**

- Both estimators support numerical and marked categorical columns. Cell embeddings, categorical-statistics
  projections, column and row stages, ICL and decoder attention run on the GPU. Column/category statistics
  and rank lookup stay on the host. Regression target normalization and the final quantile head readout also
  run on the host. Vulkan always uses fp32.
- On an RX 7900 XT, classification probabilities agree with torch CPU within 2.6e-5 with identical argmax,
  tested up to 12,000 context rows with high-cardinality categoricals. Regression quantiles agree within
  5.5e-5 relative, tested up to 40,000 context rows. Measured larger-context 1.5 workloads are 3 to 7 times
  faster than CPU; small tables can be slower. Intel and NVIDIA Vulkan drivers use the same code but have
  not been benchmarked.
- On Vulkan, one estimator's training cells (rows times features times 64 floats) must fit a single GPU buffer
  (2 GiB on the tested RX 7900 XT driver, about 8.4 million cells). For both estimators, a fit exceeding GPU
  buffer limits falls back to CPU with a warning under `"auto"` and raises `MemoryError` under explicit
  `"vulkan"`. Other context and scratch buffers, including the two-member regression cache, can impose a
  smaller limit; 8.4 million cells is an upper bound. Details in [VULKAN.md](VULKAN.md).

In `fp32`, all devices compute the same function: probabilities agree within about 3e-5.

`clf.to("cpu")` or `clf.to("cuda[:i]")` moves a fitted PyTorch classifier and its cached contexts and returns
the classifier (`clf.to("cpu")` also converts cached fp16 tensors to float32 and sets `precision_="fp32"`). It
leaves the constructor's `device` parameter unchanged; the next `fit` resolves that parameter again. A
classifier initialized with Vulkan cannot be moved with `to()`; refit it on the desired device instead.
`reg.to()` also moves only between torch devices; refit with `device="vulkan"` to use Vulkan.

## Precision

Both estimators accept `precision="auto"`, `"fp32"` or `"fp16"`:

- `precision="auto"` (default): selects `"fp16"` autocast on supported CUDA/ROCm GPUs when one estimator member
  has at least `FP16_MIN_CELLS` (2,048) context cells (`n_context_rows * n_features >= 2048`), and `"fp32"` on
  smaller tables, on CPU, and on Vulkan.
- `precision="fp32"`: forces float32 on every device.
- `precision="fp16"`: requires a CUDA or ROCm device (`ValueError` on CPU or Vulkan).
- The retrieval and moment decoders remain in float32, and regression target statistics and quantile integration
  use float64.
- Reduced precision can change predicted probabilities, class labels and quantiles slightly compared with fp32.
- If any encoded context group or prediction chunk produces non-finite values in fp16, it automatically retries
  in float32, logs a `DEBUG` message on the `"lightpfn"` logger, and increments `n_precision_fallbacks_`.
- `precision_` records the policy chosen during `fit`; changing `precision` via `set_params` takes effect on the
  next `fit`.

## Speed

Measured fit plus predict times are in [RESULTS.md](RESULTS.md#cost). In practice:

- A GPU is much faster: on an RX 7900 XT, Vulkan and ROCm are 6 to 10 times faster than a 16-thread desktop CPU
  for 1.1; in 1.5 on an RTX 5090, `precision="auto"` (fp16) and fused layers are about 1.9x to 2.0x faster than
  1.1.0 across the 8-fold x 4-member protocol.
- On CPU, `n_estimators=1` is about four times faster than 4 in classification (and twice as fast as 2 in
  regression) and loses little accuracy.
- Set `n_threads` to the number of physical cores; hyper-threads and efficiency cores help little.
- Predict in large batches: one call with many rows is faster than many calls with few rows.

## Weights, offline use and custom checkpoints

<a id="local-weights-and-migration"></a>

Without `checkpoint` or `model`, the first `fit` downloads `config.json` and `model.safetensors` from the
Hugging Face repository `ueuegio/LightPFN` at the commit pinned in `lightpfn/pretrained.json`, through
`huggingface_hub` and its cache (`HF_HOME`, `HF_HUB_CACHE`). In 1.5, the repository has separate `classifier/`
and `regressor/` subfolders, and `fit` downloads only the subfolder needed by the estimator. The download never
runs code from the repository.

```python
from lightpfn import LightPFNClassifier, LightPFNRegressor, load_pretrained, save_model

# offline: the weights must already be in the Hugging Face cache
clf = LightPFNClassifier(local_files_only=True)
reg = LightPFNRegressor(local_files_only=True)

# save a local copy of each checkpoint
save_model(load_pretrained(task="classifier"), "weights/classifier")   # writes model.safetensors and config.json
save_model(load_pretrained(task="regressor"), "weights/regressor")

clf = LightPFNClassifier(checkpoint="weights/classifier", device="cpu", precision="fp32")
reg = LightPFNRegressor(checkpoint="weights/regressor", device="cpu", precision="fp32")

# load the 1.1 classification weights in 1.5 (root layout at the 1.1 revision)
old = LightPFNClassifier(revision="bd389ab59a89dd0e05c9ecb7c642c08ee52e9637", precision="fp32")
```

Setting the environment variable `LIGHTPFN_WEIGHTS_DIR=weights` directs `load_pretrained` to read the local
`weights/classifier/` and `weights/regressor/` subfolders directly without network access.

`load_model(path)` reads safetensors with their JSON config, or PyTorch `.pt` files through
`torch.load(weights_only=True)`. Checkpoints that need arbitrary pickled objects are rejected on purpose.

## Reproducibility

With the same `random_state`, data, device, `precision` and library versions, `fit`, `predict_proba`, `predict`
and `predict_quantiles` return the same outputs. In `precision="fp32"`, across devices and between the batched
and sequential paths, predictions agree up to floating-point rounding. With `precision="auto"` on a CUDA or ROCm
GPU, fp16 is used above 2,048 context cells and predictions can differ slightly from fp32; pass
`precision="fp32"` when comparing against CPU runs or 1.1. Changing the order of the test rows or the prediction
batch size can change outputs by about 1e-7 in fp32.

## Low-level API

```python
import torch
from lightpfn import load_pretrained

model = load_pretrained(task="classifier").eval()     # a torch.nn.Module (LightPFN)
net = model.folded()                                  # the same function with normalizations folded in
X_train = torch.randn(1, 500, 8); y_train = torch.randint(0, 2, (1, 500)); X_test = torch.randn(1, 100, 8)
with torch.inference_mode():
    ctx = net.encode(X_train, y_train, n_classes=2)   # batch of tasks: (B, rows, features)
    logits = net.predict_logits(ctx, X_test)          # (B, test rows, classes)
```

`lightpfn.Config` holds the architecture; `LightPFN(Config())` builds the default (untrained) network. Further
details are in the [LightPFN 1.5 technical report](https://github.com/GioOtto/LightPFN/blob/main/paper/LightPFN15_report.pdf)
and the [report of 1.1](https://github.com/GioOtto/LightPFN/blob/main/paper/LightPFN_report.pdf).
