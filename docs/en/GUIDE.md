# User guide

**English** · [Italiano](../it/GUIDA.md)

- [How a fit works](#how-a-fit-works)
- [Parameters](#parameters)
- [Fitted attributes](#fitted-attributes)
- [Input data](#input-data)
- [Large tables](#large-tables)
- [Devices](#devices)
- [Speed](#speed)
- [Weights, offline use and custom checkpoints](#weights-offline-use-and-custom-checkpoints)
- [Reproducibility](#reproducibility)
- [Low-level API](#low-level-api)

## How a fit works

`fit(X, y)` does not train anything. It validates the data, loads the pretrained network (once per
classifier), and encodes the training set: column statistics, the inducing-point states of the two column
stages and the keys and values of the in-context blocks. `predict_proba(X)` runs only the test rows through
the network, in chunks, and attends to that cached context. A test row's prediction never depends on the
other test rows.

This describes the default `cache_context=True`. With `cache_context=False`, fit stores owned input arrays;
each prediction builds the same context and releases it afterwards. Predictions are unchanged.

With `n_estimators > 1`, the first estimator uses the original feature order and class slots and every
further one a random feature permutation and a random assignment of classes to the model's label slots; the
probabilities are averaged. On TabArena four estimators raise the mean AUC by about 0.13 points over one
and improve the mean rank among seven models from 3.37 to 2.50, at four times the cost.

## Parameters

```python
LightPFNClassifier(model=None, checkpoint=None, device="auto", n_estimators=1, max_context=20000,
                   chunk_rows=2048, n_threads=None, seed=0, *, chunk_cells="auto", batch_cells="auto",
                   fold=True, random_state=None, repo_id=None, revision=None, cache_dir=None,
                   local_files_only=False, cache_context=True)
```

| Parameter | Default | Meaning |
|---|---|---|
| `n_estimators` | `1` | estimators averaged over feature and label-slot permutations; 4 is a good accuracy setting |
| `device` | `"auto"` | `"auto"`, `"cpu"`, `"cuda[:i]"`, `"vulkan[:i]"`; see [Devices](#devices) |
| `max_context` | `20000` | above this many training rows, each estimator reads a stratified subsample of this size |
| `random_state` | `None` | seed of the permutations and subsamples (`seed` is the older alias; set one of the two) |
| `n_threads` | `None` | PyTorch CPU threads (process-wide `torch.set_num_threads`); `None` keeps PyTorch's setting |
| `chunk_rows` | `2048` | test rows per prediction chunk (memory bound) |
| `checkpoint` | `None` | a local `model.safetensors` (with `config.json` next to it), its folder, or a `.pt` file |
| `model` | `None` | a `LightPFN` torch module to use instead of loading weights (copied at fit) |
| `repo_id`, `revision` | `None` | another Hugging Face repository; a custom repository needs an explicit revision |
| `cache_dir` | `None` | Hugging Face cache folder |
| `local_files_only` | `False` | never touch the network; the weights must already be cached |
| `cache_context` | `True` | keep encoded training contexts between predictions; `False` stores owned input arrays and rebuilds then releases contexts at each prediction, useful for fold models predicted only once |
| `chunk_cells`, `batch_cells`, `fold` | `"auto"`, `"auto"`, `True` | inference implementation: cache blocking, GPU estimator batching and the folded network. They change speed and memory, not the predictions (up to rounding). `chunk_cells=None, batch_cells=0, fold=False` is the plain path |

The constructor stores the parameters and does nothing else, so `clone`, `get_params` and `set_params` work
as in any scikit-learn estimator and never download or touch a GPU.

## Fitted attributes

| Attribute | Content |
|---|---|
| `classes_` | the class labels, in the column order of `predict_proba` |
| `n_features_in_`, `feature_names_in_` | number and names (for DataFrames with string column names) of the features |
| `is_categorical_` | boolean mask of the columns encoded as categories (pandas input) |
| `categories_` | dict from column position to the fitted category vocabulary; a category dtype keeps all its declared levels, including unused ones |
| `device_` | the selected backend device, updated by `to()` |
| `fit_device_` | the device that actually runs inference, including any automatic CPU fallback |
| `model_` | the loaded `LightPFN` network |

## Input data

- **Arrays**: any numeric array-like of shape `(n_rows, n_features)`. NaN marks a missing value; infinite
  values and sparse matrices are rejected.
- **pandas DataFrames**: columns of category, string, object or bool dtype are encoded as ordinal codes of
  the categories seen in `fit` (in pandas order: sorted values for strings, the dtype's own order for a
  category dtype, including unused declared levels). Missing values and values outside that vocabulary become NaN. Numeric columns, including
  nullable `Int64` and `Float64`, are converted to float with `pd.NA` as NaN. At prediction the columns must
  have the same names and order as in `fit`.
- **Labels**: any labels scikit-learn accepts (integers, strings). The model was trained on 2 to 10 classes;
  more than 10 raise an error. With a single class, `predict_proba` returns 1 for that class.
- **Feature count**: the model was trained on up to 100 features and runs on more; wide tables cost more
  (the cell stages are linear in the number of features).
- **Scaling**: not needed. Each column is standardized and ranked on the training rows inside the model.

Categorical columns are read as ordinal codes, so the model sees their order. That is fine for low and
medium cardinality; for columns with thousands of levels, target-statistics encoders such as CatBoost's
remain stronger (see the roadmap in the README).

## Large tables

The in-context attention grows with the square of the context length, so time and memory grow quickly
with the number of training rows.

- Up to `max_context` (20,000) rows, every estimator reads the whole training set.
- Above it, each estimator reads its own stratified subsample of `max_context` rows that keeps every class
  (up to five rows of each when the class counts and context budget permit). More estimators then also cover more of the data.
- `max_context` can be raised. The report evaluates contexts of up to 100,000 rows, where the model is level
  with default CatBoost on average; the cost on CPU is high (about 75 s for 94,000 rows with one estimator on
  24 threads).

## Devices

`device="auto"` tries, in order: a CUDA or ROCm GPU through PyTorch (`torch.cuda.is_available()`), a GPU
with a Vulkan driver if `wgpu` is installed, then the CPU. The environment variable `LIGHTPFN_DEVICE`
replaces `"auto"` (for example `LIGHTPFN_DEVICE=cpu`). An explicit device is used as given, and an error is
raised if it is not available.

On Vulkan, one estimator's training cells (rows times features times 64 floats) must fit a single GPU
buffer (2 GiB on the tested RX 7900 XT driver, about 8.4 million cells). If they do not, `"auto"` falls back to the CPU with
a warning and an explicit `"vulkan"` raises a `MemoryError`. Details in [VULKAN.md](VULKAN.md).
Other context and scratch buffers can impose a smaller limit; 8.4 million cells is an upper bound.

All devices compute the same function: probabilities agree within about 3e-5.

`clf.to("cpu")` or `clf.to("cuda[:i]")` moves a fitted PyTorch classifier and its cached contexts and returns
the classifier. It leaves the constructor's `device` parameter unchanged; the next `fit` resolves that
parameter again. A classifier initialized with Vulkan cannot be moved with `to()`; refit it on the desired
device instead.

## Speed

Measured fit plus predict times are in [RESULTS.md](RESULTS.md#cost). In practice:

- A GPU is much faster: on an RX 7900 XT, Vulkan and ROCm are 6 to 10 times faster than a 16-thread desktop CPU.
- On CPU, `n_estimators=1` is about four times faster than 4 and loses little accuracy.
- Set `n_threads` to the number of physical cores; hyper-threads and efficiency cores help little.
- Predict in large batches: one call with many rows is faster than many calls with few rows.

## Weights, offline use and custom checkpoints

Without `checkpoint` or `model`, the first `fit` downloads `config.json` and `model.safetensors` from the
Hugging Face repository `ueuegio/LightPFN` at the commit pinned in `lightpfn/pretrained.json`, through
`huggingface_hub` and its cache (`HF_HOME`, `HF_HUB_CACHE`). The download never runs code from the
repository.

```python
# offline: the weights must already be in the cache
clf = LightPFNClassifier(local_files_only=True)

# a local copy
from lightpfn import load_pretrained, save_model
save_model(load_pretrained(), "lightpfn_weights/")      # writes model.safetensors and config.json
clf = LightPFNClassifier(checkpoint="lightpfn_weights/")
```

`load_model(path)` reads safetensors with their JSON config, or PyTorch files through
`torch.load(weights_only=True)`. Checkpoints that need arbitrary pickled objects are rejected on purpose.

## Reproducibility

With the same `random_state`, data, device and library versions, `fit` and `predict_proba` return the same
probabilities. Across devices, and between the batched and sequential paths, they agree up to floating-point
rounding. Changing the order of the test rows or the prediction batch size can change probabilities by about
1e-7.

## Low-level API

```python
import torch
from lightpfn import load_pretrained

model = load_pretrained().eval()                      # a torch.nn.Module (LightPFN)
net = model.folded()                                  # the same function with normalizations folded in
X_train = torch.randn(1, 500, 8); y_train = torch.randint(0, 2, (1, 500)); X_test = torch.randn(1, 100, 8)
with torch.inference_mode():
    ctx = net.encode(X_train, y_train, n_classes=2)   # batch of tasks: (B, rows, features)
    logits = net.predict_logits(ctx, X_test)          # (B, test rows, classes)
```

`lightpfn.Config` holds the architecture; `LightPFN(Config())` builds the default (untrained) network.
