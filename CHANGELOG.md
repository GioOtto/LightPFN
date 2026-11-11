# Changelog

All notable changes to this project are listed here. Versions follow [semantic versioning](https://semver.org/).

## 1.5.1 (2026-11-11)

Inference fixes and faster GPU inference, with the same weights, architecture and Hugging Face revision as 1.5.0.
Cached predictions are unchanged (bit-identical on the 31 TabArena classification and 8 regression tasks we
timed); one-pass predictions match up to float rounding.

- One-pass inference for `cache_context=False` on CUDA/ROCm: when the training and test rows fit in one piece,
  the context is encoded and the test rows predicted in a single pass that shares every row-wise operation
  (`LightPFN.encode_predict_logits`, `encode_predict_quantiles`). About half the GPU kernel launches of the
  fold models in bagging. `LIGHTPFN_FUSE_DISABLE=onepass` restores the two-step path.
- `LightPFNRegressor(cache_context=False)`, as in the classifier: the fit keeps its inputs and each prediction
  encodes them, for fold models predicted once.
- The fp16 weight view now covers every projection that inference autocast runs in fp16 (cell embedding,
  categorical adapter, attention-scale MLPs), so autocast no longer casts their weights at every call.
  Bit-identical to 1.5.0 in fp16.
- DataFrame conversion and `cat="auto"` detection work on whole dtype blocks and on category codes instead of
  one pandas call per column (2-3x less host time in `fit` on tables with many categorical columns).
  Identical outputs.
- With the TabArena protocol on an RTX 5090, training is 1.17x faster than 1.5.0 with the fixes above on 31 classification tasks
  (1.32x on tables up to 3,000 rows) and 1.42x on 8 regression tasks; prediction 1.03x. Against 1.1.0:
  1.65x training and 1.64x prediction in classification.
- Evaluate the reduced-precision attention-scale MLP with 2D GEMMs on CUDA/ROCm. This removes the fp16
  strided-batched cuBLAS operation reported failing on a torch 2.15/CUDA 13.2 stack. CPU, fp32 and training
  keep their existing computation. This does not suppress CUDA exceptions or promise recovery from a
  damaged CUDA context.
- Avoid pandas `dropna()` allocations while detecting categorical NumPy-backed numeric columns; nullable
  extension columns retain their existing handling.
- Batch the GPU activation checks with infinity-norm reductions, preserving metadata padding and float64
  target units. The check transfers a single boolean to the host.
- `LightPFNRegressor.to()` updates the fitted engine's device metadata as well as its tensors, fixing
  GPU-to-CPU inference and subsequent device moves.
- Missing modern model subfolders now raise the original download/cache error. Root-level classification
  weights are loaded only for an explicitly requested historical 1.0/1.1 revision, preventing a partial
  offline cache from selecting old weights silently.

## 1.5.0 (2026-11-10)

Update of LightPFN with regression, native categorical columns, wide tables and faster GPU inference. Same
package, Hugging Face repository and API; the 1.1 weights stay available at their pinned revision.

- New default classification weights with a categorical adapter; `cat="auto"` detects categorical inputs.
- The classification weights are fine-tuned on synthetic tables with 64 to 500 features (pretraining stopped at
  100): better on wide tables, unchanged elsewhere within noise.
- Public `LightPFNRegressor`, quantiles, two-member default, `max_context=50000`, `log_skew=2.0`.
- `precision="auto"` uses fp16 on supported CUDA/ROCm GPUs, with fp32 retries for non-finite blocks.
- Separate `classifier/` and `regressor/` safetensors folders on the same HF repository; immutable package pin.
- 1.1 root weights remain available at their original revision. Predictions change with the new defaults.
- Vulkan supports the categorical adapter and regression in fp32. Both estimators use CUDA/ROCm, then Vulkan,
  then CPU under `device="auto"` (overridden by `LIGHTPFN_DEVICE`). Fits exceeding Vulkan buffer limits fall
  back to CPU with a warning under auto and raise `MemoryError` with explicit Vulkan; `to()` uses torch devices.
- Vulkan runs cell embeddings, categorical-statistics projections, column/row stages, ICL and decoder attention
  on the GPU; column/category statistics, rank lookup, regression target normalization and quantile head readout
  stay on the host. On RX 7900 XT, probabilities agree with torch CPU within 2.6e-5 with identical argmax up to
  12,000 context rows with high-cardinality categoricals; regression quantiles agree within 5.5e-5 relative up
  to 40,000 context rows. Measured larger-context 1.5 workloads are 3 to 7 times faster than CPU; small tables
  can be slower. Intel/NVIDIA Vulkan drivers use the same code but have not been benchmarked.

## 1.1.0 (2026-10-09)

- `LightPFNClassifier` now defaults to `n_estimators=4`, the configuration reported as
  "LightPFN (default)" on TabArena. Predictions change and CPU fit is about four times slower; pass
  `n_estimators=1` for the previous behaviour.

## 1.0.0 (2026-10-08)

First public release.

- Model: LightPFN v1, 4,603,088 parameters, the final long-context checkpoint described in the technical
  report. Weights in `model.safetensors` on Hugging Face (`ueuegio/LightPFN`), pinned by commit.
- `LightPFNClassifier`: scikit-learn estimator (`fit`, `predict_proba`, `predict`, `score`), compatible with
  `Pipeline`, `GridSearchCV` and `clone`; ensembles over feature and class-slot permutations; stratified
  context subsampling above `max_context`.
- pandas input: columns of category, string, object or bool dtype are encoded as ordinal codes of the
  training categories, with missing and unseen values as NaN (`categories_`, `is_categorical_`).
- Devices: `device="auto"` picks CUDA or ROCm through PyTorch, else a Vulkan GPU, else the CPU.
- Vulkan backend (`lightpfn.vulkan`, extra `vulkan`): WGSL compute kernels run through wgpu on AMD, Intel
  and NVIDIA GPUs, Linux and Windows, with the same predictions as PyTorch up to rounding.
- Faster inference with the same predictions: cache-blocked cell stages, a folded copy of the network and
  estimator batching on the GPU.
- Optional `cache_context=False` defers context encoding to prediction and releases it afterwards, for validation fold models used once. The default keeps contexts cached.
- Safe loading: safetensors with a JSON config, or PyTorch files with `weights_only=True`; no pickle fallback.
- Training code: synthetic priors (graph and rule), chunked data streaming, data-parallel training, and the
  evaluation harness used in the report (install from source with the `train` and `eval` extras).
- `Config.cat_adapter`: an experimental categorical adapter, off by default and not used by the released
  weights.
