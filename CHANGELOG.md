# Changelog

All notable changes to this project are listed here. Versions follow [semantic versioning](https://semver.org/).

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
