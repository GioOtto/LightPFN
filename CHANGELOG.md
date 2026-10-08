# Changelog

All notable changes to this project are listed here. Versions follow [semantic versioning](https://semver.org/).

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
