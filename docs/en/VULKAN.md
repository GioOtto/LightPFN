# Vulkan backend

Version 1.5 supports classification with the categorical adapter and quantile regression on Vulkan.
Both estimators use fp32 on this backend; `precision="fp16"` requires CUDA/ROCm.

**English** · [Italiano](../it/VULKAN.md)

`lightpfn.vulkan` runs LightPFN inference on any GPU with a Vulkan driver: AMD, Intel and NVIDIA, on Linux
and Windows. It needs no CUDA, no ROCm and no Vulkan SDK: the compute kernels are written in WGSL and
compiled to SPIR-V by [wgpu](https://github.com/pygfx/wgpu-py) when the backend starts.

```bash
pip install "LightPFN[vulkan]"
python -c "import lightpfn.vulkan as v; print(v.adapters())"
```

```python
from lightpfn import LightPFNClassifier, LightPFNRegressor

clf = LightPFNClassifier(device="vulkan").fit(X_train, y_class)
reg = LightPFNRegressor(device="vulkan").fit(X_train, y_reg)
quantiles = reg.predict_quantiles(X_test, quantiles=[0.1, 0.5, 0.9])
```

For both estimators, `device="auto"` tries CUDA/ROCm through PyTorch, then Vulkan, then CPU. The environment
variable `LIGHTPFN_DEVICE` replaces `"auto"`. Vulkan is selected when PyTorch sees no CUDA/ROCm GPU and wgpu
finds a Vulkan GPU. `to()` moves only between torch devices; refit to switch to Vulkan.

## Choosing the adapter

`device="vulkan"` takes the first discrete GPU, else an integrated one. `device="vulkan:1"` picks the adapter
with index 1 in `lightpfn.vulkan.adapters()`. The environment variable `LIGHTPFN_VULKAN_ADAPTER` selects an
adapter by index or by a substring of its name; `LIGHTPFN_VULKAN_ADAPTER=llvmpipe` runs the kernels on Mesa's
CPU driver, which is how the tests run without a GPU.

## Design

The backend implements the same computation as the folded inference network: `encode` builds the context,
`predict_logits` handles classification and `predict_quantiles` handles regression.

Cell embeddings (including categorical Fourier features), categorical-statistics projections (`CatStatsEmbed`),
column stages, row refinement, the row stage, ICL transformer blocks and decoder attention run on the GPU.
Column statistics, value normalization and rank lookup (sort/search), category statistics (`context_stats` and
`query_stats`) and preparation of their projection weights run on the host in PyTorch. Regression target
normalization also runs on the host. After the GPU decoder attention computes the moments, the small final
quantile head readout (including row normalization and moment mean/scale), target rescaling, member averaging
and point-prediction integration run on the host. Target rescaling uses float64. Column, ICL key/value and
decoder-key caches stay in GPU memory between `fit` and `predict`.

- **Eight kernels.** A tiled matrix product with optional fused normalization, bias, GELU and residual;
  tiled attention with an online softmax (as in flash attention) for long key sequences; a small-attention
  kernel for the few summary and inducing tokens; row normalization; the cell features (soft-clipped z-score, empirical CDF rank, Fourier features of
  neighbouring columns, missing-value flags); rotary embeddings; the learned query scaling; and a strided copy.
- **Views instead of copies.** Every kernel reads and writes tensors through strided views (base offset, row
  length and strides), so the same kernel reads a column of cells, a row of cells, the first tokens of every
  row or a broadcast parameter without transposes or copies.
- **Precision independent of the driver.** All Vulkan kernels use float32. Sine, cosine and erf use their own range
  reduction and polynomial approximations (error below 5e-7), so results do not depend on the precision of
  each driver's transcendental functions.
- **Short GPU jobs.** Drivers reset a GPU whose job runs too long (amdgpu's ring timeout on Linux, the 2 s
  TDR on Windows). Heavy dispatches are cut into slices of workgroups and submitted in jobs of about 5e10
  floating-point operations, a few milliseconds each on a discrete GPU. A single 20,000-row encode submitted
  as one job did trigger a ring reset during development; sliced, it runs normally.
- **Memory.** Weights are uploaded once per model. The context caches of an estimator and the scratch
  buffers are allocated per fit and reused across prediction chunks.

## Agreement with PyTorch

Tests compare the backend with torch CPU fp32 on random networks, an optional historical B4 checkpoint and
the final 1.5 classification/regression checkpoints. Cases cover missing and unseen categories, absent classes,
constant columns, padded features, both regression members and buffer-limit fallback. The Vulkan tests also
run on Mesa llvmpipe without a GPU (`tests/test_vulkan.py`).

On an RX 7900 XT (Linux, Mesa RADV), the final 1.5 classifier was checked on mixed tables with three
categorical columns (5, 60 and 400 levels, 5% missing values), with 600, 4,000 and 12,000 context rows.
Probabilities agree with torch CPU within 2.6e-5 in absolute difference, with identical argmax on all checked
rows. The two-member regressor was checked on five regression sources up to 8,000 context rows, with quantile
relative differences at most 9e-6. Checks up to 40,000 context rows reached at most 5.5e-5 relative difference
on quantiles and 3.4e-5 on point predictions; RMSE was unchanged at the reported precision. These are measured
agreements on the checked tables, not universal error bounds.

## Speed

The same 1.5 checks on the RX 7900 XT measured CPU vs Vulkan fit plus predict at 0.36 vs 0.53 s for 600 context
rows, 1.86 vs 0.26 s for 4,000 and 8.24 vs 1.56 s for 12,000 in classification. Regression at 40,000 context
rows was about 3 times faster than CPU. Larger-context workloads were 3 to 7 times faster; the small
classification table was slower on Vulkan. Intel and NVIDIA Vulkan drivers use the same code but have not
been benchmarked. The first fit in a process also compiles the kernels it needs.

For historical comparison, the following measurements are for **1.1 without the categorical adapter**, with
four estimators on synthetic binary tables: Intel i7-13700KF (16 threads), AMD RX 7900 XT, Linux, Mesa RADV,
PyTorch 2.13 with ROCm 7.2. They are not 1.5 measurements.

| Train rows x features, test rows | CPU, 16 threads | ROCm (PyTorch) | Vulkan | Vulkan vs CPU |
|---|---:|---:|---:|---:|
| 1,000 x 20, 500 | 0.45 s | 0.057 s | 0.044 s | 10.3x |
| 5,000 x 50, 2,000 | 4.84 s | 0.59 s | 0.50 s | 9.6x |
| 10,000 x 100, 5,000 | 19.8 s | 2.48 s | 2.13 s | 9.3x |
| 20,000 x 50, 5,000 | 29.2 s | 4.76 s | 4.86 s | 6.0x |
| 20,000 x 200, 10,000 | 70.4 s | 10.4 s | 8.68 s | 8.1x |

## Limits

- **Buffer size.** One estimator's training cells (rows x features x 64 floats) must fit one GPU buffer
  (2 GiB on the tested RX 7900 XT driver, about 8.4 million cells, for example 84,000 rows x 100 features). Above that,
  `device="auto"` falls back to the CPU with a warning and `device="vulkan"` raises `MemoryError`. Lower
  `max_context` to stay on the GPU.
  This policy applies to classification and regression. Other context and scratch buffers, including the
  two-member regression cache, can impose a smaller limit; 8.4 million cells is an upper bound.
- **Platforms.** Linux and Windows. macOS is not supported (wgpu would use Metal, which this backend does
  not target). Tested on an AMD RX 7900 XT (Linux RADV) and on Mesa llvmpipe; Windows validation is pending;
  Intel and NVIDIA Vulkan drivers use the same code path but have not been benchmarked.
- **Training** is not supported; Vulkan is for inference, including the categorical adapter and regression.

## Alternatives we tried

We also tried two general compilers. IREE had no attention lowering for Vulkan, generated wrong code for
RDNA3 on part of the network and was slow with dynamic shapes. ONNX Runtime's WebGPU provider computed the
fused attention with errors of 2 to 17% on our shapes. The hand-written kernels match PyTorch up to rounding
and are as fast as PyTorch with ROCm on the same GPU.
