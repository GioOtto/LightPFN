# Vulkan backend

**English** · [Italiano](../it/VULKAN.md)

`lightpfn.vulkan` runs LightPFN inference on any GPU with a Vulkan driver: AMD, Intel and NVIDIA, on Linux
and Windows. It needs no CUDA, no ROCm and no Vulkan SDK: the compute kernels are written in WGSL and
compiled to SPIR-V by [wgpu](https://github.com/pygfx/wgpu-py) when the backend starts.

```bash
pip install "LightPFN[vulkan]"
python -c "import lightpfn.vulkan as v; print(v.adapters())"
```

```python
from lightpfn import LightPFNClassifier

clf = LightPFNClassifier(device="vulkan").fit(X_train, y_train)
```

`device="auto"` uses Vulkan when PyTorch sees no CUDA or ROCm GPU and wgpu finds a Vulkan GPU.

## Choosing the adapter

`device="vulkan"` takes the first discrete GPU, else an integrated one. `device="vulkan:1"` picks the adapter
with index 1 in `lightpfn.vulkan.adapters()`. The environment variable `LIGHTPFN_VULKAN_ADAPTER` selects an
adapter by index or by a substring of its name; `LIGHTPFN_VULKAN_ADAPTER=llvmpipe` runs the kernels on Mesa's
CPU driver, which is how the tests run without a GPU.

## Design

The backend implements the same function as `LightPFN.folded()`, the inference copy of the network with
normalization weights folded into the next projections, with the same two calls: `encode` builds the context
of a training set and `predict_logits` runs test rows against it. The column statistics and the rank
lookup of the cell values (a sort and a binary search per column) run on the host in PyTorch; every layer
after them runs on the GPU, and the training-set context stays in GPU memory between `fit` and `predict`.

- **Eight kernels.** A tiled matrix product with optional fused normalization, bias, GELU and residual;
  tiled attention with an online softmax (as in flash attention) for long key sequences; a small-attention
  kernel for the few summary and inducing tokens; row normalization; the cell features (soft-clipped z-score, empirical CDF rank, Fourier features of
  neighbouring columns, missing-value flags); rotary embeddings; the learned query scaling; and a strided copy.
- **Views instead of copies.** Every kernel reads and writes tensors through strided views (base offset, row
  length and strides), so the same kernel reads a column of cells, a row of cells, the first tokens of every
  row or a broadcast parameter without transposes or copies.
- **Precision independent of the driver.** Everything is float32. Sine, cosine and erf use their own range
  reduction and polynomial approximations (error below 5e-7), so results do not depend on the precision of
  each driver's transcendental functions.
- **Short GPU jobs.** Drivers reset a GPU whose job runs too long (amdgpu's ring timeout on Linux, the 2 s
  TDR on Windows). Heavy dispatches are cut into slices of workgroups and submitted in jobs of about 5e10
  floating-point operations, a few milliseconds each on a discrete GPU. A single 20,000-row encode submitted
  as one job did trigger a ring reset during development; sliced, it runs normally.
- **Memory.** Weights are uploaded once per classifier. The context caches of an estimator and the scratch
  buffers are allocated per fit and reused across prediction chunks.

## Agreement with PyTorch

Tests compare the backend with the PyTorch path on random networks and optionally on a historical B4 checkpoint, with missing
values, constant columns, padded features and several architecture variants: logits agree within about
1e-6, and probabilities through the classifier within 3e-5. The Vulkan tests run on the GPU and on llvmpipe
(`tests/test_vulkan.py`).

## Speed

Fit plus predict with four estimators of the released architecture on synthetic binary tables, desktop with an
Intel i7-13700KF (16 threads) and an AMD RX 7900 XT (Linux, Mesa RADV driver, PyTorch 2.13 with ROCm 7.2):

| Train rows x features, test rows | CPU, 16 threads | ROCm (PyTorch) | Vulkan | Vulkan vs CPU |
|---|---:|---:|---:|---:|
| 1,000 x 20, 500 | 0.45 s | 0.057 s | 0.044 s | 10.3x |
| 5,000 x 50, 2,000 | 4.84 s | 0.59 s | 0.50 s | 9.6x |
| 10,000 x 100, 5,000 | 19.8 s | 2.48 s | 2.13 s | 9.3x |
| 20,000 x 50, 5,000 | 29.2 s | 4.76 s | 4.86 s | 6.0x |
| 20,000 x 200, 10,000 | 70.4 s | 10.4 s | 8.68 s | 8.1x |

On this GPU Vulkan is as fast as or faster than PyTorch with ROCm. The first fit in a process also compiles
the kernels it needs.

## Limits

- **Buffer size.** One estimator's training cells (rows x features x 64 floats) must fit one GPU buffer
  (2 GiB on the tested RX 7900 XT driver, about 8.4 million cells, for example 84,000 rows x 100 features). Above that,
  `device="auto"` falls back to the CPU with a warning and `device="vulkan"` raises `MemoryError`. Lower
  `max_context` to stay on the GPU.
  Other context and scratch buffers can impose a smaller limit; 8.4 million cells is an upper bound.
- **Platforms.** Linux and Windows. macOS is not supported (wgpu would use Metal, which this backend does
  not target). Tested on an AMD RX 7900 XT (Linux RADV) and on Mesa llvmpipe; Windows validation is pending;
  Intel and NVIDIA Vulkan drivers use the same code path but have not been benchmarked.
- **Training** is not supported; Vulkan is for inference. With marked categorical columns, the experimental adapter
  (`Config.cat_adapter`, off in the released model) falls back to the CPU with `device="auto"`;
  an explicit `device="vulkan"` raises `MemoryError`.

## Alternatives we tried

We also tried two general compilers. IREE had no attention lowering for Vulkan, generated wrong code for
RDNA3 on part of the network and was slow with dynamic shapes. ONNX Runtime's WebGPU provider computed the
fused attention with errors of 2 to 17% on our shapes. The hand-written kernels match PyTorch up to rounding
and are as fast as PyTorch with ROCm on the same GPU.
