"""The inference scaling MLP uses 2D GEMMs without changing CPU or training."""
import math

import pytest
import torch

from lightpfn.model.layers import SoftmaxScaling


@pytest.mark.parametrize('training', [False, True])
def test_cpu_layout_and_output_unchanged(training):
    torch.manual_seed(37)
    layer = SoftmaxScaling(4, 16).train(training)
    q = torch.randn(2, 31, 4, 16).transpose(1, 2)
    shapes = []
    hook = layer.mod.register_forward_pre_hook(lambda m, inputs: shapes.append(inputs[0].shape))
    with torch.no_grad():
        out = layer(q, 80)
        raw = torch.full((1, 1), math.log(80), dtype=q.dtype)
        base = layer.effective_base(layer.base(raw).view(1, 4, 1, 16))
        expected = q * base * (1 + torch.tanh(layer.mod(q)))
    hook.remove()
    assert shapes[0] == q.shape
    assert torch.equal(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA/ROCm unavailable')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('contiguous', [False, True])
@pytest.mark.parametrize('training,grad', [(False, False), (False, True), (True, True)])
def test_gpu_modulation_dispatch(dtype, contiguous, training, grad):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip('BF16 unavailable')
    torch.manual_seed(37)
    layer = SoftmaxScaling(8, 32).cuda().train(training)
    with torch.no_grad():
        for p in layer.parameters():
            p.add_(torch.randn_like(p) * .02)
    q = torch.randn(4, 75, 8, 32, device='cuda', dtype=dtype).transpose(1, 2)
    if contiguous:
        q = q.contiguous()
    q.requires_grad_(grad)
    inputs = []
    hook = layer.mod.register_forward_pre_hook(lambda m, values: inputs.append(values[0]))
    with torch.set_grad_enabled(grad), torch.autocast('cuda', dtype=dtype):
        out = layer(q, 539)
    hook.remove()
    inference = not training and not grad
    assert inputs[0].ndim == (2 if inference else 4)
    if inference and not contiguous:
        assert inputs[0].data_ptr() == q.data_ptr()  # no copy for the actual attention layout
    # A direct four-dimensional readout is the reference arithmetic. Half/BF16
    # cuBLAS kernels may round differently, so compare against their precision.
    with torch.no_grad(), torch.autocast('cuda', dtype=dtype):
        raw = torch.full((1, 1), math.log(539), device='cuda', dtype=dtype)
        base = layer.effective_base(layer.base(raw).view(1, 8, 1, 32))
        expected = q * base * (1 + torch.tanh(layer.mod(q)))
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, expected, atol=.04 if dtype == torch.bfloat16 else .005,
                               rtol=.02 if dtype == torch.bfloat16 else .002)
    if grad:
        out.float().sum().backward()
        assert torch.isfinite(q.grad).all()
