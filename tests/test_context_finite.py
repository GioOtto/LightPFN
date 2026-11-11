"""Activation checks reject every non-finite field and preserve metadata sentinels."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from lightpfn import LightPFN, LightPFNClassifier
from lightpfn.sklearn import _context_isfinite, _floating_tensors
from test_release import small_model


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@torch.inference_mode()
def test_every_floating_context_field(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA/ROCm unavailable')
    model = LightPFN(replace(small_model().cfg, cat_adapter=True)).eval()
    x = np.random.default_rng(4).normal(size=(48, 4)).astype(np.float32)
    x[:, 0] = np.arange(48)%3
    c = LightPFNClassifier(model=model, device=device, n_estimators=1).fit(x, np.arange(48)%3,
        cat=np.array([True, False, False, False]))
    ctx = c.members_[0][1]
    assert _context_isfinite(ctx)
    for path, tensor in _floating_tensors(ctx):
        if not tensor.numel():
            continue
        index = (0,)*tensor.ndim
        old = tensor[index].clone()
        for value in (float('nan'), float('inf'), -float('inf')):
            tensor[index] = value
            allowed = path == ('extra', 'cat_table', 'keys') and value == float('inf')
            assert _context_isfinite(ctx) == allowed, (path, value)
        tensor[index] = old
    assert _context_isfinite(ctx)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA/ROCm unavailable')
@torch.inference_mode()
def test_double_target_units_and_empty_activations():
    # A finite 1e120 is a valid target-unit statistic, not float32 overflow.
    assert _context_isfinite({'scale': torch.tensor(1e120, dtype=torch.float64, device='cuda'),
                              'empty': torch.empty(0, device='cuda'),
                              'half': torch.ones(31, dtype=torch.float16, device='cuda'),
                              'single': torch.ones(23, device='cuda')})
    assert not _context_isfinite({'scale': torch.tensor(float('nan'), dtype=torch.float64, device='cuda')})
