"""The applied CCMM patch preserves the zero-weight loss/gradient path.

Executes only task_loss, never train.main or an optimizer.
"""

import torch
import torch.nn.functional as F

from lightpfn.model.ccmm import ccmm_forward, ccmm_task_loss
from lightpfn.model.lightpfn import Config, LightPFN
from lightpfn.train import task_loss

torch.set_num_threads(4)


def patched_task_loss():
    """The CCMM patch is applied in train.py; weight 0 must be the original path."""
    return lambda model, mb, device, slots, ccmm_weight=0.0: task_loss(model, mb, device, slots, ccmm_weight)


def batch():
    g = torch.Generator().manual_seed(7)
    x = torch.randn(2, 13, 5, generator=g)
    x[0, 2, 1] = float("nan")
    y = torch.stack([torch.arange(13) % 2, torch.arange(13) % 3])
    return dict(X=x, y=y, d=torch.tensor([3, 5]), n_classes=torch.tensor([2, 3]), n_train=8)


def test_review_patch_zero_weight_loss_and_all_gradients_are_exact():
    patched = patched_task_loss()
    torch.manual_seed(44)
    model = LightPFN(Config()).train()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(std=.05)
    losses, gradients = [], []
    mb = batch()
    for fn in (task_loss, patched):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(123)
        losses.append(fn(model, mb, "cpu", 16))
        losses[-1].sum().backward()
        gradients.append({n: p.grad.clone() for n, p in model.named_parameters()})
    assert torch.equal(*losses)
    assert gradients[0].keys() == gradients[1].keys()
    for name in gradients[0]:
        assert torch.equal(gradients[0][name], gradients[1][name]), name


def test_review_patch_weighted_per_task_loss_matches_ccmm():
    patched = patched_task_loss()
    model = LightPFN(Config(ccmm=True, icl_ff_reallocation=5)).train()
    mb = batch()
    torch.manual_seed(123)
    combined = patched(model, mb, "cpu", 16, ccmm_weight=.1)
    torch.manual_seed(123)
    logits, aux = ccmm_forward(model, mb, "cpu")
    expected = F.cross_entropy(logits.transpose(1, 2), mb["y"][:, 8:], reduction="none").mean(1) + .1 * aux
    assert torch.equal(combined, expected)
    assert combined.shape == (2,)
    combined.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
