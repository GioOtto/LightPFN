"""Small, opt-in length scaling for controlled v2 experiments."""

import math

import torch
from torch import nn


class LogScaling(nn.Module):
    """One unconstrained coefficient per head, multiplying log(max(n, 2)).

    Unlike NVIDIA Kumo's random coefficients, deterministic initialization makes
    the initial scale one at reference_length. This avoids changing other initial
    tensors or introducing random head temperatures in a paired experiment.
    This scale is not bounded and does not guarantee long-run stability.
    """

    def __init__(self, n_heads, reference_length=1024):
        super().__init__()
        if n_heads < 1 or not isinstance(reference_length, int) or reference_length < 2:
            raise ValueError("LogScaling requires heads > 0 and integer reference_length >= 2")
        self.n_heads = n_heads
        self.reference_length = reference_length
        self.head_scale = nn.Parameter(torch.full((n_heads,), 1 / math.log(reference_length)))

    def forward(self, q, n):
        # Keep the same (B, H, L, D) layout and dtype contract as SoftmaxScaling.
        scale = self.head_scale.to(q.dtype).view(1, self.n_heads, 1, 1)
        return q * (scale * math.log(max(n, 2)))


class BoundedHeadScaling(nn.Module):
    """Decoder scale with one positive scalar per head, bounded, no per-coordinate or query term.

    z_h(n) = bias_h + slope_h * (log(max(n, 2)) - log(reference_length)),
    s_h(n) = bound / (1 + (bound - 1) * exp(-z_h)) = bound * sigmoid(z_h - log(bound - 1)).
    Like Kumo's head scale it cannot reweight coordinates of a head against each other: it changes
    sharpness, never the geometry q'diag(.)k of the legacy/atan SoftmaxScaling. Zero initialization
    gives s=1 exactly for every n, the same initial function as legacy/atan. With non-affine q/k
    RMS norms (head dim 64) |logit| <= bound * 8 in exact arithmetic for any weights; rounding can
    exceed it slightly (fp32 ~1e-6, bf16 inputs ~0.3%, 64.2 for bound 8). The bound can still bind; it caps
    the temperature, not concentration or the rest of the network.
    """

    base_cap = 0.0
    base_mode = "head"

    def __init__(self, n_heads, bound=8.0, reference_length=1024):
        super().__init__()
        if n_heads < 1 or not 1 < bound <= 64:
            raise ValueError("BoundedHeadScaling requires heads > 0 and a bound in (1, 64]")
        if not isinstance(reference_length, int) or reference_length < 2:
            raise ValueError("reference_length must be an integer >= 2")
        self.n_heads, self.bound, self.reference_length = n_heads, float(bound), reference_length
        self.bias = nn.Parameter(torch.zeros(n_heads))
        self.slope = nn.Parameter(torch.zeros(n_heads))

    def scale(self, n):
        """(H,) float32 scale for n keys, in (0, bound). Two branches keep exp() <= 1, so neither
        the value nor the gradient overflows; z=0 takes the first and returns exactly 1."""
        x = math.log(max(n, 2)) - math.log(self.reference_length)
        z = self.bias.float() + self.slope.float() * x
        c = self.bound - 1
        hi = self.bound / (1 + c * torch.exp(-z.clamp_min(0)))
        e = torch.exp(z.clamp_max(0))
        lo = self.bound * e / (e + c)
        return torch.where(z >= 0, hi, lo)

    def forward(self, q, n):
        return q * self.scale(n).to(q.dtype).view(1, self.n_heads, 1, 1)
