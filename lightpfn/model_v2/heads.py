"""Full-bank retrieval and two separate 999-quantile regression candidates."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from lightpfn.model.layers import SoftmaxScaling, sdpa

from .scaling import BoundedHeadScaling


class Retrieval(nn.Module):
    def __init__(self, width, scale_cap=0.0, smoothing=0.0, scale_mode="legacy", qk_affine=True, head_bound=8.0):
        super().__init__()
        self.smoothing = smoothing
        self.heads, self.dim = width // 64, 64
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        # Learned gains multiply the attention logits outside the bounded scaling; at 30k steps of
        # v2_A3L_lr1e3 their per-channel product reached 7.9 (runs/v2_arch/instability_20261031/QK_GAINS.md).
        self.q_norm = nn.RMSNorm(64, eps=1e-6, elementwise_affine=qk_affine)
        self.k_norm = nn.RMSNorm(64, eps=1e-6, elementwise_affine=qk_affine)
        # head mode still builds (then drops) the legacy module, like icl_scale_mode="log": the
        # random stream and so every other initial tensor stay equal to a legacy/atan model of the same seed.
        self.scaling = SoftmaxScaling(self.heads, 64, base_cap=scale_cap,
                                      base_mode="legacy" if scale_mode == "head" else scale_mode)
        if scale_mode == "head":
            self.scaling = BoundedHeadScaling(self.heads, head_bound)
        # Opt-in z-loss (lightpfn.train --decoder-attn-zloss): while True, classification() stores
        # per-task mean over heads and query rows of logsumexp(attention logits)^2 in zloss_terms.
        # Plain attributes, never in the state dict; off, classification runs exactly as before.
        self.zloss = False
        self.zloss_terms = None

    def split(self, x):
        return x.unflatten(-1, (self.heads, self.dim)).transpose(1, 2)

    def keys(self, bank):
        return self.k_norm(self.split(self.k(bank)))

    def queries(self, rows, n):
        return self.scaling(self.q_norm(self.split(self.q(rows))), n)

    def classification(self, rows, keys, y, n_classes):
        q = self.queries(rows, keys.shape[2])
        v = F.one_hot(y, self.dim).to(q.dtype)[:, None].expand(-1, self.heads, -1, -1)
        p = sdpa(q, keys.to(q.dtype), v).float().mean(1)[..., :n_classes]
        if self.zloss:
            self.zloss_terms = attention_lse_square(q, keys.to(q.dtype), self.dim)
        diagnostic = getattr(self, "diagnostic", None)
        if diagnostic is not None:
            diagnostic.attention(self, q, keys.to(q.dtype))
            diagnostic.probabilities(p)
        if self.smoothing:
            present = v[:, 0, :, :n_classes].amax(1).float()
            p = p + self.smoothing * present[:, None]
            return (p / p.sum(-1, keepdim=True)).clamp_min(1e-8).log()
        return p.clamp_min(1e-8).log()

    def weights(self, rows, keys):
        # Caller chunks query rows; memory is O(chunk*N*H), never O(Q*999*N).
        q = self.queries(rows, keys.shape[2])
        logits = torch.matmul(q, keys.to(q.dtype).transpose(-1, -2)).float() / math.sqrt(self.dim)
        return logits.softmax(-1).mean(1)


ZLOSS_CHUNK_ELEMENTS = 1 << 24  # float32 logits per chunk: 64 MiB


def _lse_square_sum(q, k, dim):
    with torch.autocast(q.device.type, enabled=False):
        logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(dim)
        return logits.logsumexp(-1).square().sum((1, 2))


def attention_lse_square(q, k, dim):
    """Per-task mean over heads and query rows of logsumexp_keys(q k^T / sqrt(dim))^2, the logits
    sdpa normalizes in Retrieval.classification (q already scaled, no mask). Float32, in query-row
    chunks of at most ZLOSS_CHUNK_ELEMENTS logits; with autograd each chunk is recomputed in backward,
    so only q and k are saved and the extra memory stays one chunk."""
    b, h, n_q, n_k = q.shape[0], q.shape[1], q.shape[2], k.shape[2]
    step = max(1, ZLOSS_CHUNK_ELEMENTS // max(1, b * h * n_k))
    grad = torch.is_grad_enabled() and (q.requires_grad or k.requires_grad)
    total = None
    for i in range(0, n_q, step):
        part = q[:, :, i:i + step]
        s = (checkpoint(_lse_square_sum, part, k, dim, use_reentrant=False) if grad
             else _lse_square_sum(part, k, dim))
        total = s if total is None else total + s
    return total / (h * n_q)


def retrieval_moments(weights, z):
    mean = weights @ z[..., None]
    second = weights @ z.square()[..., None]
    scale = (second - mean.square()).clamp_min(1e-6).sqrt()
    return mean, scale


class QuantileHead(nn.Module):
    """999 learned outputs, conditioned on full-bank retrieved target moments."""

    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width + 2, 128), nn.GELU(), nn.Linear(128, 999))

    def forward(self, rows, weights, z):
        mean, scale = retrieval_moments(weights, z)
        delta = self.net(torch.cat([rows, mean.to(rows.dtype), scale.to(rows.dtype)], -1)).float()
        return (mean + scale * delta).sort(-1).values


class TransportHead(nn.Module):
    """R2-Transport: inverse full-bank weighted CDF plus learned residual transport.

    All 999 levels use the empirical bank directly. Linear interpolation is
    between empirical CDF knots, allowing gradients to retrieval weights; there
    is no 65-output parametric surrogate. A 999-output residual shape/shift can
    extrapolate. Final sorting enforces monotonicity, not calibration.
    """

    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width + 2, 96), nn.SiLU(), nn.Linear(96, 1000))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("taus", torch.arange(1, 1000, dtype=torch.float32) / 1000)

    def empirical(self, weights, z, sorted_z=None):
        # sorted_z: (values, indices) of z.sort(-1), computed once per context.
        sorted_z, order = z.sort(-1) if sorted_z is None else sorted_z
        sorted_w = weights.gather(2, order[:, None].expand(-1, weights.shape[1], -1))
        cdf = sorted_w.cumsum(-1)
        cdf = cdf / cdf[..., -1:].clamp_min(1e-12)
        tau = self.taus.expand(*weights.shape[:2], -1).contiguous()
        hi = torch.searchsorted(cdf.detach().contiguous(), tau).clamp_max(z.shape[1] - 1)
        lo = (hi - 1).clamp_min(0)
        zbank = sorted_z[:, None].expand(-1, weights.shape[1], -1)
        zl, zh = zbank.gather(2, lo), zbank.gather(2, hi)
        cl, ch = cdf.gather(2, lo), cdf.gather(2, hi)
        frac = ((tau - cl) / (ch - cl).clamp_min(1e-12)).clamp(0, 1)
        return zl + frac * (zh - zl)

    def forward(self, rows, weights, z, sorted_z=None):
        empirical = self.empirical(weights, z, sorted_z)
        mean, scale = retrieval_moments(weights, z)
        correction = self.net(torch.cat([rows, mean.to(rows.dtype), scale.to(rows.dtype)], -1)).float()
        return (empirical + correction[..., :1] + scale * correction[..., 1:]).sort(-1).values


def pinball_loss(quantiles, targets, reduction="mean"):
    taus = torch.arange(1, 1000, device=quantiles.device, dtype=torch.float32) / 1000
    error = targets[..., None] - quantiles.float()
    loss = torch.maximum(taus * error, (taus - 1) * error)
    if reduction == "mean":
        return loss.mean()
    if reduction == "task":
        return loss.mean((-1, -2))
    raise ValueError("unknown pinball reduction")
