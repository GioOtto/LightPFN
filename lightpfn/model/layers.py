"""Building blocks shared by the three stages of the model.

Tensors are (batch, sequence, dim) unless a name says otherwise. Attention runs through
F.scaled_dot_product_attention, so the same code uses flash kernels on GPU and fused CPU kernels.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    """Two-layer GELU feed-forward block; the output projection starts at zero so every residual
    block starts as the identity (as in TabPFN-3)."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)
        nn.init.zeros_(self.fc2.weight)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class SoftmaxScaling(nn.Module):
    """Scales attention queries as a function of the number of keys n (TabPFN-3 SoftmaxScalingMLP):
    q * base(log n) * (1 + tanh(mod(q))). Keeps attention sharp when the context grows far beyond
    the lengths seen in training. base starts at 1 so the layer starts as a no-op."""

    def __init__(self, n_heads, head_dim, hidden=64):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, head_dim
        self.base = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, n_heads * head_dim))
        self.mod = nn.Sequential(nn.Linear(head_dim, hidden), nn.GELU(), nn.Linear(hidden, head_dim))
        nn.init.zeros_(self.base[2].weight)
        nn.init.ones_(self.base[2].bias)
        nn.init.zeros_(self.mod[2].weight)
        nn.init.zeros_(self.mod[2].bias)

    def forward(self, q, n):
        """q: (B, H, L, D) queries, n: number of keys."""
        logn = torch.full((1, 1), math.log(max(n, 2)), device=q.device, dtype=q.dtype)
        base = self.base(logn).view(1, self.n_heads, 1, self.head_dim)
        return q * base * (1 + torch.tanh(self.mod(q)))


# CUDA attention kernels put the batch on a grid dimension of at most 65535 blocks: a larger batch fails
# with "invalid argument" (the row stages of a 2-task group of 50k-row tables have 100k rows in the batch).
SDPA_MAX_BATCH = 32768


def sdpa(q, k, v, mask=None):
    """F.scaled_dot_product_attention in chunks of at most SDPA_MAX_BATCH along the batch dim (a size-1
    batch dim of k, v or mask broadcasts)."""
    B = q.shape[0]
    if B <= SDPA_MAX_BATCH:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    out = []
    for i in range(0, B, SDPA_MAX_BATCH):
        part = [t if t is None or t.shape[0] == 1 else t[i:i + SDPA_MAX_BATCH] for t in (k, v, mask)]
        out.append(F.scaled_dot_product_attention(q[i:i + SDPA_MAX_BATCH], *part[:2], attn_mask=part[2]))
    return torch.cat(out)


def rope(x, pos, base=10000.0, interleaved=False):
    """Rotary position embedding (rotate-half form) on the last dim of x: (..., L, D) with
    positions pos: (L,). interleaved=True rotates the pairs (2i, 2i + 1) instead of (i, i + D/2),
    with one complex multiply: the same function for weights passed through interleave_rotary()."""
    d = x.shape[-1]
    inv_freq = base ** (-torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    ang = pos.to(torch.float32)[:, None] * inv_freq[None]  # (L, D/2)
    if interleaved and x.dtype in (torch.float32, torch.float64):
        rot = torch.complex(ang.cos(), ang.sin()).to(torch.complex64 if x.dtype == torch.float32 else torch.complex128)
        return torch.view_as_real(torch.view_as_complex(x.unflatten(-1, (d // 2, 2))) * rot).flatten(-2)
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    if interleaved:  # reduced precision (autocast): the same arithmetic as below on adjacent pairs
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    """Multi-head attention with separate query and key/value inputs.

    kv_heads_query < n_heads lets a set of queries use only the first key/value heads (multi-query
    attention for test rows, as in TabPFN-3.5): keys/values of the other heads never need to be
    cached for prediction."""

    def __init__(self, dim, n_heads, scaling=False):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.out.weight)
        self.scaling = SoftmaxScaling(n_heads, self.head_dim) if scaling else None

    def split(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)

    def keys_values(self, x):
        k, v = self.kv(x).chunk(2, dim=-1)
        return self.split(k), self.split(v)

    def attend(self, x, k, v, mask=None, q_rope=None, kv_heads=None):
        """x: (B, Lq, dim) queries; k, v: (B, Hkv, Lk, D). kv_heads=h: all query heads use the
        first h key/value heads."""
        q = self.split(self.q(x))
        if q_rope is not None:
            q = q_rope(q)
        if self.scaling is not None:
            q = self.scaling(q, k.shape[2])
        if kv_heads is not None:
            k = k[:, :kv_heads].repeat_interleave(self.n_heads // kv_heads, dim=1)
            v = v[:, :kv_heads].repeat_interleave(self.n_heads // kv_heads, dim=1)
        o = sdpa(q, k, v, mask)
        B, H, L, D = o.shape
        return self.out(o.transpose(1, 2).reshape(B, L, H * D))


class Block(nn.Module):
    """Pre-norm residual block: attention of x on a key/value sequence, then an MLP."""

    def __init__(self, dim, n_heads, ff_factor=2, scaling=False, ff_hidden=None):
        super().__init__()
        self.norm_q = nn.RMSNorm(dim)
        self.norm_kv = nn.RMSNorm(dim)
        self.norm_ff = nn.RMSNorm(dim)
        self.attn = Attention(dim, n_heads, scaling)
        self.mlp = MLP(dim, dim * ff_factor if ff_hidden is None else ff_hidden)

    def keys_values(self, kv_input):
        return self.attn.keys_values(self.norm_kv(kv_input))

    def forward(self, x, k, v, mask=None, q_rope=None, kv_heads=None):
        x = x + self.attn.attend(self.norm_q(x), k, v, mask, q_rope, kv_heads)
        return x + self.mlp(self.norm_ff(x))


class FoldedRMSNorm(nn.Module):
    """RMSNorm whose weight was folded into the linear layer that reads it (inference only): one
    reduction and one multiply instead of the unfused CPU kernels. Same eps as nn.RMSNorm (eps=None:
    that of the accumulation dtype, float32 for reduced precision), reduction in at least float32."""

    def __init__(self, dim, eps=None):
        super().__init__()
        self.dim, self.eps = dim, eps

    def forward(self, x):
        acc = torch.promote_types(x.dtype, torch.float32)
        eps = torch.finfo(acc).eps if self.eps is None else self.eps
        ms = torch.linalg.vector_norm(x, dim=-1, keepdim=True, dtype=acc).square_().div_(self.dim)
        return x * ms.add_(eps).rsqrt_().to(x.dtype)


@torch.no_grad()
def fold_norm(norm, *linears):
    """Moves the weight of an RMSNorm into the linear layers fed by it; returns the weightless norm."""
    for lin in linears:
        lin.weight.mul_(norm.weight)
    return FoldedRMSNorm(norm.weight.numel(), norm.eps)


@torch.no_grad()
def interleave_rotary(block):
    """Reorders the query and key dims of each head of a block from (i, i + D/2) pairs to adjacent
    (2i, 2i + 1) pairs: q.k is unchanged (same permutation on both), and rope(..., interleaved=True)
    rotates the same pairs."""
    a = block.attn
    H, D = a.n_heads, a.head_dim
    pairs = torch.stack([torch.arange(D // 2), torch.arange(D // 2) + D // 2], -1).flatten()
    rows = (torch.arange(H)[:, None] * D + pairs).flatten().to(a.q.weight.device)
    a.q.weight.copy_(a.q.weight[rows])
    a.kv.weight[: H * D].copy_(a.kv.weight[rows])


class OrthogonalEmbedding(nn.Embedding):
    """Label embedding with orthonormal initial rows (TabPFN-3 TrainableOrthogonalEmbedding)."""

    def __init__(self, n, dim):
        super().__init__(n, dim)
        with torch.no_grad():
            q, _ = torch.linalg.qr(torch.randn(dim, min(n, dim)))
            self.weight[: q.shape[1]] = q.T
