"""Building blocks shared by the three stages of the model.

Tensors are (batch, sequence, dim) unless a name says otherwise. Attention runs through
F.scaled_dot_product_attention, so the same code uses flash kernels on GPU and fused CPU kernels.
"""

import math
import os
from collections import OrderedDict
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


def fuse_enabled(name):
    """Private A/B switch, read at runtime; comma-separated lever names."""
    return name not in os.environ.get("LIGHTPFN_FUSE_DISABLE", "").split(",")


def _stream_key(device):
    # Tables created on one stream must not be read before completion on another.
    return torch.cuda.current_stream(device).cuda_stream if device.type == "cuda" else None


def checkpoint_training(module, function, *args, **kwargs):
    """Recompute activations only for opted-in training modules with autograd enabled.

    Non-reentrant checkpointing supports parameter gradients even when inputs do not require
    them, and preserves the forward autocast context and RNG state during recomputation.
    The flag is a plain attribute, never a parameter or buffer.
    """
    if getattr(module, "grad_checkpoint", False) and module.training and torch.is_grad_enabled():
        return checkpoint(function, *args, use_reentrant=False, **kwargs)
    return function(*args, **kwargs)


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

    def __init__(self, n_heads, head_dim, hidden=64, base_cap=0.0, base_mode="legacy"):
        super().__init__()
        if not math.isfinite(base_cap) or base_cap < 0:
            raise ValueError("base_cap must be finite and nonnegative")
        self.base_cap = base_cap
        if base_mode not in ("legacy", "atan") or (base_mode != "legacy" and base_cap):
            raise ValueError("base_mode must be legacy or atan; smooth mode excludes a hard cap")
        self.base_mode = base_mode
        self.n_heads, self.head_dim = n_heads, head_dim
        self.base = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, n_heads * head_dim))
        self.mod = nn.Sequential(nn.Linear(head_dim, hidden), nn.GELU(), nn.Linear(hidden, head_dim))
        nn.init.zeros_(self.base[2].weight)
        nn.init.ones_(self.base[2].bias)
        nn.init.zeros_(self.mod[2].weight)
        nn.init.zeros_(self.mod[2].bias)

    def effective_base(self, raw):
        """atan: (-4, 4), f(1)=f'(1)=1, no trainable/buffer state.

        With b=tan(pi/8), a=(pi/8)*(1+b*b), f(x)=8/pi*atan(a*(x-1)+b).
        f'(x)=8*a/pi/(1+(a*(x-1)+b)**2)>0 for every finite real x.
        Evaluate reduced-precision inputs in fp32; finite arithmetic still rounds
        extreme outputs to the endpoints and can underflow extreme gradients.
        """
        if self.base_mode == "atan":
            x = raw.float() if raw.dtype in (torch.bfloat16, torch.float16) else raw
            b = math.tan(math.pi / 8)
            a = math.pi / 8 * (1 + b * b)
            return ((8 / math.pi) * torch.atan(a * (x - 1) + b)).to(raw.dtype)
        return raw.clamp(-self.base_cap, self.base_cap) if self.base_cap else raw

    def _clear_base_cache(self):
        self.__dict__.pop("_base_cache", None)

    def _apply(self, fn, recurse=True):
        self._clear_base_cache()
        return super()._apply(fn, recurse)

    def train(self, mode=True):
        self._clear_base_cache()
        return super().train(mode)

    def _load_from_state_dict(self, *args, **kwargs):
        self._clear_base_cache()
        return super()._load_from_state_dict(*args, **kwargs)

    def __getstate__(self):
        state = super().__getstate__().copy()
        state.pop("_base_cache", None)
        return state

    def forward(self, q, n):
        """q: (B, H, L, D) queries, n: number of keys."""
        cached = not self.training and not torch.is_grad_enabled() and fuse_enabled("scale")
        key = None
        if cached:
            amp = torch.is_autocast_enabled(q.device.type)
            key = (max(n, 2), q.dtype, q.device, amp, torch.get_autocast_dtype(q.device.type),
                   self.base_mode, self.base_cap, _stream_key(q.device))
        cache = self.__dict__.setdefault("_base_cache", OrderedDict()) if cached else None
        base = cache.get(key) if cached else None
        if base is None:
            logn = torch.full((1, 1), math.log(max(n, 2)), device=q.device, dtype=q.dtype)
            base = self.effective_base(self.base(logn).view(1, self.n_heads, 1, self.head_dim))
            if cached:
                cache[key] = base.detach()
                if len(cache) > 32:
                    cache.popitem(last=False)
        if cached:
            cache.move_to_end(key)
        if (not self.training and not torch.is_grad_enabled() and q.device.type == "cuda"
                and q.dtype in (torch.float16, torch.bfloat16)):
            # Attention splits a contiguous (B, L, H*D) projection into (B, H, L, D).
            # Restore its physical row order for two ordinary 2D GEMMs. This avoids
            # the fp16 strided-batched cuBLAS path that failed on torch 2.15/cu132,
            # without copying the usual interleaved-head input or changing weights.
            b, h, length, d = q.shape
            modulation = self.mod(q.transpose(1, 2).reshape(-1, d)).view(b, length, h, d).transpose(1, 2)
        else:
            modulation = self.mod(q)
        return q * base * (1 + torch.tanh(modulation))


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


@lru_cache(maxsize=32)
def _rope_tables(length, d, base, dtype, device, interleaved, stream):
    # Do not retain graphs or inference tensors: an inference call must not poison autograd.
    with torch.inference_mode(False), torch.no_grad():
        pos = torch.arange(length, device=device)
        return _make_rope_tables(pos, d, base, dtype, device, interleaved)


def _make_rope_tables(pos, d, base, dtype, device, interleaved):
    inv_freq = base ** (-torch.arange(0, d, 2, device=device, dtype=torch.float32) / d)
    ang = pos.to(torch.float32)[:, None] * inv_freq[None]
    if interleaved and dtype in (torch.float32, torch.float64):
        return torch.complex(ang.cos(), ang.sin()).to(
            torch.complex64 if dtype == torch.float32 else torch.complex128)
    return ang.cos().to(dtype), ang.sin().to(dtype)


def rope(x, pos=None, base=10000.0, interleaved=False):
    """Rotary position embedding (rotate-half form) on the last dim of x: (..., L, D) with
    positions pos: (L,). interleaved=True rotates the pairs (2i, 2i + 1) instead of (i, i + D/2),
    with one complex multiply: the same function for weights passed through interleave_rotary().
    pos=None explicitly means arange(L), the only cached case. Arbitrary positions and autograd
    keep the original calculation, including gradients through caller-supplied positions."""
    d = x.shape[-1]
    if pos is None and not torch.is_grad_enabled() and fuse_enabled("rope"):
        tables = _rope_tables(x.shape[-2], d, base, x.dtype, x.device, interleaved, _stream_key(x.device))
    else:
        if pos is None:
            pos = torch.arange(x.shape[-2], device=x.device)
        tables = _make_rope_tables(pos, d, base, x.dtype, x.device, interleaved)
    if interleaved and x.dtype in (torch.float32, torch.float64):
        return torch.view_as_real(torch.view_as_complex(x.unflatten(-1, (d // 2, 2))) * tables).flatten(-2)
    cos, sin = tables
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

    def __init__(self, dim, n_heads, scaling=False, *, qk_norm=False):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.out.weight)
        self.scaling = SoftmaxScaling(n_heads, self.head_dim) if scaling else None
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=False)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=False)

    def split(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D)

    def keys_values(self, x):
        k, v = self.kv(x).chunk(2, dim=-1)
        k = self.split(k)
        if hasattr(self, "k_norm"):
            k = self.k_norm(k)
        return k, self.split(v)

    def attend(self, x, k, v, mask=None, q_rope=None, kv_heads=None):
        """x: (B, Lq, dim) queries; k, v: (B, Hkv, Lk, D). kv_heads=h: all query heads use the
        first h key/value heads."""
        q = self.split(self.q(x))
        if hasattr(self, "q_norm"):
            q = self.q_norm(q)
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
        return checkpoint_training(self, self._keys_values, kv_input)

    def _keys_values(self, kv_input):
        return self.attn.keys_values(self.norm_kv(kv_input))

    def forward(self, x, k, v, mask=None, q_rope=None, kv_heads=None):
        return checkpoint_training(self, self._forward, x, k, v, mask, q_rope, kv_heads)

    def _forward(self, x, k, v, mask=None, q_rope=None, kv_heads=None):
        x = x + self.attn.attend(self.norm_q(x), k, v, mask, q_rope, kv_heads)
        return x + self.mlp(self.norm_ff(x))

    def forward_split(self, x, split, k, v, kv_heads):
        """Inference: forward(x[:, :split], k, v) and forward(x[:, split:], k, v, kv_heads=kv_heads) in one
        call. Both parts share the norms, projections and MLP; only the two attention calls differ."""
        a = self.attn
        q = a.split(a.q(self.norm_q(x)))
        if hasattr(a, "q_norm"):
            q = a.q_norm(q)
        if a.scaling is not None:
            q = a.scaling(q, k.shape[2])
        r = a.n_heads // kv_heads
        o = torch.cat([sdpa(q[:, :, :split], k, v),
                       sdpa(q[:, :, split:], k[:, :kv_heads].repeat_interleave(r, dim=1),
                            v[:, :kv_heads].repeat_interleave(r, dim=1))], 2)
        B, H, L, D = o.shape
        x = x + a.out(o.transpose(1, 2).reshape(B, L, H * D))
        return x + self.mlp(self.norm_ff(x))


# Four members, E=64, C=4: row norms at 700x20 have 4*700*24*64=4,300,800
# elements; 1500x50 has 20,736,000. Provisional launch-bound cutoff, tune on CUDA/HIP.
RMS_FUSED_MAX_ELEMS = 8_000_000
RMS_FUSED_MAX_ELEMS_HIP = 1_000_000  # RX 7900 XT probe already regresses at 700x20.


def _fused_rms_norm(x, dim, eps):
    # Native RMSNorm accumulates reduced precision in FP32. Autocast registers this op as
    # FP32 on CUDA; disable that promotion to preserve the old path's output dtype.
    with torch.autocast(x.device.type, enabled=False):
        return F.rms_norm(x, (dim,), None, eps)


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
        limit = RMS_FUSED_MAX_ELEMS_HIP if torch.version.hip else RMS_FUSED_MAX_ELEMS
        # HIP fp32 differs by a few ulps. Preserve the explicit FP32 contract; use the fused
        # reduction only under inference autocast (residual tensors can still be FP32).
        if (x.is_cuda and not torch.is_grad_enabled() and torch.is_autocast_enabled("cuda") and
                x.numel() <= limit and fuse_enabled("rms")):
            return _fused_rms_norm(x, self.dim, eps)
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
