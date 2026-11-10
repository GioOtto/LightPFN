"""R2 SOL recurrence, corrected to always use seeds and a continuous full-bank coda."""

from dataclasses import dataclass

import torch
from torch import nn

from lightpfn.model.layers import Attention, MLP, sdpa
from .scaling import LogScaling


class QKAttention(Attention):
    def __init__(self, width, heads, kv_heads=None):
        super().__init__(width, heads, scaling=True)
        self.kv_heads = heads if kv_heads is None else kv_heads
        if self.kv_heads != heads:
            # The coda always uses MQA for both context and queries. Do not
            # retain permanently unused KV heads just to fill the budget.
            self.kv = nn.Linear(width, 2 * self.kv_heads * self.head_dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        # Exercise seeds, recurrence and retrieval at initialization too.
        nn.init.normal_(self.out.weight, std=.005)

    def keys_values(self, x):
        k, v = self.kv(x).chunk(2, -1)
        k, v = (t.unflatten(-1, (self.kv_heads, self.head_dim)).transpose(1, 2) for t in (k, v))
        return self.k_norm(k), v

    def attend(self, x, k, v, kv_heads=None):
        q = self.scaling(self.q_norm(self.split(self.q(x))), k.shape[2])
        if kv_heads is not None:
            # expand is a view; do not materialize H copies of full-context KV.
            if kv_heads != 1:
                raise ValueError("S0 query MQA uses one KV head")
            k = k[:, :1].expand(-1, self.n_heads, -1, -1)
            v = v[:, :1].expand(-1, self.n_heads, -1, -1)
        diagnostic = getattr(self, "diagnostic", None)
        if diagnostic is not None:
            diagnostic.attention(self, q, k)
        out = sdpa(q, k, v)
        return self.out(out.transpose(1, 2).flatten(-2))


class SandwichBlock(nn.Module):
    def __init__(self, width, heads, hidden, kv_heads=None):
        super().__init__()
        self.norm_q = nn.RMSNorm(width, eps=1e-6)
        self.norm_kv = nn.RMSNorm(width, eps=1e-6)
        self.norm_ff = nn.RMSNorm(width, eps=1e-6)
        self.post_attn = nn.RMSNorm(width, eps=1e-6)
        self.post_ff = nn.RMSNorm(width, eps=1e-6)
        self.attn = QKAttention(width, heads, kv_heads)
        self.mlp = MLP(width, hidden)
        nn.init.normal_(self.mlp.fc2.weight, std=.005)

    def keys_values(self, x):
        return self.attn.keys_values(self.norm_kv(x))

    def forward(self, x, k, v, kv_heads=None):
        x = x + self.post_attn(self.attn.attend(self.norm_q(x), k, v, kv_heads))
        return x + self.post_ff(self.mlp(self.norm_ff(x)))


@dataclass
class WorkspaceCache:
    prelude: tuple
    recurrent: list
    coda: tuple
    bank: torch.Tensor
    workspace: torch.Tensor
    anchor: torch.Tensor


class WorkspaceICL(nn.Module):
    def __init__(self, cfg, hidden):
        super().__init__()
        self.cfg = cfg
        d = cfg.width
        self.inducing = nn.Parameter(torch.randn(cfg.slots, d) * .02) if cfg.workspace else None
        self.thinking = nn.Parameter(torch.randn(16, d) * .02)
        self.prelude = SandwichBlock(d, d // 32, hidden)
        self.core = nn.ModuleList(SandwichBlock(d, d // 32, hidden) for _ in range(cfg.unique_blocks - 2))
        self.coda = SandwichBlock(d, d // 32, hidden, kv_heads=1)
        for block in (self.prelude, *self.core, self.coda):
            if cfg.icl_scale_mode == "log":
                # Replace only after legacy construction: with fixed FF width,
                # all common initial tensors retain the same RNG draws.
                block.attn.scaling = LogScaling(d // 32, cfg.icl_log_reference)
            else:
                block.attn.scaling.base_cap = cfg.icl_scale_cap
        self.norm = nn.RMSNorm(d, eps=1e-6)

    @staticmethod
    def small_kv(kv):
        return tuple(t[:, :1].contiguous() for t in kv)

    def encode(self, bank):
        raw = self.prelude.keys_values(bank)
        # There is no n<=S branch. Even the shortest task trains every seed.
        seed = self.inducing[None].expand(bank.shape[0], -1, -1) if self.cfg.workspace else bank
        seed = torch.cat([seed, self.thinking[None].expand(bank.shape[0], -1, -1)], 1)
        anchor = self.prelude(seed, *raw)
        w, caches = anchor, []
        for _ in range(self.cfg.repeats):
            for block in self.core:
                x = w + anchor
                kv = block.keys_values(x)
                w = block(x, *kv)
                caches.append(self.small_kv(kv))
        last = self.small_kv(self.coda.keys_values(w))
        # Same first KV head for ALL context rows and for prediction queries.
        # Always update the original bank, never w when n happens to equal S.
        h = self.norm(self.coda(bank, *last, kv_heads=1))
        return WorkspaceCache(self.small_kv(raw), caches, last, h, w, anchor)

    def query(self, rows, cache):
        x0 = self.prelude(rows, *cache.prelude, kv_heads=1)
        x = x0
        for i, kv in enumerate(cache.recurrent):
            x = self.core[i % len(self.core)](x + x0, *kv, kv_heads=1)
        return self.norm(self.coda(x, *cache.coda, kv_heads=1))
