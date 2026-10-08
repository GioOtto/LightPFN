"""LightPFN classifier: cell embedding -> column stage -> row stage -> in-context learning -> decoder.

The technical report describes the architecture in full. In short:
  cells    per-column statistics of the training rows turn each value into a z-score, an ECDF rank
           and a NaN flag; cells are grouped with circular feature shifts and embedded with learnable
           Fourier features (TabPFN-3.5), which avoids the low-rank collapse of a scalar linear
           embedding (LimiX-2M)
  column   per-column ISAB whose inducing points attend to the training cells only, with the label
           embedding added to the training cells (target-aware, TabICLv2)
  row      per-row transformer over the features with CLS tokens and RoPE; the CLS outputs are
           concatenated into the row representation (TabICLv2)
  ICL      transformer over rows: training rows (plus learned thinking rows) attend to each other,
           test rows attend to the training rows only, with multi-query attention (TabPFN-3.5) and
           learned log-n softmax scaling (TabPFN-3)
  decoder  attention of test rows over the one-hot training labels (TabPFN-3 retrieval decoder)

`forward` runs `encode` (everything that depends on the training rows, returned as a Context) and
then `predict_logits` (test rows only), so the cached inference path is the training path.
"""

import copy
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from lightpfn.model.catstats import N_COUNT, context_stats, query_stats
from lightpfn.model.layers import Block, OrthogonalEmbedding, SoftmaxScaling, fold_norm, interleave_rotary, rope


@dataclass
class Config:
    """Opt-in architecture ablations; default parameters/state names are r2-exact.

    Budget recipes at the default widths (trainable parameters, buffers excluded):
      B1 cell_embed='rbf', icl_ff_reallocation=4: 4,777,024.
      B2 ccmm=True, icl_ff_reallocation=5: 4,776,784.
      B3 row_mode='summary': 4,777,072 (no extra parameters).
      B4 row_refine=True, row_mode='summary', icl_drop_blocks=1: 4,603,088.
    One last-MLP hidden unit costs 2*icl_dim parameters (512 at default width).
    B4 uses R=3 independent rounds plus final broadcast: +372,032 parameters,
    funded by removing one 546,016-parameter ICL block (8 -> 7). Other depths/
    widths require an explicit budget choice; there is no automatic reallocation.
    In particular, row_refine does not implicitly change row_mode or ICL depth.
    """

    col_dim: int = 64
    col_blocks: int = 2
    col_heads: int = 4
    n_inducing: int = 64
    row_blocks: int = 3
    row_heads: int = 4
    n_cls: int = 4  # ICL width = n_cls * col_dim
    icl_blocks: int = 8
    icl_heads: int = 8
    icl_kv_heads_test: int = 1
    n_thinking: int = 16
    ff_factor: int = 2
    n_freq: int = 16
    n_ecdf_freq: int = 4
    group_offsets: tuple = (0, 1, 3)
    label_slots: int = 16  # max classes; training maps classes to random slots so all are trained
    decoder_heads: int = 4
    rope_base: float = 10000.0
    cell_embed: str = "fourier"  # B1: "rbf", 64 fixed uniform kernels, sigma=1
    ccmm: bool = False  # B2: training-only mask token and rank-bin readout
    ccmm_mask_fraction: float = 0.15  # observed test cells, per task; structured masks round up
    row_mode: str = "self_attention"  # B3: "summary"
    row_refine: bool = False  # B4: refinement -> independent column stage -> compression
    row_refine_rounds: int = 3
    icl_drop_blocks: int = 0  # explicit budget reallocation, e.g. 1 for B4
    icl_ff_reallocation: int = 0  # hidden units removed from the LAST retained ICL MLP
    # Categorical adapter (Kumo Tabular's separate cell embedding for categorical columns + CatBoost-style
    # ordered target statistics, lightpfn/model/catstats.py): +6,384 parameters at the default widths. It
    # starts as the identity (copies of the numerical Fourier weights, zero statistics projections) and
    # only acts on columns marked categorical; encode() without a cat mask is the plain model.
    cat_adapter: bool = False
    cat_smooth: float = 1.0  # prior strength of the target statistics, in rows

    def __post_init__(self):
        if self.cell_embed not in ("fourier", "rbf"):
            raise ValueError("cell_embed must be fourier or rbf")
        if self.cat_adapter and self.cell_embed != "fourier":
            raise ValueError("cat_adapter needs the fourier cell embedding")
        if not self.cat_smooth > 0:
            raise ValueError("cat_smooth must be positive")
        if self.row_mode not in ("self_attention", "summary"):
            raise ValueError("row_mode must be self_attention or summary")
        if not 0 <= self.ccmm_mask_fraction <= 1:
            raise ValueError("ccmm_mask_fraction must be in [0, 1]")
        if self.row_refine_rounds < 1:
            raise ValueError("row_refine_rounds must be positive")
        if not 0 <= self.icl_drop_blocks < self.icl_blocks:
            raise ValueError("icl_drop_blocks must leave at least one ICL block")
        if not 0 <= self.icl_ff_reallocation < self.icl_dim * self.ff_factor:
            raise ValueError("icl_ff_reallocation must leave a nonempty ICL MLP")

    @property
    def icl_dim(self):
        return self.n_cls * self.col_dim


@dataclass
class Context:
    """Everything prediction needs from the training rows."""

    stats: dict
    col_kv: list
    icl_kv: list
    dec_k: torch.Tensor
    y: torch.Tensor
    slots: torch.Tensor
    d: torch.Tensor | None
    n_classes: int
    extra: dict = field(default_factory=dict)
    col_refine_kv: list | None = None


class CellEmbedder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        G = len(cfg.group_offsets)
        self.offsets = cfg.group_offsets
        self.n_ecdf_freq = cfg.n_ecdf_freq
        self.cell_embed = cfg.cell_embed
        if cfg.cell_embed == "fourier":
            self.freq = nn.Parameter(torch.randn(G, cfg.n_freq) * 2.0)
            self.fourier = nn.Linear(2 * cfg.n_freq, cfg.col_dim, bias=False)
        else:
            # RaBEL's verified sweep favors 64 uniform kernels and fixed sigma=1.
            # The range is adapted to LightPFN's already soft-clipped z scores.
            self.register_buffer("rbf_centers", torch.linspace(-5.0, 5.0, 64))
            self.rbf = nn.Linear(64, cfg.col_dim, bias=False)
        self.meta = nn.Linear(G * (2 + 2 * cfg.n_ecdf_freq), cfg.col_dim, bias=False)
        self.norm = nn.LayerNorm(cfg.col_dim)
        if cfg.cat_adapter:
            # Categorical group members use their own frequencies and projection (Kumo Tabular); copies
            # of the numerical ones (init_cat_adapter) make the adapter start as the plain model.
            self.cat_freq = nn.Parameter(self.freq.detach().clone())
            self.cat_fourier = nn.Linear(2 * cfg.n_freq, cfg.col_dim, bias=False)
            self.cat_fourier.weight.data.copy_(self.fourier.weight.data)

    @staticmethod
    @torch.no_grad()
    def stats(x_train):
        """Column statistics of the training rows (NaN ignored). x_train: (B, n, m)."""
        x = x_train.float()
        valid = ~torch.isnan(x)
        cnt = valid.sum(1)  # (B, m)
        x0 = torch.where(valid, x, 0.0)
        mean = x0.sum(1) / cnt.clamp(min=1)
        var = (torch.where(valid, x - mean[:, None], 0.0) ** 2).sum(1) / (cnt - 1).clamp(min=1)
        std = var.sqrt()
        srt = torch.where(valid, x, float("inf")).transpose(1, 2).sort(-1).values.contiguous()  # (B, m, n)
        return dict(mean=mean, std=std, sorted=srt, cnt=cnt)

    @staticmethod
    def normalize(x, st):
        """Returns z-score (soft-clipped), ECDF mid-rank in [0, 1] and NaN flag, each (B, n, m)."""
        x = x.float()
        nan = torch.isnan(x)
        z = (x - st["mean"][:, None]) / (st["std"][:, None] + 1e-6)
        z = torch.where(nan | (st["std"][:, None] == 0), 0.0, 5.0 * torch.tanh(z / 5.0))
        q = torch.where(nan, 0.0, x).transpose(1, 2).contiguous()  # (B, m, n)
        lo = torch.searchsorted(st["sorted"], q, right=False)
        hi = torch.searchsorted(st["sorted"], q, right=True)
        cnt = st["cnt"][..., None]
        r = ((lo + hi).float() / 2 / cnt.clamp(min=1)).clamp(0, 1)
        r = torch.where(cnt > 0, r, 0.5).transpose(1, 2)
        r = torch.where(nan, 0.5, r)
        return z, r, nan.float()

    def group_index(self, d, B, m, device):
        j = torch.arange(m, device=device)[None]
        d = torch.full((B, 1), m, device=device) if d is None else d.to(device)[:, None]
        return torch.stack([torch.where(j < d, (j + o) % d, j) for o in self.offsets], -1)  # (B, m, G)

    def group_mask(self, cat, d=None):
        """cat: (B, m) bool categorical columns -> (B, 1, m, G): which members of each cell's group are."""
        B, m = cat.shape
        idx = self.group_index(d, B, m, cat.device)
        return torch.gather(cat, 1, idx.view(B, -1)).view(B, 1, m, -1)

    def grouped(self, x, st, d=None, mask=None, mask_token=None):
        """z, ECDF rank and NaN flag of every cell and of its group neighbors, each (B, n, m, G)."""
        B, n, m = x.shape
        if mask is None:
            z, r, nan = self.normalize(x, st)
        else:
            if mask.shape != x.shape or mask.dtype != torch.bool or mask_token is None:
                raise ValueError("cell masking needs a bool mask matching x and a mask token")
            # Neutralize BEFORE circular grouping: otherwise a hidden value leaks into
            # the embeddings of its neighbors through the grouped z/ECDF/NaN features.
            z, r, nan = self.normalize(x.masked_fill(mask, 0.0), st)
            z, r, nan = (t.masked_fill(mask, 0.0) for t in (z, r, nan))
        idx = self.group_index(d, B, m, x.device)
        G = idx.shape[-1]
        gidx = idx.view(B, 1, m * G).expand(B, n, m * G)
        return tuple(torch.gather(t, 2, gidx).view(B, n, m, G) for t in (z, r, nan))

    def forward(self, x, st, d=None, mask=None, mask_token=None, catg=None):
        out = self.embed(*self.grouped(x, st, d, mask, mask_token), catg=catg)
        if mask is not None:
            out = torch.where(mask[..., None], mask_token.to(out.dtype), out)
        return out

    def embed(self, z, r, nan, catg=None):
        """Cell embeddings (B, n, m, E) from grouped features; cells are independent, so any slice
        of rows or columns of the features gives the same slice of the embeddings. catg (B, 1, m, G)
        (cat_adapter): categorical group members, embedded with the categorical Fourier weights."""
        if self.cell_embed == "fourier":
            freq = self.freq if catg is None else torch.where(catg[..., None], self.cat_freq, self.freq)
            ang = z[..., None] * freq  # (B, n, m, G, F), fp32
            four = torch.cat([ang.sin(), ang.cos()], -1)
            if catg is None:
                value = self.fourier(four.sum(-2))  # (B, n, m, 2F) -> E
            else:
                # Numerical members exactly as without the adapter, categorical ones through their own
                # weights: a table without categorical columns gets the plain embedding, bit for bit.
                value = self.fourier(torch.where(catg[..., None], 0.0, four).sum(-2))
                value = value + self.cat_fourier(torch.where(catg[..., None], four, 0.0).sum(-2))
        else:
            kernels = torch.exp(-0.5 * (z[..., None] - self.rbf_centers).square())
            value = self.rbf(kernels.sum(-2))
        k = torch.pi * 2.0 ** torch.arange(self.n_ecdf_freq, device=z.device)
        rang = r[..., None] * k
        meta = torch.cat([z, nan, rang.sin().flatten(-2), rang.cos().flatten(-2)], -1)
        return self.norm(value + self.meta(meta))  # (B, n, m, E)


class CatStatsEmbed(nn.Module):
    """Embeds the category statistics of catstats.py: the class deviations through the label embedding of
    the column stage (so they live where the column stage adds the training labels, whatever the class ->
    slot map) and a projection, plus the count features. Both projections start at zero."""

    def __init__(self, cfg):
        super().__init__()
        self.ts = nn.Linear(cfg.col_dim, cfg.col_dim, bias=False)
        self.cnt = nn.Linear(N_COUNT, cfg.col_dim, bias=False)
        nn.init.zeros_(self.ts.weight)
        nn.init.zeros_(self.cnt.weight)

    def forward(self, feats, labels):
        """feats (B, n, m, C + N_COUNT), labels (B, C, E) label embeddings of the classes."""
        C = labels.shape[1]
        ts = torch.einsum("bnmc,bce->bnme", feats[..., :C].to(labels.dtype), labels)
        return self.ts(ts) + self.cnt(feats[..., C:].to(self.cnt.weight.dtype))


class ColumnStage(nn.Module):
    """ISAB over the rows of each column. Inducing points read the training cells only, so their
    states (col_kv) are all that test rows need."""

    def __init__(self, cfg):
        super().__init__()
        E = cfg.col_dim
        self.y_emb = OrthogonalEmbedding(cfg.label_slots, E)
        self.inducing = nn.ParameterList(nn.Parameter(torch.randn(cfg.n_inducing, E) * 0.02) for _ in range(cfg.col_blocks))
        self.ind_blocks = nn.ModuleList(Block(E, cfg.col_heads, cfg.ff_factor) for _ in range(cfg.col_blocks))
        self.cell_blocks = nn.ModuleList(Block(E, cfg.col_heads, cfg.ff_factor) for _ in range(cfg.col_blocks))

    def context(self, cells, y_slots):
        B, n, m, E = cells.shape
        x = (cells + self.y_emb(y_slots)[:, :, None]).transpose(1, 2).reshape(B * m, n, E)
        col_kv = []
        for ind, ib, cb in zip(self.inducing, self.ind_blocks, self.cell_blocks):
            h = ib(ind.expand(B * m, -1, -1), *ib.keys_values(x))
            kh, vh = cb.keys_values(h)
            x = cb(x, kh, vh)
            col_kv.append((kh, vh))
        return x.view(B, m, n, E).transpose(1, 2), col_kv

    def query(self, cells, col_kv):
        B, n, m, E = cells.shape
        x = cells.transpose(1, 2).reshape(B * m, n, E)
        for cb, (kh, vh) in zip(self.cell_blocks, col_kv):
            x = cb(x, kh, vh)
        return x.view(B, m, n, E).transpose(1, 2)


class RowStage(nn.Module):
    """Transformer over the features of each row; CLS tokens (no rotation) collect the row."""

    def __init__(self, cfg):
        super().__init__()
        self.n_cls, self.rope_base = cfg.n_cls, cfg.rope_base
        self.rope_interleaved = False  # set by LightPFN.folded()
        self.mode = cfg.row_mode
        self.cls = nn.Parameter(torch.randn(cfg.n_cls, cfg.col_dim) * 0.02)
        self.blocks = nn.ModuleList(Block(cfg.col_dim, cfg.row_heads, cfg.ff_factor) for _ in range(cfg.row_blocks))

    def _rope(self, t):
        C = self.n_cls
        pos = torch.arange(t.shape[2] - C, device=t.device)
        return torch.cat([t[:, :, :C], rope(t[:, :, C:], pos, self.rope_base, self.rope_interleaved)], dim=2)

    def forward(self, cells, d=None, has_padding=None, return_cells=False):
        B, n, m, E = cells.shape
        C = self.n_cls
        x = torch.cat([self.cls.to(cells.dtype).expand(B * n, C, E), cells.reshape(B * n, m, E)], dim=1)
        mask = None
        # Training supplies the exact decision from CPU metadata. Other callers
        # retain the original automatic padding detection.
        if d is not None and (bool((d < m).any()) if has_padding is None else has_padding):
            valid = torch.arange(m, device=cells.device)[None] < d.to(cells.device)[:, None]  # (B, m)
            valid = torch.cat([torch.ones(B, C, dtype=torch.bool, device=cells.device), valid], 1)
            mask = valid.repeat_interleave(n, 0)[:, None, None, :]  # (B*n, 1, 1, C+m)
        for blk in self.blocks:
            if self.mode == "self_attention":
                k, v = blk.keys_values(x)
                x = blk(x, self._rope(k), v, mask, q_rope=self._rope)
            else:
                # Only C queries: cells stay fixed; summaries also attend to each other.
                k, v = blk.keys_values(x)
                summary = blk(x[:, :C], self._rope(k), v, mask)
                x = torch.cat([summary, x[:, C:]], dim=1)
        rows = x[:, :C].reshape(B, n, C * E)
        if return_cells:
            return rows, x[:, C:].reshape(B, n, m, E)
        return rows


class RowRefinement(nn.Module):
    """Temporary summaries, independent broadcast/gather weights per round, final broadcast.

    Each row is processed independently. Feature attention is O(m*K), K=4;
    summaries are discarded before the second column stage and final compression.
    """

    def __init__(self, cfg):
        super().__init__()
        self.rope_base = cfg.rope_base
        self.rope_interleaved = False  # set by LightPFN.folded()
        self.summary = nn.Parameter(torch.randn(4, cfg.col_dim) * 0.02)
        self.broadcast = nn.ModuleList(
            Block(cfg.col_dim, cfg.row_heads, cfg.ff_factor) for _ in range(cfg.row_refine_rounds + 1))
        self.gather = nn.ModuleList(
            Block(cfg.col_dim, cfg.row_heads, cfg.ff_factor) for _ in range(cfg.row_refine_rounds))

    def forward(self, cells, d=None, has_padding=None):
        B, n, m, E = cells.shape
        x = cells.reshape(B * n, m, E)
        s = self.summary.to(cells.dtype).expand(B * n, -1, -1)
        mask = None
        if d is not None and (bool((d < m).any()) if has_padding is None else has_padding):
            valid = torch.arange(m, device=cells.device)[None] < d.to(cells.device)[:, None]
            mask = valid.repeat_interleave(n, 0)[:, None, None, :]
        pos = torch.arange(m, device=cells.device)
        for i, broadcast in enumerate(self.broadcast):
            x = broadcast(x, *broadcast.keys_values(s),
                          q_rope=lambda q: rope(q, pos, self.rope_base, self.rope_interleaved))
            if i < len(self.gather):
                gather = self.gather[i]
                k, v = gather.keys_values(x)
                s = gather(s, rope(k, pos, self.rope_base, self.rope_interleaved), v, mask)
        return x.reshape(B, n, m, E)


class ICLStage(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D = cfg.icl_dim
        self.kv_heads_test = cfg.icl_kv_heads_test
        self.y_emb = OrthogonalEmbedding(cfg.label_slots, D)
        self.thinking = nn.Parameter(torch.randn(cfg.n_thinking, D) * 0.02)
        depth = cfg.icl_blocks - cfg.icl_drop_blocks
        self.blocks = nn.ModuleList(
            Block(D, cfg.icl_heads, cfg.ff_factor, scaling=True,
                  ff_hidden=D * cfg.ff_factor - (cfg.icl_ff_reallocation if i == depth - 1 else 0))
            for i in range(depth))
        self.norm = nn.RMSNorm(D)

    def context(self, r, y_slots):
        B = r.shape[0]
        T = self.thinking.shape[0]
        x = torch.cat([self.thinking.to(r.dtype).expand(B, -1, -1), r + self.y_emb(y_slots)], dim=1)
        icl_kv = []
        for blk in self.blocks:
            k, v = blk.keys_values(x)
            x = blk(x, k, v)
            if not self.training:  # the cache keeps only the heads test rows read
                k, v = k[:, : self.kv_heads_test].contiguous(), v[:, : self.kv_heads_test].contiguous()
            icl_kv.append((k, v))
        return self.norm(x[:, T:]), icl_kv

    def query(self, r, icl_kv):
        x = r
        for blk, (k, v) in zip(self.blocks, icl_kv):
            x = blk(x, k, v, kv_heads=self.kv_heads_test)
        return self.norm(x)


class RetrievalDecoder(nn.Module):
    """p(class | test row) = attention-weighted average of the one-hot training labels, averaged
    over heads; logits are its log. Any number of classes up to the head dim."""

    def __init__(self, cfg):
        super().__init__()
        D, H = cfg.icl_dim, cfg.decoder_heads
        self.n_heads, self.head_dim = H, D // H
        assert cfg.label_slots <= self.head_dim
        self.q = nn.Linear(D, D, bias=False)
        self.k = nn.Linear(D, D, bias=False)
        self.scaling = SoftmaxScaling(H, self.head_dim)

    def keys(self, h_train):
        B, N, _ = h_train.shape
        return self.k(h_train).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, k, y, h_test, n_classes):
        B, M, _ = h_test.shape
        q = self.q(h_test).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.scaling(q, k.shape[2])
        # one-hot values padded to the head dim so fused attention kernels apply
        v = F.one_hot(y, self.head_dim).to(q.dtype)[:, None].expand(-1, self.n_heads, -1, -1)
        p = F.scaled_dot_product_attention(q, k.to(q.dtype), v).float().mean(1)[..., :n_classes]
        return torch.log(p.clamp(min=1e-5) + 3e-5)


class LightPFN(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg = cfg or Config()
        self.cells = CellEmbedder(cfg)
        self.col = ColumnStage(cfg)
        self.row = RowStage(cfg)
        self.icl = ICLStage(cfg)
        self.decoder = RetrievalDecoder(cfg)
        if cfg.row_refine:
            self.refine = RowRefinement(cfg)
            self.col_refine = ColumnStage(cfg)  # independent, target-aware, train-only cache
        if cfg.ccmm:
            self.ccmm_mask_token = nn.Parameter(torch.randn(cfg.col_dim) * 0.02)
            self.ccmm_head = nn.Sequential(nn.LayerNorm(cfg.col_dim), nn.Linear(cfg.col_dim, 32))
        if cfg.cat_adapter:
            self.cat_stats = CatStatsEmbed(cfg)

    def adapter_parameters(self):
        """The parameters of the categorical adapter (empty without it)."""
        if not self.cfg.cat_adapter:
            return []
        return [self.cells.cat_freq, self.cells.cat_fourier.weight, *self.cat_stats.parameters()]

    @torch.no_grad()
    def init_cat_adapter(self):
        """Make the adapter the identity on the current weights (after loading a plain checkpoint)."""
        self.cells.cat_freq.copy_(self.cells.freq)
        self.cells.cat_fourier.weight.copy_(self.cells.fourier.weight)
        for p in self.cat_stats.parameters():
            p.zero_()

    def _cat_inputs(self, X_train, y_train, d, slots, n_classes, cat, generator=None, perm=None):
        """Grouped categorical mask, context statistics, their table and the class label embeddings, or
        None when the adapter is off or no column is categorical."""
        if cat is None or not self.cfg.cat_adapter:
            return None
        cat = cat.to(X_train.device).bool()
        if d is not None:
            cat = cat & (torch.arange(cat.shape[1], device=cat.device)[None] < d.to(cat.device)[:, None])
        if not bool(cat.any()):
            return None
        feats, table = context_stats(X_train, y_train, cat, n_classes, self.cfg.cat_smooth, perm, generator)
        labels = self.col.y_emb.weight[slots[:, :n_classes]]  # (B, C, E): class -> its label embedding
        return dict(catg=self.cells.group_mask(cat, d), feats=feats, table=table, labels=labels)

    @torch.no_grad()
    def folded(self):
        """A copy for prediction only that computes the same function faster: RMSNorm weights folded
        into the linear layers that read them, and the query/key dims of the rotary blocks interleaved
        so RoPE is one complex multiply. Outputs match the original up to float rounding; the copy is
        not meant for training or for saving as a checkpoint."""
        if getattr(self, "is_folded", False):
            return self
        m = copy.deepcopy(self).eval()
        m.is_folded = True
        for blk in (b for b in m.modules() if isinstance(b, Block)):
            blk.norm_q = fold_norm(blk.norm_q, blk.attn.q)
            blk.norm_kv = fold_norm(blk.norm_kv, blk.attn.kv)
            blk.norm_ff = fold_norm(blk.norm_ff, blk.mlp.fc1)
        m.icl.norm = fold_norm(m.icl.norm, m.decoder.q, m.decoder.k)  # its output only feeds the decoder
        rotary = [m.row] + ([m.refine] if self.cfg.row_refine else [])
        for stage in rotary:
            for blk in (b for b in stage.modules() if isinstance(b, Block)):
                interleave_rotary(blk)
            stage.rope_interleaved = True
        return m

    def encode(self, X_train, y_train, d=None, slots=None, n_classes=None, has_padding=None, chunk_cells=None,
               cat=None, cat_perm=None):
        """X_train: (B, n, m) float with NaN, y_train: (B, n) labels 0..C-1, d: (B,) true feature
        counts when features are zero-padded, slots: (B, label_slots) class -> label slot map.
        n_classes: total number of classes C. The default, max(y_train) + 1, is wrong when the
        highest classes have no training row: callers that know C must pass it (the sklearn
        wrapper and the training loop do). chunk_cells (inference): run the cell stages on about
        that many cells at a time, which keeps them in the CPU cache; the result is the same.
        cat: (B, m) bool categorical columns, used by the categorical adapter (ignored without it).
        cat_perm: (B, n) the row order of its ordered statistics; default: random from the global RNG in
        training mode, otherwise one fixed order, so inference is deterministic."""
        B = X_train.shape[0]
        n_classes = int(y_train.max()) + 1 if n_classes is None else n_classes
        if n_classes > self.cfg.label_slots:
            raise ValueError(f"{n_classes} classes, the model supports {self.cfg.label_slots}")
        if slots is None:
            slots = torch.arange(self.cfg.label_slots, device=X_train.device).expand(B, -1)
        y_slots = torch.gather(slots, 1, y_train)
        stats = self.cells.stats(X_train)
        gen = None if self.training else torch.Generator().manual_seed(0)
        ci = self._cat_inputs(X_train, y_train, d, slots, n_classes, cat, gen, cat_perm)
        if chunk_cells is not None:
            if torch.is_grad_enabled():
                raise RuntimeError("chunk_cells is for inference: it writes into a shared buffer that autograd cannot track")
            rows, col_kv, col_refine_kv = self._train_rows_chunked(X_train, stats, y_slots, d, has_padding, chunk_cells, ci)
        else:
            cells = self.cells(X_train, stats, d, catg=None if ci is None else ci["catg"])
            if ci is not None:
                cells = cells + self.cat_stats(ci["feats"], ci["labels"]).to(cells.dtype)
            col, col_kv = self.col.context(cells, y_slots)
            col_refine_kv = None
            if self.cfg.row_refine:
                col = self.refine(col, d, has_padding)
                col, col_refine_kv = self.col_refine.context(col, y_slots)
            rows = self.row(col, d, has_padding)
        h, icl_kv = self.icl.context(rows, y_slots)
        extra = dict(has_padding=has_padding)
        if ci is not None:
            extra.update(catg=ci["catg"], cat_table=ci["table"], cat_labels=ci["labels"])
        return Context(stats, col_kv, icl_kv, self.decoder.keys(h), y_train, slots, d, n_classes,
                       extra=extra, col_refine_kv=col_refine_kv)

    def _train_rows_chunked(self, X, stats, y_slots, d, has_padding, chunk_cells, ci=None):
        """The cell stages of encode by pieces: column stages on groups of whole columns, row stages
        on groups of whole rows, all writing into one (B, n, m, E) buffer."""
        B, n, m = X.shape
        grouped = self.cells.grouped(X, stats, d)
        cols, rows = max(1, chunk_cells // (B * n)), max(1, chunk_cells // (B * m))
        buf = None  # (B, n, m, E), allocated with the dtype of the first column-stage output

        def column_stage(stage, cells_of):
            nonlocal buf
            parts = []
            for j in range(0, m, cols):
                out, kv = stage.context(cells_of(j, j + cols), y_slots)
                if buf is None:
                    buf = out.new_empty(B, n, m, out.shape[-1])
                buf[:, :, j : j + cols] = out
                parts.append(kv)
            # (B * columns, ...) caches of each piece, merged in the (batch, column) order of one call
            return [tuple(torch.cat([p[i][t].unflatten(0, (B, -1)) for p in parts], 1).flatten(0, 1) for t in (0, 1))
                    for i in range(len(parts[0]))]

        def cells_of(a, b):
            if ci is None:
                return self.cells.embed(*(t[:, :, a:b] for t in grouped))
            cells = self.cells.embed(*(t[:, :, a:b] for t in grouped), catg=ci["catg"][:, :, a:b])
            return cells + self.cat_stats(ci["feats"][:, :, a:b], ci["labels"]).to(cells.dtype)

        col_kv = column_stage(self.col, cells_of)
        col_refine_kv = None
        if self.cfg.row_refine:
            for i in range(0, n, rows):
                buf[:, i : i + rows] = self.refine(buf[:, i : i + rows], d, has_padding)
            col_refine_kv = column_stage(self.col_refine, lambda a, b: buf[:, :, a:b])
        out = torch.cat([self.row(buf[:, i : i + rows], d, has_padding) for i in range(0, n, rows)], 1)
        return out, col_kv, col_refine_kv

    def _test_rows(self, ctx, X_test):
        catg = ctx.extra.get("catg")
        cells = self.cells(X_test, ctx.stats, ctx.d, catg=catg)
        if catg is not None:
            feats = query_stats(ctx.extra["cat_table"], X_test)
            cells = cells + self.cat_stats(feats, ctx.extra["cat_labels"]).to(cells.dtype)
        col = self.col.query(cells, ctx.col_kv)
        if self.cfg.row_refine:
            col = self.refine(col, ctx.d, ctx.extra.get("has_padding"))
            col = self.col_refine.query(col, ctx.col_refine_kv)
        return self.row(col, ctx.d, ctx.extra.get("has_padding"))

    def predict_logits(self, ctx, X_test, chunk_cells=None):
        """Logits (B, M, n_classes) for test rows; rows are independent, so X_test can be chunked.
        chunk_cells: run the cell stages on groups of about that many cells (same result)."""
        if chunk_cells is None:
            rows = self._test_rows(ctx, X_test)
        else:
            step = max(1, chunk_cells // (X_test.shape[0] * X_test.shape[2]))
            rows = torch.cat([self._test_rows(ctx, X_test[:, i : i + step]) for i in range(0, X_test.shape[1], step)], 1)
        h = self.icl.query(rows, ctx.icl_kv)
        return self.decoder(ctx.dec_k, ctx.y, h, ctx.n_classes)

    def forward(self, X, y_train, d=None, slots=None, n_classes=None, has_padding=None, cat=None):
        n_train = y_train.shape[1]
        ctx = self.encode(X[:, :n_train], y_train, d, slots, n_classes, has_padding, cat=cat)
        return self.predict_logits(ctx, X[:, n_train:])
