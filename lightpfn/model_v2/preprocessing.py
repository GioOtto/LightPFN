"""Context-only feature geometry and bounded, exact nominal dictionaries.

Everything is fitted and applied on the whole (B, rows, M) batch at once: the
kernel count of a step must not grow with tasks x columns (OPUS S0 review M-1).
"""

from dataclasses import dataclass

import torch
from torch import nn

from lightpfn.model.lightpfn import CellEmbedder, Config as V1Config


KNOWN, OOV, MISSING, OVERFLOW = range(4)
# Above every int32 category ID: pads the int64 level dictionaries.
PAD_LEVEL = 2 ** 31


@dataclass
class Column:
    """Per-column view of a Geometry (audits/tests); the model uses the batched tensors."""

    categorical: bool
    sorted_values: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    # All identities are kept as integer metadata, but target tables have <=cap bins.
    compact: torch.Tensor | None = None
    retained: int = 0

    def codes(self, numeric, nominal, missing):
        if not self.categorical:
            miss = missing | ~torch.isfinite(numeric)
            clean = torch.where(miss, 0., numeric).contiguous()
            lo = torch.searchsorted(self.sorted_values, clean)
            hi = torch.searchsorted(self.sorted_values, clean, right=True)
            n = max(1, self.sorted_values.numel())
            rank = ((lo + hi).float() / (2 * n)).clamp(0, 1)
            rank = torch.where(miss, .5, rank)
            z = 5 * torch.tanh((clean - self.mean) / self.std / 5)
            z = torch.where(miss, 0., z)
            return z, rank, miss.long() * MISSING, lo
        # No conversion of raw category IDs to float, including under autocast.
        pos = torch.searchsorted(self.sorted_values, nominal.contiguous())
        if self.sorted_values.numel():
            safe = pos.clamp_max(self.sorted_values.numel() - 1)
            known = (pos < self.sorted_values.numel()) & (self.sorted_values[safe] == nominal)
            code = self.compact[safe]
        else:
            known = torch.zeros_like(pos, dtype=torch.bool)
            code = torch.zeros_like(pos)
        state = torch.where(known, torch.where(code >= 0, KNOWN, OVERFLOW), OOV)
        state = torch.where(missing, MISSING, state)
        # Compact ranks follow first row_uid, not integer value or label.
        rank = (code.clamp_min(0).float() + .5) / max(1, self.retained)
        rank = torch.where(state == KNOWN, rank, .5)
        z = torch.where(state == KNOWN, 2 * rank - 1, 0.)
        return z, rank, state, code


@dataclass
class Features:
    """Per-cell preprocessing of a whole table, each (B, rows, M).

    rank is the context ECDF mid-rank (numerical) or compact rank (categorical);
    code is the compact categorical level (-1 overflow), 0 for numerical cells.
    """

    z: torch.Tensor
    rank: torch.Tensor
    state: torch.Tensor
    code: torch.Tensor

    def rows(self, index):
        return Features(self.z[:, index], self.rank[:, index], self.state[:, index], self.code[:, index])


@dataclass
class Geometry:
    categorical_mask: torch.Tensor  # (B, M) bool
    column_uid: torch.Tensor        # (B, M)
    # Numerical: valid context values sorted, +inf padded (B, M, N); count (B, M).
    sorted_values: torch.Tensor
    count: torch.Tensor
    mean: torch.Tensor              # (B, M) float32
    std: torch.Tensor
    # Categorical: sorted int64 levels padded with PAD_LEVEL (B, M, N), compact
    # code per level (-1 overflow/padding), number of levels and retained (B, M).
    levels: torch.Tensor
    compact: torch.Tensor
    n_levels: torch.Tensor
    retained: torch.Tensor
    n_columns: torch.Tensor | None = None  # (B,) real columns of padded batches (Table.n_columns)

    @classmethod
    @torch.no_grad()
    def fit(cls, table, cap=4096):
        cat = table.categorical_mask[:, :, None]  # (B, M, 1)
        x = table.numerical.float().transpose(1, 2)  # (B, M, N)
        missing = table.missing.transpose(1, 2)
        valid = ~missing & torch.isfinite(x) & ~cat
        count = valid.sum(-1)
        sorted_values = torch.where(valid, x, float("inf")).sort(-1).values.contiguous()
        # FP64 accumulation avoids overflow in variance of finite fp32 inputs.
        d = torch.where(valid, x.double(), 0.)
        n = count.clamp_min(1).double()
        mean = d.sum(-1) / n
        var = torch.where(valid, d - mean[..., None], 0.).square().sum(-1) / n
        mean = torch.where(count > 0, mean, 0.).float()
        std = torch.where(count > 0, var.sqrt(), 1.).clamp_min(1e-6).float()

        # Categorical dictionaries: batched unique over (B, M) rows of int64 keys.
        valid = ~missing & cat
        key = torch.where(valid, table.categorical.long().transpose(1, 2), PAD_LEVEL)
        skey, perm = key.sort(dim=-1, stable=True)
        width = skey.shape[-1]
        new = torch.ones_like(skey, dtype=torch.bool)
        new[..., 1:] = skey[..., 1:] != skey[..., :-1]
        new &= skey != PAD_LEVEL
        seg = new.long().cumsum(-1) - 1
        dump = torch.full_like(seg, width)
        at_level = torch.where(skey != PAD_LEVEL, seg, dump)
        levels = skey.new_full(skey.shape[:-1] + (width + 1,), PAD_LEVEL)
        levels.scatter_(-1, torch.where(new, seg, dump), skey)
        counts = torch.zeros_like(levels)
        counts.scatter_add_(-1, at_level, torch.ones_like(skey))
        uid = table.row_uid[:, None].expand_as(perm).gather(-1, perm)
        first = torch.full_like(levels, torch.iinfo(torch.int64).max)
        first.scatter_reduce_(-1, at_level, uid, reduce="amin", include_self=True)
        levels, counts, first = levels[..., :width], counts[..., :width], first[..., :width]
        n_levels = new.sum(-1)
        # Frequent first; ties follow row identity, never raw IDs or y. First
        # row UIDs are distinct across levels, so this order has no ties left.
        by_first = first.argsort(dim=-1, stable=True)
        order = by_first.gather(-1, counts.gather(-1, by_first).argsort(dim=-1, descending=True, stable=True))
        retained = n_levels.clamp_max(cap)
        position = torch.arange(width, device=key.device)
        kept = torch.zeros_like(new).scatter_(-1, order, position < retained[..., None])
        # Fourier ranks of retained levels follow their first row UID.
        kept_by_first = kept.gather(-1, by_first)
        compact = torch.full_like(levels, -1).scatter_(
            -1, by_first, torch.where(kept_by_first, kept_by_first.long().cumsum(-1) - 1, -1))
        return cls(table.categorical_mask, table.column_uid, sorted_values, count, mean, std,
                   levels.contiguous(), compact, n_levels, retained, table.n_columns)

    @property
    def columns(self):
        """Per-task, per-column views for audits (host syncs; not used by the model)."""
        cols = []
        for bi in range(self.categorical_mask.shape[0]):
            batch = []
            for j in range(self.categorical_mask.shape[1]):
                if bool(self.categorical_mask[bi, j]):
                    n = int(self.n_levels[bi, j])
                    batch.append(Column(True, self.levels[bi, j, :n].to(torch.int32),
                                        self.mean.new_tensor(0.), self.mean.new_tensor(1.),
                                        self.compact[bi, j, :n], int(self.retained[bi, j])))
                else:
                    n = int(self.count[bi, j])
                    batch.append(Column(False, self.sorted_values[bi, j, :n].contiguous(),
                                        self.mean[bi, j], self.std[bi, j]))
            cols.append(batch)
        return cols

    def check(self, table):
        same_width = ((table.n_columns is None) == (self.n_columns is None)
                      and (self.n_columns is None or torch.equal(table.n_columns, self.n_columns)))
        if (not same_width or not torch.equal(table.column_uid, self.column_uid)
                or not torch.equal(table.categorical_mask, self.categorical_mask)):
            raise ValueError("query schema must match the context (including column UIDs and n_columns)")

    @torch.no_grad()
    def encode(self, table):
        """Features of every cell of `table` in one batched pass (no per-column loop)."""
        cat = self.categorical_mask[:, None]  # (B, 1, M)
        x = table.numerical.float()
        miss = table.missing | ~torch.isfinite(x)
        clean = torch.where(miss, 0., x)
        query = clean.transpose(1, 2).contiguous()
        lo = torch.searchsorted(self.sorted_values, query)
        hi = torch.searchsorted(self.sorted_values, query, right=True)
        n = self.count.clamp_min(1)[..., None]
        rank = ((lo + hi).float() / (2 * n)).clamp(0, 1).transpose(1, 2)
        rank = torch.where(miss, .5, rank)
        z = 5 * torch.tanh((clean - self.mean[:, None]) / self.std[:, None] / 5)
        z = torch.where(miss, 0., z)
        state = miss.long() * MISSING
        # No conversion of raw category IDs to float, including under autocast.
        nominal = table.categorical.long().transpose(1, 2).contiguous()
        pos = torch.searchsorted(self.levels, nominal)
        safe = pos.clamp_max(self.levels.shape[-1] - 1)
        # Padding is above every int32 ID: a match implies pos < n_levels.
        known = (self.levels.gather(-1, safe) == nominal).transpose(1, 2)
        code = self.compact.gather(-1, safe).transpose(1, 2)
        cstate = torch.where(known, torch.where(code >= 0, KNOWN, OVERFLOW), OOV)
        cstate = torch.where(table.missing, MISSING, cstate)
        # Compact ranks follow first row_uid, not integer value or label.
        crank = (code.clamp_min(0).float() + .5) / self.retained.clamp_min(1)[:, None]
        crank = torch.where(cstate == KNOWN, crank, .5)
        cz = torch.where(cstate == KNOWN, 2 * crank - 1, 0.)
        return Features(torch.where(cat, cz, z), torch.where(cat, crank, rank),
                        torch.where(cat, cstate, state), torch.where(cat, code, 0))

    def features(self, table, rows, columns):
        f = self.encode(table.rows(rows))
        return f.z[:, :, columns], f.rank[:, :, columns], f.state[:, :, columns]


class Cells(nn.Module):
    """The R2 SOL Fourier frontend, with separate int32-safe category preprocessing."""

    offsets = (0, 1, 3)

    def __init__(self):
        super().__init__()
        self.embedder = CellEmbedder(V1Config(cat_adapter=True))
        # Exceptional category states are distinct even when they have no target bin.
        self.status = nn.Embedding(4, 64)
        self.register_buffer("offset_index", torch.tensor(self.offsets), persistent=False)

    def forward(self, features, categorical_mask, columns, n_columns=None):
        """features: Features of a row slice (all M columns); columns: a contiguous range
        (built on device, no host copy) or any list of indices. n_columns (B,): circular
        groups of a padded batch wrap at each task's last real column, as in v1."""
        b, n, m = features.z.shape
        device = features.z.device
        columns = list(columns)
        contiguous = columns == list(range(columns[0], columns[0] + len(columns)))
        shape = (b, n, len(columns), len(self.offsets))
        if n_columns is not None:
            j = (torch.arange(columns[0], columns[0] + len(columns), device=device) if contiguous
                 else torch.tensor(columns, device=device))[None, :, None]
            d = n_columns[:, None, None]
            grouped = torch.where(j < d, (j + self.offset_index).remainder(d), j).flatten(1)  # (B, K*G)
            index = grouped[:, None].expand(-1, n, -1)
            z, r, state = (t.gather(2, index).reshape(shape)
                           for t in (features.z, features.rank, features.state))
            cat = categorical_mask.gather(1, grouped).reshape(b, 1, len(columns), -1)
        else:
            if contiguous:
                base = torch.arange(columns[0], columns[0] + len(columns), device=device)[:, None]
                grouped = torch.cat([base + off for off in self.offsets], 1).remainder(m).flatten()
            else:
                grouped = torch.tensor([(j + off) % m for j in columns for off in self.offsets], device=device)
            z, r, state = (t.index_select(2, grouped).reshape(shape)
                           for t in (features.z, features.rank, features.state))
            cat = categorical_mask.index_select(1, grouped).reshape(b, 1, len(columns), -1)
        out = self.embedder.embed(z, r, (state == MISSING).float(), catg=cat)
        return out + self.status(state).sum(-2)
