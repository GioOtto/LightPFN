"""Full-context, four-fold honest histograms with fold-local regression targets.

All tables of a batch live in one flat tensor: (5 views, total bins + 1 dump bin,
target channels + 1 count channel). Bin (b, j, r) occupies offsets[b, j, r] onward,
so fitting is one index_add_ per view and reading is one gather (OPUS S0 review M-1).
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .preprocessing import KNOWN
from .types import folds_from_uid


@dataclass
class TargetView:
    values: torch.Tensor
    prior: torch.Tensor
    mean: torch.Tensor
    scale: torch.Tensor


@torch.no_grad()
def target_view(y, include, regression, n_classes=2):
    """Only included labels enter moments, normalization and target ECDF.

    Excluded labels are not even read to construct normalized values. This is
    essential: subtracting fold sums after a global standardization would leak.
    """
    yy = y[include]
    if regression:
        d = yy.double()
        mean = d.mean() if d.numel() else d.new_tensor(0.)
        scale = d.std(correction=0) if d.numel() else d.new_tensor(1.)
        # A relative floor uses only this view and scales with the target units.
        # Exact constants have z=0 at every scale; retain the declared 1e-6
        # output reference used by the regression head and loss normalization.
        reference = d.abs().max() if d.numel() else d.new_tensor(0.)
        floor = 32 * torch.finfo(d.dtype).eps * reference
        scale = torch.where(scale > 0, scale.maximum(floor), scale.new_tensor(1e-6))
        z = ((d - mean) / scale).float()
        sorted_y = yy.sort().values.contiguous()
        lo = torch.searchsorted(sorted_y, yy.contiguous())
        hi = torch.searchsorted(sorted_y, yy.contiguous(), right=True)
        ecdf = (lo + hi).float() / (2 * max(1, yy.numel()))
        values = torch.stack([z, z.square(), ecdf], -1)
    else:
        values = F.one_hot(yy, n_classes).float()
        mean, scale = values.new_tensor(0.), values.new_tensor(1.)
    prior = values.mean(0) if values.shape[0] else values.new_zeros(values.shape[-1])
    return TargetView(values, prior, mean if regression else mean.float(), scale if regression else scale.float())


@torch.no_grad()
def view_targets(y, fold, regression, n_classes=2):
    """Per-view target values (5, B, N, C), inclusion (5, B, N) and priors (B, 5, C).

    Views 0-3 leave out one fold, view 4 is the full context. Values of excluded
    rows are zero and never added to a table.
    """
    include = torch.stack([fold != f for f in range(4)] + [torch.ones_like(fold, dtype=torch.bool)])
    if not regression:
        # One-hot sums are integers: the masked mean is exactly target_view's mean.
        values = F.one_hot(y, n_classes).float()[None] * include[..., None]
        count = include.sum(-1, keepdim=True)
        prior = torch.where(count > 0, values.sum(2) / count, 0.).transpose(0, 1)
        return values, include, prior
    # Fold-local standardization and ECDF stay per task and view (exact moments).
    values = y.new_zeros(include.shape + (3,), dtype=torch.float32)
    priors = []
    for bi in range(y.shape[0]):
        row = []
        for f in range(5):
            view = target_view(y[bi], include[f, bi], True)
            values[f, bi, include[f, bi]] = view.values
            row.append(view.prior)
        priors.append(torch.stack(row))
    return values, include, torch.stack(priors)


@torch.no_grad()
def view_class_mappings(y, row_uid, fold, n_classes):
    """canonical_labels for every leave-fold-out view at once: (B, 5, C).

    Slots follow the first included row by UID; absent classes come after, by
    class ID. Ranks among all rows order included rows as ranks among them do.
    """
    b, n = y.shape
    include = torch.stack([fold != f for f in range(4)] + [torch.ones_like(fold, dtype=torch.bool)], 1)
    position = torch.empty_like(row_uid).scatter_(1, row_uid.argsort(1), torch.arange(n, device=y.device).expand(b, -1))
    source = torch.where(include, position[:, None], n)
    first = torch.full((b, 5, n_classes), n, device=y.device, dtype=torch.long)
    first.scatter_reduce_(2, y[:, None].expand(-1, 5, -1), source, reduce="amin", include_self=True)
    return first.argsort(dim=-1, stable=True).argsort(-1)


@dataclass
class Histogram:
    sums: torch.Tensor   # (4 leave-fold-out views + full view, bins, target channels)
    counts: torch.Tensor


@dataclass
class HistCache:
    geometry: object
    fold: torch.Tensor
    # (5, total + 1, C + 1): target sums and, in the last channel, counts. The
    # last bin is a dump for invalid cells and is never read unmasked.
    flat: torch.Tensor
    offsets: torch.Tensor  # (B, M, 5) first bin of each (task, column, resolution)
    bins: torch.Tensor     # (B, M, 5); 0 = no table (exact for numerical, r<4 for nominal)
    priors: torch.Tensor   # (B, 5, C)
    regression: bool
    class_mapping: torch.Tensor | None

    @property
    def total(self):
        return self.flat.shape[1] - 1

    @property
    def tables(self):
        """Batch -> column -> numerical resolutions / exact categorical only (views, audits)."""
        result = []
        for offsets, bins in zip(self.offsets.tolist(), self.bins.tolist()):
            result.append([[Histogram(self.flat[:, o:o + k, :-1], self.flat[:, o:o + k, -1])
                            for o, k in zip(off, kk) if k] for off, kk in zip(offsets, bins)])
        return result

    def table_elements(self):
        return self.total * self.flat.shape[0] * self.flat.shape[2]


class ICH(nn.Module):
    rows_per_bin = (8, 32, 128, 512)

    def __init__(self, regression=True):
        super().__init__()
        self.log_a = nn.Parameter(torch.full((5,), math.log(8.)))
        self.proj = nn.Linear(5 * 65, 64, bias=False)
        self.reg_values = nn.Linear(3, 64, bias=False) if regression else None
        # Nonzero at S0: make invariant/backward tests exercise the real path.
        nn.init.normal_(self.proj.weight, std=.005)

    @classmethod
    def table_bins(cls, geometry):
        """Bins per (task, column, resolution): numerical max(1, n_valid // r), nominal exact."""
        count = geometry.count
        numerical = torch.stack([(count // r).clamp_min(1) for r in cls.rows_per_bin] + [torch.zeros_like(count)], -1)
        nominal = torch.zeros_like(numerical)
        nominal[..., 4] = geometry.retained.clamp_min(1)
        return torch.where(geometry.categorical_mask[..., None], nominal, numerical)

    @staticmethod
    def bin_index(features, offsets, bins, dump):
        """Flat table index (B, rows, K, 5) of every cell and resolution; dump if invalid."""
        bins = bins[:, None]
        # code_r = floor(rank * bins_r): reuses the context rank, no new searchsorted.
        numerical = (features.rank[..., None] * bins[..., :4].float()).long()
        numerical = numerical.clamp_min(0).minimum(bins[..., :4] - 1)
        code = torch.cat([numerical, features.code.clamp_min(0)[..., None]], -1)
        valid = (features.state == KNOWN)[..., None] & (bins > 0)
        return torch.where(valid, offsets[:, None] + code, dump), valid

    @torch.no_grad()
    def fit(self, table, geometry, y, regression=False, n_classes=2, features=None):
        features = geometry.encode(table) if features is None else features
        fold = folds_from_uid(table.row_uid)
        values, include, priors = view_targets(y, fold, regression, n_classes)
        # Canonical class slots also exclude the WHOLE held-out fold. Otherwise a
        # global y-derived slot map could change its embedded ICH feature even
        # when all raw histogram channels stay honest.
        mapping = None if regression else view_class_mappings(y, table.row_uid, fold, n_classes)
        bins = self.table_bins(geometry)
        flat_bins = bins.flatten()
        # Allocate by actual BIN COUNT, never by n+2.
        total = int(flat_bins.sum())
        offsets = (flat_bins.cumsum(0) - flat_bins).view_as(bins)
        index, valid = self.bin_index(features, offsets, bins, total)
        b, n, m, r = index.shape
        channels = values.shape[-1] + 1
        flat = values.new_zeros(5, total + 1, channels)
        # Rows are in ascending order within every bin, as in a per-table loop.
        for f in range(5):
            idx = torch.where(include[f][:, :, None, None], index, total).flatten()
            src = torch.cat([values[f], include[f, ..., None].float()], -1)
            flat[f].index_add_(0, idx, src[:, :, None, None].expand(b, n, m, r, channels).reshape(-1, channels))
        return HistCache(geometry, fold, flat, offsets, bins, priors, regression, mapping)

    def raw(self, cache, table, rows, columns, context=False, features=None):
        """Return deviations and reliability in a bounded cell slice.

        Missing/OOV/overflow have zero target deviation and zero reliability.
        They never form a pooled pseudo-level. Cells carry their distinct states.
        No target from the query is accepted by this interface.
        `features` (optional) are the Features of the whole `table`.
        """
        features = (cache.geometry.encode(table) if features is None else features).rows(rows)
        offsets, bins = cache.offsets, cache.bins
        columns = list(columns)
        if columns == list(range(columns[0], columns[0] + len(columns))):
            # Contiguous column groups (the frontend's): views, no host copy.
            cols = slice(columns[0], columns[0] + len(columns))
            offsets, bins = offsets[:, cols], bins[:, cols]
            features = type(features)(*(t[:, :, cols] for t in
                                        (features.z, features.rank, features.state, features.code)))
        else:
            cols = torch.tensor(columns, device=offsets.device)
            offsets, bins = offsets.index_select(1, cols), bins.index_select(1, cols)
            features = type(features)(*(t.index_select(2, cols) for t in
                                        (features.z, features.rank, features.state, features.code)))
        index, valid = self.bin_index(features, offsets, bins, cache.total)
        view = cache.fold[:, rows] if context else torch.full_like(table.row_uid[:, rows], 4)
        width = cache.flat.shape[1]
        flat = cache.flat.flatten(0, 1)
        gathered = flat.index_select(0, (view[:, :, None, None] * width + index).flatten()).view(*index.shape, -1)
        sm, cn = gathered[..., :-1], gathered[..., -1]
        prior = cache.priors.gather(1, view[..., None].expand(-1, -1, cache.priors.shape[-1]))
        prior = prior[:, :, None, None]
        a = self.log_a.clamp(-10, 15).exp()[:, None]
        # This form is exactly zero for an empty/OOV bin.
        dev = (sm - cn[..., None] * prior) / (cn[..., None] + a)
        dev = torch.where(valid[..., None], dev, 0.)
        reliability = torch.where(valid, cn.log1p(), 0.)
        return torch.cat([dev, reliability[..., None]], -1)

    def forward(self, cache, table, rows, columns, labels=None, context=False, features=None):
        feat = self.raw(cache, table, rows, columns, context, features)
        dev, count = feat[..., :-1], feat[..., -1:]
        if cache.regression:
            embedded = self.reg_values(dev)
        else:
            # Production labels is (B,5,C,E), with a separate honest class-slot
            # map per view. A constant (B,C,E) basis is useful for isolated audits.
            if labels.ndim == 4:
                view = cache.fold[:, rows] if context else torch.full_like(table.row_uid[:, rows], 4)
                basis = labels.gather(1, view[..., None, None].expand(-1, -1, labels.shape[2], 64))
                embedded = torch.einsum("bnmrc,bnce->bnmre", dev.to(labels.dtype), basis)
            else:
                embedded = torch.einsum("bnmrc,bce->bnmre", dev.to(labels.dtype), labels)
        features = torch.cat([embedded, count.to(embedded.dtype)], -1)
        return self.proj(features.flatten(-2))
