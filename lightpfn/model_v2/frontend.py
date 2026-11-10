"""B4 target-aware column/refinement frontend from SOL R2, independent of v1 state."""

from dataclasses import dataclass
from functools import partial
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from lightpfn.model.lightpfn import Config as V1Config, ColumnStage, RowRefinement, RowStage
from lightpfn.model.layers import Attention

from .ich import ICH
from .preprocessing import Cells, Geometry


def column_context(stage, cells, label_vectors):
    b, n, m, e = cells.shape
    x = (cells + label_vectors[:, :, None]).transpose(1, 2).reshape(b * m, n, e)
    cache = []
    for ind, ib, cb in zip(stage.inducing, stage.ind_blocks, stage.cell_blocks):
        h = ib(ind.expand(b * m, -1, -1), *ib.keys_values(x))
        kv = cb.keys_values(h)
        x = cb(x, *kv)
        cache.extend(kv)
    return (x.view(b, m, n, e).transpose(1, 2), *cache)


def paired(flat):
    return [tuple(flat[i:i + 2]) for i in range(0, len(flat), 2)]


@dataclass
class FrontendCache:
    geometry: Geometry
    hist: object
    label_embeddings: torch.Tensor | None
    column1: list
    column2: list


class Frontend(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        v1 = V1Config(n_cls=cfg.width // 64, label_slots=10, row_mode="summary",
                      row_refine=True, row_refine_rounds=3)
        self.cells = Cells()
        self.col1, self.col2 = ColumnStage(v1), ColumnStage(v1)
        self.refinement, self.row = RowRefinement(v1), RowStage(v1)
        if cfg.frontend_qk_norm:
            for module in (self.col1, self.col2, self.refinement, self.row):
                for attention in module.modules():
                    if isinstance(attention, Attention):
                        attention.q_norm = nn.RMSNorm(attention.head_dim, eps=1e-6, elementwise_affine=False)
                        attention.k_norm = nn.RMSNorm(attention.head_dim, eps=1e-6, elementwise_affine=False)
        regression = cfg.head != "classification"
        self.hist = ICH(regression=regression) if cfg.ich else None
        self.additive = nn.Linear(64, cfg.width, bias=False) if cfg.bypass else None
        if self.additive is not None:
            nn.init.normal_(self.additive.weight, std=.005)
        self.reg_col1 = nn.Linear(1, 64) if regression else None
        self.reg_col2 = nn.Linear(1, 64) if regression else None

    def chunk_cells(self):
        return self.cfg.train_chunk_cells if torch.is_grad_enabled() else self.cfg.chunk_cells

    @staticmethod
    def hist_sum(hi, cols, n_columns):
        """Sum of the ICH features over the real columns of a contiguous column group."""
        if n_columns is None:
            return hi.sum(2)
        valid = torch.arange(cols[0], cols[-1] + 1, device=hi.device) < n_columns[:, None]
        return (hi * valid[:, None, :, None].to(hi.dtype)).sum(2)

    @staticmethod
    def hist_scale(add, m, n_columns):
        return add / math.sqrt(m) if n_columns is None else add / n_columns.to(add.dtype).sqrt()[:, None, None]

    def run_chunk(self, function, *args):
        if self.cfg.checkpoint and torch.is_grad_enabled():
            return checkpoint(function, *args, use_reentrant=False, preserve_rng_state=False)
        return function(*args)

    @staticmethod
    def merge_kv(chunks, b):
        return [tuple(torch.cat([c[i][t].unflatten(0, (b, -1)) for c in chunks], 1).flatten(0, 1)
                      for t in (0, 1)) for i in range(len(chunks[0]))]

    def row_chunks(self, cells, function):
        b, n, m, _ = cells.shape
        step = max(1, self.chunk_cells() // (b * m))
        if not torch.is_grad_enabled():
            # All transforms are row-local. Reuse the single cell buffer at inference.
            for i in range(0, n, step):
                cells[:, i:i + step].copy_(function(cells[:, i:i + step]))
            return cells
        return torch.cat([self.run_chunk(function, cells[:, i:i + step])
                          for i in range(0, n, step)], 1)

    def column_groups(self, b, n, m):
        """Columns per checkpointed column-stage chunk: as many as fit in chunk_cells.

        Column stages act on each column independently, so grouping only changes
        how many ISAB launches a step needs, not the result.
        """
        width = max(1, min(m, self.chunk_cells() // max(1, b * n)))
        return [list(range(j, min(m, j + width))) for j in range(0, m, width)]

    def encode(self, table, y, slots, mapping, z=None, regression=False):
        b, n, m = table.numerical.shape
        d = table.n_columns
        refine = partial(self.refinement, d=d, has_padding=d is not None)
        row = partial(self.row, d=d, has_padding=d is not None)
        geometry = Geometry.fit(table)
        # One batched preprocessing pass; cells and ICH reuse it (no per-column codes).
        features = geometry.encode(table)
        hist = (self.hist.fit(table, geometry, y, regression, mapping.shape[1] if mapping is not None else 2,
                              features=features) if self.hist else None)
        if regression:
            lv1, lv2 = self.reg_col1(z[..., None]), self.reg_col2(z[..., None])
            labels = None
        else:
            lv1, lv2 = self.col1.y_emb(slots), self.col2.y_emb(slots)
            labels = self.col1.y_emb(hist.class_mapping) if hist is not None else None
        cache1, cache2, parts = [], [], []
        buf, add = None, None
        groups = self.column_groups(b, n, m)
        # Column groups bound full-length ISAB activations. Histograms share the
        # context geometry; no second full-table sort/copy is needed.
        for cols in groups:
            def first(label_vectors, cols=cols):
                row_parts, hist_parts = [], []
                step = max(1, self.chunk_cells() // (b * len(cols)))
                for i in range(0, n, step):
                    sl = slice(i, i + step)
                    emb = self.cells(features.rows(sl), table.categorical_mask, cols, d)
                    hi = (self.hist(hist, table, sl, cols, labels, context=True, features=features)
                          if self.hist else torch.zeros_like(emb))
                    row_parts.append(emb + hi)
                    if self.additive is not None:
                        hist_parts.append(self.hist_sum(hi, cols, d))
                encoded = column_context(self.col1, torch.cat(row_parts, 1), label_vectors)
                add = torch.cat(hist_parts, 1) if hist_parts else label_vectors.new_zeros(b, n, 64)
                return (*encoded, add)
            result = self.run_chunk(first, lv1)
            cache1.append(paired(result[1:-1]))
            if torch.is_grad_enabled():
                parts.append(result[0])
            else:
                if buf is None:
                    buf = result[0].new_empty(b, n, m, 64)
                buf[:, :, cols[0]:cols[-1] + 1].copy_(result[0])
            if self.additive is not None:
                add = result[-1] if add is None else add + result[-1]
        if torch.is_grad_enabled():
            buf = torch.cat(parts, 2)
        del parts
        buf = self.row_chunks(buf, refine)
        out_parts = []
        for cols in groups:
            def second(cells, label_vectors):
                return column_context(self.col2, cells, label_vectors)
            result = self.run_chunk(second, buf[:, :, cols[0]:cols[-1] + 1], lv2)
            cache2.append(paired(result[1:]))
            if torch.is_grad_enabled():
                out_parts.append(result[0])
            else:
                buf[:, :, cols[0]:cols[-1] + 1].copy_(result[0])
        if torch.is_grad_enabled():
            buf = torch.cat(out_parts, 2)
        step = max(1, self.chunk_cells() // (b * m))
        rows = torch.cat([self.run_chunk(row, buf[:, i:i + step])
                          for i in range(0, n, step)], 1)
        if self.additive is not None:
            rows = rows + self.additive(self.hist_scale(add, m, d))
        ctx = FrontendCache(geometry, hist, labels, self.merge_kv(cache1, b), self.merge_kv(cache2, b))
        return rows, ctx

    def query(self, table, ctx):
        ctx.geometry.check(table)
        b, n, m = table.numerical.shape
        d = table.n_columns
        features = ctx.geometry.encode(table)
        columns = list(range(m))
        step, result = max(1, self.chunk_cells() // (b * m)), []
        for i in range(0, n, step):
            sl = slice(i, i + step)
            cells = self.cells(features.rows(sl), table.categorical_mask, columns, d)
            hi = (self.hist(ctx.hist, table, sl, columns, ctx.label_embeddings, features=features)
                  if self.hist else torch.zeros_like(cells))
            cells = self.col1.query(cells + hi, ctx.column1)
            cells = self.refinement(cells, d, d is not None)
            cells = self.col2.query(cells, ctx.column2)
            rows = self.row(cells, d, d is not None)
            if self.additive is not None:
                rows = rows + self.additive(self.hist_scale(self.hist_sum(hi, columns, d), m, d))
            result.append(rows)
        return torch.cat(result, 1)
