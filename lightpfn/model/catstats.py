"""Category statistics of the context rows for the categorical adapter (Config.cat_adapter).

Every categorical cell receives the class distribution of its category among the context rows,
as a deviation from the class prior, plus count features. Query rows use every context row of
their category. A context row uses only the context rows that come before it in a random order
of the rows (CatBoost's ordered target statistics), so its own label never enters its features,
and no label is ever correlated with its own statistics beyond the true dependence (leave-one-out
statistics are: the model collapsed on them in our tests). The count of rows the
statistics rest on is a feature, so the model can weigh a context row's noisier estimate.

Missing values form one category of their own. An unseen category gets zero features.
Everything is vectorized over tasks and columns; only the categorical columns are processed.
"""

import math

import torch
import torch.nn.functional as F

N_COUNT = 3  # log1p(rows the statistics use), log1p(other rows of the category), their fraction of the rows


def _keys(x):
    """Category keys: the values, with missing values as one key that sorts last."""
    x = x.float()
    return torch.where(torch.isnan(x), torch.full_like(x, math.inf), x)


def _features(counts, n_used, n_freq, prior, n, smooth):
    """counts (..., C) class counts of the rows used, n_used/n_freq (...) row counts, prior (B, C)
    broadcastable to counts; returns (..., C + N_COUNT)."""
    delta = (counts - n_used[..., None] * prior) / (n_used[..., None] + smooth)
    cnt = torch.stack([torch.log1p(n_used) / 4, torch.log1p(n_freq) / 4, n_freq / n], -1)
    return torch.cat([delta, cnt], -1)


def context_stats(x, y, cat, n_classes, smooth=1.0, perm=None, generator=None):
    """x: (B, n, m) context values (NaN = missing), y: (B, n) classes in 0..n_classes-1, cat: (B, m) bool.
    perm: (B, n) the row order of the ordered statistics. Default: random per task from the global RNG, or,
    with a generator, one order drawn from it for every task (the same whatever the batch).

    Returns (feats, table): feats (B, n, m, n_classes + N_COUNT) float32, zero on numerical columns, and
    the table query_stats needs; (None, None) when no column is categorical."""
    B, n, m = x.shape
    C = int(n_classes)
    dev = x.device
    mc = int(cat.sum(1).max()) if cat.numel() else 0
    if mc == 0:
        return None, None
    # the categorical columns of each task first, in order; the tail of a task with fewer is padding
    cols = torch.argsort((~cat).to(torch.int8), dim=1, stable=True)[:, :mc]  # (B, mc)
    valid = torch.gather(cat, 1, cols)
    keys = _keys(torch.gather(x, 2, cols[:, None].expand(B, n, mc)))
    if perm is None:
        if generator is None:
            perm = torch.argsort(torch.rand(B, n, device=dev), 1)
        else:
            perm = torch.argsort(torch.rand(n, generator=generator)).expand(B, n)
    perm = perm.to(dev)
    pe = perm[..., None].expand(B, n, mc)
    kp = torch.gather(keys, 1, pe)
    vs, order = torch.sort(kp, dim=1, stable=True)  # stable: rows of a category keep the random order
    rows = torch.gather(pe, 1, order)  # original row of every sorted position
    ys = torch.gather(y.long()[..., None].expand(B, n, mc), 1, rows)
    oh = F.one_hot(ys, C).float()  # (B, n, mc, C)
    start = torch.ones_like(vs, dtype=torch.bool)
    start[:, 1:] = vs[:, 1:] != vs[:, :-1]
    end = torch.ones_like(start)
    end[:, :-1] = start[:, 1:]
    ar = torch.arange(n, device=dev)[None, :, None]
    first = torch.where(start, ar, 0).cummax(1).values  # first sorted position of the category
    last = (n - 1 - torch.where(end.flip(1), ar, 0).cummax(1).values).flip(1)  # last one
    incl = oh.cumsum(1)
    excl = incl - oh
    del oh
    base = torch.gather(excl, 1, first[..., None].expand_as(excl))
    prefix = excl.sub_(base)  # rows of the category before this one in the random order
    # Query counts are constant within a category. Cache one entry per category rather than n
    # repeated entries per column (hundreds of MB per estimator at 60k rows / 100 columns).
    n_keys = int((start.sum(1) * valid).max())
    category = start.long().cumsum(1) - 1
    bi, ri, ci = torch.where(end & valid[:, None])
    ki = category[bi, ri, ci]
    table_keys = torch.full((B, mc, n_keys), math.inf, device=dev)
    table_total = torch.zeros(B, mc, n_keys, C, device=dev)
    table_keys[bi, ci, ki] = vs[bi, ri, ci]
    table_total[bi, ci, ki] = incl[bi, ri, ci] - base[bi, ri, ci]
    del incl, excl, base
    prior = F.one_hot(y.long(), C).float().mean(1)[:, None, None]  # (B, 1, 1, C)
    n_used = (ar - first).float()
    feats = _features(prefix, n_used, (last - first).float(), prior, n, smooth) * valid[:, None, :, None]
    out = torch.zeros(B, n, mc, feats.shape[-1], device=dev)
    out.scatter_(1, rows[..., None].expand_as(feats), feats)
    full = torch.zeros(B, n, m, feats.shape[-1], device=dev)
    full.scatter_(2, cols[:, None, :, None].expand_as(out), out)  # cols is a permutation prefix: no repeats
    table = dict(cols=cols, valid=valid, keys=table_keys,
                 total=table_total, prior=prior, n=n, smooth=smooth)
    return full, table


def query_stats(table, x):
    """Features (B, M, m, C + N_COUNT) of query rows x: (B, M, m) from the table of context_stats."""
    if table is None:
        return None
    B, M, m = x.shape
    cols, valid, keys, total = table["cols"], table["valid"], table["keys"], table["total"]
    mc, n_keys, C = keys.shape[1], keys.shape[2], total.shape[-1]
    q = _keys(torch.gather(x, 2, cols[:, None].expand(B, M, mc))).transpose(1, 2).contiguous()  # (B, mc, M)
    pos = torch.searchsorted(keys, q).clamp(max=n_keys - 1)
    hit = torch.gather(keys, 2, pos) == q
    counts = torch.gather(total, 2, pos[..., None].expand(B, mc, M, C)) * hit[..., None]
    n_used = counts.sum(-1)
    feats = _features(counts, n_used, n_used, table["prior"], table["n"], table["smooth"])
    feats = (feats * valid[:, :, None, None]).transpose(1, 2)  # (B, M, mc, F)
    full = torch.zeros(B, M, m, feats.shape[-1], device=x.device)
    full.scatter_(2, cols[:, None, :, None].expand_as(feats), feats)
    return full
