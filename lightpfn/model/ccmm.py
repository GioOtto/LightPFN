"""CCMM-lite training helpers; encode/predict_logits never use the auxiliary head.

Unlike the original CCMM paper, this adaptation predicts 32 train-ECDF bins.
The loss is a masked-cell mean PER TASK, for task_loss's gradient accumulation.
Missing and padded cells never become reconstruction targets. Structured masks
round quotas up to whole columns/rectangles before removing missing cells.
For deployment, instantiate the same config with ccmm=False and load a state
dict excluding ccmm_mask_token and ccmm_head.*; keep the chosen ICL reallocation.
All numerical mask/loss settings here are LightPFN choices, not paper hyperparameters.
"""

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def sample_mask(x_test, d=None, fraction=0.15, scheme="mixed", generator=None):
    """Return (B, n_test, m) bool; fraction can be scalar or one quota per task.

    cells: uniform subset of observed cells. columns: full test columns.
    blocks: one contiguous row/column rectangle. mixed: random scheme per task.
    Sampling does not depend on observed magnitudes, only missingness/valid width.
    Supply a torch.Generator on x_test.device to reproduce masks.
    """
    if scheme not in ("cells", "columns", "blocks", "mixed"):
        raise ValueError("scheme must be cells, columns, blocks or mixed")
    B, n, m = x_test.shape
    widths = torch.full((B,), m, device=x_test.device) if d is None else d.to(x_test.device)
    quotas = torch.as_tensor(fraction, device=x_test.device).float().expand(B)
    if not bool(torch.isfinite(quotas).all() & ((quotas >= 0) & (quotas <= 1)).all()):
        raise ValueError("mask fractions must be finite and in [0, 1]")
    if widths.shape != (B,) or not bool(((widths >= 1) & (widths <= m)).all()):
        raise ValueError("invalid feature widths")
    mask = torch.zeros_like(x_test, dtype=torch.bool)
    for b in range(B):
        width, quota = int(widths[b]), float(quotas[b])
        if quota == 0 or n == 0:
            continue
        eligible = torch.isfinite(x_test[b, :, :width])
        count = int(eligible.sum())
        if count == 0:
            continue
        current = scheme
        if current == "mixed":
            current = ("cells", "columns", "blocks")[int(torch.randint(
                3, (), device=x_test.device, generator=generator))]
        if current == "cells":
            indices = eligible.flatten().nonzero().flatten()
            order = torch.randperm(count, device=x_test.device, generator=generator)
            selected = indices[order[:math.ceil(quota * count)]]
            local = torch.zeros((n * width,), dtype=torch.bool, device=x_test.device)
            local[selected] = True
            mask[b, :, :width] = local.reshape(n, width)
        elif current == "columns":
            indices = torch.randperm(width, device=x_test.device, generator=generator)
            mask[b, :, indices[:math.ceil(quota * width)]] = True
        else:
            # Rectangle with approximately square aspect relative to the table.
            height = min(n, max(1, math.ceil(n * math.sqrt(quota))))
            span = min(width, max(1, math.ceil(quota * n * width / height)))
            start_r = int(torch.randint(n - height + 1, (), device=x_test.device, generator=generator))
            start_c = int(torch.randint(width - span + 1, (), device=x_test.device, generator=generator))
            mask[b, start_r:start_r + height, start_c:start_c + span] = True
        mask[b, :, :width] &= eligible
    return mask


def predict_masked(model, ctx, x_test, mask):
    """Target logits and cell-bin logits; hidden values cannot enter ANY cell embedding.

    Explicit masks may include NaNs (useful for inference-like leakage checks);
    ccmm_forward excludes those cells from the auxiliary loss.
    """
    if not model.cfg.ccmm:
        raise ValueError("CCMM requires Config(ccmm=True)")
    cells = model.cells(x_test, ctx.stats, ctx.d, mask, model.ccmm_mask_token)
    cells = model.col.query(cells, ctx.col_kv)
    has_padding = ctx.extra.get("has_padding")
    if model.cfg.row_refine:
        cells = model.refine(cells, ctx.d, has_padding)
        cells = model.col_refine.query(cells, ctx.col_refine_kv)
    rows, cell_tokens = model.row(cells, ctx.d, has_padding, return_cells=True)
    bins = model.ccmm_head(cell_tokens).float()
    h = model.icl.query(rows, ctx.icl_kv)
    logits = model.decoder(ctx.dec_k, ctx.y, h, ctx.n_classes)
    return logits, bins


def ccmm_forward(model, mb, device=None, *, slots=None, mask=None,
                 mask_fraction=None, scheme="mixed", generator=None):
    """Batch contract of lightpfn.train.task_loss -> (target logits, per-task aux loss).

    y includes train AND test labels, d/n_classes are per-task CPU metadata,
    n_train is a Python int. No optimizer or training loop is run here.
    The returned target logits already exclude classes unavailable in each task.
    Auxiliary targets use only ctx.stats (training rows); true hidden values are
    read exclusively in this detached target branch. No test labels are used.
    """
    if not model.cfg.ccmm:
        raise ValueError("CCMM requires Config(ccmm=True)")
    device = next(model.parameters()).device if device is None else device
    C, m = int(mb["n_classes"].max()), int(mb["d"].max())
    has_padding = bool((mb["d"] < m).any())
    x = mb["X"][:, :, :m].to(device, non_blocking=True)
    y = mb["y"].to(device, non_blocking=True).long()
    d = mb["d"].to(device, non_blocking=True)
    nc = mb["n_classes"].to(device, non_blocking=True)
    ntr = mb["n_train"]
    if slots is None:
        slots = torch.argsort(torch.rand(x.shape[0], model.cfg.label_slots,
                                       device=device, generator=generator), dim=1)
    ctx = model.encode(x[:, :ntr], y[:, :ntr], d, slots, C, has_padding)
    x_test = x[:, ntr:]
    if mask is None:
        fraction = model.cfg.ccmm_mask_fraction if mask_fraction is None else mask_fraction
        mask = sample_mask(x_test, d, fraction, scheme, generator)
    else:
        mask = mask.to(device)
        if mask.shape != x_test.shape or mask.dtype != torch.bool:
            raise ValueError("mask must be bool and match the cropped test tensor")
        if bool((mask & (torch.arange(m, device=device)[None, None] >= d[:, None, None])).any()):
            raise ValueError("cannot mask padded cells")
    logits, bins = predict_masked(model, ctx, x_test, mask)
    with torch.no_grad():
        _, ranks, _ = model.cells.normalize(x_test, ctx.stats)
        target = (ranks * 32).long().clamp(0, 31)
        active = mask & torch.isfinite(x_test)
    cell_loss = F.cross_entropy(bins.flatten(0, 2), target.flatten(), reduction="none")
    cell_loss = cell_loss.reshape_as(target).masked_fill(~active, 0)
    auxiliary = cell_loss.sum((1, 2)) / active.sum((1, 2)).clamp(min=1)
    logits = logits.float().masked_fill(torch.arange(C, device=device)[None, None] >= nc[:, None, None], -1e4)
    return logits, auxiliary


def ccmm_task_loss(model, mb, device, label_slots, weight):
    """Drop-in per-task classification + weighted CCMM loss for the review diff."""
    if label_slots != model.cfg.label_slots or weight < 0:
        raise ValueError("invalid CCMM label slots or weight")
    logits, auxiliary = ccmm_forward(model, mb, device)
    targets = mb["y"][:, mb["n_train"]:].to(device, non_blocking=True).long()
    ce = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none").mean(1)
    return ce + weight * auxiliary
