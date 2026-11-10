"""Regression head of the v1.5 trunk (Config.head="quantiles"): 999 quantiles from the retrieval decoder.

The targets are normalized with the context rows only (mean and standard deviation in float64, floor 1e-6).
The column and ICL stages receive the normalized target through a linear label embedding (in place of the
class label embeddings), the ICL rows a learned regression task vector. The retrieval decoder attends with
the same queries and keys as in classification and reads the attention-weighted mean and second moment of
the context targets (one SDPA call, values [z, z^2], as the class one-hot in classification). Fused SDPA
backends avoid materializing a query x context matrix; a math fallback may allocate it. A small MLP maps
the test row and those moments to 999
quantile offsets, mean + scale * delta, sorted (the v2 "quantiles" head chosen on 2026-11-04,
runs/v2_arch/reg_heads_20261104/). Every module starts from its own random stream, so a model with the
head has the same classification initial tensors as one without it at the same seed, and a classification
checkpoint loads into it with only these modules missing (warm start).
"""

import torch
from torch import nn

TAUS = torch.arange(1, 1000, dtype=torch.float32) / 1000


class QuantileHead(nn.Module):
    """999 outputs conditioned on the retrieved target moments (lightpfn.model_v2.heads.QuantileHead)."""

    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width + 2, 128), nn.GELU(), nn.Linear(128, 999))

    def forward(self, h, mean, scale):
        delta = self.net(torch.cat([h, mean.to(h.dtype), scale.to(h.dtype)], -1)).float()
        return (mean + scale * delta).sort(-1).values


class RegressionModules(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.col_label = nn.Linear(1, cfg.col_dim)
        self.refine_label = nn.Linear(1, cfg.col_dim) if cfg.row_refine else None
        self.icl_label = nn.Linear(1, cfg.icl_dim)
        self.task = nn.Parameter(torch.zeros(cfg.icl_dim))
        self.head = QuantileHead(cfg.icl_dim)


def normalize_targets(y):
    """y: (B, n) context targets -> float32 z, float64 mean and scale (B,). Finite targets only."""
    if y.ndim != 2 or not y.is_floating_point() or not bool(torch.isfinite(y).all()):
        raise ValueError("regression targets must be a finite float (B, n) tensor")
    y64 = y.double()
    mean = y64.mean(1)
    scale = y64.std(1, correction=0).clamp_min(1e-6)
    return ((y64 - mean[:, None]) / scale[:, None]).float(), mean, scale


def pinball_loss(quantiles, targets, reduction="task"):
    """Mean pinball loss over the 999 levels: quantiles (B, M, 999), targets (B, M) in the same units.
    reduction "task": (B,) mean over query rows; "mean": a scalar."""
    taus = TAUS.to(quantiles.device)
    # Training normalizes query targets in float64: with a nearly constant context they can exceed
    # float32 even when the original targets are finite (e.g. 1e35 / 1e-6). Keep that precision.
    error = targets[..., None] - quantiles.float()
    loss = torch.maximum(taus * error, (taus - 1) * error)
    if reduction == "task":
        return loss.mean((-1, -2))
    if reduction == "mean":
        return loss.mean()
    raise ValueError("unknown pinball reduction")
