"""Context-only target scaling for experimental v2 regression."""

import torch


def normalize_targets(y, n_context):
    """Return float64 z, mean and scale. Query labels never affect the fit.

    Near-constant targets use a declared absolute floor of 1e-6. Metrics report
    these separately. No clipping of normalized context or query labels.
    """
    if y.ndim != 2 or not 1 <= n_context <= y.shape[1] or not y.is_floating_point() or not torch.isfinite(y).all():
        raise ValueError("invalid regression targets/context")
    y64 = y.double()
    mean = y64[:, :n_context].mean(1)
    scale = y64[:, :n_context].std(1, correction=0).clamp_min(1e-6)
    return (y64 - mean[:, None]) / scale[:, None], mean, scale


def inverse_targets(z, mean, scale):
    shape = (len(mean),) + (1,) * (z.ndim - 1)
    return mean.reshape(shape) + scale.reshape(shape) * z.double()
