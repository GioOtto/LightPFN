"""Shared inference-only fp16 views of the projections that inference autocast runs in float16.

Source parameters remain fp32 for retries. In-place mutation of fitted weights is unsupported;
device moves and replaced parameters invalidate the shared view on the next fit.
"""

import copy
import weakref

import torch


# fp16 views of the projections, per source network: classifiers sharing one folded network (AutoGluon
# SharedWeights, one classifier per bagging fold) reuse the same view instead of rebuilding it at every fit
# (~9 ms on an RX 7900 XT, the whole fp16 fit overhead on small tables). The signature catches device moves and
# replaced parameters; in-place edits of the weights of a fitted network remain unsupported (see _inference_net).
_FP16_VIEWS = weakref.WeakKeyDictionary()
# Modules whose projections inference keeps in float32 (autocast disabled): never converted.
_FP32_MODULES = ("decoder", "reg")


def _half_linears(net):
    """The projections an fp16 view stores in half precision: every nn.Linear that inference autocast runs in
    float16 (blocks, cell embedding, categorical adapter, attention scaling MLPs). Autocast would cast their
    float32 weights at every call, with the same rounding as one cast here; the retrieval decoder and the
    regression modules run in float32 and stay shared."""
    skip = set()
    for name in _FP32_MODULES:
        sub = getattr(net, name, None)
        if isinstance(sub, torch.nn.Module):
            skip.update(id(m) for m in sub.modules())
    return [m for m in net.modules() if isinstance(m, torch.nn.Linear) and id(m) not in skip]


def _half_signature(net):
    return tuple((p.data_ptr(), p.device, p.dtype, tuple(p.shape)) for linear in _half_linears(net)
                 for p in (linear.weight, linear.bias) if p is not None)


def shared_fp16_view(net):
    """Return the cached view, sharing untouched modules and tensors with net."""
    signature = _half_signature(net)
    cached = _FP16_VIEWS.get(net)
    if cached is not None and cached[1] == signature:
        half = cached[0]
    else:
        # Copy only the converted projections and the modules that contain them, sharing every other module
        # and all weights/buffers. Extra storage is 2 bytes per converted parameter, not a second network.
        converted = {id(m) for m in _half_linears(net)}
        memo = {id(t): t for t in list(net.parameters()) + list(net.buffers())}
        for module in net.modules():
            if not any(id(m) in converted for m in module.modules()):
                memo[id(module)] = module
        half = copy.deepcopy(net, memo)
        for linear in _half_linears(half):
            linear.weight = torch.nn.Parameter(linear.weight.detach().half(), requires_grad=False)
            if linear.bias is not None:
                linear.bias = torch.nn.Parameter(linear.bias.detach().half(), requires_grad=False)
        half.eval()  # inference-only view, whatever the source network's mode when it was built
        _FP16_VIEWS[net] = (half, signature)
    return half
