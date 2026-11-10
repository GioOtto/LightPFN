"""Shared inference-only views of attention/MLP projections.

Source parameters remain fp32 for retries. In-place mutation of fitted weights is unsupported;
device moves and replaced parameters invalidate the shared view on the next fit.
"""

import copy
import weakref

import torch

from lightpfn.model.layers import Block

# fp16 views of the Block projections, per source network: classifiers sharing one folded network (AutoGluon
# SharedWeights, one classifier per bagging fold) reuse the same view instead of rebuilding it at every fit
# (~9 ms on an RX 7900 XT, the whole fp16 fit overhead on small tables). The signature catches device moves and
# replaced parameters; in-place edits of the weights of a fitted network remain unsupported (see _inference_net).
_FP16_VIEWS = weakref.WeakKeyDictionary()
_HALF_LINEARS = ("attn.q", "attn.kv", "attn.out", "mlp.fc1", "mlp.fc2")


def _half_linears(net):
    return [block.get_submodule(name) for block in net.modules() if isinstance(block, Block) for name in _HALF_LINEARS]


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
        # Copy the module structure, sharing all untouched weights/buffers. Extra tensor
        # storage is only 2 bytes per converted parameter, not a second full FP32 network.
        memo = {id(t): t for t in list(net.parameters()) + list(net.buffers())}
        blocks = [b for b in net.modules() if isinstance(b, Block)]
        inside = {id(m) for b in blocks for m in b.modules()}
        for module in net.modules():
            if id(module) not in inside and not any(isinstance(b, Block) for b in module.modules()):
                memo[id(module)] = module
        half = copy.deepcopy(net, memo)
        for linear in _half_linears(half):
            linear.weight = torch.nn.Parameter(linear.weight.detach().half(), requires_grad=False)
            if linear.bias is not None:
                linear.bias = torch.nn.Parameter(linear.bias.detach().half(), requires_grad=False)
        half.eval()  # inference-only view, whatever the source network's mode when it was built
        _FP16_VIEWS[net] = (half, signature)
    return half
