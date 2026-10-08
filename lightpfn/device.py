"""Inference device: "auto" takes torch's GPU backend when there is one (CUDA on NVIDIA, also ROCm builds of
torch, which use the same "cuda" device), otherwise a GPU with a Vulkan driver (AMD, Intel or NVIDIA,
through wgpu), otherwise the CPU. Each can be asked for explicitly: "cuda[:i]", "vulkan[:i]" (index into
lightpfn.vulkan.adapters()), "cpu". The environment variable LIGHTPFN_DEVICE replaces "auto" with any of
these, e.g. LIGHTPFN_DEVICE=cpu to keep a GPU free.
"""

import os

import torch

DEVICE_ENV = "LIGHTPFN_DEVICE"


def resolve_device(device="auto"):
    """The device string LightPFNClassifier runs on: "cuda[:i]", "vulkan[:i]", "cpu" (or "mps")."""
    device = "auto" if device is None else str(device).strip().lower()
    if device == "auto":
        device = os.environ.get(DEVICE_ENV, "").strip().lower() or "auto"
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        from lightpfn import vulkan

        return "vulkan" if vulkan.is_available() else "cpu"
    kind, _, index = device.partition(":")
    if ":" in device and not index:
        raise ValueError(f"device {device!r}: missing index after ':'")
    if index and not index.isdigit():
        raise ValueError(f"device {device!r}: the index after ':' must be a number")
    if kind == "cpu":
        return "cpu"
    if kind == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device 'cuda' requested, but this torch build sees no CUDA/ROCm GPU")
        return device
    if kind == "vulkan":
        from lightpfn import vulkan

        if not vulkan.is_available(int(index) if index else None):
            raise RuntimeError(f"device {device!r} requested, but no Vulkan adapter was found: it needs a GPU "
                               "driver with Vulkan and `pip install wgpu` (adapters: lightpfn.vulkan.adapters())")
        return device
    if kind == "mps":
        return device
    raise ValueError(f"unknown device {device!r}: use 'auto', 'cpu', 'cuda[:i]' or 'vulkan[:i]'")


def kind(device):
    """"cpu", "cuda", "vulkan" or "mps" of a resolved device string."""
    return str(device).partition(":")[0]
