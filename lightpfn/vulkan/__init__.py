"""Vulkan backend: LightPFN inference on any GPU with a Vulkan driver (AMD, Intel, NVIDIA; Linux and
Windows), through wgpu (`pip install wgpu`). The WGSL kernels in kernels.py are compiled to SPIR-V when the
backend starts; no Vulkan SDK or compiler is needed.

    from lightpfn.vulkan import VulkanLightPFN, is_available
    net = VulkanLightPFN(model)              # first GPU with a Vulkan driver
    ctx = net.encode(X_train, y_train, n_classes=C)
    logits = net.predict_logits(ctx, X_test)

LIGHTPFN_VULKAN_ADAPTER selects the adapter by index or name ("llvmpipe" runs the kernels on the CPU,
which is how the tests run without a GPU).
"""

from lightpfn.vulkan.engine import ADAPTER_ENV, GPU_TYPES, pick_adapter, vulkan_adapters


def adapters():
    """The Vulkan adapters wgpu sees, as dicts (index, name, type, vendor, driver)."""
    return [dict(index=i, name=info.get("device"), type=info.get("adapter_type"), vendor=info.get("vendor"),
                 driver=info.get("description")) for i, (_, info) in enumerate(vulkan_adapters())]


def is_available(adapter=None):
    """True when wgpu is installed and finds a Vulkan GPU (or the adapter selected by `adapter` or by
    LIGHTPFN_VULKAN_ADAPTER, which may be a CPU driver)."""
    try:
        return pick_adapter(adapter) is not None
    except Exception:
        return False


def __getattr__(name):  # VulkanLightPFN imports torch and the model: load it on first use
    if name in ("VulkanLightPFN", "VulkanContext"):
        from lightpfn.vulkan import model

        return getattr(model, name)
    raise AttributeError(name)


__all__ = ["ADAPTER_ENV", "GPU_TYPES", "VulkanLightPFN", "VulkanContext", "adapters", "is_available"]
