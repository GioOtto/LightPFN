"""Vulkan device, buffers and kernel dispatch through wgpu (WebGPU on the Vulkan backend).

Operations are recorded into one compute pass and submitted on flush() or before a read; WebGPU orders the
dispatches of a pass and inserts the barriers between them, so a later kernel sees what an earlier one
wrote. Parameters share one storage buffer, at offsets aligned to the device (at least 256 bytes).
"""

import math
import os

import numpy as np

from lightpfn.vulkan import kernels

try:
    import wgpu
except ImportError:  # optional dependency: pip install wgpu
    wgpu = None

ADAPTER_ENV = "LIGHTPFN_VULKAN_ADAPTER"
GPU_TYPES = ("DiscreteGPU", "IntegratedGPU", "VirtualGPU")
PARAM_SLOT = 256  # bytes of parameters per operation; u32 63 = slice start
MAX_GROUPS = 65535
# GPU work per submission, in floating-point operations: a few ms on a discrete GPU, well under a second on
# an integrated one. Drivers reset a GPU whose job runs too long (amdgpu's ring timeout, Windows TDR after
# 2 s), so heavy dispatches are cut into slices of workgroups and submitted in several jobs.
SUBMIT_FLOPS = 5e10
SUBMIT_OPS = 4096


def vulkan_adapters():
    """Vulkan adapters seen by wgpu, as (adapter, info) pairs; empty without wgpu or a Vulkan driver."""
    if wgpu is None:
        return []
    try:
        found = wgpu.gpu.enumerate_adapters_sync()
    except Exception:  # no loader, no driver
        return []
    return [(a, a.info) for a in found if a.info.get("backend_type") == "Vulkan"]


def pick_adapter(adapter=None):
    """adapter: None (LIGHTPFN_VULKAN_ADAPTER, else the first discrete GPU, else any GPU), an index into
    vulkan_adapters() or a substring of the device name (e.g. "llvmpipe", the CPU driver, for tests)."""
    found = vulkan_adapters()
    if adapter is None:
        adapter = os.environ.get(ADAPTER_ENV) or None
    if adapter is None:
        for kind in GPU_TYPES:
            for a, info in found:
                if info.get("adapter_type") == kind:
                    return a
        return None
    if isinstance(adapter, int) or str(adapter).isdigit():
        i = int(adapter)
        return found[i][0] if 0 <= i < len(found) else None
    for a, info in found:
        if str(adapter).lower() in str(info.get("device", "")).lower():
            return a
    return None


class View:
    """Rows of a buffer for the kernels: row r = (z, i) with z = r // L, i = r % L, at float offset
    base + (z // zin) * so + (z % zin) * si + i * ss."""

    __slots__ = ("buf", "base", "L", "zin", "so", "si", "ss")

    def __init__(self, buf, base=0, L=1, zin=1, so=0, si=0, ss=0):
        self.buf, self.base, self.L, self.zin, self.so, self.si, self.ss = buf, base, L, zin, so, si, ss

    def params(self):
        return [self.base, self.L, self.zin, self.so, self.si, self.ss]


def rows(buf, width, base=0):
    """Contiguous rows of `width` floats."""
    return View(buf, base, L=1, zin=1, so=width)


class Engine:
    def __init__(self, adapter=None):
        if wgpu is None:
            raise RuntimeError("the Vulkan backend needs wgpu: pip install wgpu")
        ad = pick_adapter(adapter)
        if ad is None:
            raise RuntimeError("no Vulkan adapter found" + (f" matching {adapter!r}" if adapter is not None else ""))
        self.adapter, self.info = ad, ad.info
        lim = ad.limits
        want = ("max-storage-buffer-binding-size", "max-buffer-size", "max-storage-buffers-per-shader-stage",
                "max-compute-workgroup-storage-size", "max-compute-invocations-per-workgroup",
                "max-compute-workgroups-per-dimension")
        self.device = ad.request_device_sync(required_limits={k: lim[k] for k in want if k in lim})
        lim = self.device.limits
        self.max_binding = min(lim["max-storage-buffer-binding-size"], lim["max-buffer-size"], (1 << 34) - 16)
        self.max_groups = min(MAX_GROUPS, lim["max-compute-workgroups-per-dimension"])
        self.param_slot = max(PARAM_SLOT, lim["min-storage-buffer-offset-alignment"])
        self.submit_ops = min(SUBMIT_OPS, self.max_binding // self.param_slot)
        self.usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
        self.pipes = {}
        self.ops = []  # (pipeline, buffers by binding, params u32, workgroups)
        self.pending = 0.0  # estimated flops recorded since the last submission
        self.submit_flops = SUBMIT_FLOPS
        self.scratch = {}
        self.sync = self.empty(4)

    # buffers -----------------------------------------------------------------------------------
    def upload(self, a, dtype=np.float32):
        a = np.ascontiguousarray(a, dtype=dtype)
        if a.nbytes == 0:
            a = np.zeros(4, dtype)
        if a.nbytes > self.max_binding:
            raise MemoryError(f"{a.nbytes} bytes exceed the device's {self.max_binding}-byte buffer limit")
        return self.device.create_buffer_with_data(data=a, usage=self.usage)

    def empty(self, n_floats):
        size = (max(4, int(n_floats)) + 3) // 4 * 16
        if size > self.max_binding:
            raise MemoryError(f"{size} bytes exceed the device's {self.max_binding}-byte buffer limit")
        return self.device.create_buffer(size=size, usage=self.usage)

    def temp(self, role, n_floats):
        """A scratch buffer reused by role (dispatches are ordered, so reuse is safe)."""
        buf = self.scratch.get(role)
        if buf is None or buf.size < 4 * n_floats:
            grow = min(buf.size // 2, self.max_binding // 4) if buf is not None else 0  # doubling, within the limit
            buf = self.scratch[role] = self.empty(max(n_floats, grow))
        return buf

    def download(self, buf, shape, offset=0):
        self.flush()
        count = int(np.prod(shape))
        if count == 0:
            return np.empty(shape, np.float32)
        data = self.device.queue.read_buffer(buf, 4 * offset, 4 * count)
        return np.frombuffer(data, dtype=np.float32).reshape(shape).copy()

    # kernels -----------------------------------------------------------------------------------
    def pipeline(self, name, src, flags=(), **consts):
        key = (name, tuple(sorted(flags)), tuple(sorted(consts.items())))
        p = self.pipes.get(key)
        if p is None:
            code = kernels.render(src, flags, **consts)
            module = self.device.create_shader_module(code=code)
            p = self.pipes[key] = self.device.create_compute_pipeline(
                layout="auto", compute={"module": module, "entry_point": "main"})
        return p

    def dispatch(self, pipe, buffers, params, groups, flops=0.0):
        """buffers: {binding: buffer}; params: list of u32 (floats as float32 bits), P[1] and P[63] filled
        here; groups: workgroups (over two grid dims past 65535); flops: estimated cost, which cuts the
        dispatch into slices of consecutive workgroups and the work into submissions of ~submit_flops."""
        if groups <= 0:
            return
        params = [int(v) for v in params]
        if len(params) > 63 or any(v < 0 or v > 0xFFFFFFFF for v in params) or groups > 0xFFFFFFFF:
            raise ValueError("kernel parameters must fit u32 and leave P[63] for the slice start")
        parts = max(1, min(groups, math.ceil(flops / self.submit_flops)))
        step = math.ceil(groups / parts)
        start = 0
        while start < groups:
            count = min(step, groups - start)
            gx = min(count, self.max_groups)
            gy = min(count // gx, self.max_groups)
            if parts == 1 and groups <= self.max_groups ** 2:
                gy = math.ceil(count / gx)  # the global kernel bound covers a single dispatch's padding
            else:
                count = gx * gy  # no padded workgroups may spill into the next slice
            p = list(params) + [0] * (PARAM_SLOT // 4 - len(params))
            p[1], p[63] = gx, start
            cost = flops * count / groups
            if self.pending + cost > self.submit_flops:
                self.flush()
            self.ops.append((pipe, buffers, p, (gx, gy, 1)))
            self.pending += cost
            start += count
            if self.pending >= self.submit_flops or len(self.ops) >= self.submit_ops:
                self.flush()

    def flush(self):
        if not self.ops:
            return
        slot = self.param_slot // 4
        P = np.zeros(slot * len(self.ops), np.uint32)
        for k, (_, _, params, _) in enumerate(self.ops):
            P[k * slot : k * slot + len(params)] = params
        pbuf = self.device.create_buffer_with_data(data=P, usage=wgpu.BufferUsage.STORAGE)
        enc = self.device.create_command_encoder()
        cp = enc.begin_compute_pass()
        for k, (pipe, buffers, _, grid) in enumerate(self.ops):
            entries = [{"binding": 0, "resource": {"buffer": pbuf, "offset": k * self.param_slot, "size": PARAM_SLOT}}]
            entries += [{"binding": b, "resource": {"buffer": buf, "offset": 0, "size": buf.size}}
                        for b, buf in sorted(buffers.items())]
            bg = self.device.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=entries)
            cp.set_pipeline(pipe)
            cp.set_bind_group(0, bg)
            cp.dispatch_workgroups(*grid)
        cp.end()
        self.device.queue.submit([enc.finish()])
        self.ops = []
        self.pending = 0.0

    def finish(self):
        """Waits for the submitted work (timing, synchronization with the host)."""
        self.flush()
        self.device.queue.read_buffer(self.sync, 0, 4)


def f32bits(x):
    return int(np.array([x], np.float32).view(np.uint32)[0])
