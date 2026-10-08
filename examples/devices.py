"""Run the same fit on every available device and check that the predictions agree.

"auto" picks a CUDA or ROCm GPU through PyTorch, else a Vulkan GPU (pip install "lightpfn[vulkan]"), else the
CPU. Times include the one-time setup of each device (kernel compilation for Vulkan).

    python examples/devices.py
"""

import time

import numpy as np
import torch

from lightpfn import LightPFNClassifier

rng = np.random.default_rng(0)
X = rng.normal(size=(6000, 30)).astype(np.float32)
y = (X[:, 0] * X[:, 1] + X[:, 2] > 0).astype(int)
X_train, y_train, X_test = X[:5000], y[:5000], X[5000:]

devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")
try:
    import lightpfn.vulkan as vk

    if vk.is_available():
        devices.append("vulkan")
        print("Vulkan adapters:", [a["name"] for a in vk.adapters()])
except ImportError:
    pass

reference = None
for device in devices:
    t0 = time.perf_counter()
    proba = LightPFNClassifier(device=device, n_estimators=4, random_state=0).fit(X_train, y_train).predict_proba(X_test)
    seconds = time.perf_counter() - t0
    reference = proba if reference is None else reference
    print(f"{device:7s} {seconds:6.2f} s   max |p - p_cpu| = {np.abs(proba - reference).max():.1e}")
