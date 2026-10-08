"""Vulkan/torch parity, including the persistent caches (run with llvmpipe in CI).

Use ``-s`` to print the largest absolute errors for each numerical test.
No test selects a GPU for the device resolver: its availability checks are mocked.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("wgpu")
from lightpfn import vulkan

if not vulkan.is_available():
    pytest.skip("no selected Vulkan adapter", allow_module_level=True)

from lightpfn.device import resolve_device
from lightpfn.model.lightpfn import CellEmbedder
from lightpfn.sklearn import LightPFNClassifier, load_model
from lightpfn.vulkan import VulkanLightPFN
from test_inference_paths import CONFIGS, _model

VARIANTS = {**CONFIGS, "unaligned_mlp": dict(icl_ff_reallocation=5),
            "full_kv": dict(icl_kv_heads_test=8)}


@pytest.fixture(scope="module")
def networks():
    # Compile each shader variant once; contexts must survive later scratch reuse.
    cache = {}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(4)

    def get(name):
        if name not in cache:
            plain = _model(VARIANTS[name])
            folded = plain.folded()
            cache[name] = plain, folded, VulkanLightPFN(folded)
        return cache[name]

    yield get
    cache.clear()
    torch.set_num_threads(old_threads)


@pytest.fixture
def errors(request, record_property):
    maxima = {}

    def check(actual, expected, category, atol=1e-5):
        actual, expected = torch.as_tensor(actual), torch.as_tensor(expected)
        assert actual.shape == expected.shape
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
        delta = (actual - expected).abs().max().item() if actual.numel() else 0.0
        maxima[category] = max(maxima.get(category, 0.0), delta)
        torch.testing.assert_close(actual, expected, rtol=0, atol=atol)

    yield check
    for category, value in sorted(maxima.items()):
        record_property("max_abs_" + category, value)
    if maxima:
        print(f"\n{request.node.name}: " + ", ".join(f"{k}={v:.9g}" for k, v in sorted(maxima.items())))


def task(B, n, M, m, C=3, seed=19, absent=False):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(B, n + M, m, generator=g)
    X[0, 0, 0] = float("nan")
    if M:
        X[0, n, 0] = float("nan")
    if m > 1:
        X[-1, :, -1] = 1.0
    if m > 2:
        X[0, :, 1] = float("nan")  # an entirely missing column
    y = torch.randint(0, min(C, 2) if absent else C, (B, n), generator=g)
    slots = torch.stack([torch.randperm(16, generator=g) for _ in range(B)])
    return X[:, :n], y, X[:, n:], slots


def cache_parity(net, ctx, ref, check, atol=1e-5):
    E, B, m = net.E, ctx.B, ctx.m
    for gpu_list, cpu_list in ((ctx.col_kv, ref.col_kv),
                               (ctx.col_refine_kv or [], ref.col_refine_kv or [])):
        assert len(gpu_list) == len(cpu_list)
        for buf, (k, v) in zip(gpu_list, cpu_list, strict=True):
            I = k.shape[2]
            packed = torch.from_numpy(net.eng.download(buf, (B * m, I, 2 * E)))
            for i, target in enumerate((k, v)):
                out = packed[..., i * E:(i + 1) * E].reshape(B * m, I, k.shape[1], -1).transpose(1, 2)
                check(out, target, "column_cache", atol=atol)
    assert len(ctx.icl_kv) == len(ref.icl_kv)
    for buf, (k, v) in zip(ctx.icl_kv, ref.icl_kv, strict=True):
        H, L, hd = k.shape[1:]
        packed = torch.from_numpy(net.eng.download(buf, (B, L, 2, H, hd)))
        check(packed[:, :, 0].transpose(1, 2), k, "icl_cache", atol=atol)
        check(packed[:, :, 1].transpose(1, 2), v, "icl_cache", atol=atol)
    H, hd = ref.dec_k.shape[1], ref.dec_k.shape[-1]
    keys = net.eng.download(ctx.dec_k, (B, ctx.n, H, hd))
    check(torch.from_numpy(keys).transpose(1, 2), ref.dec_k, "decoder_cache", atol=atol)
    check(net.eng.download(ctx.onehot, (B, ctx.n, net.dec_hd)),
          torch.nn.functional.one_hot(ref.y, net.dec_hd).float(), "onehot", atol=0)
    for key in ref.stats:
        torch.testing.assert_close(ctx.stats[key], ref.stats[key], rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("name", list(VARIANTS))
@torch.inference_mode()
def test_configs_caches_and_independent_chunks(name, networks, errors):
    plain, folded, net = networks(name)
    B = 1 + list(VARIANTS).index(name) % 3
    X, y, Xt, slots = task(B, 5, 2, 3)
    args = dict(slots=slots, n_classes=3)
    ref = plain.encode(X, y, **args)
    fref = folded.encode(X, y, **args)
    expected = plain.predict_logits(ref, Xt)
    errors(folded.predict_logits(fref, Xt), expected, "folded_logits", atol=2e-5)
    # Encode chunks and prediction chunks are varied independently, not in lockstep.
    contexts = []
    for chunk in (None, 1, B * 5 * 2 + 1, 10**6):
        ctx = net.encode(X.double(), y, chunk_cells=chunk, **args)
        cache_parity(net, ctx, ref, errors)
        cache_parity(net, ctx, fref, errors)
        errors(net.predict_logits(ctx, Xt), expected, "logits", atol=2e-5)
        contexts.append(ctx)
    for chunk in (None, 1, B * 3 + 1, 10**6):
        errors(net.predict_logits(contexts[0], Xt.double(), chunk_cells=chunk), expected, "logits", atol=2e-5)
    # A later encode and all scratch reallocations must leave the oldest context intact.
    errors(net.predict_logits(contexts[0], Xt), expected, "retained_context", atol=2e-5)


@pytest.mark.parametrize("B,n,m", [(1, 1, 1), (2, 2, 1), (3, 5, 2)])
@torch.inference_mode()
def test_tiny_and_empty_queries(B, n, m, networks, errors):
    plain, _, net = networks("B4")
    X, y, Xt, slots = task(B, n, 1, m, C=10, absent=True)
    ref = plain.encode(X, y, slots=slots, n_classes=10)
    ctx = net.encode(X, y, slots=slots, n_classes=10, chunk_cells=1)
    cache_parity(net, ctx, ref, errors)
    errors(net.predict_logits(ctx, Xt, chunk_cells=1), plain.predict_logits(ref, Xt), "logits", atol=2e-5)
    # Torch's chunked empty path concatenates an empty list; its plain path is defined.
    expected = plain.predict_logits(ref, Xt[:, :0])
    for chunk in (None, 1, 17, 10**6):
        errors(net.predict_logits(ctx, Xt[:, :0], chunk_cells=chunk), expected, "empty_logits", atol=0)


@pytest.mark.parametrize("C", range(2, 11))
@torch.inference_mode()
def test_classes_slots_and_absent_classes(C, networks, errors):
    plain, _, net = networks("summary")
    X, y, Xt, slots = task(2, C + 1, 1, 2, C=C, seed=C)
    y[:, :C] = torch.arange(C)
    for labels in (y, y % (C - 1)):
        # First every class appears, then the highest class is absent, including C=2.
        ref = plain.encode(X, labels, slots=slots, n_classes=C)
        ctx = net.encode(X, labels, slots=slots, n_classes=C, chunk_cells=7)
        errors(net.predict_logits(ctx, Xt, chunk_cells=1), plain.predict_logits(ref, Xt), "logits", atol=2e-5)


@torch.inference_mode()
def test_tiled_attention_partial_tiles_and_long_icl(networks, errors):
    # n > 64, M >= 32 and m+C >= 32 exercise both flash kernels, partial query/key
    # tiles, >64 cell keys per column, and multiple ICL key tiles (16+n=81).
    plain, folded, net = networks("refine_self_attention")
    X, y, Xt, slots = task(2, 65, 33, 29, C=7)
    args = dict(slots=slots, n_classes=7)
    ref = plain.encode(X, y, **args)
    ctx = net.encode(X, y, **args, chunk_cells=2 * 65 * 17)
    cache_parity(net, ctx, ref, errors)
    expected = plain.predict_logits(ref, Xt)
    errors(net.predict_logits(ctx, Xt), expected, "logits", atol=2e-5)
    errors(net.predict_logits(ctx, Xt, chunk_cells=2 * 29 * 7), expected, "chunked_logits", atol=2e-5)
    fref = folded.encode(X, y, **args)
    errors(net.predict_logits(ctx, Xt), folded.predict_logits(fref, Xt), "folded_logits", atol=2e-5)
    assert {key[0] for key in net.eng.pipes} >= {"attn_small", "attn_tiled"}


@torch.inference_mode()
def test_infinite_test_cells_with_finite_training(networks, errors):
    plain, _, net = networks("default")
    X, y, Xt, slots = task(1, 5, 2, 2)
    Xt[0, :, 0] = torch.tensor([float("inf"), -float("inf")])
    st = CellEmbedder.stats(X)
    assert all(torch.isfinite(t).all() for t in CellEmbedder.normalize(Xt, st))
    # Infinite *training* values give NaN mean/std in torch and are not a defined
    # finite inference case; do not claim support for them.
    bad = X.clone()
    bad[0, 1, 0] = float("inf")
    assert torch.isnan(CellEmbedder.normalize(bad, CellEmbedder.stats(bad))[0]).any()
    ref = plain.encode(X, y, slots=slots, n_classes=3)
    ctx = net.encode(X, y, slots=slots, n_classes=3)
    errors(net.predict_logits(ctx, Xt), plain.predict_logits(ref, Xt), "logits", atol=2e-5)


@torch.inference_mode()
def test_real_B4_checkpoint(errors):
    path = Path(__file__).resolve().parents[1] / "runs/train/B4/ema_step008000.pt"
    if not path.exists():
        pytest.skip("B4 checkpoint is not present")
    plain = load_model(path).eval()
    net = VulkanLightPFN(plain)
    X, y, Xt, slots = task(1, 17, 5, 3, C=4)
    ref = plain.encode(X, y, slots=slots, n_classes=4)
    ctx = net.encode(X, y, slots=slots, n_classes=4, chunk_cells=19)
    # Trained cache magnitudes reach 10.5 (randomized ones are about 0.1).
    # Observed accumulated float32 error is 1.54e-5; logits stay below 1.5e-6.
    # Only this checkpoint's caches use 2e-5; logits retain the same 2e-5 bound.
    cache_parity(net, ctx, ref, errors, atol=2e-5)
    folded = plain.folded()
    fref = folded.encode(X, y, slots=slots, n_classes=4)
    cache_parity(net, ctx, fref, errors, atol=2e-5)
    errors(net.predict_logits(ctx, Xt, chunk_cells=7), plain.predict_logits(ref, Xt), "logits", atol=2e-5)
    errors(net.predict_logits(ctx, Xt), folded.predict_logits(fref, Xt), "folded_logits", atol=2e-5)


def wrapper_task():
    rng = np.random.default_rng(41)
    X = rng.normal(size=(29, 3)).astype(np.float32)
    X[2, 0] = np.nan
    y = np.array(["alpha"] * 20 + ["beta"] * 7 + ["rare"] * 2)
    return X, y


@pytest.mark.parametrize("estimators", [1, 3])
def test_classifier_stratification_and_buffer_grouping(estimators, networks, errors, monkeypatch):
    plain, _, _ = networks("B4")
    X, y = wrapper_task()
    opts = dict(model=plain, n_estimators=estimators, max_context=11, seed=4, n_threads=4, chunk_rows=3)
    cpu = LightPFNClassifier(device="cpu", **opts).fit(X, y)
    gpu = LightPFNClassifier(device="vulkan", **opts)
    gpu._initialize_backend()
    # Two estimators per buffer, then a remainder, independent of batch_cells.
    monkeypatch.setattr(gpu.net_, "max_train_cells", 2 * 11 * 3)
    gpu.fit(X, y)
    assert [ctx.B for _, ctx in gpu.members_] == ([1] if estimators == 1 else [2, 1])
    for _, ctx in gpu.members_:
        assert (gpu.net_.eng.download(ctx.onehot, (ctx.B, ctx.n, gpu.net_.dec_hd)).sum(1)[..., :3] > 0).all()
    errors(gpu.predict_proba(X[:5]), cpu.predict_proba(X[:5]), "probabilities", atol=1e-6)
    np.testing.assert_array_equal(gpu.predict(X[:5]), cpu.predict(X[:5]))
    np.testing.assert_array_equal(gpu.classes_, cpu.classes_)


def test_classifier_auto_fallback_and_explicit_error(networks, errors, monkeypatch):
    import lightpfn.sklearn as wrapper

    plain, _, _ = networks("B4")
    X, y = wrapper_task()
    cpu = LightPFNClassifier(model=plain, device="cpu", max_context=11, n_estimators=3).fit(X, y)
    monkeypatch.setattr(wrapper, "resolve_device", lambda device: "vulkan")
    auto = LightPFNClassifier(model=plain, device="auto", max_context=11, n_estimators=3)
    auto._initialize_backend()
    monkeypatch.setattr(auto.net_, "max_train_cells", 1)
    with pytest.warns(RuntimeWarning, match="running this fit on the CPU"):
        auto.fit(X, y)
    assert auto.fit_device_ == "cpu" and auto.fit_net_ is auto.cpu_net_
    errors(auto.predict_proba(X[:5]), cpu.predict_proba(X[:5]), "probabilities", atol=1e-6)
    np.testing.assert_array_equal(auto.predict(X[:5]), cpu.predict(X[:5]))
    # A subsequent smaller fit returns to Vulkan, rather than using stale CPU contexts.
    monkeypatch.setattr(auto.net_, "max_train_cells", 1000)
    auto.fit(X, y)
    assert auto.fit_device_ == "vulkan"
    errors(auto.predict_proba(X[:5]), cpu.predict_proba(X[:5]), "refit_probabilities", atol=1e-6)
    explicit = LightPFNClassifier(model=plain, device="vulkan", max_context=11)
    explicit._initialize_backend()
    monkeypatch.setattr(explicit.net_, "max_train_cells", 1)
    with pytest.raises(MemoryError, match="training cells"):
        explicit.fit(X, y)


def test_uncached_classifier_releases_context(networks, errors):
    plain, _, _ = networks("B4")
    X, y = wrapper_task()
    opts = dict(model=plain, device="vulkan", n_estimators=3, max_context=11, seed=4, chunk_rows=3)
    cached = LightPFNClassifier(**opts).fit(X, y)
    uncached = LightPFNClassifier(**opts, cache_context=False).fit(X, y)
    assert uncached.members_ == []
    expected = cached.predict_proba(X[:5])
    for _ in range(2):
        errors(uncached.predict_proba(X[:5]), expected, "uncached_probabilities", atol=1e-7)
        assert uncached.members_ == []


def test_kernel_math_over_trained_input_range(errors):
    from lightpfn.vulkan.engine import Engine

    eng = Engine()
    # B4's largest trained frequency is 4.37: |z*f| <= 21.85; ECDF <= 8*pi.
    x = torch.linspace(-32, 32, 4097)
    src = """
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> O: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let i = wgid(wg) * 64u + li;
  if (i >= P[0]) { return; }
  let x = X[i]; let sc = sincos(x);
  O[i*5u] = sc.x; O[i*5u+1u] = sc.y; O[i*5u+2u] = erf_(x);
  O[i*5u+3u] = gelu(x); O[i*5u+4u] = tanh_(x);
}
"""
    inp, out = eng.upload(x.numpy()), eng.empty(x.numel() * 5)
    eng.dispatch(eng.pipeline("math_probe", src), {1: inp, 2: out}, [x.numel(), 0], (x.numel() + 63) // 64)
    got = torch.from_numpy(eng.download(out, (x.numel(), 5)))
    for i, (expected, label, atol) in enumerate(((x.sin(), "sin", 2e-6), (x.cos(), "cos", 2e-6),
            (x.erf(), "erf", 5e-7), (torch.nn.functional.gelu(x), "gelu", 3e-6), (x.tanh(), "tanh", 2e-7))):
        errors(got[:, i], expected, label, atol=atol)


def test_engine_parameter_slots_slicing_and_2d_grid(errors):
    from lightpfn.vulkan.engine import Engine

    eng = Engine()
    src = """
@group(0) @binding(1) var<storage, read_write> O: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) wg: vec3u) {
  let i = wgid(wg);
  if (i >= P[0]) { return; }
  O[i] += pf(2u);
}
"""
    pipe = eng.pipeline("grid_probe", src)
    from lightpfn.vulkan.engine import f32bits

    # Multiple outstanding parameter slots, a rectangular 2D grid with padding.
    count = 65537
    buf = eng.empty(count)
    for value in (1.0, 2.0):
        eng.dispatch(pipe, {1: buf}, [count, 0, f32bits(value)], count)
    assert len(eng.ops) == 2  # one dispatch per operation when no FLOPS slicing is needed
    errors(eng.download(buf, (count,)), torch.full((count,), 3.0), "grid", atol=0)
    # The updated engine also slices expensive dispatches across submissions.
    if hasattr(eng, "submit_flops"):
        eng.submit_flops = 1.0
        buf = eng.empty(257)
        eng.dispatch(pipe, {1: buf}, [257, 0, f32bits(1.0)], 257, flops=7.0)
        errors(eng.download(buf, (257,)), torch.ones(257), "sliced_grid", atol=0)


@pytest.mark.parametrize("cuda,vk,expected", [(True, True, "cuda"), (True, False, "cuda"),
                                            (False, True, "vulkan"), (False, False, "cpu")])
def test_resolve_auto_order(cuda, vk, expected, monkeypatch):
    monkeypatch.delenv("LIGHTPFN_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(vulkan, "is_available", lambda adapter=None: vk)
    assert resolve_device("auto") == resolve_device(None) == expected


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:2", "vulkan", "vulkan:1", "mps"])
def test_resolve_explicit_and_environment(device, monkeypatch):
    seen = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(vulkan, "is_available", lambda adapter=None: seen.append(adapter) or True)
    monkeypatch.setenv("LIGHTPFN_DEVICE", device.upper())
    assert resolve_device("auto") == device
    assert resolve_device(" " + device.upper() + " ") == device
    assert resolve_device("cpu") == "cpu"  # explicit beats the environment
    if device.startswith("vulkan"):
        assert seen == ([1, 1] if ":" in device else [None, None])


@pytest.mark.parametrize("device", ["bad", "auto:1", "vulkan:-1", "cuda:abc", "vulkan:1:2", "",
                                   "vulkan:", "cuda:", "cpu:", "mps:"])
def test_resolve_bad_strings(device, monkeypatch):
    monkeypatch.delenv("LIGHTPFN_DEVICE", raising=False)
    with pytest.raises(ValueError):
        resolve_device(device)


@pytest.mark.parametrize("device", ["vulkan", "vulkan:2", "cuda", "cuda:0"])
def test_resolve_unavailable(device, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(vulkan, "is_available", lambda adapter=None: False)
    with pytest.raises(RuntimeError, match="requested"):
        resolve_device(device)


def test_large_inplace_dispatch_slices(errors, monkeypatch):
    from lightpfn.vulkan.engine import Engine

    eng = Engine()
    eng.submit_flops = 1.0
    src = """
@group(0) @binding(1) var<storage, read_write> O: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) wg: vec3u) {
  let i = wgid(wg);
  if (i >= P[0]) { return; }
  O[i] += 1.0;
}
"""
    grids = []
    flush = eng.flush

    def capture():
        grids.extend(grid for _, _, _, grid in eng.ops)
        flush()

    monkeypatch.setattr(eng, "flush", capture)
    count = 262144  # two nominal slices of 131072: both exceed one grid dimension
    buf = eng.empty(count)
    eng.dispatch(eng.pipeline("large_inplace", src), {1: buf}, [count, 0], count, flops=2.0)
    errors(eng.download(buf, (count,)), torch.ones(count), "sliced_grid", atol=0)
    assert any(x * y > 65535 for x, y, _ in grids)
    assert sum(x * y for x, y, _ in grids) == count


@pytest.mark.parametrize("B,n,m", [(3, 20000, 1), (1, 70000, 1), (1, 1, 5000)])
def test_training_capacity_covers_all_buffers(B, n, m, networks, monkeypatch):
    _, _, net = networks("default")
    monkeypatch.setattr(net.eng, "max_binding", 128 * 1024**2)
    assert B * n * m <= net.max_train_cells
    if n == 20000:
        # This batch fits once the row pieces are limited, not as one row piece.
        assert net.train_capacity(n, m) >= B
        _, cost = net._piece_costs(n, m)
        assert B * n * cost * 4 > net.eng.max_binding
    else:
        assert net.train_capacity(n, m) == 0
        X, y = torch.zeros(B, n, m), torch.zeros(B, n, dtype=torch.long)
        with pytest.raises(MemoryError, match="cache/scratch"):
            net.encode(X, y, n_classes=2)
        assert not net.eng.ops


@torch.inference_mode()
def test_buffer_limited_row_pieces(networks, errors, monkeypatch):
    plain, _, net = networks("default")
    X, y, Xt, slots = task(2, 100, 103, 1)
    ref = plain.encode(X, y, slots=slots, n_classes=3)
    monkeypatch.setattr(net.eng, "max_binding", 500000)
    assert net.train_capacity(100, 1) == 2
    ctx = net.encode(X, y, slots=slots, n_classes=3, chunk_cells=10**6)
    cache_parity(net, ctx, ref, errors)
    errors(net.predict_logits(ctx, Xt, chunk_cells=10**6), plain.predict_logits(ref, Xt), "logits", atol=2e-5)
    with pytest.raises(MemoryError, match="fewer query rows"):
        net.predict_logits(ctx, torch.zeros(2, 123, 1))
    assert not net.eng.ops


def test_classifier_cache_capacity_and_grouping(networks, errors, monkeypatch):
    import lightpfn.sklearn as wrapper

    plain, _, _ = networks("default")
    X, y = wrapper_task()
    opts = dict(model=plain, n_estimators=3, max_context=11, seed=4)
    cpu = LightPFNClassifier(device="cpu", **opts).fit(X, y)
    gpu = LightPFNClassifier(device="vulkan", **opts)
    gpu._initialize_backend()
    monkeypatch.setattr(gpu.net_.eng, "max_binding", 200000)
    assert gpu.net_.train_capacity(11, 3) == 2
    gpu.fit(X, y)
    assert [ctx.B for _, ctx in gpu.members_] == [2, 1]
    errors(gpu.predict_proba(X[:5]), cpu.predict_proba(X[:5]), "probabilities", atol=1e-6)
    monkeypatch.setattr(wrapper, "resolve_device", lambda device: "vulkan")
    auto = LightPFNClassifier(device="auto", **opts)
    auto._initialize_backend()
    monkeypatch.setattr(auto.net_.eng, "max_binding", 65536)
    with pytest.warns(RuntimeWarning, match="cache/scratch.*running this fit on the CPU"):
        auto.fit(X, y)
    errors(auto.predict_proba(X[:5]), cpu.predict_proba(X[:5]), "fallback_probabilities", atol=1e-6)
    monkeypatch.setattr(gpu.net_.eng, "max_binding", 65536)
    with pytest.raises(MemoryError, match="cache/scratch"):
        gpu.fit(X, y)


@pytest.mark.parametrize("device", ["auto", "vulkan"])
def test_classifier_allocation_failure_clears_partial_fit(device, networks, errors, monkeypatch):
    import lightpfn.sklearn as wrapper
    from lightpfn.vulkan.engine import rows
    import wgpu

    plain, _, _ = networks("B4")
    X, y = wrapper_task()
    cpu = LightPFNClassifier(model=plain, device="cpu", n_estimators=3, batch_cells=0, max_context=11).fit(X, y)
    monkeypatch.setattr(wrapper, "resolve_device", lambda device: "vulkan")
    clf = LightPFNClassifier(model=plain, device=device, n_estimators=3, batch_cells=0, max_context=11)
    clf._initialize_backend()
    encode = clf.net_.encode
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            buf = clf.net_.eng.temp("failed_fit", 4)
            clf.net_.copy(rows(buf, 4), rows(clf.net_.eng.empty(4), 4), 1, 4)
            raise wgpu.GPUOutOfMemoryError("simulated total memory exhaustion")
        return encode(*args, **kwargs)

    monkeypatch.setattr(clf.net_, "encode", fail_second)
    if device == "auto":
        with pytest.warns(RuntimeWarning, match="running this fit on the CPU"):
            clf.fit(X, y)
        errors(clf.predict_proba(X[:5]), cpu.predict_proba(X[:5]), "probabilities", atol=1e-6)
    else:
        with pytest.raises(MemoryError, match="total memory exhaustion"):
            clf.fit(X, y)
        assert not clf.members_
    assert not clf.net_.eng.ops and not clf.net_.eng.scratch and clf.net_.eng.pending == 0
    monkeypatch.setattr(clf.net_, "encode", encode)
    clf.fit(X, y)
    assert clf.fit_device_ == "vulkan" and len(clf.members_) == 3


@pytest.mark.parametrize("name", ["default", "B4"])
@torch.inference_mode()
def test_empty_training_context(name, networks, errors):
    plain, _, net = networks(name)
    X, y = torch.zeros(2, 0, 3), torch.zeros(2, 0, dtype=torch.long)
    Xt = torch.randn(2, 5, 3)
    ref = plain.encode(X, y, n_classes=2)
    for chunk in (None, 1):
        ctx = net.encode(X, y, n_classes=2, chunk_cells=chunk)
        cache_parity(net, ctx, ref, errors)
        errors(net.predict_logits(ctx, Xt, chunk_cells=chunk), plain.predict_logits(ref, Xt), "logits", atol=2e-5)
    with pytest.raises(ValueError, match="n_classes"):
        net.encode(X, y)


@torch.inference_mode()
def test_negative_and_large_group_offsets(errors):
    plain = _model(dict(group_offsets=(0, -1, (1 << 40) + 3)))
    net = VulkanLightPFN(plain)
    for m in (2, 3):
        X, y, Xt, slots = task(2, 5, 3, m)
        ref = plain.encode(X, y, slots=slots, n_classes=3)
        ctx = net.encode(X, y, slots=slots, n_classes=3, chunk_cells=1)
        cache_parity(net, ctx, ref, errors)
        errors(net.predict_logits(ctx, Xt), plain.predict_logits(ref, Xt), "logits", atol=2e-5)


@torch.inference_mode()
def test_host_scaling_owns_cpu_float_copy(errors):
    plain = _model({}).double().folded()
    net = VulkanLightPFN(plain)
    scales = [b.attn.scaling for b in plain.icl.blocks] + [plain.decoder.scaling]
    for sc, gpu in zip(scales, [b.scaling for b in net.icl_blocks] + [net.dec_scaling], strict=True):
        assert next(sc.base.parameters()).dtype == torch.float64
        assert next(gpu.base_module.parameters()).device.type == "cpu"
        assert next(gpu.base_module.parameters()).dtype == torch.float32
        expected = sc.base(torch.tensor([[np.log(7)]], dtype=torch.float64)).float().view(-1)
        sc.base.to("meta")  # emulate a base on another device; the host copy must stay independent
        errors(net.eng.download(gpu.base(7), (gpu.H * gpu.hd,)), expected, "scaling_base", atol=1e-6)


@pytest.mark.parametrize("cfg", [dict(col_heads=32), dict(row_heads=32), dict(icl_heads=128),
                                 dict(decoder_heads=128, label_slots=2), dict(n_ecdf_freq=32), dict(n_inducing=0)])
def test_unsupported_kernel_dimensions(cfg):
    with pytest.raises(NotImplementedError, match="head dimensions|n_ecdf_freq|inducing token"):
        VulkanLightPFN(_model(cfg))


@pytest.mark.parametrize("shape", [(0, 2, 1), (1, 2, 0)])
def test_empty_batch_or_features(shape, networks):
    _, _, net = networks("default")
    with pytest.raises(ValueError, match="nonempty batch"):
        net.encode(torch.zeros(shape), torch.zeros(shape[:2], dtype=torch.long), n_classes=2)


def test_engine_parameter_and_allocation_limits(monkeypatch):
    from lightpfn.vulkan.engine import Engine

    eng = Engine()
    for params in ([-1], [1 << 32], [0] * 64):
        with pytest.raises(ValueError, match="u32"):
            eng.dispatch(None, {}, params, 1)
    with pytest.raises(ValueError, match="u32"):
        eng.dispatch(None, {}, [], 1 << 32)
    monkeypatch.setattr(eng, "max_binding", 31)
    with pytest.raises(MemoryError):
        eng.empty(5)  # 20 bytes round up to 32
    monkeypatch.setattr(eng, "max_binding", 15)
    with pytest.raises(MemoryError):
        eng.upload(np.zeros(0, np.float32))
    assert not eng.ops


def test_engine_op_limit_and_parameter_alignment(errors, monkeypatch):
    from lightpfn.vulkan.engine import Engine, f32bits

    eng = Engine()
    monkeypatch.setattr(eng, "submit_ops", 2)
    monkeypatch.setattr(eng, "param_slot", 512)
    src = """
@group(0) @binding(1) var<storage, read_write> O: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) wg: vec3u) { O[P[2]] += pf(3u); }
"""
    pipe = eng.pipeline("op_limit", src)
    buf = eng.empty(4)
    for i in range(7):
        eng.dispatch(pipe, {1: buf}, [1, 0, i % 4, f32bits(i + 1)], 1)
        assert len(eng.ops) < 2
    errors(eng.download(buf, (4,)), torch.tensor([6., 8., 10., 4.]), "slots", atol=0)


@pytest.mark.parametrize("src", ["#else", "#endif", "#if FLAG\nfoo", "const N = {MISSING}u;"])
def test_invalid_shader_templates(src):
    from lightpfn.vulkan.kernels import render

    with pytest.raises(ValueError, match="shader template"):
        render(src)


def test_supplied_model_eval_on_vulkan_and_cpu():
    for device in ("cpu", "vulkan"):
        model = _model({}).train()
        clf = LightPFNClassifier(model=model, device=device, fold=False)
        assert clf.model is model and model.training
        clf._initialize_backend()
        assert not clf.model_.training and model.training


def test_engine_uses_device_limits(monkeypatch):
    from lightpfn.vulkan import engine

    real = engine.Engine()
    limits = dict(real.device.limits)
    limits.update({"max-storage-buffer-binding-size": 65536, "max-buffer-size": 131072,
                   "min-storage-buffer-offset-alignment": 512, "max-compute-workgroups-per-dimension": 1024})

    class Device:
        def __getattr__(self, name):
            return getattr(real.device, name)

    device = Device()
    device.limits = limits

    class Adapter:
        info, limits = real.info, real.adapter.limits

        def request_device_sync(self, **kwargs):
            return device

    monkeypatch.setattr(engine, "pick_adapter", lambda adapter=None: Adapter())
    eng = engine.Engine()
    assert eng.max_binding == 65536 and eng.max_groups == 1024 and eng.param_slot == 512
    assert eng.submit_ops * eng.param_slot <= eng.max_binding
    with pytest.raises(MemoryError):
        eng.empty(16385)
