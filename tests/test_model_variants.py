"""CPU-only architecture, cache, budget and masking invariants (no training run)."""

import pytest
import torch
import torch.nn.functional as F
from dataclasses import replace

from lightpfn.model.ccmm import ccmm_forward, ccmm_task_loss, predict_masked, sample_mask
from lightpfn.model.lightpfn import Config, LightPFN

torch.set_num_threads(4)

VARIANTS = [
    ({"cell_embed": "rbf", "icl_ff_reallocation": 4}, 4_777_024),
    ({"ccmm": True, "icl_ff_reallocation": 5}, 4_776_784),
    ({"row_mode": "summary"}, 4_777_072),
    ({"row_mode": "summary", "row_refine": True, "icl_drop_blocks": 1}, 4_603_088),
    ({"row_mode": "summary", "row_refine": True, "icl_drop_blocks": 1,
      "ccmm": True, "cell_embed": "rbf", "icl_ff_reallocation": 9}, 4_602_752),
]


def task():
    g = torch.Generator().manual_seed(71)
    x = torch.randn(2, 13, 5, generator=g)
    x[0, :, 1] = float("nan")  # all-missing training column
    x[1, :, 2] = 1.0  # constant column
    x[1, 11, 4] = float("nan")
    y = torch.arange(13)[None].expand(2, -1).clone() % 3
    y[0] = torch.arange(13) % 2
    return dict(X=x, y=y, d=torch.tensor([3, 5]), n_classes=torch.tensor([2, 3]), n_train=8)


def randomized(kw):
    torch.manual_seed(35)
    model = LightPFN(Config(**kw))
    with torch.no_grad():
        for p in model.parameters():
            # Initialized residual projections are zero; activate every attention/MLP.
            p.normal_(std=0.10)
    return model


@pytest.mark.parametrize("kw,count", VARIANTS)
def test_budget_and_finite_backward(kw, count):
    model = randomized(kw).train()
    assert sum(p.numel() for p in model.parameters()) == count
    assert 4_800_000 * .95 <= count <= 4_800_000 * 1.05
    mb = task()
    logits = model(mb["X"], mb["y"][:, :8], d=mb["d"], n_classes=3, has_padding=True)
    loss = F.cross_entropy(logits.transpose(1, 2), mb["y"][:, 8:])
    if kw.get("ccmm"):
        logits_masked, aux = ccmm_forward(model, mb, scheme="cells")
        assert logits_masked.shape == (2, 5, 3)
        assert aux.shape == (2,) and torch.isfinite(aux).all()
        loss = loss + aux.mean()
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


@pytest.mark.parametrize("kw,count", VARIANTS)
@pytest.mark.parametrize("training", [False, True])
@torch.no_grad()
def test_cache_chunks_independence_and_padding(kw, count, training):
    model = randomized(kw).train(training)
    mb = task()
    x, y, d = mb["X"], mb["y"][:, :8], mb["d"]
    base = model(x, y, d=d, n_classes=3)
    ctx = model.encode(x[:, :8], y, d=d, n_classes=3, has_padding=True)
    assert torch.equal(base, model.predict_logits(ctx, x[:, 8:]))
    blocks = torch.cat([model.predict_logits(ctx, x[:, 8 + i:8 + i + 2]) for i in range(0, 5, 2)], 1)
    torch.testing.assert_close(base, blocks, rtol=2e-5, atol=2e-6)
    changed = x[:, 8:].clone()
    changed[:, 2:] = 100 * torch.randn_like(changed[:, 2:])
    out = model.predict_logits(ctx, changed)
    torch.testing.assert_close(base[:, :2], out[:, :2], rtol=0, atol=0)
    perm = torch.tensor([4, 0, 2, 1, 3])
    torch.testing.assert_close(model.predict_logits(ctx, x[:, 8:][:, perm]), base[:, perm], rtol=2e-5, atol=2e-6)
    padded = torch.cat([x, torch.full((2, 13, 3), float("nan"))], 2)
    padded[0, :, 3:5] = 1000.0  # padding of the narrower task must be ignored too
    torch.testing.assert_close(model(padded, y, d=d, n_classes=3, has_padding=True), base, rtol=2e-5, atol=2e-6)
    for i, width in enumerate(d):
        single = model(x[i:i + 1, :, :width], y[i:i + 1], n_classes=3)
        torch.testing.assert_close(single, base[i:i + 1], rtol=2e-5, atol=2e-6)
    if kw.get("row_refine"):
        assert ctx.col_refine_kv is not None
        assert len(ctx.col_refine_kv) == model.cfg.col_blocks
        assert ctx.col_refine_kv[0][0].shape[2] == model.cfg.n_inducing
    else:
        assert ctx.col_refine_kv is None


@pytest.mark.parametrize("kw,count", [VARIANTS[1], VARIANTS[4],
    ({"ccmm": True, "cell_embed": "rbf", "icl_ff_reallocation": 9}, 4_776_736)])
@torch.no_grad()
def test_ccmm_no_hidden_value_leak_anywhere(kw, count):
    model = randomized(kw).eval()
    mb = task()
    x, y = mb["X"], mb["y"][:, :8]
    ctx = model.encode(x[:, :8], y, d=mb["d"], n_classes=3)
    mask = torch.zeros_like(x[:, 8:], dtype=torch.bool)
    mask[0, 0, 0] = True
    mask[1, 2, 3] = True
    original = x[:, 8:].clone()
    base = predict_masked(model, ctx, original, mask)
    # Compare ALL cells, including grouped neighbors and summaries, as well as y.
    for value in (12345., -98765., float("nan")):
        changed = original.masked_fill(mask, value)
        new = predict_masked(model, ctx, changed, mask)
        for before, after in zip(base, new):
            assert torch.equal(before, after)
    embeddings = model.cells(original, ctx.stats, ctx.d, mask, model.ccmm_mask_token)
    assert torch.equal(embeddings[mask], model.ccmm_mask_token.expand(int(mask.sum()), -1))
    changed = original.clone()
    changed[1, 0, 0] += 15  # visible values must still matter (non-vacuous test)
    assert not torch.equal(base[1], predict_masked(model, ctx, changed, mask)[1])
    # Masked path also preserves test-row independence and chunked queries.
    blocks = [predict_masked(model, ctx, original[:, i:i + 2], mask[:, i:i + 2]) for i in range(0, 5, 2)]
    for j in range(2):
        torch.testing.assert_close(torch.cat([b[j] for b in blocks], 1), base[j], rtol=2e-5, atol=2e-6)
    changed[:, 3:] = 200.
    independent = predict_masked(model, ctx, changed, mask)
    for j in range(2):
        torch.testing.assert_close(independent[j][:, 1:3], base[j][:, 1:3], rtol=0, atol=0)


@pytest.mark.parametrize("scheme", ["cells", "columns", "blocks", "mixed"])
def test_ccmm_schemes_quota_missing_and_padding(scheme):
    x = torch.arange(2 * 7 * 6).float().reshape(2, 7, 6)
    x[1, 2, 1] = float("nan")
    d = torch.tensor([3, 6])
    a = sample_mask(x, d, [.0, .25], scheme, torch.Generator().manual_seed(3))
    b = sample_mask(x, d, [.0, .25], scheme, torch.Generator().manual_seed(3))
    assert torch.equal(a, b) and not a[0].any() and a[1].any()
    assert not (a & ~torch.isfinite(x)).any()
    assert not (a & (torch.arange(6)[None, None] >= d[:, None, None])).any()
    full = sample_mask(x, d, 1., scheme)
    valid = torch.isfinite(x) & (torch.arange(6)[None, None] < d[:, None, None])
    assert torch.equal(full, valid)
    no_missing = torch.ones(1, 7, 6)
    structured = sample_mask(no_missing, fraction=.25, scheme=scheme, generator=torch.Generator().manual_seed(3))[0]
    if scheme == "cells":
        assert int(structured.sum()) == 11
    elif scheme == "columns":
        assert torch.equal(structured, structured[:1].expand_as(structured))
        assert int(structured[0].sum()) == 2
    elif scheme == "blocks":
        rows, cols = structured.nonzero(as_tuple=True)
        assert structured[rows.min():rows.max() + 1, cols.min():cols.max() + 1].all()
        assert int(structured.sum()) >= 11


def test_ccmm_loss_bins_train_only_and_zero_quota():
    model = randomized({"ccmm": True}).train()
    mb = task()
    slots = torch.arange(16)[None].expand(2, -1)
    ctx = model.encode(mb["X"][:, :8], mb["y"][:, :8], d=mb["d"], slots=slots, n_classes=3)
    mask = sample_mask(mb["X"][:, 8:], mb["d"], 1.)
    logits, aux = ccmm_forward(model, mb, slots=slots, mask=mask)
    target_logits, bins = predict_masked(model, ctx, mb["X"][:, 8:], mask)
    _, rank, _ = model.cells.normalize(mb["X"][:, 8:], ctx.stats)
    targets = (rank * 32).long().clamp(0, 31)
    expected = torch.stack([F.cross_entropy(bins[b][mask[b]], targets[b][mask[b]]) for b in range(2)])
    torch.testing.assert_close(aux, expected)
    torch.testing.assert_close(logits[1], target_logits[1])
    assert (logits[0, :, 2] == -1e4).all()
    out, zero = ccmm_forward(model, mb, slots=slots, mask_fraction=0.)
    assert torch.equal(zero, torch.zeros(2))
    torch.testing.assert_close(out[1], model(mb["X"], mb["y"][:, :8], d=mb["d"], slots=slots, n_classes=3)[1])
    # Empty observed targets remain finite and connected to autograd.
    missing = dict(mb, X=mb["X"].clone())
    missing["X"][:, 8:] = float("nan")
    _, empty = ccmm_forward(model, missing, slots=slots)
    assert torch.equal(empty, torch.zeros(2)) and empty.requires_grad
    combined = ccmm_task_loss(model, mb, "cpu", 16, .1)
    assert combined.shape == (2,) and torch.isfinite(combined).all()


@torch.no_grad()
def test_ccmm_known_ecdf_bins_ties_and_boundaries():
    model = randomized({"ccmm": True}).eval()
    # Train ECDF: four observations, including a tie. Test outliers MUST NOT
    # alter its mid-ranks; the reconstruction bins are specified independently.
    x = torch.tensor([0., 1., 1., 3., -10., 0., 1., 2., 3., 10., float("nan")]).view(1, 11, 1)
    y = (torch.arange(11) % 2)[None]
    mb = dict(X=x, y=y, d=torch.tensor([1]), n_classes=torch.tensor([2]), n_train=4)
    slots = torch.arange(16)[None]
    mask = torch.isfinite(x[:, 4:])
    ctx = model.encode(x[:, :4], y[:, :4], d=mb["d"], slots=slots, n_classes=2)
    _, bins = predict_masked(model, ctx, x[:, 4:], mask)
    _, loss = ccmm_forward(model, mb, slots=slots, mask=mask)
    expected = F.cross_entropy(bins[mask], torch.tensor([0, 4, 16, 24, 28, 31]))
    torch.testing.assert_close(loss[0], expected)


@pytest.mark.parametrize("kw,count", [VARIANTS[1], VARIANTS[4]])
@torch.no_grad()
def test_ccmm_head_can_be_removed_for_inference(kw, count):
    model = randomized(kw).eval()
    mb = task()
    base = model(mb["X"], mb["y"][:, :8], d=mb["d"], n_classes=3)
    inference = LightPFN(replace(model.cfg, ccmm=False)).eval()
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("ccmm_")}
    inference.load_state_dict(state, strict=True)
    assert not hasattr(inference, "ccmm_head") and not hasattr(inference, "ccmm_mask_token")
    assert sum(p.numel() for p in inference.parameters()) == count - 2272
    out = inference(mb["X"], mb["y"][:, :8], d=mb["d"], n_classes=3)
    assert torch.equal(base, out)


@torch.no_grad()
def test_summary_and_refinement_attention_shapes_and_train_only_gather():
    model = randomized(VARIANTS[3][0]).eval()
    mb = task()
    row_shapes, refine_shapes, col_reads = [], [], []
    handles = []
    for blk in model.row.blocks:
        handles.append(blk.register_forward_pre_hook(lambda mod, args: row_shapes.append((args[0].shape[1], args[1].shape[2]))))
    for blk in list(model.refine.broadcast) + list(model.refine.gather):
        handles.append(blk.register_forward_pre_hook(lambda mod, args: refine_shapes.append((args[0].shape[1], args[1].shape[2]))))
    for blk in model.col_refine.ind_blocks:
        handles.append(blk.register_forward_pre_hook(lambda mod, args: col_reads.append(args[1].shape[2])))
    try:
        ctx = model.encode(mb["X"][:, :8], mb["y"][:, :8], d=mb["d"], n_classes=3)
        model.predict_logits(ctx, mb["X"][:, 8:])
        model.predict_logits(ctx, mb["X"][:, 8:10])
    finally:
        for h in handles:
            h.remove()
    assert row_shapes == [(4, 9)] * 9  # 3 layers, 3 calls, four summary queries
    assert set(refine_shapes) == {(5, 4), (4, 5)}
    assert col_reads == [8, 8]  # the second inducing stage never reads test cells
    assert set(map(id, model.col.parameters())).isdisjoint(map(id, model.col_refine.parameters()))
    assert model.refine.summary is not model.row.cls


@pytest.mark.parametrize("kw", [
    {"cell_embed": "bad"}, {"row_mode": "bad"}, {"icl_drop_blocks": 8},
    {"icl_ff_reallocation": 512}, {"row_refine_rounds": 0}, {"ccmm_mask_fraction": -1},
])
def test_invalid_config(kw):
    with pytest.raises(ValueError):
        Config(**kw)


def test_attention_batch_chunks_match(monkeypatch):
    # CUDA kernels fail above 65535 batch rows, so attention runs in batch chunks: same logits and grads.
    from lightpfn.model import layers

    mb = task()
    out = []
    for limit in (layers.SDPA_MAX_BATCH, 3):
        monkeypatch.setattr(layers, "SDPA_MAX_BATCH", limit)
        model = randomized(VARIANTS[3][0]).train()
        logits = model(mb["X"], mb["y"][:, :8], d=mb["d"], n_classes=3, has_padding=True)
        F.cross_entropy(logits.transpose(1, 2), mb["y"][:, 8:]).backward()
        out.append((logits.detach(), [p.grad for p in model.parameters()]))
    torch.testing.assert_close(out[1][0], out[0][0], rtol=1e-5, atol=1e-6)
    for a, b in zip(out[1][1], out[0][1]):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
