"""Categorical adapter: identity at initialization, untouched numerical tables, leak-free ordered
statistics, consistent cached/chunked/batched inference, and gradients reaching only the adapter."""

import numpy as np
import pytest
import torch

from lightpfn import LightPFNClassifier
from lightpfn.model.catstats import N_COUNT, context_stats, query_stats
from lightpfn.model.lightpfn import CellEmbedder, Config, LightPFN

torch.set_num_threads(4)

B4 = dict(row_refine=True, row_mode="summary", icl_drop_blocks=1)


def models(seed=0, **kw):
    """A plain model and an adapter model holding the same weights, adapter at the identity."""
    torch.manual_seed(seed)
    plain = LightPFN(Config(**(B4 | kw))).eval()
    with torch.no_grad():  # zero-initialized output layers would make an untrained model constant
        for p in plain.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    adapted = LightPFN(Config(**(B4 | kw), cat_adapter=True)).eval()
    missing, unexpected = adapted.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and all("cat_" in k for k in missing)
    adapted.init_cat_adapter()
    return plain, adapted


def table(seed=0, B=2, n=60, M=17, m=7, C=3, ncat=3, levels=5):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(B, n + M, m, generator=g)
    X[:, :, :ncat] = torch.randint(0, levels, (B, n + M, ncat), generator=g).float()
    X[0, 3, 0] = float("nan")
    X[1, n + 2, 1] = float("nan")
    y = torch.randint(0, C, (B, n), generator=g)
    y[:, :C] = torch.arange(C)
    cat = torch.zeros(B, m, dtype=torch.bool)
    cat[:, :ncat] = True
    return X[:, :n], y, X[:, n:], cat, C


def test_parameter_budget():
    plain, adapted = models()
    extra = sum(p.numel() for p in adapted.parameters()) - sum(p.numel() for p in plain.parameters())
    assert extra == sum(p.numel() for p in adapted.adapter_parameters()) == 6384


def test_identity_at_init_and_numeric_tables_bit_identical():
    plain, adapted = models()
    X, y, Xt, cat, C = table()
    with torch.no_grad():
        ref = plain.predict_logits(plain.encode(X, y, n_classes=C), Xt)
        # no categorical column: the plain path, bit for bit
        out = adapted.predict_logits(adapted.encode(X, y, n_classes=C, cat=torch.zeros_like(cat)), Xt)
        assert torch.equal(out, ref)
        out = adapted.predict_logits(adapted.encode(X, y, n_classes=C), Xt)
        assert torch.equal(out, ref)
        # categorical columns, adapter at init: the same function up to float rounding
        out = adapted.predict_logits(adapted.encode(X, y, n_classes=C, cat=cat), Xt)
        torch.testing.assert_close(out, ref, atol=2e-5, rtol=0)


def test_context_stats_leak_free_and_exact():
    X, y, Xt, cat, C = table(n=40, ncat=2, levels=3)
    perm = torch.stack([torch.randperm(40, generator=torch.Generator().manual_seed(b)) for b in range(2)])
    feats, tab = context_stats(X, y, cat, C, smooth=1.0, perm=perm)
    assert feats.shape == (2, 40, X.shape[2], C + N_COUNT)
    assert torch.equal(feats[:, :, 2:], torch.zeros_like(feats[:, :, 2:]))  # numerical columns
    prior = torch.nn.functional.one_hot(y, C).float().mean(1)
    for b in range(2):
        rank = torch.empty(40, dtype=torch.long)
        rank[perm[b]] = torch.arange(40)
        for j in range(2):
            key = torch.nan_to_num(X[b, :, j], nan=1e9)
            for i in range(40):
                same = (key == key[i]) & (rank < rank[i])  # earlier rows of the category, never row i
                counts = torch.nn.functional.one_hot(y[b, same], C).float().sum(0)
                used = same.sum().float()
                delta = (counts - used * prior[b]) / (used + 1.0)
                torch.testing.assert_close(feats[b, i, j, :C], delta)
                torch.testing.assert_close(feats[b, i, j, C], torch.log1p(used) / 4)
                torch.testing.assert_close(feats[b, i, j, C + 1], torch.log1p((key == key[i]).sum().float() - 1) / 4)
    # one row's label reaches that row's own statistics only through the class prior, which is the same
    # for every row: the counts behind them do not change
    y2 = y.clone()
    y2[:, 5] = (y2[:, 5] + 1) % C
    feats2, _ = context_stats(X, y2, cat, C, smooth=1.0, perm=perm)
    prior2 = torch.nn.functional.one_hot(y2, C).float().mean(1)
    used = torch.expm1(feats[:, 5, :2, C] * 4)[..., None]
    counts = feats[:, 5, :2, :C] * (used + 1) + used * prior[:, None]
    counts2 = feats2[:, 5, :2, :C] * (used + 1) + used * prior2[:, None]
    torch.testing.assert_close(counts2, counts)
    assert torch.equal(feats2[:, 5, :, C:], feats[:, 5, :, C:])


def test_query_stats_use_every_context_row():
    X, y, Xt, cat, C = table(n=40, ncat=2, levels=3)
    feats, tab = context_stats(X, y, cat, C)
    q = query_stats(tab, Xt)
    prior = torch.nn.functional.one_hot(y, C).float().mean(1)
    for b in range(2):
        for j in range(2):
            key = torch.nan_to_num(X[b, :, j], nan=1e9)
            qkey = torch.nan_to_num(Xt[b, :, j], nan=1e9)
            for i in range(Xt.shape[1]):
                same = key == qkey[i]
                counts = torch.nn.functional.one_hot(y[b, same], C).float().sum(0)
                used = same.sum().float()
                torch.testing.assert_close(q[b, i, j, :C], (counts - used * prior[b]) / (used + 1.0))
                torch.testing.assert_close(q[b, i, j, C + 1], torch.log1p(used) / 4)
    unseen = Xt.clone()
    unseen[:, :, :2] = 99.0
    assert torch.equal(query_stats(tab, unseen), torch.zeros_like(q))


def trained_adapter(adapted):
    with torch.no_grad():
        for p in adapted.adapter_parameters():
            p.add_(torch.randn_like(p) * 0.3)
    return adapted


def test_chunked_cached_and_padded_inference_agree():
    _, adapted = models(1)
    adapted = trained_adapter(adapted)
    X, y, Xt, cat, C = table(seed=3)
    with torch.no_grad():
        ref = adapted.predict_logits(adapted.encode(X, y, n_classes=C, cat=cat), Xt)
        ctx = adapted.encode(X, y, n_classes=C, cat=cat, chunk_cells=50)
        torch.testing.assert_close(adapted.predict_logits(ctx, Xt, chunk_cells=30), ref, atol=2e-5, rtol=0)
        # zero-padded features with d: the same logits
        pad = lambda t: torch.cat([t, torch.zeros(*t.shape[:2], 2)], 2)  # noqa: E731
        d = torch.full((2,), X.shape[2])
        catp = torch.cat([cat, torch.zeros(2, 2, dtype=torch.bool)], 1)
        out = adapted.predict_logits(adapted.encode(pad(X), y, d=d, n_classes=C, cat=catp), pad(Xt))
        torch.testing.assert_close(out, ref, atol=2e-5, rtol=0)
        # the trained adapter changes the categorical tables
        plain = adapted.predict_logits(adapted.encode(X, y, n_classes=C), Xt)
        assert (plain - ref).abs().max() > 1e-3


def test_only_adapter_gets_gradients_when_frozen():
    _, adapted = models(2)
    adapted.train()
    keep = {id(p) for p in adapted.adapter_parameters()}
    for p in adapted.parameters():
        p.requires_grad_(id(p) in keep)
    X, y, Xt, cat, C = table(seed=4)
    logits = adapted(torch.cat([X, Xt], 1), y, n_classes=C, cat=cat)
    logits.logsumexp(-1).mean().backward()
    grads = {n for n, p in adapted.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert grads and all("cat_" in n for n in grads)
    assert any("cat_stats.ts" in n for n in grads)


def test_sklearn_wrapper_permutes_the_mask_with_the_features():
    _, adapted = models(5)
    adapted = trained_adapter(adapted)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 5)).astype(np.float32)
    X[:, 1] = rng.integers(0, 4, 80)
    y = (X[:, 1] % 2).astype(int)
    cat = np.array([False, True, False, False, False])
    clf = LightPFNClassifier(model=adapted, device="cpu", n_estimators=3, batch_cells=10**6).fit(X, y, cat)
    one = [LightPFNClassifier(model=adapted, device="cpu", n_estimators=3, batch_cells=0).fit(X, y, cat)]
    np.testing.assert_allclose(clf.predict_proba(X), one[0].predict_proba(X), atol=1e-5)
    plain = LightPFNClassifier(model=adapted, device="cpu", n_estimators=3).fit(X, y)
    assert np.abs(plain.predict_proba(X) - clf.predict_proba(X)).max() > 1e-4
    # a mask without categorical columns is the plain model
    none = LightPFNClassifier(model=adapted, device="cpu", n_estimators=3).fit(X, y, np.zeros(5, bool))
    np.testing.assert_array_equal(none.predict_proba(X), plain.predict_proba(X))


def test_config_validation():
    with pytest.raises(ValueError, match="fourier"):
        Config(cat_adapter=True, cell_embed="rbf")
    with pytest.raises(ValueError, match="cat_smooth"):
        Config(cat_adapter=True, cat_smooth=0)


def naive_stats(x, y, cat, C, perm, query, smooth):
    """Independent reference: match categories directly, never sort or accumulate prefixes."""
    B, n, m = x.shape
    ctx = torch.zeros(B, n, m, C + N_COUNT)
    qry = torch.zeros(B, query.shape[1], m, C + N_COUNT)
    for b in range(B):
        prior = torch.nn.functional.one_hot(y[b], C).float().mean(0)
        rank = torch.empty(n, dtype=torch.long)
        rank[perm[b]] = torch.arange(n)
        for j in range(m):
            if not cat[b, j]:
                continue
            for values, out, ordered in ((x[b], ctx[b], True), (query[b], qry[b], False)):
                for i in range(len(values)):
                    value = values[i, j]
                    same = torch.isnan(x[b, :, j]) if torch.isnan(value) else x[b, :, j] == value
                    used = same & (rank < rank[i]) if ordered else same
                    counts = torch.nn.functional.one_hot(y[b, used], C).float().sum(0)
                    nu = used.sum().float()
                    nf = same.sum().float() - int(ordered)
                    out[i, j] = torch.cat([(counts - nu * prior) / (nu + smooth),
                                          torch.stack([torch.log1p(nu) / 4, torch.log1p(nf) / 4, nf / n])])
    return ctx, qry


@pytest.mark.parametrize("n,B,kind", [(1, 1, "missing"), (1, 3, "random"), (9, 3, "random"),
                                     (9, 1, "single"), (9, 3, "missing")])
def test_stats_edge_cases_against_naive(n, B, kind):
    for seed in range(4):
        g = torch.Generator().manual_seed(seed)
        x = torch.randint(0, 4, (B, n, 5), generator=g).float()
        q = torch.randint(0, 6, (B, 4, 5), generator=g).float()
        x[0, :, 0] = float("nan")
        q[:, 0] = float("nan")
        if kind == "missing":
            x.fill_(float("nan"))
        elif kind == "single":
            x.fill_(2)
        y = torch.randint(0, 2, (B, n), generator=g)  # eight classes absent
        cat = torch.ones(B, 5, dtype=torch.bool)
        if B > 1:
            cat[0] = False
            cat[1, 1:] = False
        perm = torch.stack([torch.randperm(n, generator=g) for _ in range(B)])
        feats, tab = context_stats(x, y, cat, 10, smooth=2.3, perm=perm)
        ref, qref = naive_stats(x, y, cat, 10, perm, q, 2.3)
        torch.testing.assert_close(feats, ref)
        torch.testing.assert_close(query_stats(tab, q), qref)


def test_no_categories_and_empty_queries():
    x = torch.zeros(1, 1, 3)
    y = torch.zeros(1, 1, dtype=torch.long)
    assert context_stats(x, y, torch.zeros(1, 3, dtype=torch.bool), 10) == (None, None)
    _, tab = context_stats(x, y, torch.ones(1, 3, dtype=torch.bool), 10)
    assert query_stats(tab, x[:, :0]).shape == (1, 0, 3, 13)


@pytest.mark.parametrize("chunk", [None, 11])
@torch.no_grad()
def test_identity_and_trained_adapter_folded_batched_heterogeneous_padding(chunk):
    plain, adapted = models(8)
    X, y, Xt, cat, C = table(seed=7, n=11, M=5, m=5)
    d = torch.tensor([3, 5])
    X[0, :, 3:] = Xt[0, :, 3:] = 0
    cat[0] = False  # one numerical task in a mixed batch
    cat[0, 3:] = True  # padding falsely marked categorical: must be ignored
    slots = torch.stack([torch.randperm(16), torch.randperm(16)])
    kw = dict(d=d, n_classes=C, slots=slots, chunk_cells=chunk)
    ref = plain.predict_logits(plain.encode(X, y, **kw), Xt, chunk_cells=chunk)
    out = adapted.predict_logits(adapted.encode(X, y, cat=cat, **kw), Xt, chunk_cells=chunk)
    torch.testing.assert_close(out, ref, atol=2e-5, rtol=0)
    for mask in (None, torch.zeros_like(cat)):
        out = adapted.predict_logits(adapted.encode(X, y, cat=mask, **kw), Xt, chunk_cells=chunk)
        assert torch.equal(out, ref)
    trained_adapter(adapted)
    ref = adapted.predict_logits(adapted.encode(X, y, cat=cat, **kw), Xt, chunk_cells=chunk)
    folded = adapted.folded()
    out = folded.predict_logits(folded.encode(X, y, cat=cat, **kw), Xt, chunk_cells=chunk)
    torch.testing.assert_close(out, ref, atol=2e-5, rtol=0)
    for b in range(2):
        db = int(d[b])
        ctx = adapted.encode(X[b:b+1, :, :db], y[b:b+1], slots=slots[b:b+1], n_classes=C,
                             cat=cat[b:b+1, :db], chunk_cells=chunk)
        out = adapted.predict_logits(ctx, Xt[b:b+1, :, :db], chunk_cells=chunk)
        torch.testing.assert_close(out, ref[b:b+1], atol=2e-5, rtol=0)


@pytest.mark.parametrize("active", [False, True])
def test_allreduce_handles_unused_adapter_parameters(monkeypatch, active):
    from lightpfn.train import allreduce_grads

    params = [torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(3))]
    if active:
        params[0].grad = torch.tensor([1., 2.])
    calls = []
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, **kw: calls.append(tensor.clone()))
    allreduce_grads(params)
    assert len(calls) == (2 if active else 1)
    assert params[1].grad is None


def test_query_table_stores_counts_once_per_category():
    x = torch.arange(48).remainder(3).float().view(1, 48, 1).expand(2, -1, 4).clone()
    x[1, :, 1] = float("nan")
    y = torch.arange(48).remainder(2).expand(2, -1)
    cat = torch.tensor([[True, False, False, False], [True, True, False, False]])
    _, tab = context_stats(x, y, cat, 10)
    assert tab["keys"].shape == (2, 2, 3)
    assert tab["total"].shape == (2, 2, 3, 10)
    perm = torch.arange(48).expand(2, -1)
    query = x[:, :4].clone()
    query[:, 3] = 99
    _, ref = naive_stats(x, y, cat, 10, perm, query, 1.)
    torch.testing.assert_close(query_stats(tab, query), ref)


def test_allreduce_supplies_zeros_for_locally_unused_parameters(monkeypatch):
    from lightpfn.train import allreduce_grads

    params = [torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(3))]
    params[0].grad = torch.tensor([1., 2.])
    calls = []

    def reduce(tensor, **kw):
        calls.append(tensor.clone())
        if len(calls) == 1:
            tensor.fill_(1)  # both parameters are live on another rank
        else:
            tensor.add_(3)  # that rank contributes a gradient of 3 to every element

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    allreduce_grads(params)
    torch.testing.assert_close(params[0].grad, torch.tensor([4., 5.]))
    torch.testing.assert_close(params[1].grad, torch.full((3,), 3.))


def test_adapter_optimizer_ema_and_main_resume(tmp_path, monkeypatch):
    """Exercise the actual CLI loop on CPU; redirect only its CUDA setup/transport."""
    import copy
    import json
    import sys
    from contextlib import nullcontext
    from dataclasses import asdict
    from lightpfn import train

    cfg = Config(col_dim=16, col_blocks=1, row_blocks=1, n_inducing=4, n_cls=2, decoder_heads=2,
                 icl_blocks=2, n_thinking=2, row_mode="summary")
    torch.manual_seed(73)
    plain = LightPFN(cfg)
    with torch.no_grad():
        for p in plain.parameters():
            p.add_(torch.randn_like(p) * .05)
    initial = copy.deepcopy(plain.state_dict())
    source = tmp_path / "plain.pt"
    torch.save(dict(model=initial, config=asdict(cfg)), source)
    X = torch.arange(36).remainder(3).float().view(2, 6, 3)
    y = torch.arange(6).remainder(2).expand(2, -1)
    mb = dict(X=X, y=y, d=torch.tensor([2, 3]), n_classes=torch.tensor([2, 2]), n_train=4,
              cat_mask=torch.tensor([[True, False, False], [False, True, False]]))
    numeric = dict(mb, cat_mask=torch.zeros_like(mb["cat_mask"]))
    numeric["cat_mask"][0, 2] = True  # categorical flag only on padding: still a numerical batch

    class CPUModel(LightPFN):
        def to(self, device, *args, **kw):
            return super().to("cpu" if device == "cuda" else device, *args, **kw)

    zeros, load, loss = torch.zeros, torch.load, train.task_loss
    monkeypatch.setattr(train, "LightPFN", CPUModel)
    monkeypatch.setattr(torch, "zeros", lambda *args, **kw: zeros(*args, **dict(kw, device="cpu"))
                        if kw.get("device") == "cuda" else zeros(*args, **kw))
    monkeypatch.setattr(torch, "autocast", lambda *args, **kw: nullcontext())
    monkeypatch.setattr(torch, "load", lambda *args, **kw: load(*args, **dict(kw, map_location="cpu"))
                        if kw.get("map_location") == "cuda" else load(*args, **kw))
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda *args: None)
    monkeypatch.setattr(train, "task_loss", lambda model, batch, device, *args: loss(model, batch, "cpu", *args))
    monkeypatch.setattr(train, "DataLoader", lambda *args, **kw: iter([numeric, mb, mb]))
    monkeypatch.setattr(train, "lite_eval", lambda *args: pytest.fail("--no-lite must skip evaluation"))
    out = tmp_path / "run"
    conf = asdict(cfg) | dict(cat_adapter=True)
    argv = ["train", "--run", "cpu_review", "--output-dir", str(out), "--pools", "unused:1",
            "--steps", "3", "--stop-at", "2", "--groups-per-step", "1", "--warmup", "0",
            "--workers", "0", "--save-every", "1", "--eval-every", "1", "--no-lite",
            "--weight-decay", ".2", "--clip", ".01", "--trainable", "adapter", "--init", str(source),
            "--config", json.dumps(conf)]
    monkeypatch.setattr(sys, "argv", argv)
    train.main()
    ck = torch.load(out / "ckpt.pt", weights_only=True)
    assert ck["step"] == 2 and len(ck["opts"]) == 1
    assert len(ck["opts"][0]["state"]) == 4  # adapter only, no Muon optimizer
    assert all("cat_" in k or torch.equal(v, initial[k]) for k, v in ck["model"].items())
    assert all("cat_" in k or torch.equal(v, initial[k]) for k, v in ck["ema"].items())
    assert ck["model"]["cat_stats.ts.weight"].abs().sum() > 0
    first = torch.load(out / "ema_step000001.pt", weights_only=True)
    assert torch.count_nonzero(first["ema"]["cat_stats.ts.weight"]) == 0  # skipped batch
    assert (out / "ema_step000002.pt").exists()
    argv[argv.index("--stop-at") + 1] = "3"
    argv[argv.index("--init") + 1] = str(tmp_path / "does_not_exist.pt")
    monkeypatch.setattr(train, "DataLoader", lambda *args, **kw: iter([mb]))
    train.main()  # resumes the optimizer; nonexistent --init must be ignored
    resumed = torch.load(out / "ckpt.pt", weights_only=True)
    assert resumed["step"] == 3
    assert all("cat_" in k or torch.equal(v, initial[k]) for k, v in resumed["model"].items())
    assert all("cat_" in k or torch.equal(v, initial[k]) for k, v in resumed["ema"].items())
    assert not torch.equal(resumed["model"]["cat_stats.ts.weight"], ck["model"]["cat_stats.ts.weight"])
    assert {int(v["step"]) for v in resumed["opts"][0]["state"].values()} == {2}


def test_ccmm_cannot_silently_bypass_the_adapter():
    from lightpfn.train import freeze_for, task_loss

    cfg = Config(col_dim=16, col_blocks=1, row_blocks=1, n_inducing=4, n_cls=2, decoder_heads=2,
                 icl_blocks=2, n_thinking=2, ccmm=True, cat_adapter=True)
    adapted = LightPFN(cfg).train()
    freeze_for(adapted, "adapter")
    mb = dict(X=torch.zeros(1, 6, 2), y=torch.arange(6)[None] % 2,
              d=torch.tensor([2]), n_classes=torch.tensor([2]), n_train=4,
              cat_mask=torch.ones(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="CCMM.*categorical adapter"):
        task_loss(adapted, mb, "cpu", 16, ccmm_weight=.1)


@pytest.mark.parametrize("kw", [dict(row_refine=False, row_mode="self_attention", icl_drop_blocks=0),
                                dict(row_refine=False), {}])
@torch.no_grad()
def test_adapter_single_row_all_missing_and_absent_classes(kw):
    plain, adapted = models(11, **kw)
    x = torch.full((1, 1, 3), float("nan"))
    y = torch.zeros(1, 1, dtype=torch.long)
    cat = torch.ones(1, 3, dtype=torch.bool)
    ref = plain.predict_logits(plain.encode(x, y, n_classes=10), x)
    for chunk in (None, 1):
        ctx = adapted.encode(x, y, n_classes=10, cat=cat, chunk_cells=chunk)
        out = adapted.predict_logits(ctx, x, chunk_cells=chunk)
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out, ref, atol=2e-5, rtol=0)


def test_adapter_gradients_under_cpu_bfloat16_autocast():
    from lightpfn.train import freeze_for, task_loss

    cfg = Config(col_dim=16, col_blocks=1, row_blocks=1, n_inducing=4, n_cls=2, decoder_heads=2,
                 icl_blocks=2, n_thinking=2, cat_adapter=True)
    model = LightPFN(cfg).train()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * .05)
    model.init_cat_adapter()
    freeze_for(model, "adapter")
    mb = dict(X=torch.arange(18).remainder(4).float().view(1, 6, 3), y=torch.arange(6)[None] % 2,
              d=torch.tensor([3]), n_classes=torch.tensor([2]), n_train=4,
              cat_mask=torch.tensor([[True, False, True]]))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = task_loss(model, mb, "cpu", 16).mean()
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        else:
            assert p.grad is None, name


def test_adapter_does_not_double_the_fourier_activation_work(monkeypatch):
    cfg = Config(cat_adapter=True, n_ecdf_freq=0)
    cells = CellEmbedder(cfg)
    z = torch.randn(2, 4, 5, 3)
    catg = torch.rand(2, 1, 5, 3) > .5
    work = []
    sin = torch.Tensor.sin

    def record_sin(t):
        work.append(t.numel())
        return sin(t)

    monkeypatch.setattr(torch.Tensor, "sin", record_sin)
    out = cells.embed(z, torch.rand_like(z), torch.zeros_like(z), catg=catg)
    assert out.shape == (2, 4, 5, cfg.col_dim)
    assert sum(work) == z.numel() * cfg.n_freq  # one Fourier expansion per group member
