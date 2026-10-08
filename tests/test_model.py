"""Invariance tests for the LightPFN model. Run: python -m pytest tests/test_model.py -q"""

import torch

from lightpfn.model.lightpfn import Config, LightPFN

torch.set_num_threads(4)


def _task(B=3, n_train=60, n_test=20, m=7, C=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(B, n_train + n_test, m, generator=g)
    X[0, 5, 2] = float("nan")
    X[1, :, 3] = 1.0  # constant column
    y = torch.randint(0, C, (B, n_train), generator=g)
    y[:, :C] = torch.arange(C)
    return X, y


def _model():
    torch.manual_seed(0)
    model = LightPFN(Config()).eval()
    for p in model.parameters():  # zero-initialized outputs would hide leakage: randomize them
        torch.nn.init.normal_(p, std=0.05)
    return model


def test_shapes_and_params():
    model = LightPFN(Config())
    n = sum(p.numel() for p in model.parameters())
    assert 3e6 < n < 6e6, n
    X, y = _task()
    out = model(X, y)
    assert out.shape == (3, 20, 4)
    assert torch.isfinite(out).all()


@torch.no_grad()
def test_test_rows_are_independent():
    model = _model()
    X, y = _task()
    base = model(X, y)
    X2 = X.clone()
    X2[:, 60 + 5 :] = torch.randn_like(X2[:, 60 + 5 :]) * 10  # change all test rows but the first 5
    out = model(X2, y)
    torch.testing.assert_close(out[:, :5], base[:, :5], rtol=1e-4, atol=1e-5)


@torch.no_grad()
def test_chunked_prediction_matches_forward():
    model = _model()
    X, y = _task()
    base = model(X, y)
    ctx = model.encode(X[:, :60], y)
    chunks = [model.predict_logits(ctx, X[:, 60 + i : 60 + i + 7]) for i in range(0, 20, 7)]
    torch.testing.assert_close(torch.cat(chunks, 1), base, rtol=1e-4, atol=1e-5)


@torch.no_grad()
def test_padded_features_are_ignored():
    model = _model()
    X, y = _task(m=5)
    base = model(X, y)
    Xp = torch.cat([X, torch.zeros(3, 80, 4)], dim=2)  # zero-padded to 9 features
    out = model(Xp, y, d=torch.tensor([5, 5, 5]))
    torch.testing.assert_close(out, base, rtol=1e-4, atol=1e-5)


@torch.no_grad()
def test_train_row_order_invariance():
    model = _model()
    X, y = _task()
    base = model(X, y)
    perm = torch.randperm(60)
    Xp = torch.cat([X[:, perm], X[:, 60:]], 1)
    out = model(Xp, y[:, perm])
    torch.testing.assert_close(out, base, rtol=1e-3, atol=1e-4)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
