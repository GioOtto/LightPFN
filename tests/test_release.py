"""Release checkpoints must be safe, exact, and usable without research dependencies."""

import json
import pickle
import subprocess
import sys
from dataclasses import asdict

import numpy as np
import pytest
import torch

from lightpfn import Config, LightPFN, load_model, save_model
from lightpfn.checkpoint import load_pretrained


def small_model():
    torch.manual_seed(13)
    model = LightPFN(Config(col_dim=16, col_blocks=1, col_heads=4, n_inducing=4,
                            row_blocks=1, row_heads=4, n_cls=2, icl_blocks=2,
                            icl_heads=4, decoder_heads=2, n_thinking=2, n_freq=4, n_ecdf_freq=2)).eval()
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    return model


def test_legacy_and_safetensors_roundtrip_predictions(tmp_path):
    model = small_model()
    path = tmp_path / "ema.pt"
    torch.save(dict(config=asdict(model.cfg), ema=model.state_dict()), path)
    legacy = load_model(path)
    release = save_model(legacy, tmp_path / "release")
    restored = load_model(release)
    assert restored.cfg == model.cfg
    assert set(json.loads((release / "config.json").read_text())) == set(asdict(model.cfg))
    g = torch.Generator().manual_seed(3)
    X = torch.randn(1, 22, 3, generator=g)
    X[0, 1, 0] = float("nan")
    y = torch.arange(16).remainder(3).unsqueeze(0)
    with torch.inference_mode():
        expected = model(X, y, n_classes=3)
        torch.testing.assert_close(legacy(X, y, n_classes=3), expected, rtol=0, atol=0)
        torch.testing.assert_close(restored(X, y, n_classes=3), expected, rtol=0, atol=0)


def test_pickle_payload_is_rejected_without_execution(tmp_path):
    class Payload:
        def __reduce__(self):
            return eval, ("__import__('pathlib').Path(%r).touch()" % str(tmp_path / "executed"),)
    path = tmp_path / "untrusted.pt"
    torch.save(dict(config={}, model=Payload()), path)
    with pytest.raises(pickle.UnpicklingError):
        load_model(path)
    assert not (tmp_path / "executed").exists()


def test_reject_malformed_and_folded_checkpoints(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save(dict(config={}, model={"not_a_tensor": 5}), path)
    with pytest.raises(ValueError, match="tensor state"):
        load_model(path)
    with pytest.raises(ValueError, match="folded"):
        save_model(small_model().folded(), tmp_path)


def test_hub_files_use_same_resolved_commit(tmp_path, monkeypatch):
    revision = "a" * 40
    folder = save_model(small_model(), tmp_path / "snapshots" / revision)
    calls = []
    def download(filename, **kwargs):
        calls.append((filename, kwargs))
        return str(folder / filename)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    load_pretrained(repo_id="test/model", revision="main", local_files_only=True)
    assert calls[0][1]["revision"] == "main"
    assert calls[1][1]["revision"] == revision
    assert all(options["local_files_only"] for _, options in calls)
    with pytest.raises(ValueError, match="revision"):
        load_pretrained(repo_id="test/other")


def test_core_import_does_not_load_optional_dependencies():
    code = """
import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'sklearn', 'tabicl', 'pandas', 'scipy', 'wgpu', 'huggingface_hub', 'safetensors'}:
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, Block())
from lightpfn import LightPFN, Config, load_model
assert LightPFN(Config()).cfg.label_slots == 16
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
