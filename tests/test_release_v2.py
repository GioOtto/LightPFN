"""Both release subfolders, immutable resolution, and offline placeholder guard."""
import json
from pathlib import Path

import pytest

from lightpfn import LightPFNClassifier, LightPFNRegressor, save_model
from lightpfn.checkpoint import load_pretrained, validate_pretrained_pin
from test_regressor_api import network, task


def test_placeholder_guard(monkeypatch):
    monkeypatch.delenv('LIGHTPFN_WEIGHTS_DIR', raising=False)
    monkeypatch.setattr('lightpfn.checkpoint.pretrained_spec', lambda: {'repo_id': 'ueuegio/LightPFN', 'revision': 'PENDING'})
    with pytest.raises(ValueError, match='placeholder'):
        validate_pretrained_pin()
    with pytest.raises(ValueError, match='placeholder'):
        load_pretrained()


@pytest.mark.parametrize('kind', ['classifier', 'regressor'])
def test_local_subfolders(tmp_path, monkeypatch, kind):
    model = network()
    if kind == 'classifier':
        from lightpfn import Config, LightPFN
        from dataclasses import asdict
        model = LightPFN(Config(**(asdict(model.cfg) | {'head': 'classification'}))).eval()
    save_model(model, tmp_path / kind)
    monkeypatch.setenv('LIGHTPFN_WEIGHTS_DIR', str(tmp_path))
    assert load_pretrained(task=kind).cfg == model.cfg
    x, y = task()
    cls = LightPFNClassifier if kind == 'classifier' else LightPFNRegressor
    r = cls(device='cpu', n_estimators=1).fit(x, (y > 2).astype(int) if kind == 'classifier' else y)
    assert len(r.predict(x[:3])) == 3


@pytest.mark.parametrize('kind', ['classifier', 'regressor'])
def test_hf_subfolders_same_commit(tmp_path, monkeypatch, kind):
    revision = 'a' * 40
    folder = save_model(network(), tmp_path / 'snapshots' / revision / kind)
    calls = []
    def download(filename, **options):
        calls.append(options.copy())
        assert options['subfolder'] == kind
        return str(folder / filename)
    monkeypatch.setattr('huggingface_hub.hf_hub_download', download)
    load_pretrained(repo_id='test/model', revision='main', task=kind, local_files_only=True)
    assert calls[0]['revision'] == 'main' and calls[1]['revision'] == revision


@pytest.mark.parametrize('revision', ['bd389ab5', 'v1.1.0'])
def test_legacy_revision_uses_root(tmp_path, monkeypatch, revision):
    from huggingface_hub.utils import EntryNotFoundError
    folder = save_model(network(), tmp_path / 'bd389ab5')
    calls = []
    def download(filename, **options):
        calls.append(options.copy())
        if 'subfolder' in options:  # the 1.x layout has no classifier/ or regressor/ folder
            raise EntryNotFoundError('missing')
        return str(folder / filename)
    monkeypatch.setattr('huggingface_hub.hf_hub_download', download)
    load_pretrained(repo_id='ueuegio/LightPFN', revision=revision)
    assert 'subfolder' in calls[0] and all('subfolder' not in c for c in calls[1:])
    assert calls[-1]['revision'] == 'bd389ab5'
    with pytest.raises(ValueError, match='no regression'):
        load_pretrained(repo_id='ueuegio/LightPFN', revision=revision, task='regressor')
