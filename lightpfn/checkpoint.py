"""Safe local checkpoints and revision-pinned Hugging Face downloads.

The release format is model.safetensors plus config.json. Legacy tensor/dict
checkpoints remain supported using PyTorch's restricted weights-only loader.
There is deliberately no fallback to unrestricted pickle.
"""

import json
from collections.abc import Mapping
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path

import torch

from lightpfn.model.lightpfn import Config, LightPFN


def _safetensors():
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise ImportError("Safetensors weights require `pip install safetensors`.") from exc
    return load_file, save_file


def _config(values):
    if not isinstance(values, Mapping):
        raise ValueError("Checkpoint config must be a mapping of architecture parameters.")
    values = dict(values)
    if "group_offsets" in values:
        values["group_offsets"] = tuple(values["group_offsets"])
    return Config(**values)


def load_model(path, device="cpu"):
    """Load a safetensors directory/file or a weights-only compatible .pt file.

    Safetensors files need config.json in the same directory. Full training
    checkpoints containing arbitrary Python objects are intentionally rejected.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "model.safetensors"
    if path.suffix == ".safetensors":
        load_file, _ = _safetensors()
        config = json.loads(path.with_name("config.json").read_text(encoding="utf-8"))
        state = load_file(str(path), device="cpu")
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, Mapping) or "config" not in checkpoint:
            raise ValueError("Checkpoint must contain 'config' and 'ema' or 'model' weights.")
        config = checkpoint["config"]
        state = checkpoint.get("ema", checkpoint.get("model"))
    if not isinstance(state, Mapping) or not state or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise ValueError("Checkpoint weights must be a nonempty tensor state dictionary.")
    model = LightPFN(_config(config))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def save_model(model, directory):
    """Export unfused inference weights and JSON config, excluding optimizer state."""
    if getattr(model, "is_folded", False):
        raise ValueError("Export the original model; folded weights use a different layout.")
    _, save_file = _safetensors()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().contiguous().clone() for key, value in model.state_dict().items()}
    save_file(state, str(directory / "model.safetensors"), metadata={"format": "pt"})
    (directory / "config.json").write_text(json.dumps(asdict(model.cfg), indent=2) + "\n", encoding="utf-8")
    return directory


def pretrained_spec():
    """Return the packaged model repository and immutable revision."""
    return json.loads(files("lightpfn").joinpath("pretrained.json").read_text(encoding="utf-8"))


def load_pretrained(*, repo_id=None, revision=None, device="cpu", cache_dir=None, local_files_only=False):
    """Load the released model, or an explicit repo and revision using cached HF authentication.

    A custom repository requires a revision. No remote Python code is executed.
    For private repositories authenticate with `hf auth login` or HF_TOKEN.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("Hugging Face downloads require `pip install huggingface-hub`.") from exc
    spec = pretrained_spec()
    repo_id = spec["repo_id"] if repo_id is None else repo_id
    if revision is None:
        if repo_id != spec["repo_id"]:
            raise ValueError("Provide revision when using a custom Hugging Face repository.")
        revision = spec["revision"]
    if not revision:
        raise ValueError("No pretrained revision configured; provide a checkpoint or explicit revision.")
    options = dict(repo_id=repo_id, revision=revision, cache_dir=cache_dir, local_files_only=local_files_only)
    config_path = Path(hf_hub_download(filename="config.json", **options))
    # Use the resolved commit from the snapshot path for both files, even when a
    # caller explicitly chose a mutable branch or tag.
    resolved = config_path.parent.name
    options["revision"] = resolved
    weights_path = Path(hf_hub_download(filename="model.safetensors", **options))
    return load_model(weights_path, device=device)
