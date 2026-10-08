"""LightPFN: compact tabular inference, pretrained only on synthetic data."""

__version__ = "1.0.0"
__all__ = ["Config", "LightPFN", "LightPFNClassifier", "load_model", "load_pretrained", "save_model", "__version__"]


def __getattr__(name):
    if name in ("Config", "LightPFN"):
        from lightpfn.model import lightpfn
        value = getattr(lightpfn, name)
    elif name == "LightPFNClassifier":
        try:
            from lightpfn.sklearn import LightPFNClassifier
        except ModuleNotFoundError as exc:
            if exc.name != "sklearn":
                raise
            raise ImportError("LightPFNClassifier requires scikit-learn: `pip install scikit-learn`.") from exc
        value = LightPFNClassifier
    elif name in ("load_model", "load_pretrained", "save_model"):
        from lightpfn import checkpoint
        value = getattr(checkpoint, name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
