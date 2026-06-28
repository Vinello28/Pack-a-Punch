"""Inference module.

Engine classes are exposed lazily so importing this package does not pull heavy,
backend-specific dependencies (e.g. torch for the PyTorch engine) that a given
deployment may not have installed. The Triton gateway image, for instance, ships
without torch and only needs the Triton engine.
"""

from .server import app

__all__ = [
    "PyTorchInferenceEngine",
    "create_pytorch_engine",
    "TritonInferenceEngine",
    "create_triton_engine",
    "app",
]

_LAZY = {
    "PyTorchInferenceEngine": "pytorch_engine",
    "create_pytorch_engine": "pytorch_engine",
    "TritonInferenceEngine": "triton_engine",
    "create_triton_engine": "triton_engine",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    mod = importlib.import_module(f".{module}", __name__)
    return getattr(mod, name)
