from importlib import import_module

from .abc import AbsInferenceArguments
from .inference import TextEmbedder, BaseEmbedderInferenceEngine, TextReranker, BaseRerankerInferenceEngine

__all__ = [
    "AbsInferenceArguments",
    "TextEmbedder",
    "BaseEmbedderInferenceEngine",
    "TextReranker",
    "BaseRerankerInferenceEngine",
    "MultiModalEmbedder",
    "MultimodalEmbedder",
]


def __getattr__(name):
    if name in {"MultiModalEmbedder", "MultimodalEmbedder"}:
        module = import_module("Nexus.inference.embedder.multimodal_retrieval")
        return getattr(module, name, getattr(module, "MultimodalEmbedder"))
    raise AttributeError(f"module 'Nexus' has no attribute {name!r}")
