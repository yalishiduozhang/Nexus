from importlib import import_module

from .embedder.text_retrieval import TextEmbedder, BaseEmbedderInferenceEngine
from .reranker.text_retrieval import TextReranker, BaseRerankerInferenceEngine

__all__ = [
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
    raise AttributeError(f"module 'Nexus.inference' has no attribute {name!r}")
