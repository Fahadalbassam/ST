"""A small, dependency-free retrieval-augmented question answering system."""

from .core import (
    Answer,
    BM25Retriever,
    DenseRetriever,
    Document,
    HybridRetriever,
    RAGSystem,
    RetrievalResult,
)

__all__ = [
    "Answer",
    "BM25Retriever",
    "DenseRetriever",
    "Document",
    "HybridRetriever",
    "RAGSystem",
    "RetrievalResult",
]
