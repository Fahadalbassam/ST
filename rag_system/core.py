"""Retrieval and evidence-grounded answer generation.

The implementation intentionally uses only the Python standard library.  Dense
retrieval is represented by TF-IDF vectors, which provides a deterministic
semantic-search baseline without requiring an external model or network access.
The retriever interface makes it possible to replace it with a real embedding
model for experiments.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Callable, Iterable, Protocol, Sequence


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "before", "does", "for", "how", "in", "is",
    "of", "on", "the", "to", "what", "when", "where", "which", "who",
}


def tokenize(text: str) -> list[str]:
    """Return normalized terms used by all built-in retrievers."""

    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: dict[str, str] | None = None


@dataclass(frozen=True)
class RetrievalResult:
    document: Document
    score: float
    method: str


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    sources: tuple[RetrievalResult, ...]
    response_time_ms: float


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        ...


class BM25Retriever:
    """Keyword retrieval using the Okapi BM25 ranking function."""

    def __init__(
        self,
        documents: Iterable[Document],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = list(documents)
        if not self.documents:
            raise ValueError("at least one document is required")
        self.k1, self.b = k1, b
        self._tokens = [tokenize(document.text) for document in self.documents]
        self._lengths = [len(tokens) for tokens in self._tokens]
        self._average_length = sum(self._lengths) / len(self._lengths) or 1.0
        self._term_frequency = [Counter(tokens) for tokens in self._tokens]
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            document_frequency.update(set(tokens))
        self._idf = {
            term: math.log(1 + (len(self.documents) - frequency + 0.5) /
                           (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if top_k <= 0:
            return []
        query_terms = tokenize(query)
        scores: list[RetrievalResult] = []
        for document, frequencies, length in zip(
            self.documents, self._term_frequency, self._lengths
        ):
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency:
                    denominator = frequency + self.k1 * (
                        1 - self.b + self.b * length / self._average_length
                    )
                    score += self._idf.get(term, 0.0) * (
                        frequency * (self.k1 + 1) / denominator
                    )
            scores.append(RetrievalResult(document, score, "bm25"))
        return sorted(scores, key=lambda result: result.score, reverse=True)[:top_k]


class DenseRetriever:
    """Deterministic dense-style retrieval using TF-IDF cosine similarity."""

    def __init__(self, documents: Iterable[Document]) -> None:
        self.documents = list(documents)
        if not self.documents:
            raise ValueError("at least one document is required")
        tokenized = [tokenize(document.text) for document in self.documents]
        document_frequency: Counter[str] = Counter()
        for tokens in tokenized:
            document_frequency.update(set(tokens))
        self._idf = {
            term: math.log((1 + len(self.documents)) / (1 + frequency)) + 1
            for term, frequency in document_frequency.items()
        }
        self._vectors = [self._vector(tokens) for tokens in tokenized]

    def _vector(self, tokens: Sequence[str]) -> dict[str, float]:
        counts = Counter(tokens)
        return {
            term: (1 + math.log(count)) * self._idf[term]
            for term, count in counts.items()
            if term in self._idf
        }

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        denominator = math.sqrt(sum(value * value for value in left.values()))
        denominator *= math.sqrt(sum(value * value for value in right.values()))
        if not denominator:
            return 0.0
        return sum(left.get(term, 0.0) * value for term, value in right.items()) / denominator

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if top_k <= 0:
            return []
        query_vector = self._vector(tokenize(query))
        return sorted(
            (
                RetrievalResult(
                    document, self._cosine(query_vector, vector), "dense"
                )
                for document, vector in zip(self.documents, self._vectors)
            ),
            key=lambda result: result.score,
            reverse=True,
        )[:top_k]


class HybridRetriever:
    """Weighted reciprocal-rank fusion of BM25 and dense retrieval."""

    def __init__(
        self,
        documents: Iterable[Document],
        dense_weight: float = 0.5,
    ) -> None:
        if not 0 <= dense_weight <= 1:
            raise ValueError("dense_weight must be between 0 and 1")
        documents = list(documents)
        self.bm25 = BM25Retriever(documents)
        self.dense = DenseRetriever(documents)
        self.dense_weight = dense_weight

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if top_k <= 0:
            return []
        count = len(self.bm25.documents)
        bm25 = self.bm25.retrieve(query, count)
        dense = self.dense.retrieve(query, count)
        bm25_rank = {result.document.id: rank for rank, result in enumerate(bm25)}
        dense_rank = {result.document.id: rank for rank, result in enumerate(dense)}
        weight = self.dense_weight
        fused = [
            RetrievalResult(
                document,
                (1 - weight) / (60 + bm25_rank[document.id])
                + weight / (60 + dense_rank[document.id]),
                "hybrid",
            )
            for document in self.bm25.documents
        ]
        return sorted(fused, key=lambda result: result.score, reverse=True)[:top_k]


def _extractive_answer(question: str, results: Sequence[RetrievalResult]) -> str:
    """Create an answer exclusively from retrieved evidence."""

    if not results:
        return "I could not find evidence to answer this question."
    terms = set(tokenize(question)) - _STOPWORDS
    candidates: list[tuple[float, str, str]] = []
    for result in results:
        sentences = re.split(r"(?<=[.!?])\s+", result.document.text.strip())
        for sentence in sentences:
            overlap = len(terms.intersection(tokenize(sentence)))
            if overlap:
                candidates.append((overlap, sentence, result.document.id))
    if not candidates:
        return "The retrieved sources do not contain enough evidence to answer this question."
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    seen: set[str] = set()
    for _, sentence, document_id in candidates:
        if sentence not in seen:
            selected.append(f"{sentence} [{document_id}]")
            seen.add(sentence)
        if len(selected) == 2:
            break
    return " ".join(selected)


class RAGSystem:
    """Retrieve evidence, then answer using only that evidence."""

    def __init__(
        self,
        documents: Iterable[Document],
        generator: Callable[[str, Sequence[RetrievalResult]], str] | None = None,
    ) -> None:
        self.documents = list(documents)
        if not self.documents:
            raise ValueError("at least one document is required")
        self.retrievers: dict[str, Retriever] = {
            "bm25": BM25Retriever(self.documents),
            "dense": DenseRetriever(self.documents),
            "hybrid": HybridRetriever(self.documents),
        }
        self.generator = generator or _extractive_answer

    def answer(self, question: str, method: str = "hybrid", top_k: int = 5) -> Answer:
        if not question.strip():
            raise ValueError("question must not be empty")
        if method not in self.retrievers:
            raise ValueError(f"unknown retrieval method: {method}")
        import time

        started = time.perf_counter()
        sources = tuple(self.retrievers[method].retrieve(question, top_k))
        text = self.generator(question, sources)
        elapsed = (time.perf_counter() - started) * 1000
        return Answer(question, text, sources, elapsed)
