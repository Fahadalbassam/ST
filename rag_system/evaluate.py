"""Small evaluation helpers for comparing retrieval methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .core import RAGSystem


@dataclass(frozen=True)
class Evaluation:
    method: str
    accuracy: float
    faithfulness: float
    retrieval_quality: float
    response_time_ms: float


def evaluate(
    system: RAGSystem,
    questions: Sequence[tuple[str, str]],
    method: str,
) -> Evaluation:
    """Evaluate exact answer coverage and evidence coverage for a method.

    ``questions`` contains (question, expected_terms).  This lightweight
    baseline is deterministic and can be replaced by human or model grading.
    """

    if not questions:
        raise ValueError("at least one question is required")
    correct = faithful = retrieved = 0
    total_ms = 0.0
    for question, expected in questions:
        answer = system.answer(question, method)
        answer_terms = set(answer.text.lower().split())
        expected_terms = set(expected.lower().split())
        correct += bool(expected_terms and expected_terms <= answer_terms)
        evidence = " ".join(source.document.text.lower() for source in answer.sources)
        faithful += bool(answer.text and any(term in evidence for term in expected_terms))
        retrieved += bool(answer.sources)
        total_ms += answer.response_time_ms
    count = len(questions)
    return Evaluation(
        method,
        correct / count,
        faithful / count,
        retrieved / count,
        total_ms / count,
    )
