"""Command-line entry point: ``python -m rag_system``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import Document, RAGSystem


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidence-grounded RAG question answering")
    parser.add_argument("dataset", type=Path, help="JSONL file with id and text fields")
    parser.add_argument("question")
    parser.add_argument("--method", choices=("bm25", "dense", "hybrid"), default="hybrid")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    documents = [
        Document(str(item["id"]), str(item["text"]), item.get("metadata"))
        for item in map(json.loads, args.dataset.read_text(encoding="utf-8").splitlines())
        if item.get("text")
    ]
    answer = RAGSystem(documents).answer(args.question, args.method, args.top_k)
    print(answer.text)
    print("\nSources:")
    for source in answer.sources:
        print(f"- {source.document.id} ({source.score:.4f})")
    print(f"\nResponse time: {answer.response_time_ms:.2f} ms")


if __name__ == "__main__":
    main()
