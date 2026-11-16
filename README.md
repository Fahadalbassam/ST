# CSC 403 — RAG Question Answering System

This repository contains a runnable, dependency-free RAG baseline for comparing
BM25 keyword retrieval, TF-IDF dense-style retrieval, and hybrid retrieval.
Answers are extractive and include the IDs of the retrieved evidence sources,
so they cannot silently rely on information outside the knowledge base.

## Run

Create a UTF-8 JSONL dataset with one `id` and `text` object per line:

```json
{"id": "intro", "text": "Retrieval augmented generation retrieves evidence before answering."}
```

Ask a question with any retrieval method:

```bash
python -m rag_system knowledge.jsonl "What happens before answering?" --method hybrid
```

The same question can be run with `bm25`, `dense`, and `hybrid` to compare
accuracy, faithfulness, retrieval coverage, and response time. The Python API
is also available:

```python
from rag_system import Document, RAGSystem

system = RAGSystem([Document("doc-1", "Your knowledge base text.")])
answer = system.answer("What is in the knowledge base?", method="hybrid")
```

Run the focused tests with:

```bash
python -m unittest discover -s tests -v
```