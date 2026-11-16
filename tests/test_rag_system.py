import unittest

from rag_system import Document, RAGSystem
from rag_system.evaluate import evaluate


DOCUMENTS = [
    Document("python", "Python is a programming language used for data science."),
    Document("bm25", "BM25 ranks documents by matching query terms."),
    Document("dense", "Dense retrieval finds semantically similar text."),
]


class RAGSystemTests(unittest.TestCase):
    def setUp(self):
        self.system = RAGSystem(DOCUMENTS)

    def test_all_retrieval_methods_return_relevant_evidence(self):
        for method in ("bm25", "dense", "hybrid"):
            answer = self.system.answer("How does BM25 rank documents?", method)
            self.assertEqual("bm25", answer.sources[0].document.id)
            self.assertIn("[bm25]", answer.text)

    def test_answer_is_grounded_when_no_terms_match(self):
        answer = self.system.answer("What is quantum gravity?", "hybrid")
        self.assertIn("not contain enough evidence", answer.text)

    def test_evaluation_reports_metrics(self):
        result = evaluate(self.system, [("What is Python?", "Python programming")], "bm25")
        self.assertEqual("bm25", result.method)
        self.assertGreaterEqual(result.retrieval_quality, 0)
        self.assertGreaterEqual(result.response_time_ms, 0)

    def test_invalid_method_and_question_are_rejected(self):
        with self.assertRaises(ValueError):
            self.system.answer("", "hybrid")
        with self.assertRaises(ValueError):
            self.system.answer("question", "unknown")


if __name__ == "__main__":
    unittest.main()
