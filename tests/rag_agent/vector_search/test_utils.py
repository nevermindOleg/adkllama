# tests/rag_agent/vector_search/test_utils.py

import unittest
from unittest.mock import MagicMock
from typing import List, Any, Dict
# Adjust import path based on your project structure
from rag_agent.vector_search.utils import evaluate_retrieval
# Need actual types for creating mock objects that behave like LlamaIndex Nodes
from llama_index.core.schema import NodeWithScore, TextNode


class TestUtils(unittest.TestCase):

    def test_evaluate_retrieval_perfect_match(self):
        """Test evaluation with perfect precision and recall."""
        query = "test query"
        # Create mock documents with metadata including 'doc_id'
        retrieved_docs = [
            MagicMock(spec=NodeWithScore, metadata={'doc_id': 'doc1'}),
            MagicMock(spec=NodeWithScore, metadata={'doc_id': 'doc2'}),
            MagicMock(spec=NodeWithScore, metadata={'doc_id': 'doc3'}),
        ]
        relevant_doc_ids = ['doc1', 'doc2', 'doc3']

        metrics = evaluate_retrieval(query, retrieved_docs, relevant_doc_ids)

        self.assertIsInstance(metrics, dict)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)

    def test_evaluate_retrieval_some_relevant_retrieved(self):
        """Test evaluation when some but not all relevant docs are retrieved."""
        query = "test query"
        retrieved_docs = [
            MagicMock(spec=NodeWithScore, metadata={'doc_id': 'doc1'}), # Relevant
            MagicMock(spec=NodeWithScore, metadata={'doc_id': 'doc4'}), # Not relevant
            MagicMock(spec=NodeWithScore, metadata={'doc_id': 'doc2'}), # Relevant
        ]
        relevant_doc_ids = ['doc1', 'doc2', 'doc3'] # doc3 is relevant but not retrieved

        metrics = evaluate_retrieval(query, retrieved_docs, relevant_doc_ids)

        # Retrieved: doc1, doc4, doc2 (Total 3)
        # Relevant: doc1, doc2, doc3 (Total 3)
        # Relevant and Retrieved: doc1, doc2 (Count 2)
        # Precision: 2 / 3
        # Recall: 2 / 3
        self.assertAlmostEqual(metrics["precision"], 2/3)
        self.assertAlmostEqual(metrics["recall"], 2/3)

    def test_evaluate_retrieval_irrelevant_retrieved(self):
        """Test evaluation when irrelevant docs are retrieved."""
        query = "test query"
        retrieved_docs = [
            MagicMock(spec=NodeWithScore, metadata={'doc_id': 'doc4'}), # Not relevant
            MagicMock(spec=NodeWithScore, metadata={'doc_id': 'doc5'}), # Not relevant
        ]
        relevant_doc_ids = ['doc1', 'doc2'] # No relevant docs retrieved

        metrics = evaluate_retrieval(query, retrieved_docs, relevant_doc_ids)

        # Retrieved: doc4, doc5 (Total 2)
        # Relevant: doc1, doc2 (Total 2)
        # Relevant and Retrieved: None (Count 0)
        # Precision: 0 / 2 = 0.0
        # Recall: 0 / 2 = 0.0
        self.assertAlmostEqual(metrics["precision"], 0.0)
        self.assertAlmostEqual(metrics["recall"], 0.0)

    def test_evaluate_retrieval_empty_retrieved(self):
        """Test evaluation with no retrieved documents."""
        query = "test query"
        retrieved_docs = []
        relevant_doc_ids = ['doc1', 'doc2']

        metrics = evaluate_retrieval(query, retrieved_docs, relevant_doc_ids)

        # Retrieved: [] (Total 0)
        # Relevant: doc1, doc2 (Total 2)
        # Relevant and Retrieved: None (Count 0)
        # Precision: 0 / 0 = 0.0 (by function logic)
        # Recall: 0 / 2 = 0.0
        self.assertAlmostEqual(metrics["precision"], 0.0)
        self.assertAlmostEqual(metrics["recall"], 0.0)

    def test_evaluate_retrieval_empty_relevant(self):
        """Test evaluation with no relevant documents."""
        query = "test query"
        retrieved_docs = [
             MagicMock(spec=NodeWithScore, metadata={'doc_id': 'doc1'}),
             MagicMock(spec=NodeWithScore, metadata={'doc_id': 'doc2'}),
        ]
        relevant_doc_ids = []

        metrics = evaluate_retrieval(query, retrieved_docs, relevant_doc_ids)

        # Retrieved: doc1, doc2 (Total 2)
        # Relevant: [] (Total 0)
        # Relevant and Retrieved: None (Count 0)
        # Precision: 0 / 2 = 0.0
        # Recall: 0 / 0 = 0.0 (by function logic)
        self.assertAlmostEqual(metrics["precision"], 0.0)
        self.assertAlmostEqual(metrics["recall"], 0.0)

    def test_evaluate_retrieval_empty_both(self):
        """Test evaluation with no retrieved and no relevant documents."""
        query = "test query"
        retrieved_docs = []
        relevant_doc_ids = []

        metrics = evaluate_retrieval(query, retrieved_docs, relevant_doc_ids)

        # Retrieved: [] (Total 0)
        # Relevant: [] (Total 0) Non (Count 0)
        # Precision: 0 / 0 = 0.0
        # Recall: 0 / 0 = 0.0
        self.assertAlmostEqual(metrics["precision"], 0.0)
        self.assertAlmostEqual(metrics["recall"], 0.0)

    def test_evaluate_retrieval_missing_metadata(self):
        """Test evaluation with documents missing metadata or doc_id."""
        query = "test query"
        retrieved_docs = [
             MagicMock(spec=NodeWithScore, metadata={'doc_id': 'doc1'}), # Relevant
             MagicMock(spec=NodeWithScore, metadata={}), # Missing doc_id
             MagicMock(spec=NodeWithScore), # Missing metadata attribute
             MagicMock(spec=NodeWithScore, metadata={'doc_id': 'doc2'}), # Relevant
        ]
        relevant_doc_ids = ['doc1', 'doc2']

        metrics = evaluate_retrieval(query, retrieved_docs, relevant_doc_ids)

        # Retrieved IDs extracted: ['doc1', 'doc2'] (from valid nodes)
        # Total retrieved considered for precision: 2 (doc1, doc2 based on successful ID extraction)
        # Relevant retrieved: doc1, doc2 (Count 2)
        # Precision: 2 / 2 = 1.0
        # Recall: 2 / 2 = 1.0
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
