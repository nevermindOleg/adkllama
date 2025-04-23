# tests/rag_agent/vector_search/test_hybrid_retriever.py

import unittest
from unittest.mock import MagicMock, patch
from typing import List
from llama_index.core.retrievers import VectorIndexRetriever # Need the actual type for spec
from llama_index.retrievers.bm25 import BM25Retriever # Need the actual type for spec
from llama_index.core.schema import NodeWithScore, TextNode # Need actual types
# Adjust import path based on your project structure
from rag_agent.vector_search.hybrid_retriever import HybridRetriever
from rag_agent.vector_search.reranker import Reranker # Need the actual type for spec


class TestHybridRetriever(unittest.TestCase):

    def setUp(self):
        """Set up mock retrievers and reranker."""
        self.mock_vector_retriever = MagicMock(spec=VectorIndexRetriever)
        self.mock_bm25_retriever = MagicMock(spec=BM25Retriever)
        self.mock_reranker = MagicMock(spec=Reranker)

        # Create some mock NodeWithScore objects
        self.node1 = NodeWithScore(node=TextNode(text="Node 1 content", id_="node1"), score=0.8)
        self.node2 = NodeWithScore(node=TextNode(text="Node 2 content", id_="node2"), score=0.6)
        self.node3 = NodeWithScore(node=TextNode(text="Node 3 content", id_="node3"), score=0.9)
        self.node4 = NodeWithScore(node=TextNode(text="Node 4 content", id_="node4"), score=0.7)
        self.node5_duplicate = NodeWithScore(node=TextNode(text="Node 5 content", id_="node1"), score=0.5) # Same ID as node1

        self.mock_vector_results = [self.node1, self.node2]
        self.mock_bm25_results = [self.node3, self.node4, self.node5_duplicate]

        self.mock_vector_retriever.retrieve.return_value = self.mock_vector_results
        self.mock_bm25_retriever.retrieve.return_value = self.mock_bm25_results

        # Mock the reranker's return value - a sorted and truncated list
        self.mock_reranked_results = [self.node3, self.node4, self.node1] # Example reranked order

        self.mock_reranker.rerank.return_value = self.mock_reranked_results

        self.retriever = HybridRetriever(
            vector_retriever=self.mock_vector_retriever,
            bm25_retriever=self.mock_bm25_retriever,
            reranker=self.mock_reranker,
            vector_weight=0.7,
            bm25_weight=0.3
        )

    def test_retrieve_hybrid(self):
        """Test the _retrieve method for correct hybrid logic."""
        query = "test query"

        retrieved_nodes = self.retriever._retrieve(query)

        # Check that both retrievers were called
        self.mock_vector_retriever.retrieve.assert_called_once_with(query)
        self.mock_bm25_retriever.retrieve.assert_called_once_with(query)

        # Check if weights were applied (approximately)
        # Original scores: node1=0.8, node2=0.6, node3=0.9, node4=0.7, node5_duplicate=0.5
        # Weighted: node1=0.8*0.7=0.56, node2=0.6*0.7=0.42
        #           node3=0.9*0.3=0.27, node4=0.7*0.3=0.21, node5_duplicate=0.5*0.3=0.15

        # Since node1 (id_="node1") is in both, the one from bm25_results (last seen)
        # might overwrite the score if using a dictionary for unique.
        # Let's check the nodes passed to reranker - it should be unique nodes.
        combined_nodes_passed_to_reranker = self.mock_reranker.rerank.call_args[0][1] # Second argument to rerank

        # Check de-duplication - node1 should appear only once
        self.assertEqual(len(combined_nodes_passed_to_reranker), 4) # node1, node2, node3, node4
        node_ids_passed_to_reranker = {node.node_id for node in combined_nodes_passed_to_reranker}
        self.assertIn("node1", node_ids_passed_to_reranker)
        self.assertIn("node2", node_ids_passed_to_reranker)
        self.assertIn("node3", node_ids_passed_to_reranker)
        self.assertIn("node4", node_ids_passed_to_reranker)
        # Check the scores passed to reranker - scores should have weights applied
        # We can't assert the exact list order passed to rerank, but we can check if the scores are correct
        # for the unique nodes. The score for node1 passed to reranker should be the weighted score
        # from the LAST appearance in the combined list (bm25_results in this case).
        found_node1_in_combined = next((n for n in combined_nodes_passed_to_reranker if n.node_id == "node1"), None)
        self.assertIsNotNone(found_node1_in_combined)
        self.assertAlmostEqual(found_node1_in_combined.score, 0.15) # Weighted score from node5_duplicate

        # Check that the reranker was called with the correct query and combined nodes
        self.mock_reranker.rerank.assert_called_once()
        call_args, call_kwargs = self.mock_reranker.rerank.call_args
        self.assertEqual(call_args[0], query) # First arg is query
        # Check that the second arg (nodes) is a list of NodeWithScore
        self.assertIsInstance(call_args[1], list)
        self.assertTrue(all(isinstance(node, NodeWithScore) for node in call_args[1]))

        # Check the final result is the one from the reranker
        self.assertEqual(retrieved_nodes, self.mock_reranked_results)


    def test_retrieve_hybrid_empty_results(self):
        """Test _retrieve when underlying retrievers return empty lists."""
        self.mock_vector_retriever.retrieve.return_value = []
        self.mock_bm25_retriever.retrieve.return_value = []
        self.mock_reranker.rerank.return_value = [] # Reranker should return empty if no nodes

        query = "test query"
        retrieved_nodes = self.retriever._retrieve(query)

        self.mock_vector_retriever.retrieve.assert_called_once_with(query)
        self.mock_bm25_retriever.retrieve.assert_called_once_with(query)
        self.mock_reranker.rerank.assert_called_once() # Reranker should still be called with an empty list
        self.assertEqual(retrieved_nodes, [])


    def test_retrieve_hybrid_reranker_none(self):
        """Test _retrieve when no reranker is provided (uses default)."""
        # Create a new retriever instance without a provided reranker
        retriever_no_reranker = HybridRetriever(
            vector_retriever=self.mock_vector_retriever,
            bm25_retriever=self.mock_bm25_retriever,
            vector_weight=0.7,
            bm25_weight=0.3
        )

        # Mock the default Reranker that would be initialized internally
        with patch('rag_agent.vector_search.hybrid_retriever.Reranker') as MockDefaultReranker:
             mock_default_reranker_instance = MockDefaultReranker.return_value
             mock_default_reranker_instance.rerank.return_value = self.mock_reranked_results # Still mock the rerank behavior

             query = "test query"
             retrieved_nodes = retriever_no_reranker._retrieve(query)

             MockDefaultReranker.assert_called_once() # Check that the default reranker was instantiated
             mock_default_reranker_instance.rerank.assert_called_once() # Check rerank was called on the default instance

             self.assertEqual(retrieved_nodes, self.mock_reranked_results)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
