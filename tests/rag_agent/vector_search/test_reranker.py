# tests/rag_agent/vector_search/test_reranker.py

import unittest
from unittest.mock import MagicMock, patch
from typing import List
from llama_index.core.schema import NodeWithScore, TextNode
# Adjust import path based on your project structure
from rag_agent.vector_search.reranker import Reranker
# Need the actual type for spec and sometimes for instantiation if not fully mocked
from sentence_transformers import CrossEncoder

class TestReranker(unittest.TestCase):

    @patch('rag_agent.vector_search.reranker.CrossEncoder')
    def test_reranker_init(self, MockCrossEncoder):
        """Test Reranker initialization."""
        model_name = "test-model"
        reranker = Reranker(model_name)

        MockCrossEncoder.assert_called_once_with(model_name)
        self.assertEqual(reranker.model, MockCrossEncoder.return_value)
        self.assertIsNotNone(reranker.model) # Ensure model attribute is set


    @patch('rag_agent.vector_search.reranker.CrossEncoder')
    def test_rerank_success(self, MockCrossEncoder):
        """Test reranking with valid nodes and query."""
        # Configure the mock CrossEncoder instance
        mock_cross_encoder_instance = MockCrossEncoder.return_value
        # Mock the predict method to return scores
        # Scores should correspond to nodes in the order they are passed
        mock_scores = [0.9, 0.1, 0.7, 0.5] # Scores for node1, node2, node3, node4

        mock_cross_encoder_instance.predict.return_value = mock_scores

        # Create NodeWithScore objects
        node1 = NodeWithScore(node=TextNode(text="Node 1 content"), score=0.5) # Initial score
        node2 = NodeWithScore(node=TextNode(text="Node 2 content"), score=0.2)
        node3 = NodeWithScore(node=TextNode(text="Node 3 content"), score=0.8)
        node4 = NodeWithScore(node=TextNode(text="Node 4 content"), score=0.3)

        nodes_to_rerank = [node1, node2, node3, node4]
        query = "test query"
        top_k = 3

        reranker = Reranker() # Use default model name for init
        # Patch the instance's model attribute with our mock
        reranker.model = mock_cross_encoder_instance

        reranked_nodes = reranker.rerank(query, nodes_to_rerank, top_k=top_k)

        # Check that predict was called with the correct pairs
        expected_pairs = [
            [query, node1.text],
            [query, node2.text],
            [query, node3.text],
            [query, node4.text],
        ]
        mock_cross_encoder_instance.predict.assert_called_once_with(expected_pairs)

        # Check that the nodes' scores were updated
        self.assertAlmostEqual(node1.score, 0.9)
        self.assertAlmostEqual(node2.score, 0.1)
        self.assertAlmostEqual(node3.score, 0.7)
        self.assertAlmostEqual(node4.score, 0.5)

        # Check that the results are sorted by the new scores and truncated to top_k
        self.assertEqual(len(reranked_nodes), top_k)
        self.assertEqual(reranked_nodes[0].node.text, "Node 1 content") # Highest score 0.9
        self.assertEqual(reranked_nodes[1].node.text, "Node 3 content") # Next highest score 0.7
        self.assertEqual(reranked_nodes[2].node.text, "Node 4 content") # Next highest score 0.5
        # Check scores of the returned nodes
        self.assertAlmostEqual(reranked_nodes[0].score, 0.9)
        self.assertAlmostEqual(reranked_nodes[1].score, 0.7)
        self.assertAlmostEqual(reranked_nodes[2].score, 0.5)


    @patch('rag_agent.vector_search.reranker.CrossEncoder')
    def test_rerank_empty_nodes(self, MockCrossEncoder):
        """Test reranking with an empty list of nodes."""
        mock_cross_encoder_instance = MockCrossEncoder.return_value
        reranker = Reranker()
        reranker.model = mock_cross_encoder_instance # Patch the instance

        query = "test query"
        nodes_to_rerank = []
        top_k = 5

        reranked_nodes = reranker.rerank(query, nodes_to_rerank, top_k=top_k)

        # Predict should not be called with empty nodes
        mock_cross_encoder_instance.predict.assert_not_called()
        # The method should return an empty list
        self.assertEqual(reranked_nodes, [])


    @patch('rag_agent.vector_search.reranker.CrossEncoder')
    def test_rerank_score_is_none(self, MockCrossEncoder):
        """Test reranking when a node's score is None."""
        mock_cross_encoder_instance = MockCrossEncoder.return_value
        # Mock scores, including where the prediction might result in None (unlikely for CrossEncoder, but testing robustness)
        mock_scores = [0.9, None, 0.7]

        mock_cross_encoder_instance.predict.return_value = mock_scores

        # Create NodeWithScore objects, one with a None score
        node1 = NodeWithScore(node=TextNode(text="Node 1"), score=0.5)
        node2 = NodeWithScore(node=TextNode(text="Node 2"), score=None) # Initial score None
        node3 = NodeWithScore(node=TextNode(text="Node 3"), score=0.8)

        nodes_to_rerank = [node1, node2, node3]
        query = "test query"
        top_k = 3

        reranker = Reranker()
        reranker.model = mock_cross_encoder_instance

        reranked_nodes = reranker.rerank(query, nodes_to_rerank, top_k=top_k)

        # Check scores are updated, None predicted score should become 0.0 due to key=lambda logic
        self.assertAlmostEqual(node1.score, 0.9)
        self.assertIsNone(node2.score) # The score attribute is updated with None
        self.assertAlmostEqual(node3.score, 0.7)

        # Check sorting: 0.9, 0.7, 0.0 (from None)
        self.assertEqual(len(reranked_nodes), top_k)
        self.assertEqual(reranked_nodes[0].node.text, "Node 1")
        self.assertEqual(reranked_nodes[1].node.text, "Node 3")
        self.assertEqual(reranked_nodes[2].node.text, "Node 2")

        # Check scores of the returned nodes (should match the values assigned or defaulted in key lambda)
        self.assertAlmostEqual(reranked_nodes[0].score, 0.9)
        self.assertAlmostEqual(reranked_nodes[1].score, 0.7)
        # The node2 object in reranked_nodes still has score=None,
        # but it was sorted based on the lambda key (which treated None as 0.0).
        self.assertIsNone(reranked_nodes[2].score) # Still None


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
