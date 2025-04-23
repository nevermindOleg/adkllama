# rag_agent/vector_search/reranker.py

from sentence_transformers import CrossEncoder
from typing import List
from llama_index.core.schema import NodeWithScore
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)

class Reranker:
    """
    A reranker using a CrossEncoder model to reorder retrieved documents based on relevance.

    This class takes a list of retrieved documents (NodesWithScore) and a query,
    uses a Sentence-Transformers CrossEncoder model to predict relevance scores
    for each query-document pair, and then reorders the documents based on these
    new scores.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initializes the Reranker with a specified CrossEncoder model.

        Args:
            model_name: The name of the CrossEncoder model to use for reranking.
                        Defaults to "cross-encoder/ms-marco-MiniLM-L-6-v2".
        """
        # Load the specified CrossEncoder model
        self.model = CrossEncoder(model_name)
        logger.info(f"Reranker initialized with model: {model_name}")

    def rerank(self, query: str, nodes: List[NodeWithScore], top_k: int = 5) -> List[NodeWithScore]:
        """
        Reranks a list of nodes based on their relevance to the query.

        Uses the CrossEncoder model to score query-document pairs and then
        sorts the nodes by these scores, returning the top_k results.

        Args:
            query: The user's query string.
            nodes: A list of NodeWithScore objects (retrieved documents) to rerank.
            top_k: The number of top reranked documents to return. Defaults to 5.

        Returns:
            A list of NodeWithScore objects, sorted by relevance score in descending order,
            limited to the top_k results.
        """
        if not nodes:
            logger.info("No nodes to rerank. Returning empty list.")
            return []

        logger.debug(f"Reranking {len(nodes)} nodes for query: {query} (returning top {top_k})")

        # Create pairs of [query, document_text] for the CrossEncoder model
        pairs = [[query, node.text] for node in nodes]

        # Predict relevance scores using the CrossEncoder model
        scores = self.model.predict(pairs)

        # Assign the predicted scores back to the nodes
        for node, score in zip(nodes, scores):
            # Convert score to float explicitly for type consistency
            node.score = float(score)

        # Sort the nodes by their new scores in descending order and take the top_k
        # Use a default score of 0.0 if node.score is None (though CrossEncoder should provide scores)
        reranked = sorted(nodes, key=lambda x: x.score if x.score is not None else 0.0, reverse=True)[:top_k]

        logger.info(f"Reranked and returned top {len(reranked)} documents.")
        # Log the IDs and scores of the reranked documents for debugging
        for i, node in enumerate(reranked):
            logger.debug(f"  Rank {i+1}: Node ID: {node.node_id}, Score: {node.score}")

        return reranked
