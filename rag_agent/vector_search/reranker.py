from sentence_transformers import CrossEncoder
from typing import List
from llama_index.core.schema import NodeWithScore
import logging

logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
        logger.info(f"Reranker initialized with model: {model_name}")

    def rerank(self, query: str, nodes: List[NodeWithScore], top_k: int = 5) -> List[NodeWithScore]:
        if not nodes:
            return []

        logger.debug(f"Reranking {len(nodes)} nodes for query: {query}")

        pairs = [[query, node.text] for node in nodes]

        scores = self.model.predict(pairs)

        for node, score in zip(nodes, scores):
            node.score = float(score)

        reranked = sorted(nodes, key=lambda x: x.score if x.score is not None else 0.0, reverse=True)[:top_k]

        logger.info(f"Reranked and returned top {len(reranked)} documents.")
        for i, node in enumerate(reranked):
            logger.debug(f"  Rank {i+1}: Node ID: {node.node_id}, Score: {node.score}")

        return reranked
