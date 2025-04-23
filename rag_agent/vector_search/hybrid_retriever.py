from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever

from typing import List
from .reranker import Reranker
import logging

logger = logging.getLogger(__name__)

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever: VectorIndexRetriever, bm25_retriever: BM25Retriever, reranker: Reranker = None, vector_weight: float = 0.7, bm25_weight: float = 0.3):
        super().__init__()
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker or Reranker()
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

    def _retrieve(self, query: str, **kwargs) -> List:
        vector_results = self.vector_retriever.retrieve(query, **kwargs)
        bm25_results = self.bm25_retriever.retrieve(query, **kwargs)
        
        for node in vector_results:
            node.score = node.score * self.vector_weight if node.score is not None else 0.0
        for node in bm25_results:
            node.score = node.score * self.bm25_weight if node.score is not None else 0.0
        
        combined_results = vector_results + bm25_results
        unique_results = {node.node_id: node for node in combined_results}
        combined_nodes = list(unique_results.values())
        
        reranked_nodes = self.reranker.rerank(query, combined_nodes, top_k=5)
        logger.info(f"Retrieved and reranked {len(reranked_nodes)} documents for query: {query}")
        return reranked_nodes
