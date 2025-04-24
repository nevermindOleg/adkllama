from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from typing import List
from .reranker import Reranker
import logging

logger = logging.getLogger(__name__)

class HybridRetriever(BaseRetriever):
    """
    A hybrid retriever that combines vector search and BM25 search, with optional reranking.

    This retriever takes results from both a VectorIndexRetriever and a BM25Retriever,
    combines and potentially re-ranks them to provide a comprehensive set of relevant
    documents for a given query.
    """
    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        bm25_retriever: BM25Retriever,
        reranker: Reranker = None,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ):
        """
        Initializes the HybridRetriever with the necessary components.

        Args:
            vector_retriever: The retriever for vector similarity search.
            bm25_retriever: The retriever for keyword (BM25) search.
            reranker: An optional Reranker instance to reorder the combined results.
                      If None, a default Reranker is used.
            vector_weight: The weight assigned to vector search scores for initial combination.
            bm25_weight: The weight assigned to BM25 search scores for initial combination.
        """
        super().__init__()
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        # Initialize reranker, using a default one if none is provided
        self.reranker = reranker or Reranker()
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight


    def _retrieve(self, query: str, **kwargs):
        """
        Performs the hybrid retrieval process.

        Executes queries on both the vector and BM25 retrievers, combines their
        results, re-ranks the combined list, and returns the top results.

        Args:
            query: The user's query string.
            **kwargs: Additional keyword arguments passed to the underlying retrievers.

        Returns:
            A list of relevant LlamaIndex Nodes after hybrid retrieval and reranking.
        """
        
        vector_results = self.vector_retriever.retrieve(query, **kwargs)
        bm25_results = self.bm25_retriever.retrieve(query, **kwargs)
       
        for node in vector_results:
            node.score = (node.score * self.vector_weight) if node.score is not None else 0.0
        for node in bm25_results:
            node.score = (node.score * self.bm25_weight) if node.score is not None else 0.0

        combined_results = vector_results + bm25_results
        unique_results = {node.node_id: node for node in combined_results}
        combined_nodes = list(unique_results.values())

        reranked_nodes = self.reranker.rerank(query, combined_nodes, top_k=5)
        
        logger.debug(f"HybridRetriever initialized with vector_weight={self.vector_weight}, "
            f"bm25_weight={self.bm25_weight}, reranker={self.reranker is not None}")
        logger.debug(f"Vector retriever returned {len(vector_results)} results. BM25 retriever returned {len(bm25_results)} results.")
        logger.debug(f"Combined and de-duplicated results: {len(combined_nodes)} nodes.")
        logger.info(f"Retrieved and reranked {len(reranked_nodes)} documents for query: '{query}'")
        return reranked_nodes
    
   
