# rag_agent/vector_search/utils.py

"""
Utility functions for evaluating retrieval performance.

This module provides functions to calculate metrics such as precision and recall
for a given retrieval result against a set of known relevant documents.
"""

from typing import List, Dict, Any # Added Dict and Any for return type hint
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)

def evaluate_retrieval(
    query: str,
    retrieved_docs: List[Any], # Use Any as the exact type of retrieved_docs elements isn't strictly defined here
    relevant_doc_ids: List[str]
) -> Dict[str, float]:
    """
    Evaluates the precision and recall of a list of retrieved documents for a query.

    Calculates precision and recall based on the retrieved document IDs compared
    to a list of known relevant document IDs. Assumes retrieved documents have
    an 'doc_id' field in their metadata.

    Args:
        query: The query string for which retrieval was performed.
        retrieved_docs: A list of retrieved document objects (e.g., LlamaIndex Nodes
                        with a 'metadata' attribute containing 'doc_id').
        relevant_doc_ids: A list of strings representing the IDs of documents
                          that are considered relevant to the query.

    Returns:
        A dictionary containing the calculated 'precision' and 'recall' scores.
        Returns 0.0 for precision or recall if the denominators are zero.
    """
    # Extract document IDs from the retrieved documents' metadata
    # Assumes each document in retrieved_docs has a .metadata attribute with 'doc_id'
    retrieved_ids = [doc.metadata.get('doc_id') for doc in retrieved_docs if hasattr(doc, 'metadata') and doc.metadata is not None]

    # Calculate the number of relevant documents that were retrieved
    # Use sets for efficient intersection
    relevant_retrieved_count = len(set(retrieved_ids) & set(relevant_doc_ids))

    # Calculate Precision: (Relevant retrieved) / (Total retrieved)
    precision = relevant_retrieved_count / len(retrieved_ids) if retrieved_ids else 0.0

    # Calculate Recall: (Relevant retrieved) / (Total relevant)
    recall = relevant_retrieved_count / len(relevant_doc_ids) if relevant_doc_ids else 0.0

    logger.info(f"Retrieval evaluation for query: '{query}' - Precision: {precision:.4f}, Recall: {recall:.4f}")
    return {"precision": precision, "recall": recall}
