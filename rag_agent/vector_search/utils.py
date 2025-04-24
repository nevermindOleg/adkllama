from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def evaluate_retrieval(
    query: str,
    retrieved_docs: List[Any],
    relevant_doc_ids: List[str]
) -> Dict[str, float]:

    retrieved_ids = [doc.metadata.get('doc_id') for doc in retrieved_docs if hasattr(doc, 'metadata') and doc.metadata is not None]

    relevant_retrieved_count = len(set(retrieved_ids) & set(relevant_doc_ids))

    precision = relevant_retrieved_count / len(retrieved_ids) if retrieved_ids else 0.0

    recall = relevant_retrieved_count / len(relevant_doc_ids) if relevant_doc_ids else 0.0

    logger.debug(f"Retrieval evaluation for query: '{query}' - Precision: {precision:.4f}, Recall: {recall:.4f}")
    return {"precision": precision, "recall": recall}
