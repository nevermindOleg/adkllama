from typing import List
import logging

logger = logging.getLogger(__name__)

def evaluate_retrieval(query: str, retrieved_docs: List, relevant_doc_ids: List[str]):
    retrieved_ids = [doc.metadata.get('doc_id') for doc in retrieved_docs]
    relevant_retrieved = len(set(retrieved_ids) & set(relevant_doc_ids))
    precision = relevant_retrieved / len(retrieved_ids) if retrieved_ids else 0.0
    recall = relevant_retrieved / len(relevant_doc_ids) if relevant_doc_ids else 0.0
    logger.info(f"Query: {query}, Precision: {precision}, Recall: {recall}")
    return {"precision": precision, "recall": recall}
