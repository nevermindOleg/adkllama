from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import logging



logger = logging.getLogger(__name__)

def setup_qdrant_vector_store(
    collection_name: str,
    openai_api_key: str, # Although not directly used here, kept for consistency with potential embedding setup
    qdrant_url: str,
    qdrant_api_key: str = None
) -> QdrantVectorStore:
    
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    
    logger.debug("Qdrant client initialized and connection verified.")
    
    
    try:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "default": VectorParams(size=1536, distance=Distance.DOT)
            }
        )
        logger.info(f"Collection '{collection_name}' created.")
    except Exception as e:
        # Log if the collection already exists or if another error occurred during creation.
        logger.info(f"Collection '{collection_name}' already exists or error during creation: {e}")

    vector_store = QdrantVectorStore(
        collection_name=collection_name,
        client=qdrant_client,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25",
        batch_size=20
    )
    logger.debug("Hybrid QdrantVectorStore initialized successfully.")
    return vector_store
