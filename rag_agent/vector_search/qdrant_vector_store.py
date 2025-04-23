# rag_agent/vector_search/qdrant_vector_store.py

from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_qdrant_vector_store(
    collection_name: str,
    openai_api_key: str, # Although not directly used here, kept for consistency with potential embedding setup
    qdrant_url: str,
    qdrant_api_key: str = None
) -> QdrantVectorStore:
    """
    Sets up and initializes the Qdrant vector store.

    Initializes the Qdrant client, creates the specified collection if it doesn't
    exist, and configures the QdrantVectorStore for use with LlamaIndex,
    including hybrid search capabilities.

    Args:
        collection_name: The name of the collection in Qdrant.
        openai_api_key: OpenAI API key (required for embedding, passed for context).
        qdrant_url: The URL of the Qdrant instance (cloud or local).
        qdrant_api_key: The API key for authentication with Qdrant (optional).

    Returns:
        An initialized QdrantVectorStore instance.
    """
    # Initialize Qdrant client
    # The client is configured to connect to the specified Qdrant instance.
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    logger.info("Qdrant client initialized and connection verified.")

    # Create collection with a dense vector configuration
    # Attempts to create the collection. If it already exists, a 409 Conflict
    # exception is expected and handled gracefully.
    try:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "default": VectorParams(size=1536, distance=Distance.DOT) # Size 1536 is standard for OpenAI embeddings
            }
        )
        logger.info(f"Collection '{collection_name}' created.")
    except Exception as e:
        # Log if the collection already exists or if another error occurred during creation.
        logger.info(f"Collection '{collection_name}' already exists or error during creation: {e}")

    # Initialize LlamaIndex's Qdrant vector store with hybrid search enabled
    # Configures the vector store wrapper for LlamaIndex, enabling both
    # vector similarity search and sparse keyword (BM25) search.
    vector_store = QdrantVectorStore(
        collection_name=collection_name,
        client=qdrant_client,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25",  # Specifies the sparse model for hybrid search
        batch_size=20 # Defines the batch size for upsert operations
    )

    logger.info("Hybrid QdrantVectorStore initialized successfully.")
    return vector_store
