from llama_index.vector_stores.qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse
from qdrant_client import QdrantClient, models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_qdrant_vector_store(collection_name: str, openai_api_key: str, qdrant_url: str, qdrant_api_key: str = None):
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    logger.info("Qdrant client initialized and connection verified.")
    
    try:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(size=1536, distance=models.Distance.DOT),
                "sparse": models.SparseVectorParams()
            }
        )
        logger.info(f"Collection {collection_name} created.")
    except Exception as e:
        logger.info(f"Collection {collection_name} already exists or error: {e}")

    dense_embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    logger.info("Embeddings initialized successfully.")

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode="hybrid"
    )
    return vector_store
