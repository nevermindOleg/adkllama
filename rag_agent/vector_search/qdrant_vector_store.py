from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_qdrant_vector_store(
        collection_name: str,
        openai_api_key: str,
        qdrant_url: str,
        qdrant_api_key: str = None
):
    # Инициализация клиента
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    logger.info("Qdrant client initialized and connection verified.")

    # Создание коллекции только с dense-вектором
    try:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "default": VectorParams(size=1536, distance=Distance.DOT)
            }
        )
        logger.info(f"Collection '{collection_name}' created.")
    except Exception as e:
        logger.info(f"Collection '{collection_name}' already exists or error: {e}")

    # Инициализация vector store с гибридным поиском
    vector_store = QdrantVectorStore(
        collection_name=collection_name,
        client=qdrant_client,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25",  # подключается FastEmbedSparse под капотом
        batch_size=20
    )

    logger.info("Hybrid QdrantVectorStore initialized successfully.")
    return vector_store
