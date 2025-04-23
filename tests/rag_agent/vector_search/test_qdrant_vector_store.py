# tests/rag_agent/vector_search/test_qdrant_vector_store.py

import unittest
from unittest.mock import MagicMock, patch, call
# Adjust import path based on your project structure
from rag_agent.vector_search.qdrant_vector_store import setup_qdrant_vector_store
from qdrant_client import QdrantClient # Need the actual type for spec
from qdrant_client.models import Distance, VectorParams # Need actual types
from llama_index.vector_stores.qdrant import QdrantVectorStore # Need the actual type for spec


class TestQdrantVectorStoreSetup(unittest.TestCase):

    @patch('rag_agent.vector_search.qdrant_vector_store.QdrantClient')
    @patch('rag_agent.vector_search.qdrant_vector_store.QdrantVectorStore')
    def test_setup_qdrant_vector_store_success(self, MockQdrantVectorStore, MockQdrantClient):
        """Test successful setup of Qdrant vector store."""
        mock_qdrant_client_instance = MockQdrantClient.return_value
        mock_vector_store_instance = MockQdrantVectorStore.return_value

        collection_name = "test_collection"
        openai_api_key = "fake_openai_key"
        qdrant_url = "http://localhost:6333"
        qdrant_api_key = "fake_qdrant_key"

        # Mock create_collection to not raise an error (simulating success or already exists)
        mock_qdrant_client_instance.create_collection.return_value = None

        vector_store = setup_qdrant_vector_store(
            collection_name=collection_name,
            openai_api_key=openai_api_key,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key
        )

        # Check if QdrantClient was initialized with correct arguments
        MockQdrantClient.assert_called_once_with(url=qdrant_url, api_key=qdrant_api_key)

        # Check if create_collection was called with correct arguments
        mock_qdrant_client_instance.create_collection.assert_called_once_with(
            collection_name=collection_name,
            vectors_config={
                "default": VectorParams(size=1536, distance=Distance.DOT)
            }
        )

        # Check if QdrantVectorStore was initialized with correct arguments
        MockQdrantVectorStore.assert_called_once_with(
            collection_name=collection_name,
            client=mock_qdrant_client_instance,
            enable_hybrid=True,
            fastembed_sparse_model="Qdrant/bm25",
            batch_size=20
        )

        # Check if the function returned the mock vector store instance
        self.assertEqual(vector_store, mock_vector_store_instance)


    @patch('rag_agent.vector_search.qdrant_vector_store.QdrantClient')
    @patch('rag_agent.vector_search.qdrant_vector_store.QdrantVectorStore')
    def test_setup_qdrant_vector_store_collection_exists(self, MockQdrantVectorStore, MockQdrantClient):
        """Test setup when collection already exists (409 Conflict)."""
        mock_qdrant_client_instance = MockQdrantClient.return_value
        mock_vector_store_instance = MockQdrantVectorStore.return_value

        collection_name = "existing_collection"
        openai_api_key = "fake_openai_key"
        qdrant_url = "http://localhost:6333"
        qdrant_api_key = "fake_qdrant_key"

        # Mock create_collection to raise a conflict exception
        mock_qdrant_client_instance.create_collection.side_effect = Exception("409 Conflict")

        vector_store = setup_qdrant_vector_store(
            collection_name=collection_name,
            openai_api_key=openai_api_key,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key
        )

        # Check if QdrantClient was initialized
        MockQdrantClient.assert_called_once()
        # Check if create_collection was attempted
        mock_qdrant_client_instance.create_collection.assert_called_once()
        # Check if QdrantVectorStore was still initialized (it should be)
        MockQdrantVectorStore.assert_called_once_with(
            collection_name=collection_name,
            client=mock_qdrant_client_instance,
            enable_hybrid=True,
            fastembed_sparse_model="Qdrant/bm25",
            batch_size=20
        )
        # Check the function returned the mock vector store instance
        self.assertEqual(vector_store, mock_vector_store_instance)

    # Note: Testing actual connection errors (e.g., invalid URL) would require
    # more sophisticated mocking or integration tests. This focuses on the logic
    # within setup_qdrant_vector_store assuming the client initialization might fail later.


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
