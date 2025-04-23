# tests/rag_agent/vector_search/test_document_loader.py

import unittest
from unittest.mock import MagicMock, patch, call
import os
from typing import List
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document
# Adjust import path based on your project structure
from rag_agent.vector_search.document_loader import DocumentLoader

class TestDocumentLoader(unittest.TestCase):

    def setUp(self):
        """Set up a DocumentLoader instance."""
        self.test_dir = "test_documents_for_loader"
        self.loader = DocumentLoader(directory_path=self.test_dir)

    def tearDown(self):
        """Clean up test files and directory."""
        if os.path.exists(self.test_dir):
            for filename in os.listdir(self.test_dir):
                os.remove(os.path.join(self.test_dir, filename))
            os.rmdir(self.test_dir)

    @patch('rag_agent.vector_search.document_loader.SimpleDirectoryReader')
    def test_load_documents_success(self, MockSimpleDirectoryReader):
        """Test loading documents successfully."""
        # Create a dummy directory and file
        os.makedirs(self.test_dir, exist_ok=True)
        with open(os.path.join(self.test_dir, "doc1.txt"), "w") as f:
            f.write("test content 1")
        with open(os.path.join(self.test_dir, "doc2.txt"), "w") as f:
            f.write("test content 2")

        mock_reader_instance = MockSimpleDirectoryReader.return_value
        mock_documents = [MagicMock(spec=Document), MagicMock(spec=Document)] # Mock Document objects
        mock_reader_instance.load_data.return_value = mock_documents

        documents = self.loader.load_documents()

        MockSimpleDirectoryReader.assert_called_once_with(input_dir=self.test_dir)
        mock_reader_instance.load_data.assert_called_once()
        self.assertEqual(documents, mock_documents)

    @patch('rag_agent.vector_search.document_loader.SimpleDirectoryReader')
    def test_load_documents_with_arg_overrides_init(self, MockSimpleDirectoryReader):
        """Test load_documents argument overrides init path."""
        override_dir = "override_test_dir"
        os.makedirs(override_dir, exist_ok=True)
        with open(os.path.join(override_dir, "doc1.txt"), "w") as f:
            f.write("test content")

        mock_reader_instance = MockSimpleDirectoryReader.return_value
        mock_documents = [MagicMock(spec=Document)]
        mock_reader_instance.load_data.return_value = mock_documents

        documents = self.loader.load_documents(directory_path=override_dir)

        MockSimpleDirectoryReader.assert_called_once_with(input_dir=override_dir)
        mock_reader_instance.load_data.assert_called_once()
        self.assertEqual(documents, mock_documents)

        # Clean up override directory
        os.remove(os.path.join(override_dir, "doc1.txt"))
        os.rmdir(override_dir)


    def test_load_documents_no_path(self):
        """Test loading documents without any directory path provided."""
        loader_no_path = DocumentLoader()
        with self.assertRaisesRegex(ValueError, "Directory path must be provided"):
             loader_no_path.load_documents()

    @patch('rag_agent.vector_search.document_loader.VectorStoreIndex')
    @patch('rag_agent.vector_search.document_loader.DocumentLoader.load_documents')
    def test_create_index_success(self, mock_load_documents, MockVectorStoreIndex):
        """Test creating index successfully."""
        mock_documents = [MagicMock(spec=Document), MagicMock(spec=Document)]
        mock_load_documents.return_value = mock_documents
        mock_vector_store = MagicMock()
        mock_vector_store.get_storage_context.return_value = "mock_storage_context" # Based on original create_index code
        mock_index = MagicMock(spec=VectorStoreIndex)
        MockVectorStoreIndex.from_documents.return_value = mock_index

        index = self.loader.create_index(documents=mock_documents, vector_store=mock_vector_store) # Pass documents directly as in revised code

        mock_load_documents.assert_not_called() # Load is assumed to be done before calling create_index
        # MockVectorStoreIndex.from_documents.assert_called_once_with( # Based on original code
        #     mock_documents,
        #     storage_context=mock_vector_store.get_storage_context.return_value
        # )
        MockVectorStoreIndex.from_documents.assert_called_once_with( # Based on revised code
            mock_documents,
            vector_store=mock_vector_store # Should pass vector_store directly
        )
        self.assertEqual(index, mock_index)

    @patch('rag_agent.vector_search.document_loader.DocumentLoader.load_documents')
    def test_create_index_no_documents(self, mock_load_documents):
        """Test creating index with no documents."""
        mock_documents = []
        mock_load_documents.return_value = mock_documents
        mock_vector_store = MagicMock()

        index = self.loader.create_index(documents=mock_documents, vector_store=mock_vector_store)

        mock_load_documents.assert_not_called() # Load is assumed to be done
        self.assertIsNone(index)

    @patch('rag_agent.vector_search.document_loader.DocumentLoader.load_documents')
    def test_create_index_no_vector_store(self, mock_load_documents):
        """Test creating index with no vector store."""
        mock_documents = [MagicMock(spec=Document)]
        mock_load_documents.return_value = mock_documents
        mock_vector_store = None

        index = self.loader.create_index(documents=mock_documents, vector_store=mock_vector_store)

        mock_load_documents.assert_not_called() # Load is assumed to be done
        self.assertIsNone(index)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
