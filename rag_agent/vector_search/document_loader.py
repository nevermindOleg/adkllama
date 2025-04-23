from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from typing import List
import logging

logger = logging.getLogger(__name__)

class DocumentLoader:
    def __init__(self, directory_path: str = None):
        self.directory_path = directory_path

    def load_documents(self) -> List:
        if not self.directory_path:
            raise ValueError("Directory path must be provided for loading documents")
        reader = SimpleDirectoryReader(input_dir=self.directory_path)
        documents = reader.load_data()
        logger.info(f"Loaded {len(documents)} documents from {self.directory_path}")
        return documents

    def create_index(self, vector_store) -> VectorStoreIndex:
        documents = self.load_documents()
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=vector_store.get_storage_context()
        )
        logger.info(f"Index created with {len(documents)} documents")
        return index
