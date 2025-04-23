# rag_agent/vector_search/document_loader.py

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from typing import List, Optional # Imported Optional as directory_path is optional in __init__ (implicitly by check)
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)

class DocumentLoader:
    """
    A utility class for loading documents from a directory and creating a LlamaIndex.

    This class simplifies the process of reading documents from a specified directory
    and initializing a VectorStoreIndex from these documents using a provided
    vector store.
    """
    def __init__(self, directory_path: Optional[str] = None):
        """
        Initializes the DocumentLoader with an optional directory path.

        Args:
            directory_path: The path to the directory containing the documents.
                            If None, the path must be provided during load_documents.
        """
        self.directory_path = directory_path
        logger.info(f"DocumentLoader initialized with directory_path: {self.directory_path}")


    def load_documents(self, directory_path: Optional[str] = None) -> List[Document]:
        """
        Loads documents from the specified directory using SimpleDirectoryReader.

        Uses the directory_path provided during initialization or the one passed
        as an argument to this method.

        Args:
            directory_path: Optional. Overrides the directory_path provided in __init__.

        Returns:
            A list of LlamaIndex Document objects.

        Raises:
            ValueError: If no directory path is provided during initialization or to this method.
        """
        # Determine the actual directory path to use
        current_directory_path = directory_path if directory_path is not None else self.directory_path

        if not current_directory_path:
            logger.error("No directory path provided for loading documents.")
            raise ValueError("Directory path must be provided for loading documents")

        logger.info(f"Loading documents from directory: {current_directory_path}")
        try:
            reader = SimpleDirectoryReader(input_dir=current_directory_path)
            documents = reader.load_data()
            logger.info(f"Loaded {len(documents)} documents from {current_directory_path}")
            return documents
        except Exception as e:
            logger.error(f"Error loading documents from {current_directory_path}: {e}", exc_info=True)
            return [] # Return empty list or re-raise, depending on desired behavior


    def create_index(self, documents: List[Document], vector_store) -> Optional[VectorStoreIndex]:
        """
        Creates a LlamaIndex from a list of documents using a provided vector store.

        Args:
            documents: A list of LlamaIndex Document objects.
            vector_store: An initialized LlamaIndex-compatible vector store instance
                          (e.g., QdrantVectorStore).

        Returns:
            An initialized VectorStoreIndex instance, or None if indexing fails.
        """
        if not documents:
            logger.warning("No documents provided for indexing. Skipping index creation.")
            return None

        if vector_store is None:
             logger.error("Vector store not initialized. Cannot create index.")
             return None

        logger.info(f"Creating index with {len(documents)} documents using the provided vector store.")
        try:
            # Create index using the vector store.
            # Note: Passing the vector_store directly is the standard way to integrate
            # a VectorStore with VectorStoreIndex.from_documents.
            # Using storage_context might have been a cause for previous issues.
            index = VectorStoreIndex.from_documents(
                documents,
                vector_store=vector_store, # Directly pass the vector_store instance
                # storage_context=vector_store.get_storage_context() # Avoid this when passing vector_store directly
            )
            logger.info(f"Index created successfully with {len(documents)} documents.")
            return index
        except Exception as e:
            logger.error(f"Error creating index: {e}", exc_info=True)
            return None


# Note: The original code in create_index was using storage_context=vector_store.get_storage_context().
# While this can work in some LlamaIndex setups, directly passing vector_store=vector_store
# to VectorStoreIndex.from_documents is the more common and often more reliable method
# when integrating with a dedicated vector database like Qdrant.
# The provided combined script uses the direct vector_store=vector_store approach.
