from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class DocumentLoader:
    
    """    
    This class simplifies the process of reading documents from a specified directory
    and initializing a VectorStoreIndex from these documents using a provided
    vector store.
    """
    def __init__(self, directory_path: Optional[str] = None):
        """
        Initializes the DocumentLoader with an optional directory path.
        """
        
        
        
        
        self.directory_path = directory_path
        logger.debug(f"DocumentLoader initialized with directory_path: {self.directory_path}")
        """
        self.directory_path = directory_path
        logger.info(f"DocumentLoader initialized with directory_path: {self.directory_path}")


    def load_documents(self, directory_path: Optional[str] = None) -> List[Document]:
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

        logger.debug(f"Loading documents from directory: {current_directory_path}")
        try:
            reader = SimpleDirectoryReader(input_dir=current_directory_path)
            documents = reader.load_data()
            logger.debug(f"Loaded {len(documents)} documents from {current_directory_path}")
            return documents
        except Exception as e:
            logger.error(f"Error loading documents from {current_directory_path}: {e}", exc_info=True)
            return []


    def create_index(self, documents: List[Document], vector_store) -> Optional[VectorStoreIndex]:
        
        if not documents:
            logger.warning("No documents provided for indexing. Skipping index creation.")
            return None

        if vector_store is None:
            logger.error("Vector store not initialized. Cannot create index.")
            return None

        logger.debug(f"Creating index with {len(documents)} documents using the provided vector store.")
        try:
            index = VectorStoreIndex.from_documents(documents,vector_store=vector_store)
            logger.debug(f"Index created successfully with {len(documents)} documents.")
            return index
        except Exception as e:
            logger.error(f"Error creating index: {e}", exc_info=True)
            return None
