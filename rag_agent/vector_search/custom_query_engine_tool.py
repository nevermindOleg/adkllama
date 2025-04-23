# rag_agent/vector_search/custom_query_engine_tool.py

import logging
from typing import Any, List, Optional
import pandas as pd # Assuming pandas is used for metadata processing, although dropped_duplicates might be problematic if metadata structure changes
from llama_index.core.tools import QueryEngineTool, ToolOutput
from llama_index.core.query_engine import BaseQueryEngine
# Assuming evaluate_retrieval is part of your vector_search utils
from vector_search.utils import evaluate_retrieval # Check if this import is actually used in this file

# Configure logging for this module
logger = logging.getLogger(__name__)

class CustomQueryEngineTool(QueryEngineTool):
    """
    Custom tool to wrap a LlamaIndex QueryEngine, providing metadata in the output.

    This tool extends the standard QueryEngineTool to include document metadata
    (like file name and URL) in the final response content, making it suitable
    for RAG applications where source attribution is important.
    """

    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """
        Performs a synchronous query to the wrapped query_engine.

        Retrieves the query string from arguments, executes the query, and formats
        the response to include relevant document metadata. Error handling is included
        to catch issues during the query process.

        Args:
            *args: Positional arguments passed to the tool, expected to contain the query string.
            **kwargs: Keyword arguments passed to the tool.

        Returns:
            ToolOutput: An object containing the query result, tool name,
                        raw input, and formatted output including metadata.
                        Returns an error message in content if an exception occurs.
        """
        query_str = self._get_query_str(*args, **kwargs)
        logger.debug(f"Synchronous query received: {query_str}")
        try:
            # Execute the query using the wrapped query engine
            response = self._query_engine.query(query_str)
            logger.debug(f"Query engine response received. Type: {type(response)}")
            logger.debug(f"Response text: {response.response[:100]}...")
            logger.debug(f"Response metadata: {response.metadata}")

            # Format the response to include document metadata
            response_with_docs_info = self.get_response_with_metadata(
                response.response,
                response.metadata
            )

            logger.info(f"Query processed successfully. Returning ToolOutput.")
            return ToolOutput(
                content=response_with_docs_info,
                tool_name=self.metadata.name,
                raw_input={"input": query_str},
                raw_output=response,
            )
        except Exception as e:
            # Log the error and return a ToolOutput with an error message
            logger.error(f"Error processing query '{query_str}': {e}", exc_info=True) # exc_info=True logs traceback
            return ToolOutput(
                content=f"Error processing query: {str(e)}",
                tool_name=self.metadata.name,
                raw_input={"input": query_str},
                raw_output=None, # Indicate no raw output on error
            )

    async def acall(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """
        Performs an asynchronous query to the wrapped query_engine.

        Retrieves the query string from arguments, executes the query asynchronously,
        and formats the response to include relevant document metadata. Error handling
        is included to catch issues during the asynchronous query process.

        Args:
            *args: Positional arguments passed to the tool, expected to contain the query string.
            **kwargs: Keyword arguments passed to the tool.

        Returns:
            ToolOutput: An object containing the asynchronous query result, tool name,
                        raw input, and formatted output including metadata.
                        Returns an error message in content if an exception occurs.
        """\
        query_str = self._get_query_str(*args, **kwargs)
        logger.debug(f"Asynchronous query received: {query_str}")
        try:
            # Execute the query asynchronously
            response = await self._query_engine.aquery(query_str)
            logger.debug(f"Async query engine response received. Type: {type(response)}")
            logger.debug(f"Response text: {response.response[:100]}...")
            logger.debug(f"Response metadata: {response.metadata}")

            # Format the response to include document metadata
            response_with_docs_info = self.get_response_with_metadata(
                response.response,
                response.metadata
            )

            logger.info(f"Async query processed successfully. Returning ToolOutput.")
            return ToolOutput(
                content=response_with_docs_info,
                tool_name=self.metadata.name,
                raw_input={"input": query_str},
                raw_output=response,
            )
        except Exception as e:
            # Log the error and return a ToolOutput with an error message
            logger.error(f"Error processing async query '{query_str}': {e}", exc_info=True) # exc_info=True logs traceback
            return ToolOutput(
                content=f"Error processing query: {str(e)}",
                tool_name=self.metadata.name,
                raw_input={"input": query_str},
                raw_output=None, # Indicate no raw output on error
            )


    def get_response_with_metadata(self, response: str, metadata: Optional[dict]) -> str:
        """
        Formats the query response to include source document information from metadata.

        Extracts relevant fields (like file_name and url) from the document metadata
        provided by the query engine and appends them to the generated text response.

        Args:
            response: The primary text response generated by the query engine.
            metadata: A dictionary containing metadata about the retrieved documents.
                      Expected structure depends on the LlamaIndex retriever/node parser.

        Returns:
            str: The combined response string, including the generated text and
                 formatted source document information.
        """
        documents_info = "Source documents:
"
        # Check if metadata is a non-empty dictionary
        if metadata and isinstance(metadata, dict):
            try:
                # Convert metadata values to a pandas DataFrame for easier processing
                # Note: This assumes metadata values are structured consistently
                documents_df = pd.DataFrame(list(metadata.values()))

                # Drop duplicates based on file_name to avoid listing the same document multiple times
                if 'file_name' in documents_df.columns:
                     documents_df.drop_duplicates(subset='file_name', inplace=True)

                # Iterate through the unique documents and format their info
                for _, document in documents_df.iterrows():
                    document_info = (
                        f"  Document name: {document.get('file_name', 'N/A')}
"
                        f"  Document link: {document.get('url', 'N/A')}
"
                    )
                    documents_info += document_info
            except Exception as e:
                # Log error if metadata processing fails
                logger.error(f"Error processing document metadata: {e}", exc_info=True)
                documents_info += "  Error processing document metadata.
"
        else:
            # Indicate if no relevant documents were found based on metadata
            documents_info += "  No relevant documents found.
"

        # Combine the generated response and the formatted document information
        response_with_docs_info = f"{response}

{documents_info}"
        # Log the final formatted response (potentially large, use DEBUG for full content if needed)
        logger.info("Response formatted with document metadata.")
        logger.debug(f"Formatted response content: {response_with_docs_info}")
        return response_with_docs_info

    def _get_query_str(self, *args: Any, **kwargs: Any) -> str:
        """Helper to extract the query string from args or kwargs."""
        if args:
            return str(args[0])
        if kwargs and 'query' in kwargs:
             return str(kwargs['query'])
        # Fallback or raise error if query string is not found
        logger.warning("Query string not found in arguments.")
        return "" # Or raise ValueError("Query string must be provided.")


# Note: The evaluate_retrieval function imported from utils is not used in this file.
# You might want to remove the import if it's not utilized here.
