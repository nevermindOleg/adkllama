import logging
from typing import Any, List, Optional
import pandas as pd
from llama_index.core.tools import QueryEngineTool, ToolOutput

logger = logging.getLogger(__name__)

class CustomQueryEngineTool(QueryEngineTool):

    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        

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
        try:
            response = self._query_engine.query(query_str)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Query engine response received. Response metadata: {response.metadata}")
                logger.debug(f"Response text: {response.response[:100]}...")
                
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
            logger.error(f"Error processing query '{query_str}': {e}", exc_info=True)
            return ToolOutput(
                content=f"Error processing query: {str(e)}",
                tool_name=self.metadata.name,
                raw_input={"input": query_str},
                raw_output=None,
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
        try:
            response = await self._query_engine.aquery(query_str)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Async query engine response received. Response metadata: {response.metadata}")
                logger.debug(f"Response text: {response.response[:100]}...")
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
            logger.error(f"Error processing async query '{query_str}': {e}", exc_info=True)
            return ToolOutput(
                content=f"Error processing query: {str(e)}",
                tool_name=self.metadata.name,
                raw_input={"input": query_str},
                raw_output=None,
            )

    def get_response_with_metadata(self, response: str, metadata: Optional[dict]) -> str:

        Returns:
            str: The combined response string, including the generated text and
                 formatted source document information.
        """
        documents_info = "Source documents:
"
        if metadata and isinstance(metadata, dict):
            try:
                documents_df = pd.DataFrame(list(metadata.values()))
                if 'file_name' in documents_df.columns:
                     documents_df.drop_duplicates(subset='file_name', inplace=True)
                for _, document in documents_df.iterrows():
                    document_info = (
                        f"  Document name: {document.get('file_name', 'N/A')}
"
                        f"  Document link: {document.get('url', 'N/A')}
"
                    )                    
                    documents_info += document_info
            except Exception as e:
                logger.error(f"Error processing document metadata: {e}", exc_info=True)
                documents_info += "  Error processing document metadata.
"
        else:
            documents_info += "  No relevant documents found.
"
        response_with_docs_info = f"{response}

{documents_info}"
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Formatted response content: {response_with_docs_info}")
        return response_with_docs_info

    def _get_query_str(self, *args: Any, **kwargs: Any) -> str:
        """Helper to extract the query string from args or kwargs."""
        if args:
            return str(args[0])
        if kwargs and 'query' in kwargs:
             return str(kwargs['query'])
        logger.warning("Query string not found in arguments.")
        return ""
