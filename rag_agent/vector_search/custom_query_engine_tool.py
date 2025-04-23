import logging
from typing import Any, List, Optional
import pandas as pd
from llama_index.core.tools import QueryEngineTool, ToolOutput
from llama_index.core.query_engine import BaseQueryEngine
from vector_search.utils import evaluate_retrieval


logger = logging.getLogger(__name__)

class CustomQueryEngineTool(QueryEngineTool):
    def call(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """
        Выполняет синхронный запрос к query_engine и возвращает результат с метаданными документов.

        Args:
            *args: Позиционные аргументы, включая query_str.
            **kwargs: Дополнительные аргументы для query.

        Returns:
            ToolOutput: Объект с результатом запроса, именем инструмента, входными и выходными данными.
        """
        query_str = self._get_query_str(*args, **kwargs)
        try:
            response = self._query_engine.query(query_str)
            response_with_docs_info = self.get_response_with_metadata(response.response, response.metadata)
            logger.info(f"Query: {query_str}, Response: {response_with_docs_info}")
            return ToolOutput(
                content=response_with_docs_info,
                tool_name=self.metadata.name,
                raw_input={"input": query_str},
                raw_output=response,
            )
        except Exception as e:
            logger.error(f"Error processing query: {query_str}, Error: {e}")
            return ToolOutput(
                content=f"Error processing query: {str(e)}",
                tool_name=self.metadata.name,
                raw_input={"input": query_str},
                raw_output=None,
            )

    async def acall(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """
        Выполняет асинхронный запрос к query_engine и возвращает результат с метаданными документов.

        Args:
            *args: Позиционные аргументы, включая query_str.
            **kwargs: Дополнительные аргументы для query.

        Returns:
            ToolOutput: Объект с результатом запроса, именем инструмента, входными и выходными данными.
        """
        query_str = self._get_query_str(*args, **kwargs)
        try:
            response = await self._query_engine.aquery(query_str)
            response_with_docs_info = self.get_response_with_metadata(response.response, response.metadata)
            logger.info(f"Async query: {query_str}, Response: {response_with_docs_info}")
            return ToolOutput(
                content=response_with_docs_info,
                tool_name=self.metadata.name,
                raw_input={"input": query_str},
                raw_output=response,
            )
        except Exception as e:
            logger.error(f"Error processing async query: {query_str}, Error: {e}")
            return ToolOutput(
                content=f"Error processing query: {str(e)}",
                tool_name=self.metadata.name,
                raw_input={"input": query_str},
                raw_output=None,
            )

    def get_response_with_metadata(self, response: str, metadata: Optional[dict]) -> str:
        """
        Форматирует ответ с информацией о документах из метаданных.

        Args:
            response: Текстовый ответ от query_engine.
            metadata: Словарь метаданных документов.

        Returns:
            str: Ответ, объединенный с информацией о документах.
        """
        documents_info = "Source documents:\n"
        if metadata and isinstance(metadata, dict) and metadata.values():
            try:
                documents = pd.DataFrame(metadata.values())
                documents.drop_duplicates(subset='file_name', inplace=True)
                for _, document in documents.iterrows():
                    document_info = (
                        f"  Document name: {document.get('file_name', 'N/A')}\n"
                        f"  Document link: {document.get('url', 'N/A')}\n"
                    )
                    documents_info += document_info
            except Exception as e:
                logger.error(f"Error processing metadata: {e}")
                documents_info += "  Error processing document metadata.\n"
        else:
            documents_info += "  No relevant documents found.\n"

        response_with_docs_info = f"{response}\n\n{documents_info}"
        logger.info(f"Formatted response: {response_with_docs_info}")
        return response_with_docs_info