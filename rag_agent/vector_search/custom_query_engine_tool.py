from llama_index.core.tools import QueryEngineTool, ToolOutput
from typing import Any, List
import pandas as pd
import logging
from .utils import evaluate_retrieval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomQueryEngineTool(QueryEngineTool):
    def call(self, *args: Any, doc_ids: List[str] = None, **kwargs: Any) -> ToolOutput:
        query_str = self._get_query_str(*args, **kwargs)
        response = self._query_engine.query(query_str, doc_ids=doc_ids)
        response_with_docs_info = self.get_response_with_metadata(response.response, response.metadata)
        if doc_ids:
            metrics = evaluate_retrieval(query_str, response.source_nodes, doc_ids)
            logger.info(f"Retrieval metrics: {metrics}")
        logger.info(f"Query: {query_str}, Retrieved documents: {[doc.metadata for doc in response.source_nodes]}")
        return ToolOutput(
            content=response_with_docs_info,
            tool_name=self.metadata.name,
            raw_input={"input": query_str},
            raw_output=response,
        )

    async def acall(self, *args: Any, doc_ids: List[str] = None, **kwargs: Any) -> ToolOutput:
        query_str = self._get_query_str(*args, **kwargs)
        response = await self._query_engine.aquery(query_str, doc_ids=doc_ids)
        response_with_docs_info = self.get_response_with_metadata(response.response, response.metadata)
        if doc_ids:
            metrics = evaluate_retrieval(query_str, response.source_nodes, doc_ids)
            logger.info(f"Retrieval metrics: {metrics}")
        logger.info(f"Query: {query_str}, Retrieved documents: {[doc.metadata for doc in response.source_nodes]}")
        return ToolOutput(
            content=response_with_docs_info,
            tool_name=self.metadata.name,
            raw_input={"input": query_str},
            raw_output=response,
        )

    def get_response_with_metadata(self, response, metadata):
        documents = pd.DataFrame(metadata.values())
        documents.drop_duplicates(subset='file_name', inplace=True)
        
        if 'score' in documents.columns:
            documents = documents[documents['score'] > 0.7].sort_values(by='score', ascending=False).head(5)
        
        documents_info = []
        for _, document in documents.iterrows():
            document_info = f"""
                Document name: {document.get('file_name')}
                Document link: {document.get('url')}
                Relevance score: {document.get('score', 'N/A')}
            """
            documents_info.append(document_info)

        documents_info = "Source documents:
" + "
".join(documents_info) if documents_info else "No relevant documents found."
        response_with_docs_info = f"{response}
{documents_info}"
        logger.info(response_with_docs_info)
        return response_with_docs_info
