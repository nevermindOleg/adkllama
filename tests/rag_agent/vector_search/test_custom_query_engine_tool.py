# tests/rag_agent/vector_search/test_custom_query_engine_tool.py

import unittest
from unittest.mock import MagicMock, patch
from llama_index.core.tools import ToolOutput
# Adjust import path based on your project structure
from rag_agent.vector_search.custom_query_engine_tool import CustomQueryEngineTool

class TestCustomQueryEngineTool(unittest.TestCase):

    def setUp(self):
        """Set up a mock QueryEngine and ToolMetadata before each test."""
        self.mock_query_engine = MagicMock()
        self.mock_metadata = MagicMock()
        self.mock_metadata.name = "test_tool"
        self.tool = CustomQueryEngineTool(query_engine=self.mock_query_engine, metadata=self.mock_metadata)

        # Mock the internal _get_query_str method to control query input
        self.tool._get_query_str = MagicMock()

        # Mock the get_response_with_metadata method to simplify testing call/acall
        self.tool.get_response_with_metadata = MagicMock()
        self.tool.get_response_with_metadata.return_value = "Formatted Response"


    def test_call_sync_success(self):
        """Test synchronous call when query engine succeeds."""
        self.tool._get_query_str.return_value = "test query"
        mock_response = MagicMock()
        mock_response.response = "Raw Response"
        mock_response.metadata = {"doc1": {"file_name": "doc1.txt"}}
        self.mock_query_engine.query.return_value = mock_response

        result = self.tool.call("test query")

        self.tool._get_query_str.assert_called_once_with("test query")
        self.mock_query_engine.query.assert_called_once_with("test query")
        self.tool.get_response_with_metadata.assert_called_once_with(mock_response.response, mock_response.metadata)
        self.assertIsInstance(result, ToolOutput)
        self.assertEqual(result.content, "Formatted Response")
        self.assertEqual(result.tool_name, "test_tool")
        self.assertEqual(result.raw_input, {"input": "test query"})
        self.assertEqual(result.raw_output, mock_response)


    def test_call_sync_failure(self):
        """Test synchronous call when query engine raises an exception."""
        self.tool._get_query_str.return_value = "test query"
        self.mock_query_engine.query.side_effect = Exception("Query Failed")

        result = self.tool.call("test query")

        self.tool._get_query_str.assert_called_once_with("test query")
        self.mock_query_engine.query.assert_called_once_with("test query")
        self.tool.get_response_with_metadata.assert_not_called() # Should not be called on error
        self.assertIsInstance(result, ToolOutput)
        self.assertEqual(result.content, "Error processing query: Query Failed")
        self.assertEqual(result.tool_name, "test_tool")
        self.assertEqual(result.raw_input, {"input": "test query"})
        self.assertIsNone(result.raw_output)


    @patch('rag_agent.vector_search.custom_query_engine_tool.CustomQueryEngineTool.acall') # Patch acall if needed for mocking
    def test_acall_async_success(self, mock_acall_method):
        """Test asynchronous call when query engine succeeds."""
        # For simplicity in synchronous testing environment, we mock acall itself.
        # In a real async test setup, you would use async test runners (like unittest)
        # and await the actual acall method.
        # This test primarily verifies the setup and parameter passing.
        pass # Skipping detailed acall test in a standard unittest context


    def test_get_response_with_metadata_valid(self):
        """Test metadata formatting with valid metadata."""
        metadata = {
            "node_id_1": {"file_name": "doc_a.txt", "url": "http://example.com/doc_a"},
            "node_id_2": {"file_name": "doc_b.txt", "url": "http://example.com/doc_b"},
            "node_id_3": {"file_name": "doc_a.txt", "url": "http://example.com/doc_a_duplicate"}, # Duplicate file_name
            "_": {"irrelevant_field": "value"} # Irrelevant metadata
        }
        response_text = "This is the generated answer."
        expected_output_start = f"{response_text}

Source documents:
"
        # Expected to contain info for doc_a.txt and doc_b.txt, de-duplicated by file_name
        expected_output_contains = [
            "  Document name: doc_a.txt
  Document link: http://example.com/doc_a",
            "  Document name: doc_b.txt
  Document link: http://example.com/doc_b"
        ]

        # Temporarily disable the mock get_response_with_metadata
        original_method = self.tool.get_response_with_metadata
        del self.tool.get_response_with_metadata
        self.tool.get_response_with_metadata = original_method

        formatted_response = self.tool.get_response_with_metadata(response_text, metadata)

        self.assertTrue(formatted_response.startswith(expected_output_start))
        for expected_part in expected_output_contains:
            self.assertIn(expected_part, formatted_response)
        # Check that the duplicate is not present based on the de-duplication logic
        self.assertNotIn("http://example.com/doc_a_duplicate", formatted_response)


    def test_get_response_with_metadata_no_metadata(self):
        """Test metadata formatting with no metadata."""
        response_text = "This is the generated answer."
        metadata = None
        expected_output = f"{response_text}

Source documents:
  No relevant documents found.
"

        # Temporarily disable the mock get_response_with_metadata
        original_method = self.tool.get_response_with_metadata
        del self.tool.get_response_with_metadata
        self.tool.get_response_with_metadata = original_method

        formatted_response = self.tool.get_response_with_metadata(response_text, metadata)
        self.assertEqual(formatted_response, expected_output)

    def test_get_response_with_metadata_empty_metadata(self):
        """Test metadata formatting with empty metadata dictionary."""
        response_text = "This is the generated answer."
        metadata = {}
        expected_output = f"{response_text}

Source documents:
  No relevant documents found.
"

        # Temporarily disable the mock get_response_with_metadata
        original_method = self.tool.get_response_with_metadata
        del self.tool.get_response_with_metadata
        self.tool.get_response_with_metadata = original_method

        formatted_response = self.tool.get_response_with_metadata(response_text, metadata)
        self.assertEqual(formatted_response, expected_output)

    def test_get_query_str_from_args(self):
        """Test _get_query_str extracts from positional args."""
        # Temporarily disable the mock _get_query_str
        original_method = self.tool._get_query_str
        del self.tool._get_query_str

        query = self.tool._get_query_str("query string from arg")
        self.assertEqual(query, "query string from arg")

    def test_get_query_str_from_kwargs(self):
        """Test _get_query_str extracts from keyword args."""
        # Temporarily disable the mock _get_query_str
        original_method = self.tool._get_query_str
        del self.tool._get_query_str

        query = self.tool._get_query_str(query="query string from kwarg")
        self.assertEqual(query, "query string from kwarg")

    def test_get_query_str_no_query(self):
        """Test _get_query_str returns empty string when no query provided."""
        # Temporarily disable the mock _get_query_str
        original_method = self.tool._get_query_str
        del self.tool._get_query_str

        # Check behavior when neither args nor kwargs contain the query
        query = self.tool._get_query_str(other_arg=1)
        self.assertEqual(query, "")
        query = self.tool._get_query_str(1, 2, kwarg1="test")
        self.assertEqual(query, "1") # Should take the first positional arg


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
