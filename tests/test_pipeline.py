"""Tests for specllm.pipeline.request module - full request pipeline."""
import unittest
from unittest.mock import Mock, patch, MagicMock

from specllm.pipeline.request import RequestPipeline
from specllm.spec.parser import Endpoint

class TestRequestPipeline(unittest.TestCase):
    """Test full request pipeline orchestration."""

    def setUp(self):
        """Set up pipeline with mock LLM provider."""
        self.endpoint = Endpoint(
            path="/users",
            method="post",
            description="Create a new user",
            request_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                },
                "required": ["name", "email"],
            },
            response_schema={
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                },
                "required": ["id", "name", "email"],
            },
            parameters=[],
        )

    @patch("specllm.pipeline.request.Cache")
    @patch("specllm.pipeline.request.validate_schema")
    def test_successful_flow(self, mock_validate, mock_cache_cls):
        """Successful flow: input valid -> cache miss -> LLM called -> output valid -> response returned."""
        mock_cache = Mock()
        mock_cache.get.return_value = None  # cache miss
        mock_cache_cls.return_value = mock_cache

        mock_provider = Mock()
        mock_provider.call.return_value = {"id": 1, "name": "Alice", "email": "a@b.com"}

        mock_validate.return_value = []  # no validation errors

        pipeline = RequestPipeline(provider=mock_provider)
        result = pipeline.handle(self.endpoint, {"name": "Alice", "email": "a@b.com"})

        self.assertIn("id", result)
        mock_provider.call.assert_called()

    @patch("specllm.pipeline.request.Cache")
    @patch("specllm.pipeline.request.validate_schema")
    def test_cache_hit_skips_llm(self, mock_validate, mock_cache_cls):
        """Cache hit: LLM not called, cached response returned."""
        cached_response = {"id": 1, "name": "Alice", "email": "a@b.com"}
        mock_cache = Mock()
        mock_cache.get.return_value = cached_response
        mock_cache_cls.return_value = mock_cache

        mock_validate.return_value = []

        mock_provider = Mock()
        pipeline = RequestPipeline(provider=mock_provider)
        result = pipeline.handle(self.endpoint, {"name": "Alice", "email": "a@b.com"})

        self.assertEqual(result, cached_response)
        mock_provider.call.assert_not_called()

    @patch("specllm.pipeline.request.Cache")
    @patch("specllm.pipeline.request.validate_schema")
    def test_input_validation_failure(self, mock_validate, mock_cache_cls):
        """Input validation failure returns error with INPUT_VALIDATION_FAILED code."""
        mock_cache = Mock()
        mock_cache.get.return_value = None
        mock_cache_cls.return_value = mock_cache

        # Input validation fails
        mock_validate.return_value = [
            Mock(path="email", message="required field missing")
        ]

        mock_provider = Mock()
        pipeline = RequestPipeline(provider=mock_provider)
        result = pipeline.handle(self.endpoint, {"name": "Alice"})

        self.assertIn("error", result)
        self.assertEqual(result["error"]["code"], "INPUT_VALIDATION_FAILED")
        mock_provider.call.assert_not_called()

    @patch("specllm.pipeline.request.Cache")
    @patch("specllm.pipeline.request.validate_schema")
    def test_output_validation_failure_triggers_retry(self, mock_validate, mock_cache_cls):
        """Output validation failure triggers retry."""
        mock_cache = Mock()
        mock_cache.get.return_value = None
        mock_cache_cls.return_value = mock_cache

        # First call: input valid, output invalid; second: output valid
        mock_validate.side_effect = [
            [],  # input validation passes
            [Mock(path="id", message="required field missing")],  # output fails
            [],  # retry output passes
        ]

        mock_provider = Mock()
        mock_provider.call.side_effect = [
            {"name": "Alice"},  # missing id
            {"id": 1, "name": "Alice", "email": "a@b.com"},  # valid
        ]

        pipeline = RequestPipeline(provider=mock_provider)
        result = pipeline.handle(self.endpoint, {"name": "Alice", "email": "a@b.com"})

        self.assertNotIn("error", result)
        self.assertGreaterEqual(mock_provider.call.call_count, 2)

    @patch("specllm.pipeline.request.Cache")
    @patch("specllm.pipeline.request.validate_schema")
    def test_all_retries_exhausted_returns_error(self, mock_validate, mock_cache_cls):
        """All retries exhausted returns OUTPUT_SCHEMA_VIOLATION error."""
        mock_cache = Mock()
        mock_cache.get.return_value = None
        mock_cache_cls.return_value = mock_cache

        # Input valid, but output always invalid
        mock_validate.side_effect = [
            [],  # input validation passes
        ] + [
            [Mock(path="id", message="required")]  # output always fails
        ] * 10

        mock_provider = Mock()
        mock_provider.call.return_value = {"bad": "data"}

        pipeline = RequestPipeline(provider=mock_provider)
        result = pipeline.handle(self.endpoint, {"name": "Alice", "email": "a@b.com"})

        self.assertIn("error", result)
        self.assertEqual(result["error"]["code"], "OUTPUT_SCHEMA_VIOLATION")

if __name__ == "__main__":
    unittest.main()
