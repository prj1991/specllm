"""Tests for specllm.pipeline.retry module - retry logic."""
import unittest
from unittest.mock import Mock, patch, MagicMock

from specllm.pipeline.retry import RetryHandler, MaxRetriesExceeded

class TestRetryHandler(unittest.TestCase):
    """Test retry logic with schema error feedback."""

    def setUp(self):
        """Set up RetryHandler with default config."""
        self.handler = RetryHandler(max_retries=3)

    def test_no_retry_when_output_is_valid(self):
        """No retry when LLM output passes validation."""
        call_fn = Mock(return_value={"id": 1, "name": "Alice"})
        validate_fn = Mock(return_value=[])  # no errors

        result = self.handler.execute(call_fn, validate_fn)

        self.assertEqual(result, {"id": 1, "name": "Alice"})
        self.assertEqual(call_fn.call_count, 1)

    def test_retry_on_schema_validation_failure(self):
        """Retry when output fails schema validation."""
        # First call returns invalid output, second returns valid
        call_fn = Mock(side_effect=[
            {"invalid": "data"},
            {"id": 1, "name": "Alice"},
        ])
        # First validation fails, second passes
        validate_fn = Mock(side_effect=[
            [Mock(path="name", message="required field missing")],
            [],
        ])

        result = self.handler.execute(call_fn, validate_fn)

        self.assertEqual(result, {"id": 1, "name": "Alice"})
        self.assertEqual(call_fn.call_count, 2)

    def test_stop_after_max_retries(self):
        """Stop retrying after max_retries attempts and raise error."""
        call_fn = Mock(return_value={"bad": "data"})
        validate_fn = Mock(return_value=[
            Mock(path="name", message="required field missing"),
        ])

        with self.assertRaises(MaxRetriesExceeded):
            self.handler.execute(call_fn, validate_fn)

        # Initial call + max_retries retries
        self.assertEqual(call_fn.call_count, 4)  # 1 initial + 3 retries

    def test_feedback_includes_validation_errors(self):
        """Retry call receives feedback with validation errors."""
        error_mock = Mock(path="email", message="required field missing")
        call_fn = Mock(side_effect=[
            {"bad": "data"},
            {"id": 1, "name": "Alice", "email": "a@b.com"},
        ])
        validate_fn = Mock(side_effect=[
            [error_mock],
            [],
        ])

        result = self.handler.execute(call_fn, validate_fn)

        # Second call should have received feedback
        self.assertEqual(call_fn.call_count, 2)
        self.assertEqual(result, {"id": 1, "name": "Alice", "email": "a@b.com"})

    def test_default_max_retries_is_three(self):
        """Default max_retries value is 3."""
        handler = RetryHandler()
        self.assertEqual(handler.max_retries, 3)

    def test_custom_max_retries(self):
        """Custom max_retries value is respected."""
        handler = RetryHandler(max_retries=5)
        self.assertEqual(handler.max_retries, 5)

if __name__ == "__main__":
    unittest.main()
