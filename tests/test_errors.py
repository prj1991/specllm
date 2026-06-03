"""Tests for specllm.errors.codes module - error codes and error responses."""
import unittest

from specllm.errors.codes import (
    ErrorCode,
    build_error_response,
    HTTP_STATUS_MAP,
)

class TestErrorCodes(unittest.TestCase):
    """Test error codes enum."""

    def test_input_validation_failed_exists(self):
        """INPUT_VALIDATION_FAILED error code exists."""
        self.assertTrue(hasattr(ErrorCode, "INPUT_VALIDATION_FAILED"))

    def test_output_schema_violation_exists(self):
        """OUTPUT_SCHEMA_VIOLATION error code exists."""
        self.assertTrue(hasattr(ErrorCode, "OUTPUT_SCHEMA_VIOLATION"))

    def test_provider_timeout_exists(self):
        """PROVIDER_TIMEOUT error code exists."""
        self.assertTrue(hasattr(ErrorCode, "PROVIDER_TIMEOUT"))

    def test_provider_unavailable_exists(self):
        """PROVIDER_UNAVAILABLE error code exists."""
        self.assertTrue(hasattr(ErrorCode, "PROVIDER_UNAVAILABLE"))

    def test_rate_limited_exists(self):
        """RATE_LIMITED error code exists."""
        self.assertTrue(hasattr(ErrorCode, "RATE_LIMITED"))

    def test_cost_limit_reached_exists(self):
        """COST_LIMIT_REACHED error code exists."""
        self.assertTrue(hasattr(ErrorCode, "COST_LIMIT_REACHED"))

    def test_endpoint_not_found_exists(self):
        """ENDPOINT_NOT_FOUND error code exists."""
        self.assertTrue(hasattr(ErrorCode, "ENDPOINT_NOT_FOUND"))

    def test_all_providers_failed_exists(self):
        """ALL_PROVIDERS_FAILED error code exists."""
        self.assertTrue(hasattr(ErrorCode, "ALL_PROVIDERS_FAILED"))

    def test_provider_rate_limited_exists(self):
        """PROVIDER_RATE_LIMITED error code exists."""
        self.assertTrue(hasattr(ErrorCode, "PROVIDER_RATE_LIMITED"))

    def test_endpoint_disabled_exists(self):
        """ENDPOINT_DISABLED error code exists."""
        self.assertTrue(hasattr(ErrorCode, "ENDPOINT_DISABLED"))

    def test_output_safety_violation_exists(self):
        """OUTPUT_SAFETY_VIOLATION error code exists."""
        self.assertTrue(hasattr(ErrorCode, "OUTPUT_SAFETY_VIOLATION"))

    def test_all_codes_count(self):
        """All 11 error codes are defined."""
        codes = [m for m in dir(ErrorCode) if not m.startswith("_")]
        self.assertGreaterEqual(len(codes), 11)

class TestBuildErrorResponse(unittest.TestCase):
    """Test structured error response building."""

    def test_error_response_structure(self):
        """Error response has correct nested structure."""
        response = build_error_response(
            ErrorCode.INPUT_VALIDATION_FAILED,
            message="Missing required field: email",
            request_id="req-123",
        )
        self.assertIn("error", response)
        error = response["error"]
        self.assertIn("code", error)
        self.assertIn("status", error)
        self.assertIn("message", error)
        self.assertIn("request_id", error)
        self.assertIn("timestamp", error)

    def test_error_response_code_value(self):
        """Error response code matches the ErrorCode name."""
        response = build_error_response(
            ErrorCode.INPUT_VALIDATION_FAILED,
            message="Bad input",
            request_id="req-456",
        )
        self.assertEqual(
            response["error"]["code"], "INPUT_VALIDATION_FAILED"
        )

    def test_error_response_message(self):
        """Error response message matches input."""
        response = build_error_response(
            ErrorCode.PROVIDER_TIMEOUT,
            message="Provider timed out after 30s",
            request_id="req-789",
        )
        self.assertEqual(
            response["error"]["message"], "Provider timed out after 30s"
        )

    def test_error_response_request_id(self):
        """Error response includes request_id."""
        response = build_error_response(
            ErrorCode.RATE_LIMITED,
            message="Too many requests",
            request_id="req-abc",
        )
        self.assertEqual(response["error"]["request_id"], "req-abc")

    def test_error_response_timestamp_present(self):
        """Error response includes a timestamp string."""
        response = build_error_response(
            ErrorCode.ENDPOINT_NOT_FOUND,
            message="Not found",
            request_id="req-def",
        )
        self.assertIsInstance(response["error"]["timestamp"], str)
        self.assertGreater(len(response["error"]["timestamp"]), 0)

class TestHTTPStatusMap(unittest.TestCase):
    """Test HTTP status code mapping."""

    def test_input_validation_failed_is_400(self):
        """INPUT_VALIDATION_FAILED maps to 400."""
        self.assertEqual(
            HTTP_STATUS_MAP[ErrorCode.INPUT_VALIDATION_FAILED], 400
        )

    def test_output_schema_violation_is_422(self):
        """OUTPUT_SCHEMA_VIOLATION maps to 422."""
        self.assertEqual(
            HTTP_STATUS_MAP[ErrorCode.OUTPUT_SCHEMA_VIOLATION], 422
        )

    def test_endpoint_not_found_is_404(self):
        """ENDPOINT_NOT_FOUND maps to 404."""
        self.assertEqual(
            HTTP_STATUS_MAP[ErrorCode.ENDPOINT_NOT_FOUND], 404
        )

    def test_rate_limited_is_429(self):
        """RATE_LIMITED maps to 429."""
        self.assertEqual(HTTP_STATUS_MAP[ErrorCode.RATE_LIMITED], 429)

    def test_provider_timeout_is_504(self):
        """PROVIDER_TIMEOUT maps to 504."""
        self.assertEqual(
            HTTP_STATUS_MAP[ErrorCode.PROVIDER_TIMEOUT], 504
        )

if __name__ == "__main__":
    unittest.main()
