"""Tests for specllm.observability.headers module - observability headers."""
import unittest

from specllm.observability.headers import build_headers

class TestBuildHeaders(unittest.TestCase):
    """Test observability headers builder."""

    def setUp(self):
        """Set up sample header parameters."""
        self.headers = build_headers(
            request_id="req-123-abc",
            provider="openai",
            model="gpt-4",
            latency_ms=250,
            tokens_used=150,
            retries=1,
            cache_hit=False,
        )

    def test_request_id_header_present(self):
        """X-SpecLLM-Request-Id header is present."""
        self.assertIn("X-SpecLLM-Request-Id", self.headers)

    def test_provider_header_present(self):
        """X-SpecLLM-Provider header is present."""
        self.assertIn("X-SpecLLM-Provider", self.headers)

    def test_model_header_present(self):
        """X-SpecLLM-Model header is present."""
        self.assertIn("X-SpecLLM-Model", self.headers)

    def test_latency_header_present(self):
        """X-SpecLLM-Latency-Ms header is present."""
        self.assertIn("X-SpecLLM-Latency-Ms", self.headers)

    def test_tokens_header_present(self):
        """X-SpecLLM-Tokens-Used header is present."""
        self.assertIn("X-SpecLLM-Tokens-Used", self.headers)

    def test_retries_header_present(self):
        """X-SpecLLM-Retries header is present."""
        self.assertIn("X-SpecLLM-Retries", self.headers)

    def test_cache_hit_header_present(self):
        """X-SpecLLM-Cache-Hit header is present."""
        self.assertIn("X-SpecLLM-Cache-Hit", self.headers)

    def test_request_id_value(self):
        """Request-Id header has correct value."""
        self.assertEqual(self.headers["X-SpecLLM-Request-Id"], "req-123-abc")

    def test_provider_value(self):
        """Provider header has correct value."""
        self.assertEqual(self.headers["X-SpecLLM-Provider"], "openai")

    def test_model_value(self):
        """Model header has correct value."""
        self.assertEqual(self.headers["X-SpecLLM-Model"], "gpt-4")

    def test_latency_as_int_string(self):
        """Latency-Ms is formatted as integer string."""
        self.assertEqual(self.headers["X-SpecLLM-Latency-Ms"], "250")

    def test_tokens_as_int_string(self):
        """Tokens-Used is formatted as integer string."""
        self.assertEqual(self.headers["X-SpecLLM-Tokens-Used"], "150")

    def test_retries_as_int_string(self):
        """Retries is formatted as integer string."""
        self.assertEqual(self.headers["X-SpecLLM-Retries"], "1")

    def test_cache_hit_false_as_string(self):
        """Cache-Hit false is formatted as 'false' string."""
        self.assertEqual(self.headers["X-SpecLLM-Cache-Hit"], "false")

    def test_cache_hit_true_as_string(self):
        """Cache-Hit true is formatted as 'true' string."""
        headers = build_headers(
            request_id="req-456",
            provider="anthropic",
            model="claude-3",
            latency_ms=100,
            tokens_used=50,
            retries=0,
            cache_hit=True,
        )
        self.assertEqual(headers["X-SpecLLM-Cache-Hit"], "true")

    def test_all_headers_have_prefix(self):
        """All headers have X-SpecLLM- prefix."""
        for key in self.headers:
            self.assertTrue(
                key.startswith("X-SpecLLM-"),
                f"Header '{key}' missing X-SpecLLM- prefix",
            )

    def test_returns_dict(self):
        """build_headers returns a dictionary."""
        self.assertIsInstance(self.headers, dict)

if __name__ == "__main__":
    unittest.main()
