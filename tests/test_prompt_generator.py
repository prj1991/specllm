"""Tests for specllm.prompts.generator module - prompt generation."""
import unittest

from specllm.spec.parser import Endpoint
from specllm.prompts.generator import generate_prompt

class TestGeneratePrompt(unittest.TestCase):
    """Test prompt generation from endpoint spec."""

    def setUp(self):
        """Set up sample Endpoint fixtures."""
        self.endpoint = Endpoint(
            path="/users",
            method="post",
            description="Create a new user account",
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
                    "created_at": {"type": "string"},
                },
                "required": ["id", "name", "email"],
            },
            parameters=[],
        )

        self.endpoint_no_description = Endpoint(
            path="/health",
            method="get",
            description="",
            request_schema=None,
            response_schema={
                "type": "object",
                "properties": {"status": {"type": "string"}},
            },
            parameters=[],
        )

    def test_prompt_includes_endpoint_description(self):
        """Generated prompt includes the endpoint description."""
        prompt = generate_prompt(self.endpoint)
        self.assertIn("Create a new user account", prompt)

    def test_prompt_includes_json_format_instructions(self):
        """Generated prompt includes JSON response format instructions."""
        prompt = generate_prompt(self.endpoint)
        self.assertIn("JSON", prompt)

    def test_prompt_includes_response_schema(self):
        """Generated prompt includes the response schema."""
        prompt = generate_prompt(self.endpoint)
        self.assertIn("id", prompt)
        self.assertIn("name", prompt)
        self.assertIn("email", prompt)

    def test_prompt_includes_request_body_context(self):
        """Generated prompt includes request body context when provided."""
        request_body = {"name": "Alice", "email": "alice@example.com"}
        prompt = generate_prompt(self.endpoint, request_body=request_body)
        self.assertIn("Alice", prompt)
        self.assertIn("alice@example.com", prompt)

    def test_handle_endpoint_with_no_description(self):
        """Prompt handles endpoint with empty description gracefully."""
        prompt = generate_prompt(self.endpoint_no_description)
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)

    def test_prompt_is_non_empty_string(self):
        """Generated prompt is a non-empty string."""
        prompt = generate_prompt(self.endpoint)
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)

    def test_prompt_without_request_body(self):
        """Prompt works without request_body argument."""
        prompt = generate_prompt(self.endpoint)
        self.assertIsInstance(prompt, str)

if __name__ == "__main__":
    unittest.main()
