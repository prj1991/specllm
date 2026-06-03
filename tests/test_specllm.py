"""Tests for main specllm module - public API (SpecLLM class)."""
import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock

from specllm import SpecLLM

class TestSpecLLM(unittest.TestCase):
    """Test main SpecLLM class - the public API."""

    def setUp(self):
        """Set up sample spec fixtures."""
        self.spec_dict = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {
                "/users": {
                    "post": {
                        "description": "Create user",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                        },
                                        "required": ["name"],
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "integer"},
                                                "name": {"type": "string"},
                                            },
                                        }
                                    }
                                }
                            }
                        },
                    }
                }
            },
        }

    def test_from_openapi_with_dict(self):
        """from_openapi with dict creates SpecLLM instance."""
        app = SpecLLM.from_openapi(self.spec_dict)
        self.assertIsInstance(app, SpecLLM)

    def test_from_openapi_with_file_path(self):
        """from_openapi with file path loads and creates instance."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(self.spec_dict, f)
            f.flush()
            tmp_path = f.name

        try:
            app = SpecLLM.from_openapi(tmp_path)
            self.assertIsInstance(app, SpecLLM)
        finally:
            os.unlink(tmp_path)

    def test_prompt_decorator_registers_custom_prompt(self):
        """prompt decorator registers custom prompt function for endpoint."""
        app = SpecLLM.from_openapi(self.spec_dict)

        @app.prompt("/users", "post")
        def custom_prompt(endpoint, request_body):
            return "Custom prompt for user creation"

        # Verify the prompt was registered
        self.assertIsNotNone(app._custom_prompts.get(("/users", "post")))

    def test_validate_decorator_registers_custom_validator(self):
        """validate decorator registers custom validator function for endpoint."""
        app = SpecLLM.from_openapi(self.spec_dict)

        @app.validate("/users", "post")
        def check_input(body):
            return None

        self.assertIsNotNone(app._custom_validators.get(("/users", "post")))

    def test_validate_decorator_defaults_to_post(self):
        """validate decorator defaults method to 'post'."""
        app = SpecLLM.from_openapi(self.spec_dict)

        @app.validate("/users")
        def check_input(body):
            return None

        self.assertIsNotNone(app._custom_validators.get(("/users", "post")))

    def test_test_client_returns_test_client(self):
        """test_client returns a TestClient that can call endpoints."""
        app = SpecLLM.from_openapi(self.spec_dict)
        client = app.test_client()
        self.assertIsNotNone(client)
        self.assertTrue(hasattr(client, "post"))

    @patch("specllm.testing.contract.ContractTestRunner")
    def test_contract_test_runs_and_returns_results(self, mock_runner_cls):
        """contract_test runs tests and returns results."""
        mock_results = Mock()
        mock_results.total = 2
        mock_results.passed = 2
        mock_results.failed = 0
        mock_runner = Mock()
        mock_runner.run.return_value = mock_results
        mock_runner_cls.return_value = mock_runner

        app = SpecLLM.from_openapi(self.spec_dict)
        results = app.contract_test(
            samples=[
                {"name": "Alice"},
                {"name": "Bob"},
            ]
        )
        self.assertEqual(results.total, 2)
        self.assertEqual(results.passed, 2)

if __name__ == "__main__":
    unittest.main()
