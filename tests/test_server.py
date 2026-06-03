"""Tests for specllm.server.app module - HTTP server integration."""
import json
import threading
import time
import unittest
import urllib.request
import urllib.error
from unittest.mock import Mock, patch, MagicMock

from specllm.server.app import SpecLLMServer

class TestSpecLLMServer(unittest.TestCase):
    """Test HTTP server integration."""

    @classmethod
    def setUpClass(cls):
        """Start a test server instance in a background thread."""
        cls.mock_spec = {
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
        cls.mock_provider = Mock()
        cls.mock_provider.call.return_value = {"id": 1, "name": "Alice"}

        cls.server = SpecLLMServer(
            spec=cls.mock_spec,
            provider=cls.mock_provider,
            host="127.0.0.1",
            port=0,  # Let OS assign port
        )
        cls.server_thread = threading.Thread(target=cls.server.serve, daemon=True)
        cls.server_thread.start()
        time.sleep(0.5)  # Wait for server to start
        cls.base_url = f"http://127.0.0.1:{cls.server.port}"

    @classmethod
    def tearDownClass(cls):
        """Stop the test server."""
        cls.server.shutdown()

    def test_post_returns_json_200(self):
        """Server responds to POST with JSON response and 200."""
        data = json.dumps({"name": "Alice"}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/users",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as response:
            self.assertEqual(response.status, 200)
            body = json.loads(response.read().decode("utf-8"))
            self.assertIn("id", body)

    def test_unknown_path_returns_404(self):
        """Unknown path returns 404 with structured error."""
        data = json.dumps({"key": "value"}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/nonexistent",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req)
            self.fail("Expected HTTPError 404")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)
            body = json.loads(e.read().decode("utf-8"))
            self.assertIn("error", body)

    def test_wrong_method_returns_405(self):
        """GET on POST-only endpoint returns 405."""
        req = urllib.request.Request(
            f"{self.base_url}/users",
            method="GET",
        )
        try:
            urllib.request.urlopen(req)
            self.fail("Expected HTTPError 405")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 405)

    def test_malformed_json_returns_400(self):
        """Malformed JSON body returns 400."""
        data = b"not valid json{{"
        req = urllib.request.Request(
            f"{self.base_url}/users",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req)
            self.fail("Expected HTTPError 400")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)

    def test_response_content_type_json(self):
        """Response includes Content-Type: application/json."""
        data = json.dumps({"name": "Alice"}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/users",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as response:
            content_type = response.headers.get("Content-Type")
            self.assertIn("application/json", content_type)

    def test_response_includes_observability_headers(self):
        """Response includes X-SpecLLM-* observability headers."""
        data = json.dumps({"name": "Alice"}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/users",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as response:
            header_keys = [k for k in response.headers if k.startswith("X-SpecLLM-")]
            self.assertGreater(len(header_keys), 0)
            self.assertIn("X-SpecLLM-Request-Id", response.headers)

if __name__ == "__main__":
    unittest.main()
