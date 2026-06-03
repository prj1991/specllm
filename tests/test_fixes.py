"""Tests covering the 12 review issues fixed in specllm."""
import unittest
from unittest.mock import Mock, patch
from collections import OrderedDict

from specllm import SpecLLM, TestClient
from specllm.spec.parser import parse_openapi_spec, Endpoint, _resolve_refs
from specllm.pipeline.cache import Cache
from specllm.pipeline.request import RequestPipeline, ProviderError
from specllm.llm.providers import MockProvider
from specllm.testing.contract import ContractTestRunner

class TestFromOpenapiFileHandling(unittest.TestCase):
    """YAML and JSON spec file support."""

    def test_from_openapi_nonexistent_file_raises_error(self):
        """from_openapi raises FileNotFoundError for missing files."""
        with self.assertRaises(FileNotFoundError):
            SpecLLM.from_openapi("./nonexistent.json")

    def test_from_openapi_nonexistent_yaml_raises_error(self):
        """from_openapi raises FileNotFoundError for missing YAML files."""
        with self.assertRaises(FileNotFoundError):
            SpecLLM.from_openapi("./nonexistent.yaml")

    def test_from_openapi_nonexistent_unknown_ext_raises_error(self):
        """from_openapi raises FileNotFoundError for missing files with unknown extension."""
        with self.assertRaises(FileNotFoundError):
            SpecLLM.from_openapi("./spec.txt")

    def test_from_openapi_yaml_loads_successfully(self):
        """from_openapi loads a valid YAML spec file."""
        import tempfile, os

        yaml_content = "openapi: '3.0.0'\ninfo:\n  title: Test\n  version: '1'\npaths: {}\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            app = SpecLLM.from_openapi(path)
            self.assertEqual(app.spec["openapi"], "3.0.0")
        finally:
            os.unlink(path)

class TestCustomPromptUsedByPipeline(unittest.TestCase):
    """Issue 3: Custom prompt decorator should be used by the pipeline."""

    def setUp(self):
        self.spec_dict = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {
                "/summarize": {
                    "post": {
                        "description": "Summarize text",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {"text": {"type": "string"}},
                                        "required": ["text"],
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
                                            "properties": {"summary": {"type": "string"}},
                                            "required": ["summary"],
                                        }
                                    }
                                }
                            }
                        },
                    }
                }
            },
        }

    def test_custom_prompt_is_used_by_pipeline(self):
        """Registered custom prompt function is called during request handling."""
        provider = MockProvider(responses=[{"summary": "Short version"}])
        app = SpecLLM.from_openapi(self.spec_dict, provider=provider)

        custom_prompt_called = []

        @app.prompt("/summarize", "post")
        def custom_prompt(request_body):
            custom_prompt_called.append(request_body)
            return "Custom: summarize this: " + request_body.get("text", "")

        client = app.test_client()
        result = client.post("/summarize", json_body={"text": "Hello world"})

        # Verify custom prompt was called
        self.assertEqual(len(custom_prompt_called), 1)
        self.assertEqual(custom_prompt_called[0], {"text": "Hello world"})

        # Verify the provider received the custom prompt
        self.assertIn("Custom: summarize this:", provider._calls[0]["prompt"])

class TestRefResolutionInParser(unittest.TestCase):
    """Issue 4: $ref resolution in parser."""

    def test_ref_resolution_in_request_schema(self):
        """Spec with $ref in request schema resolves correctly."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0.0"},
            "paths": {
                "/users": {
                    "post": {
                        "description": "Create user",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/User"}
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {"type": "object"}
                                    }
                                }
                            }
                        },
                    }
                }
            },
            "components": {
                "schemas": {
                    "User": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": "string"},
                        },
                        "required": ["name", "email"],
                    }
                }
            },
        }
        endpoints = parse_openapi_spec(spec)
        self.assertEqual(len(endpoints), 1)
        schema = endpoints[0].request_schema
        self.assertNotIn("$ref", schema)
        self.assertEqual(schema["type"], "object")
        self.assertIn("name", schema["properties"])
        self.assertIn("email", schema["properties"])
        self.assertEqual(schema["required"], ["name", "email"])

    def test_ref_resolution_in_response_schema(self):
        """Spec with $ref in response schema resolves correctly."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0.0"},
            "paths": {
                "/users": {
                    "get": {
                        "description": "List users",
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/User"},
                                        }
                                    }
                                }
                            }
                        },
                    }
                }
            },
            "components": {
                "schemas": {
                    "User": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    }
                }
            },
        }
        endpoints = parse_openapi_spec(spec)
        schema = endpoints[0].response_schema
        self.assertEqual(schema["type"], "array")
        items = schema["items"]
        self.assertNotIn("$ref", items)
        self.assertEqual(items["type"], "object")
        self.assertIn("name", items["properties"])

    def test_nested_ref_resolution(self):
        """Nested $ref references are resolved recursively."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0.0"},
            "paths": {
                "/orders": {
                    "post": {
                        "description": "Create order",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Order"}
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {"type": "object"}
                                    }
                                }
                            }
                        },
                    }
                }
            },
            "components": {
                "schemas": {
                    "Order": {
                        "type": "object",
                        "properties": {
                            "user": {"$ref": "#/components/schemas/User"},
                            "total": {"type": "number"},
                        },
                    },
                    "User": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    },
                }
            },
        }
        endpoints = parse_openapi_spec(spec)
        schema = endpoints[0].request_schema
        self.assertEqual(schema["type"], "object")
        user_schema = schema["properties"]["user"]
        self.assertNotIn("$ref", user_schema)
        self.assertEqual(user_schema["type"], "object")
        self.assertIn("name", user_schema["properties"])

    def test_resolve_refs_returns_none_for_none(self):
        """_resolve_refs handles None input."""
        self.assertIsNone(_resolve_refs(None, {}))

class TestCacheMaxSizeEviction(unittest.TestCase):
    """Issue 5: Cache should evict oldest entries when max_size is reached."""

    def test_cache_max_size_eviction(self):
        """Cache evicts oldest entry when max_size is reached."""
        cache = Cache(default_ttl=3600, max_size=3)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        # All three present
        self.assertEqual(cache.get("key1"), "value1")
        self.assertEqual(cache.get("key2"), "value2")
        self.assertEqual(cache.get("key3"), "value3")

        # Add a 4th entry - should evict key1 (oldest)
        cache.set("key4", "value4")
        self.assertIsNone(cache.get("key1"))
        self.assertEqual(cache.get("key2"), "value2")
        self.assertEqual(cache.get("key3"), "value3")
        self.assertEqual(cache.get("key4"), "value4")

    def test_cache_max_size_default(self):
        """Default max_size is 1000."""
        cache = Cache()
        self.assertEqual(cache.max_size, 1000)

    def test_cache_eviction_order(self):
        """Eviction follows insertion order (oldest first)."""
        cache = Cache(default_ttl=3600, max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # evicts "a"
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.get("b"), 2)
        self.assertEqual(cache.get("c"), 3)

        cache.set("d", 4)  # evicts "b"
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("c"), 3)
        self.assertEqual(cache.get("d"), 4)

    def test_cache_update_existing_key_does_not_evict(self):
        """Updating an existing key does not trigger eviction."""
        cache = Cache(default_ttl=3600, max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        # Update "a" -- should not evict anything
        cache.set("a", 10)
        self.assertEqual(cache.get("a"), 10)
        self.assertEqual(cache.get("b"), 2)

class TestProviderErrorReturnsStructuredError(unittest.TestCase):
    """Issue 6: Provider errors should return PROVIDER_UNAVAILABLE."""

    def test_provider_error_returns_provider_unavailable(self):
        """When provider.call() raises, pipeline returns PROVIDER_UNAVAILABLE error."""
        provider = Mock()
        provider.call.side_effect = RuntimeError("Connection refused")

        pipeline = RequestPipeline(provider=provider)
        endpoint = Endpoint(
            path="/test",
            method="post",
            description="Test",
            request_schema=None,
            response_schema={"type": "object", "properties": {"result": {"type": "string"}}},
        )

        result = pipeline.handle(endpoint, {})
        self.assertIn("error", result)
        self.assertEqual(result["error"]["code"], "PROVIDER_UNAVAILABLE")
        self.assertIn("Connection refused", result["error"]["message"])

    def test_provider_error_includes_request_id(self):
        """Provider error response includes request_id."""
        provider = Mock()
        provider.call.side_effect = Exception("timeout")

        pipeline = RequestPipeline(provider=provider)
        endpoint = Endpoint(
            path="/test",
            method="post",
            description="Test",
            request_schema=None,
            response_schema=None,
        )

        result = pipeline.handle(endpoint, {})
        self.assertIn("error", result)
        self.assertIn("request_id", result["error"])

class TestTestClientNoProviderError(unittest.TestCase):
    """Issue 8: TestClient should return PROVIDER_NOT_CONFIGURED when no provider."""

    def setUp(self):
        self.spec_dict = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {
                "/users": {
                    "post": {
                        "description": "Create user",
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {"type": "object"}
                                    }
                                }
                            }
                        },
                    }
                }
            },
        }

    def test_test_client_no_provider_error_message(self):
        """TestClient without provider returns PROVIDER_NOT_CONFIGURED error."""
        app = SpecLLM.from_openapi(self.spec_dict)  # No provider
        client = app.test_client()
        result = client.post("/users")

        self.assertIn("error", result)
        self.assertEqual(result["error"]["code"], "PROVIDER_NOT_CONFIGURED")
        self.assertIn("No LLM provider configured", result["error"]["message"])

    def test_test_client_missing_endpoint_still_returns_not_found(self):
        """TestClient returns ENDPOINT_NOT_FOUND for nonexistent endpoints."""
        app = SpecLLM.from_openapi(self.spec_dict)
        client = app.test_client()
        result = client.post("/nonexistent")

        self.assertIn("error", result)
        self.assertEqual(result["error"]["code"], "ENDPOINT_NOT_FOUND")

class TestContractTestSpecificEndpoint(unittest.TestCase):
    """Issue 12: contract_test should support targeting specific endpoints."""

    def setUp(self):
        self.spec_dict = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {
                "/users": {
                    "post": {
                        "description": "Create user",
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {"id": {"type": "integer"}},
                                            "required": ["id"],
                                        }
                                    }
                                }
                            }
                        },
                    },
                    "get": {
                        "description": "List users",
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {"users": {"type": "array"}},
                                            "required": ["users"],
                                        }
                                    }
                                }
                            }
                        },
                    },
                },
                "/items": {
                    "get": {
                        "description": "List items",
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {"items": {"type": "array"}},
                                            "required": ["items"],
                                        }
                                    }
                                }
                            }
                        },
                    }
                },
            },
        }

    def test_contract_test_specific_endpoint(self):
        """contract_test with path/method targets correct endpoint."""
        provider = MockProvider(responses=[{"id": 1}])
        app = SpecLLM.from_openapi(self.spec_dict, provider=provider)

        results = app.contract_test(
            samples=[{"name": "Alice"}],
            path="/users",
            method="post",
        )
        self.assertIsNotNone(results)
        self.assertEqual(results.total, 1)
        self.assertEqual(results.passed, 1)

    def test_contract_test_specific_endpoint_not_found(self):
        """contract_test with unknown path/method returns None."""
        provider = MockProvider(responses=[{"id": 1}])
        app = SpecLLM.from_openapi(self.spec_dict, provider=provider)

        results = app.contract_test(
            samples=[{"name": "Alice"}],
            path="/nonexistent",
            method="post",
        )
        self.assertIsNone(results)

    def test_contract_test_all_endpoints(self):
        """contract_test without path/method runs against all endpoints."""
        provider = MockProvider(responses=[{"id": 1, "users": [], "items": []}])
        app = SpecLLM.from_openapi(self.spec_dict, provider=provider)

        results = app.contract_test(samples=[{"name": "Alice"}])
        # With 3 endpoints, should return a list
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)

    def test_contract_test_single_endpoint_returns_single_result(self):
        """contract_test with single-endpoint spec returns single result (not list)."""
        single_spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0.0"},
            "paths": {
                "/only": {
                    "post": {
                        "description": "Only endpoint",
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {"ok": {"type": "boolean"}},
                                        }
                                    }
                                }
                            }
                        },
                    }
                }
            },
        }
        provider = MockProvider(responses=[{"ok": True}])
        app = SpecLLM.from_openapi(single_spec, provider=provider)
        results = app.contract_test(samples=[{}])
        # Single endpoint: returns single ContractTestResults, not a list
        self.assertNotIsInstance(results, list)
        self.assertEqual(results.total, 1)
        self.assertEqual(results.passed, 1)

class TestBuildErrorResponseUsed(unittest.TestCase):
    """Issue 10: build_error_response is now used in pipeline."""

    def test_error_response_has_structured_format(self):
        """Pipeline error responses include status, timestamp, and request_id."""
        provider = Mock()
        provider.call.return_value = {"bad": "data"}

        pipeline = RequestPipeline(provider=provider)
        endpoint = Endpoint(
            path="/test",
            method="post",
            description="Test",
            request_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            response_schema=None,
        )

        # Input validation failure
        result = pipeline.handle(endpoint, {})
        self.assertIn("error", result)
        error = result["error"]
        self.assertEqual(error["code"], "INPUT_VALIDATION_FAILED")
        self.assertIn("status", error)
        self.assertEqual(error["status"], 400)
        self.assertIn("request_id", error)
        self.assertIn("timestamp", error)

    def test_output_schema_error_has_structured_format(self):
        """OUTPUT_SCHEMA_VIOLATION error includes structured fields."""
        provider = Mock()
        provider.call.return_value = {"wrong": "data"}

        pipeline = RequestPipeline(provider=provider, max_retries=1)
        endpoint = Endpoint(
            path="/test",
            method="post",
            description="Test",
            request_schema=None,
            response_schema={
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "required": ["result"],
            },
        )

        result = pipeline.handle(endpoint, {})
        self.assertIn("error", result)
        error = result["error"]
        self.assertEqual(error["code"], "OUTPUT_SCHEMA_VIOLATION")
        self.assertIn("status", error)
        self.assertEqual(error["status"], 422)
        self.assertIn("request_id", error)
        self.assertIn("timestamp", error)

if __name__ == "__main__":
    unittest.main()
