"""Tests for specllm.spec.parser module - OpenAPI JSON spec parsing."""
import unittest

from specllm.spec.parser import parse_openapi_spec, Endpoint

class TestParseOpenAPISpec(unittest.TestCase):
    """Test parsing OpenAPI JSON specs into Endpoint dataclass instances."""

    def setUp(self):
        """Set up sample OpenAPI spec fixtures."""
        self.single_endpoint_spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {
                "/users": {
                    "get": {
                        "description": "List all users",
                        "parameters": [
                            {
                                "name": "limit",
                                "in": "query",
                                "schema": {"type": "integer"},
                            }
                        ],
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "id": {"type": "integer"},
                                                    "name": {"type": "string"},
                                                },
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

        self.multi_endpoint_spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {
                "/users": {
                    "get": {
                        "description": "List users",
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {"type": "array"}
                                    }
                                }
                            }
                        },
                    },
                    "post": {
                        "description": "Create user",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "email": {"type": "string"},
                                        },
                                        "required": ["name", "email"],
                                    }
                                }
                            }
                        },
                        "responses": {
                            "201": {
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
                    },
                },
                "/users/{id}": {
                    "get": {
                        "description": "Get user by ID",
                        "parameters": [
                            {
                                "name": "id",
                                "in": "path",
                                "required": True,
                                "schema": {"type": "integer"},
                            }
                        ],
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
                },
            },
        }

    def test_parse_single_endpoint(self):
        """Parse a single endpoint from spec dict."""
        endpoints = parse_openapi_spec(self.single_endpoint_spec)
        self.assertEqual(len(endpoints), 1)
        endpoint = endpoints[0]
        self.assertIsInstance(endpoint, Endpoint)
        self.assertEqual(endpoint.path, "/users")
        self.assertEqual(endpoint.method, "get")

    def test_parse_multiple_endpoints(self):
        """Parse multiple endpoints from spec dict."""
        endpoints = parse_openapi_spec(self.multi_endpoint_spec)
        self.assertEqual(len(endpoints), 3)
        paths_methods = [(e.path, e.method) for e in endpoints]
        self.assertIn(("/users", "get"), paths_methods)
        self.assertIn(("/users", "post"), paths_methods)
        self.assertIn(("/users/{id}", "get"), paths_methods)

    def test_extract_request_schema(self):
        """Extract request_schema correctly from requestBody."""
        endpoints = parse_openapi_spec(self.multi_endpoint_spec)
        post_endpoint = [e for e in endpoints if e.method == "post"][0]
        self.assertIsNotNone(post_endpoint.request_schema)
        self.assertEqual(post_endpoint.request_schema["type"], "object")
        self.assertIn("name", post_endpoint.request_schema["properties"])

    def test_extract_response_schema(self):
        """Extract response_schema correctly from responses."""
        endpoints = parse_openapi_spec(self.single_endpoint_spec)
        endpoint = endpoints[0]
        self.assertIsNotNone(endpoint.response_schema)
        self.assertEqual(endpoint.response_schema["type"], "array")

    def test_handle_missing_description(self):
        """Handle missing optional description field."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0.0"},
            "paths": {
                "/health": {
                    "get": {
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {
                                        "schema": {"type": "object"}
                                    }
                                }
                            }
                        }
                    }
                }
            },
        }
        endpoints = parse_openapi_spec(spec)
        self.assertEqual(len(endpoints), 1)
        self.assertEqual(endpoints[0].description, "")

    def test_handle_missing_parameters(self):
        """Handle missing optional parameters field."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0.0"},
            "paths": {
                "/health": {
                    "get": {
                        "description": "Health check",
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
        endpoints = parse_openapi_spec(spec)
        self.assertEqual(endpoints[0].parameters, [])

    def test_parse_path_parameters(self):
        """Parse path parameters correctly."""
        endpoints = parse_openapi_spec(self.multi_endpoint_spec)
        get_by_id = [e for e in endpoints if e.path == "/users/{id}"][0]
        self.assertEqual(len(get_by_id.parameters), 1)
        self.assertEqual(get_by_id.parameters[0]["name"], "id")
        self.assertEqual(get_by_id.parameters[0]["in"], "path")

    def test_parse_query_parameters(self):
        """Parse query parameters correctly."""
        endpoints = parse_openapi_spec(self.single_endpoint_spec)
        endpoint = endpoints[0]
        self.assertEqual(len(endpoint.parameters), 1)
        self.assertEqual(endpoint.parameters[0]["name"], "limit")
        self.assertEqual(endpoint.parameters[0]["in"], "query")

    def test_endpoint_has_correct_fields(self):
        """Endpoint dataclass has all expected fields."""
        endpoints = parse_openapi_spec(self.single_endpoint_spec)
        endpoint = endpoints[0]
        self.assertTrue(hasattr(endpoint, "path"))
        self.assertTrue(hasattr(endpoint, "method"))
        self.assertTrue(hasattr(endpoint, "description"))
        self.assertTrue(hasattr(endpoint, "request_schema"))
        self.assertTrue(hasattr(endpoint, "response_schema"))
        self.assertTrue(hasattr(endpoint, "parameters"))

    def test_request_schema_none_when_no_request_body(self):
        """request_schema is None when endpoint has no requestBody."""
        endpoints = parse_openapi_spec(self.single_endpoint_spec)
        endpoint = endpoints[0]
        self.assertIsNone(endpoint.request_schema)

if __name__ == "__main__":
    unittest.main()
