"""Tests for specllm.spec.validator module - JSON Schema validation."""
import unittest

from specllm.spec.validator import validate_schema, ValidationError

class TestValidateSchema(unittest.TestCase):
    """Test JSON Schema validation functionality."""

    def test_valid_data_passes_with_no_errors(self):
        """Valid data produces no validation errors."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        data = {"name": "Alice", "age": 30}
        errors = validate_schema(data, schema)
        self.assertEqual(errors, [])

    def test_validate_required_fields_missing(self):
        """Missing required field produces a validation error."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
            },
            "required": ["name", "email"],
        }
        data = {"name": "Alice"}
        errors = validate_schema(data, schema)
        self.assertGreater(len(errors), 0)
        self.assertIsInstance(errors[0], ValidationError)
        self.assertIn("email", errors[0].message)

    def test_validate_type_string(self):
        """String type violation produces error."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        data = {"name": 123}
        errors = validate_schema(data, schema)
        self.assertGreater(len(errors), 0)
        self.assertIn("name", errors[0].path)

    def test_validate_type_integer(self):
        """Integer type violation produces error."""
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        }
        data = {"count": "not_a_number"}
        errors = validate_schema(data, schema)
        self.assertGreater(len(errors), 0)

    def test_validate_type_number(self):
        """Number type violation produces error."""
        schema = {
            "type": "object",
            "properties": {"price": {"type": "number"}},
        }
        data = {"price": "expensive"}
        errors = validate_schema(data, schema)
        self.assertGreater(len(errors), 0)

    def test_validate_type_boolean(self):
        """Boolean type violation produces error."""
        schema = {
            "type": "object",
            "properties": {"active": {"type": "boolean"}},
        }
        data = {"active": "yes"}
        errors = validate_schema(data, schema)
        self.assertGreater(len(errors), 0)

    def test_validate_type_array(self):
        """Array type violation produces error."""
        schema = {
            "type": "object",
            "properties": {"tags": {"type": "array"}},
        }
        data = {"tags": "not_an_array"}
        errors = validate_schema(data, schema)
        self.assertGreater(len(errors), 0)

    def test_validate_type_object(self):
        """Object type violation produces error."""
        schema = {
            "type": "object",
            "properties": {"metadata": {"type": "object"}},
        }
        data = {"metadata": "not_an_object"}
        errors = validate_schema(data, schema)
        self.assertGreater(len(errors), 0)

    def test_validate_enum_values(self):
        """Enum violation produces error when value not in allowed set."""
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["active", "inactive"]}
            },
        }
        data = {"status": "unknown"}
        errors = validate_schema(data, schema)
        self.assertGreater(len(errors), 0)
        self.assertIn("enum", errors[0].message.lower())

    def test_validate_enum_valid_value(self):
        """Valid enum value produces no error."""
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["active", "inactive"]}
            },
        }
        data = {"status": "active"}
        errors = validate_schema(data, schema)
        self.assertEqual(errors, [])

    def test_validate_minimum(self):
        """Number below minimum produces error."""
        schema = {
            "type": "object",
            "properties": {"age": {"type": "integer", "minimum": 0}},
        }
        data = {"age": -1}
        errors = validate_schema(data, schema)
        self.assertGreater(len(errors), 0)
        self.assertIn("minimum", errors[0].message.lower())

    def test_validate_maximum(self):
        """Number above maximum produces error."""
        schema = {
            "type": "object",
            "properties": {"age": {"type": "integer", "maximum": 150}},
        }
        data = {"age": 200}
        errors = validate_schema(data, schema)
        self.assertGreater(len(errors), 0)
        self.assertIn("maximum", errors[0].message.lower())

    def test_validate_nested_objects(self):
        """Nested object validation catches deep violations."""
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "zip": {"type": "string"},
                    },
                    "required": ["street"],
                }
            },
        }
        data = {"address": {"zip": "12345"}}
        errors = validate_schema(data, schema)
        self.assertGreater(len(errors), 0)
        self.assertIn("street", errors[0].message)

    def test_validate_arrays_with_items_schema(self):
        """Array items are validated against items schema."""
        schema = {
            "type": "object",
            "properties": {
                "scores": {
                    "type": "array",
                    "items": {"type": "integer"},
                }
            },
        }
        data = {"scores": [1, 2, "three"]}
        errors = validate_schema(data, schema)
        self.assertGreater(len(errors), 0)

    def test_multiple_violations_reported(self):
        """Multiple violations are all reported in a single call."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        data = {}
        errors = validate_schema(data, schema)
        self.assertGreaterEqual(len(errors), 2)

    def test_validation_error_has_path_and_message(self):
        """ValidationError has path and message fields."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        data = {}
        errors = validate_schema(data, schema)
        self.assertTrue(hasattr(errors[0], "path"))
        self.assertTrue(hasattr(errors[0], "message"))
        self.assertIsInstance(errors[0].path, str)
        self.assertIsInstance(errors[0].message, str)

if __name__ == "__main__":
    unittest.main()
