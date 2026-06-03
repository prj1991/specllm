"""JSON Schema validator.

Supported JSON Schema keywords:
    - type: string, integer, number, boolean, array, object, null
    - required: list of required property names (for objects)
    - properties: property name to sub-schema mapping (for objects)
    - items: sub-schema for array items
    - enum: list of allowed values
    - minimum / maximum: numeric range constraints

Not yet implemented (planned):
    - oneOf, anyOf, allOf: schema composition
    - pattern: regex pattern matching for strings
    - format: semantic format validation (e.g., date-time, email, uri)
    - minLength / maxLength: string length constraints
    - minItems / maxItems: array length constraints
    - additionalProperties: control over extra object keys
    - $ref resolution (handled in parser instead)
"""

from dataclasses import dataclass
from typing import Any, List


@dataclass
class ValidationError:
    """Represents a schema validation error."""

    path: str
    message: str


def validate_schema(data: Any, schema: dict) -> List[ValidationError]:
    """Validate data against a JSON Schema."""
    errors: List[ValidationError] = []
    _validate(data, schema, "", errors)
    return errors


def _validate(data: Any, schema: dict, path: str, errors: List[ValidationError]) -> None:
    """Recursively validate data against schema."""
    schema_type = schema.get("type")
    if schema_type:
        if not _check_type(data, schema_type):
            errors.append(
                ValidationError(
                    path=path or ".", message=f"Expected type '{schema_type}' but got '{type(data).__name__}'"
                )
            )
            return

    if "enum" in schema:
        if data not in schema["enum"]:
            errors.append(ValidationError(path=path or ".", message=f"Value not in enum: {schema['enum']}"))

    if "minimum" in schema and isinstance(data, (int, float)) and not isinstance(data, bool):
        if data < schema["minimum"]:
            errors.append(
                ValidationError(path=path or ".", message=f"Value {data} is below minimum {schema['minimum']}")
            )

    if "maximum" in schema and isinstance(data, (int, float)) and not isinstance(data, bool):
        if data > schema["maximum"]:
            errors.append(
                ValidationError(path=path or ".", message=f"Value {data} exceeds maximum {schema['maximum']}")
            )

    if schema_type == "object" and isinstance(data, dict):
        required = schema.get("required", [])
        for field_name in required:
            if field_name not in data:
                field_path = f"{path}.{field_name}" if path else field_name
                errors.append(ValidationError(path=field_path, message=f"Required field '{field_name}' is missing"))

        properties = schema.get("properties", {})
        for prop_name, prop_schema in properties.items():
            if prop_name in data:
                field_path = f"{path}.{prop_name}" if path else prop_name
                _validate(data[prop_name], prop_schema, field_path, errors)

    if schema_type == "array" and isinstance(data, list):
        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(data):
                item_path = f"{path}[{i}]" if path else f"[{i}]"
                _validate(item, items_schema, item_path, errors)


def _check_type(value: Any, schema_type: str) -> bool:
    """Check if a value matches the expected JSON Schema type."""
    if schema_type == "string":
        return isinstance(value, str)
    elif schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    elif schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    elif schema_type == "boolean":
        return isinstance(value, bool)
    elif schema_type == "array":
        return isinstance(value, list)
    elif schema_type == "object":
        return isinstance(value, dict)
    elif schema_type == "null":
        return value is None
    return True
