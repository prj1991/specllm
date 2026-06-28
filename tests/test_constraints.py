"""Unit tests for resolve_constraints (x-constrain-from schema resolution)."""

from specllm.pipeline.constraints import resolve_constraints


class TestResolveConstraints:
    def test_enum_from_request_field(self):
        schema = {"type": "object", "properties": {
            "intent": {"type": "string", "x-constrain-from": "options"},
        }}
        result = resolve_constraints(schema, {"options": ["billing", "shipping"]})
        assert result["properties"]["intent"]["enum"] == ["billing", "shipping"]
        assert "x-constrain-from" not in result["properties"]["intent"]

    def test_numeric_range(self):
        schema = {"type": "object", "properties": {
            "score": {"type": "integer", "x-constrain-from": {"minimum": "min", "maximum": "max"}},
        }}
        result = resolve_constraints(schema, {"min": 1, "max": 10})
        assert result["properties"]["score"]["minimum"] == 1
        assert result["properties"]["score"]["maximum"] == 10

    def test_array_items_enum(self):
        schema = {"type": "object", "properties": {
            "tags": {"type": "array", "items": {"type": "string", "x-constrain-from": "labels"}},
        }}
        result = resolve_constraints(schema, {"labels": ["bug", "feature"]})
        assert result["properties"]["tags"]["items"]["enum"] == ["bug", "feature"]

    def test_does_not_mutate_original(self):
        schema = {"type": "object", "properties": {"x": {"type": "string", "x-constrain-from": "opts"}}}
        resolve_constraints(schema, {"opts": ["a"]})
        assert "x-constrain-from" in schema["properties"]["x"]
        assert "enum" not in schema["properties"]["x"]

    def test_missing_field_no_enum_set(self):
        schema = {"type": "object", "properties": {"x": {"type": "string", "x-constrain-from": "opts"}}}
        result = resolve_constraints(schema, {"other": ["a"]})
        assert "enum" not in result["properties"]["x"]

    def test_no_constraints_noop(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        assert resolve_constraints(schema, {"a": 1}) == schema

    def test_none_schema(self):
        assert resolve_constraints(None, {"x": 1}) is None
