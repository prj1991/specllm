"""Resolves x-constrain-from in response schemas at request time.

In OpenAPI spec, add to any response field:

    intent:
      type: string
      x-constrain-from: "options"           # enum from body["options"]

    score:
      type: integer
      x-constrain-from:
        minimum: "min_score"                # maps schema keyword → body field
        maximum: "max_score"
"""

import copy
from typing import Any


def resolve_constraints(response_schema: dict, request_body: dict) -> dict:
    """Return a schema copy with x-constrain-from resolved from request_body."""
    if not response_schema or not request_body:
        return response_schema
    result = copy.deepcopy(response_schema)
    _resolve(result, request_body)
    return result


def _resolve(schema: dict, body: dict) -> None:
    if not isinstance(schema, dict):
        return
    constraint = schema.pop("x-constrain-from", None)
    if constraint is not None:
        mapping = {"enum": constraint} if isinstance(constraint, str) else constraint
        for keyword, field_ref in mapping.items():
            val = body.get(field_ref)
            if val is not None:
                schema[keyword] = val
    for prop in schema.get("properties", {}).values():
        _resolve(prop, body)
    if isinstance(schema.get("items"), dict):
        _resolve(schema["items"], body)
