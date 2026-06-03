"""OpenAPI JSON spec parser."""

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Endpoint:
    """Represents a parsed API endpoint."""

    path: str
    method: str
    description: str
    request_schema: Optional[Dict] = None
    response_schema: Optional[Dict] = None
    parameters: List[Dict] = field(default_factory=list)


def _resolve_refs(schema: Any, root_spec: dict) -> Any:
    """Resolve $ref references in a schema against the root spec.

    Follows JSON Pointer paths (e.g., "#/components/schemas/User") and
    recursively resolves any nested $ref values.
    """
    if schema is None:
        return None
    if isinstance(schema, list):
        return [_resolve_refs(item, root_spec) for item in schema]
    if not isinstance(schema, dict):
        return schema
    if "$ref" in schema:
        ref_path = schema["$ref"]
        if ref_path.startswith("#/"):
            parts = ref_path[2:].split("/")
            resolved = root_spec
            for part in parts:
                resolved = resolved.get(part, {})
            return _resolve_refs(copy.deepcopy(resolved), root_spec)
        return schema
    result = {}
    for key, value in schema.items():
        result[key] = _resolve_refs(value, root_spec)
    return result


def parse_openapi_spec(spec: dict) -> List[Endpoint]:
    """Parse an OpenAPI 3.0 spec dict into a list of Endpoint objects."""
    endpoints: List[Endpoint] = []
    paths = spec.get("paths", {})

    for path, path_item in paths.items():
        for method, operation in path_item.items():
            if method.startswith("x-"):
                continue

            description: str = operation.get("description", "")
            parameters: List[Dict] = operation.get("parameters", [])

            request_schema: Optional[Dict] = None
            request_body = operation.get("requestBody")
            if request_body:
                content = request_body.get("content", {})
                json_content = content.get("application/json", {})
                request_schema = json_content.get("schema")

            response_schema: Optional[Dict] = None
            responses = operation.get("responses", {})
            for status_code, response_obj in responses.items():
                content = response_obj.get("content", {})
                json_content = content.get("application/json", {})
                schema = json_content.get("schema")
                if schema:
                    response_schema = schema
                    break

            if request_schema:
                request_schema = _resolve_refs(request_schema, spec)
            if response_schema:
                response_schema = _resolve_refs(response_schema, spec)

            endpoints.append(
                Endpoint(
                    path=path,
                    method=method,
                    description=description,
                    request_schema=request_schema,
                    response_schema=response_schema,
                    parameters=parameters,
                )
            )

    return endpoints
