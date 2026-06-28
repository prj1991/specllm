"""Prompt generator for LLM calls based on endpoint specs."""

import json
from typing import Optional

from specllm.spec.parser import Endpoint


def generate_prompt(
    endpoint: Endpoint,
    request_body: Optional[dict] = None,
    response_schema: Optional[dict] = None,
) -> str:
    """Generate a prompt from an endpoint spec and optional request body.

    If response_schema is provided, it overrides endpoint.response_schema
    (used for dynamic constraints).
    """
    parts: list = []

    if endpoint.description:
        parts.append(f"Task: {endpoint.description}")

    parts.append("")
    parts.append("You must respond with valid JSON matching the following schema:")
    parts.append("")

    schema = response_schema or endpoint.response_schema
    if schema:
        parts.append(json.dumps(schema, indent=2))

    if request_body is not None:
        parts.append("")
        parts.append("Request body:")
        parts.append(json.dumps(request_body, indent=2))

    return "\n".join(parts)
