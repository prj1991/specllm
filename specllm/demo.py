"""specllm demo — try it in 30 seconds, no API key needed.

Usage:
    python -m specllm.demo

Starts a local server with a mock LLM. Try the curl commands printed to the console.
"""

import json
import threading
import time
from specllm import SpecLLM
from specllm.llm.providers import MockProvider

SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "specllm Demo", "version": "1.0"},
    "paths": {
        "/v1/route-ticket": {
            "post": {
                "description": "Route a support ticket to the right team with priority",
                "requestBody": {"content": {"application/json": {"schema": {
                    "type": "object",
                    "required": ["title", "body"],
                    "properties": {
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                    },
                }}}},
                "responses": {"200": {"content": {"application/json": {"schema": {
                    "type": "object",
                    "required": ["team", "priority"],
                    "properties": {
                        "team": {"type": "string", "enum": ["billing", "engineering", "support"]},
                        "priority": {"type": "integer", "minimum": 1, "maximum": 5},
                    },
                }}}}},
            }
        },
        "/v1/classify-intent": {
            "post": {
                "description": "Classify text into one of the caller-provided intents",
                "requestBody": {"content": {"application/json": {"schema": {
                    "type": "object",
                    "required": ["text", "intents"],
                    "properties": {
                        "text": {"type": "string"},
                        "intents": {"type": "array", "items": {"type": "string"}},
                    },
                }}}},
                "responses": {"200": {"content": {"application/json": {"schema": {
                    "type": "object",
                    "required": ["intent", "confidence"],
                    "properties": {
                        "intent": {"type": "string", "x-constrain-from": "intents"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                }}}}},
            }
        },
    },
}

MOCK_RESPONSES = [
    {"team": "billing", "priority": 2},
    {"intent": "refund", "confidence": 0.92},
]


def main():
    provider = MockProvider(responses=MOCK_RESPONSES)
    app = SpecLLM(spec=SPEC, provider=provider)

    port = 8080
    print("""
╔══════════════════════════════════════════════════════════════╗
║  specllm demo — running on http://127.0.0.1:8080           ║
╚══════════════════════════════════════════════════════════════╝

Try these:

  curl -s -X POST http://127.0.0.1:8080/v1/route-ticket \\
    -H "Content-Type: application/json" \\
    -d '{"title": "Charged twice", "body": "My card was charged twice for order #123"}' | python -m json.tool

  curl -s -X POST http://127.0.0.1:8080/v1/classify-intent \\
    -H "Content-Type: application/json" \\
    -d '{"text": "I want my money back", "intents": ["refund", "shipping", "account"]}' | python -m json.tool

  # This will fail validation (missing required field):
  curl -s -X POST http://127.0.0.1:8080/v1/route-ticket \\
    -H "Content-Type: application/json" \\
    -d '{"title": "no body field"}' | python -m json.tool

Press Ctrl+C to stop.
""")

    from specllm.server.app import SpecLLMServer
    server = SpecLLMServer(app=app, host="127.0.0.1", port=port)
    server.serve()


if __name__ == "__main__":
    main()
