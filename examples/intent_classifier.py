"""Example: Intent classifier with caller-provided options.

The caller controls which intents are valid per request using x-constrain-from.
"""

from specllm import SpecLLM
from specllm.llm.providers import MockProvider

SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Intent Classifier", "version": "1.0"},
    "paths": {
        "/v1/classify": {
            "post": {
                "description": "Classify the text into one of the provided intents.",
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
        }
    },
}

# Replace MockProvider with a real LLM provider for production use
provider = MockProvider(responses=[{"intent": "billing", "confidence": 0.92}])
app = SpecLLM(spec=SPEC, provider=provider)

result = app.test_client().post("/v1/classify", json_body={
    "text": "I was charged twice on my credit card",
    "intents": ["billing", "shipping", "account_cancellation", "technical_support"],
})

print(result)
# → {"intent": "billing", "confidence": 0.92}

# To serve as a live HTTP API:
# app.serve(port=8080)
