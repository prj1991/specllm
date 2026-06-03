"""
Integration test for specllm using a real LLM (Claude via AWS Bedrock).

Runs the full pipeline end-to-end: spec parsing → input validation → prompt
generation → LLM call → output validation → retry → caching.

REQUIREMENTS:
    - AWS credentials with Bedrock access (via environment, profile, or IAM role)
    - boto3 installed
    - Network access to AWS Bedrock

This script is NOT part of the standard test suite. It is meant to be run
manually to validate real LLM behavior. The standard test suite
(pytest tests/) uses MockProvider and requires no credentials or network.

Usage:
    python test_integration_real_llm.py
"""
import json
import os
import sys
import traceback

try:
    import boto3
except ImportError:
    print("SKIP: boto3 not installed. Install with: pip install boto3")
    sys.exit(0)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from specllm import SpecLLM
from specllm.llm.providers import LLMProvider


class BedrockProvider(LLMProvider):
    """Real Claude provider via AWS Bedrock."""

    def __init__(self, model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0", region="us-east-1"):
        import boto3
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.call_count = 0
        self.total_tokens = 0

    def call(self, prompt: str, system_prompt: str = None) -> dict:
        self.call_count += 1
        system = system_prompt or (
            "You are a backend API. Respond ONLY with valid JSON matching the schema described. "
            "No markdown, no code fences, no explanation. Just the raw JSON object."
        )

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            }),
            contentType="application/json",
        )
        body = json.loads(response["body"].read())
        text = "".join(b["text"] for b in body.get("content", []) if b.get("type") == "text")
        self.total_tokens += body.get("usage", {}).get("input_tokens", 0)
        self.total_tokens += body.get("usage", {}).get("output_tokens", 0)

        # Strip markdown fences if LLM wraps in ```json
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Return something the validator will reject, triggering retry
            return {"__parse_error__": text[:200]}


# ─── Test Spec ────────────────────────────────────────────────────────────────

SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Test API", "version": "1.0.0"},
    "paths": {
        "/v1/route-ticket": {
            "post": {
                "description": "Route a support ticket to the appropriate engineering team based on the title and body content.",
                "requestBody": {"content": {"application/json": {"schema": {
                    "type": "object", "required": ["title", "body"],
                    "properties": {
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                        "customer_tier": {"type": "string", "enum": ["free", "pro", "enterprise"]},
                    },
                }}}},
                "responses": {"200": {"content": {"application/json": {"schema": {
                    "type": "object", "required": ["team", "priority"],
                    "properties": {
                        "team": {"type": "string"},
                        "priority": {"type": "integer", "minimum": 1, "maximum": 5},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                }}}}},
            }
        },
        "/v1/classify-sentiment": {
            "post": {
                "description": "Classify the sentiment of the given text as positive, negative, or neutral. Return a confidence score between 0 and 1.",
                "requestBody": {"content": {"application/json": {"schema": {
                    "type": "object", "required": ["text"],
                    "properties": {"text": {"type": "string"}},
                }}}},
                "responses": {"200": {"content": {"application/json": {"schema": {
                    "type": "object", "required": ["sentiment", "confidence"],
                    "properties": {
                        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                }}}}},
            }
        },
        "/v1/extract-entities": {
            "post": {
                "description": "Extract named entities (people, organizations, locations) from the given text.",
                "requestBody": {"content": {"application/json": {"schema": {
                    "type": "object", "required": ["text"],
                    "properties": {"text": {"type": "string"}},
                }}}},
                "responses": {"200": {"content": {"application/json": {"schema": {
                    "type": "object", "required": ["entities"],
                    "properties": {"entities": {"type": "array", "items": {
                        "type": "object", "required": ["name", "type"],
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": ["person", "organization", "location"]},
                        },
                    }}},
                }}}}},
            }
        },
    },
}


# ─── Test Runner ──────────────────────────────────────────────────────────────

def run_tests():
    print("=" * 70)
    print("specllm REAL LLM INTEGRATION TEST (Bedrock Claude Haiku 4.5)")
    print("=" * 70)

    provider = BedrockProvider()
    app = SpecLLM(spec=SPEC, provider=provider)
    client = app.test_client()

    results = {"passed": 0, "failed": 0, "errors": []}

    def check(name, condition, detail=""):
        if condition:
            results["passed"] += 1
            print(f"  ✅ {name}")
        else:
            results["failed"] += 1
            results["errors"].append(f"{name}: {detail}")
            print(f"  ❌ {name}: {detail}")

    # ─── TEST 1: Ticket routing (happy path) ──────────────────────────────────
    print("\n── TEST 1: Ticket routing ──")
    resp = client.post("/v1/route-ticket", json_body={
        "title": "Payment failed on checkout",
        "body": "I tried to pay with my credit card but got error code E-5021 during checkout",
        "customer_tier": "pro",
    })
    print(f"  Response: {json.dumps(resp, indent=2)}")
    check("No error", "error" not in resp, str(resp.get("error", ""))[:100])
    if "error" not in resp:
        check("Has 'team'", "team" in resp)
        check("Has 'priority'", "priority" in resp)
        check("priority is int", isinstance(resp.get("priority"), int), f"type={type(resp.get('priority'))}")
        check("priority in [1,5]", 1 <= resp.get("priority", 0) <= 5, f"val={resp.get('priority')}")
        if "tags" in resp:
            check("tags is string[]", isinstance(resp["tags"], list) and all(isinstance(t, str) for t in resp["tags"]))

    # ─── TEST 2: Input validation (missing required) ──────────────────────────
    print("\n── TEST 2: Missing required field ──")
    resp = client.post("/v1/route-ticket", json_body={"title": "Only title"})
    print(f"  Response: {json.dumps(resp, indent=2)}")
    check("Returns error", "error" in resp)
    check("Code=INPUT_VALIDATION_FAILED", resp.get("error", {}).get("code") == "INPUT_VALIDATION_FAILED")

    # ─── TEST 3: Sentiment positive ───────────────────────────────────────────
    print("\n── TEST 3: Sentiment (positive text) ──")
    resp = client.post("/v1/classify-sentiment", json_body={
        "text": "I absolutely love this product! Best purchase I've ever made."
    })
    print(f"  Response: {json.dumps(resp, indent=2)}")
    check("No error", "error" not in resp, str(resp.get("error", ""))[:100])
    if "error" not in resp:
        check("Has sentiment", "sentiment" in resp)
        check("Has confidence", "confidence" in resp)
        check("Valid enum", resp.get("sentiment") in ["positive", "negative", "neutral"], f"got {resp.get('sentiment')}")
        check("Confidence in [0,1]", 0 <= resp.get("confidence", -1) <= 1, f"got {resp.get('confidence')}")
        check("Detected positive", resp.get("sentiment") == "positive", f"got {resp.get('sentiment')}")

    # ─── TEST 4: Sentiment negative ───────────────────────────────────────────
    print("\n── TEST 4: Sentiment (negative text) ──")
    resp = client.post("/v1/classify-sentiment", json_body={
        "text": "This is terrible. Worst experience ever. I want a refund immediately."
    })
    print(f"  Response: {json.dumps(resp, indent=2)}")
    check("No error", "error" not in resp, str(resp.get("error", ""))[:100])
    if "error" not in resp:
        check("Detected negative", resp.get("sentiment") == "negative", f"got {resp.get('sentiment')}")

    # ─── TEST 5: Entity extraction ────────────────────────────────────────────
    print("\n── TEST 5: Entity extraction (nested schema) ──")
    resp = client.post("/v1/extract-entities", json_body={
        "text": "Tim Cook announced that Apple will open a new office in Austin, Texas."
    })
    print(f"  Response: {json.dumps(resp, indent=2)}")
    check("No error", "error" not in resp, str(resp.get("error", ""))[:100])
    if "error" not in resp:
        check("Has entities[]", isinstance(resp.get("entities"), list))
        entities = resp.get("entities", [])
        if entities:
            check("Entity has name+type", all("name" in e and "type" in e for e in entities))
            types = [e.get("type") for e in entities]
            check("All types valid enum", all(t in ["person", "organization", "location"] for t in types), f"types={types}")
            names_lower = [e.get("name", "").lower() for e in entities]
            check("Found Tim Cook", any("tim" in n or "cook" in n for n in names_lower), f"names={names_lower}")
            check("Found Apple", any("apple" in n for n in names_lower), f"names={names_lower}")

    # ─── TEST 6: Endpoint not found ───────────────────────────────────────────
    print("\n── TEST 6: Unknown endpoint ──")
    resp = client.post("/v1/nonexistent", json_body={"x": 1})
    check("Returns error", "error" in resp)
    check("Code=ENDPOINT_NOT_FOUND", resp.get("error", {}).get("code") == "ENDPOINT_NOT_FOUND")

    # ─── TEST 7: Caching ─────────────────────────────────────────────────────
    print("\n── TEST 7: Caching (same request → no LLM call) ──")
    app._pipeline.cache._store.clear()  # fresh start
    before = provider.call_count
    r1 = client.post("/v1/classify-sentiment", json_body={"text": "The sky is blue."})
    calls_first = provider.call_count - before

    before = provider.call_count
    r2 = client.post("/v1/classify-sentiment", json_body={"text": "The sky is blue."})
    calls_second = provider.call_count - before

    print(f"  First: {calls_first} LLM call(s), Second: {calls_second} LLM call(s)")
    check("Second call uses cache (0 calls)", calls_second == 0, f"got {calls_second}")
    check("Cached result matches", r1 == r2)
    check("Metadata reports cache_hit", app._pipeline.last_metadata.get("cache_hit") is True)

    # ─── TEST 8: Custom prompt ────────────────────────────────────────────────
    print("\n── TEST 8: Custom prompt decorator ──")

    @app.prompt("/v1/classify-sentiment", "post")
    def custom_prompt(body):
        return (
            f"Classify sentiment. Respond ONLY with JSON.\n"
            f"Text: \"{body['text']}\"\n"
            f'Schema: {{"sentiment": "positive"|"negative"|"neutral", "confidence": 0.0-1.0}}'
        )

    app._pipeline.cache._store.clear()
    resp = client.post("/v1/classify-sentiment", json_body={"text": "Meh, it's okay I guess."})
    print(f"  Response: {json.dumps(resp, indent=2)}")
    check("No error", "error" not in resp, str(resp.get("error", ""))[:100])
    if "error" not in resp:
        check("Sentiment is neutral", resp.get("sentiment") == "neutral", f"got {resp.get('sentiment')}")

    # ─── TEST 9: Validate retry kicks in on schema violation ──────────────────
    print("\n── TEST 9: Retry mechanism (force schema violation) ──")
    # Use a provider that returns bad output first, then good output
    from specllm.llm.providers import MockProvider
    bad_then_good = MockProvider(responses=[
        {"team": "billing", "priority": "HIGH"},  # wrong type: string not int
        {"team": "billing", "priority": 99},       # out of range
        {"team": "billing", "priority": 3, "tags": ["payment"]},  # valid
    ])
    test_app = SpecLLM(spec=SPEC, provider=bad_then_good)
    test_client = test_app.test_client()
    resp = test_client.post("/v1/route-ticket", json_body={"title": "test", "body": "test"})
    print(f"  Response: {json.dumps(resp, indent=2)}")
    check("Retry recovered valid response", "error" not in resp and resp.get("priority") == 3)
    check("Provider called 3 times", bad_then_good._call_count == 3, f"calls={bad_then_good._call_count}")

    # ─── TEST 10: Retry exhaustion → 422 ─────────────────────────────────────
    print("\n── TEST 10: Retry exhaustion → OUTPUT_SCHEMA_VIOLATION ──")
    always_bad = MockProvider(responses=[{"team": "x", "priority": "not_an_int"}])
    test_app2 = SpecLLM(spec=SPEC, provider=always_bad)
    resp = test_app2.test_client().post("/v1/route-ticket", json_body={"title": "t", "body": "b"})
    print(f"  Response: {json.dumps(resp, indent=2)}")
    check("Returns error", "error" in resp)
    check("Code=OUTPUT_SCHEMA_VIOLATION", resp.get("error", {}).get("code") == "OUTPUT_SCHEMA_VIOLATION")

    # ─── TEST 11: $ref resolution (if used) ───────────────────────────────────
    print("\n── TEST 11: $ref resolution in spec ──")
    spec_with_ref = {
        "openapi": "3.0.0", "info": {"title": "t", "version": "1"},
        "components": {"schemas": {"Priority": {"type": "integer", "minimum": 1, "maximum": 5}}},
        "paths": {"/v1/test-ref": {"post": {
            "description": "Return a priority value",
            "requestBody": {"content": {"application/json": {"schema": {"type": "object", "properties": {"q": {"type": "string"}}}}}},
            "responses": {"200": {"content": {"application/json": {"schema": {
                "type": "object", "required": ["priority"],
                "properties": {"priority": {"$ref": "#/components/schemas/Priority"}},
            }}}}},
        }}},
    }
    ref_app = SpecLLM(spec=spec_with_ref, provider=MockProvider(responses=[{"priority": 3}]))
    resp = ref_app.test_client().post("/v1/test-ref", json_body={"q": "test"})
    check("$ref resolves correctly", resp == {"priority": 3})

    # Bad value should fail validation
    ref_app2 = SpecLLM(spec=spec_with_ref, provider=MockProvider(responses=[{"priority": 10}]))
    resp2 = ref_app2.test_client().post("/v1/test-ref", json_body={"q": "test"})
    check("$ref validation enforced (priority>5 rejected)", "error" in resp2, f"resp={resp2}")

    # ─── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"RESULTS: {results['passed']} passed, {results['failed']} failed")
    print(f"LLM calls: {provider.call_count} | Tokens: ~{provider.total_tokens}")
    print("=" * 70)

    if results["errors"]:
        print("\nFAILURES:")
        for e in results["errors"]:
            print(f"  • {e}")

    return results["failed"] == 0


if __name__ == "__main__":
    try:
        success = run_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 FATAL: {e}")
        traceback.print_exc()
        sys.exit(2)
