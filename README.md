# specllm

**The production framework for spec-first LLM APIs.**

You define a REST API spec. specllm turns it into a live service where every response is powered by an LLM — but validated, cached, and retried automatically. The caller never knows an LLM is involved. It just looks like a normal, reliable API.

```
┌──────────┐       ┌─────────────────────────────────────┐       ┌─────────┐
│ Your App │──────▶│ specllm                             │──────▶│   LLM   │
│          │◀──────│ (validates, retries, caches, serves) │◀──────│(any LLM)│
└──────────┘ JSON  └─────────────────────────────────────┘ JSON  └─────────┘
```

- Your app sends a normal HTTP request
- specllm checks if the request is valid (rejects garbage instantly, no LLM cost)
- Builds a prompt from your spec, calls the LLM
- Checks if the LLM response matches your schema — if not, retries with error feedback
- Returns a guaranteed JSON response, or a structured error explaining what went wrong

```python
from specllm import SpecLLM

app = SpecLLM.from_openapi("./api.yaml", provider=my_provider)
app.serve(port=8080)
# Your spec is now a live, schema-validated, observable REST API.
```

## Quick Start

```bash
pip install specllm
pip install specllm[yaml]  # optional, for YAML specs
```

### Serve a live API

```python
from specllm import SpecLLM

app = SpecLLM.from_openapi("./my-api.yaml", provider=my_provider)
app.serve(port=8080)
```

### Test without a server

```python
from specllm import SpecLLM
from specllm.llm.providers import MockProvider

provider = MockProvider(responses=[{"team": "billing", "priority": 2}])
app = SpecLLM.from_openapi("./my-api.json", provider=provider)

result = app.test_client().post("/v1/route-ticket", json_body={"title": "Help", "body": "..."})
```

### Test with a real LLM but no credentials in CI

```python
from specllm.testing.record_replay import RecordReplayProvider

# Record (locally, with credentials):
recorder = RecordReplayProvider(provider=real_provider, cassette="tests/tape.json")

# Replay (in CI, no credentials):
replayer = RecordReplayProvider(cassette="tests/tape.json")
app = SpecLLM.from_openapi("spec.json", provider=replayer)
```

## How It Works

```
Request → [Input Validation] → [Cache Check] → [Build Prompt] → [Call LLM]
                                                                      ↓
Response ← [Cache Store] ← [Output Validation] ←──── Valid? ←── [LLM Response]
                                                       ↓ No
                                              [Retry with feedback] (up to 3x)
                                                       ↓ Still invalid
                                              [422 with clear error]
```

1. **Validate input** against your request schema — bad requests get instant 400, zero LLM cost
2. **Check cache** — identical requests return cached responses
3. **Build prompt** from your endpoint description + request body + response schema
4. **Call LLM** with timeout enforcement and fallback provider support
5. **Validate output** against your response schema
6. **Retry with feedback** if output is invalid ("priority must be integer, you gave string")
7. **Return guaranteed schema** or a structured error — nothing in between

## Features

### Spec-First — Your OpenAPI spec is the implementation

```yaml
paths:
  /v1/parse-invoice:
    post:
      description: "Extract structured data from raw invoice text"
      requestBody:
        content:
          application/json:
            schema:
              type: object
              required: [raw_text]
              properties:
                raw_text: { type: string }
      responses:
        200:
          content:
            application/json:
              schema:
                type: object
                required: [vendor, amount, due_date, line_items]
                properties:
                  vendor: { type: string }
                  amount: { type: number, minimum: 0 }
                  due_date: { type: string }
                  line_items:
                    type: array
                    items:
                      type: object
                      required: [description, amount]
                      properties:
                        description: { type: string }
                        amount: { type: number }
```

That's it. specllm gives you a working `/v1/parse-invoice` endpoint. Feed it messy invoice text, get clean structured JSON back — guaranteed to match the schema or you get a clear error.

### Provider-Agnostic

Any LLM works. Implement one method:

```python
from specllm.llm.providers import LLMProvider

class MyProvider(LLMProvider):
    def call(self, prompt: str, system_prompt=None) -> dict:
        # Call any LLM, return parsed JSON dict
        ...
```

Swap providers without changing your API contract. Configure fallback for auto-failover:

```python
app = SpecLLM.from_openapi("spec.json", provider=primary, config={
    "fallback_provider": backup,  # used if primary crashes or times out
})
```

### Per-Endpoint Model Selection

```python
app = SpecLLM.from_openapi("spec.json", provider=provider, config={
    "endpoint_models": {
        "/v1/route-ticket": "haiku",      # fast, cheap
        "/v1/analyze-contract": "sonnet",  # powerful
    }
})
```

### Custom Input Validation (Business Rules)

Schema validates structure. `@app.validate()` adds business rules — runs before the LLM, zero cost on rejection:

```python
@app.validate("/v1/route-ticket")
def only_enterprise(body):
    if body.get("customer_tier") != "enterprise":
        return "Only enterprise tickets accepted"
```

### Custom Prompts

Override auto-generated prompts where needed:

```python
@app.prompt("/v1/moderate")
def moderate_prompt(body):
    return f"Evaluate against community guidelines:\n{body['text']}"
```

### Observability (every response)

```
X-SpecLLM-Request-Id: req_abc123
X-SpecLLM-Latency-Ms: 1240
X-SpecLLM-Tokens-Used: 387
X-SpecLLM-Retries: 0
X-SpecLLM-Cache-Hit: false
```

### Cost Controls

```python
app = SpecLLM.from_openapi("spec.json", provider=provider, config={
    "cost_limit_daily": 50.0  # auto-stops at $50/day with 503
})
```

### Async Server

```python
app.serve_async(port=8080)  # asyncio, handles concurrent LLM calls
```

### Webhooks for Long-Running Operations

```python
from specllm.pipeline.webhook import WebhookManager

mgr = WebhookManager()
job_id = mgr.submit(job_fn, callback_url="https://your-service.com/hook")
status = mgr.get_status(job_id)  # {"status": "completed", "result": {...}}
```

## Error Handling

Every failure maps to standard HTTP with structured, machine-readable bodies:

| Scenario | HTTP | Code |
|----------|------|------|
| Bad request body | 400 | `INPUT_VALIDATION_FAILED` |
| Custom validator rejects | 400 | `INPUT_VALIDATION_FAILED` |
| Output failed schema after retries | 422 | `OUTPUT_SCHEMA_VIOLATION` |
| Provider timed out | 504 | `PROVIDER_TIMEOUT` |
| Provider down (+ fallback failed) | 503 | `PROVIDER_UNAVAILABLE` |
| Daily cost limit hit | 503 | `COST_LIMIT_REACHED` |

Error response format:

```json
{
  "error": {
    "code": "OUTPUT_SCHEMA_VIOLATION",
    "status": 422,
    "message": "LLM output failed schema validation after 3 attempts",
    "request_id": "req_abc123",
    "timestamp": "2026-06-02T10:30:00Z"
  }
}
```

## Use Cases

**Content Moderation** — Replace regex rule engines with LLM understanding. Same REST contract, better intelligence.

**Document Parsing** — Extract structured data from resumes, contracts, invoices. One endpoint replaces brittle format-specific parsers.

**Ticket Routing** — Route support tickets by intent, not keywords. Returns team + priority + tags as guaranteed schema.

**Compliance Checks** — Flag suspicious transactions with structured verdicts. Sits between payment initiation and settlement.

**Multi-Language Intent Detection** — One endpoint handles 20+ languages. No per-language model maintenance.

In every case: the calling system doesn't know an LLM is involved. It's just a REST API with reliable responses.

## Configuration

```python
app = SpecLLM.from_openapi("./api.yaml", provider=my_provider, config={
    # Reliability
    "max_retries": 3,
    "timeout_seconds": 30,
    "fallback_provider": backup_provider,

    # Per-endpoint models
    "endpoint_models": {"/v1/simple": "haiku", "/v1/complex": "sonnet"},

    # Cost
    "cost_limit_daily": 50.0,
    "cache_ttl": 3600,

    # Observability
    "log_prompts": False,
})
```

## Architecture

```
specllm (1,121 lines, zero required dependencies)

├── spec/         → OpenAPI parser + $ref resolution + JSON Schema validator
├── pipeline/     → Request orchestration, cache, retry, webhook, cost tracking
├── llm/          → Provider ABC + MockProvider
├── server/       → ThreadingHTTPServer + async server
├── prompts/      → Auto-prompt generation from endpoint specs
├── errors/       → Structured error codes + HTTP status mapping
├── observability/ → Response headers
└── testing/      → Contract tests + record/replay
```

## Roadmap

- ✅ OpenAPI spec parsing (JSON + YAML) with $ref resolution
- ✅ Schema validation + retry with error feedback
- ✅ Custom input validation (`@app.validate`)
- ✅ Provider abstraction + fallback chain + per-endpoint models
- ✅ Thread-safe caching, timeout enforcement, cost limits
- ✅ Async server, webhooks, record/replay testing
- ⬜ Built-in providers (Anthropic, OpenAI, Google, Ollama)
- ⬜ Redis cache backend
- ⬜ Prometheus metrics + OpenTelemetry tracing
- ⬜ Schema evolution detection

## License

Apache License 2.0

## Contributing

We welcome contributions! See `CONTRIBUTING.md` for guidelines.
