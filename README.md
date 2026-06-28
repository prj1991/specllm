# specllm

**The production framework for spec-first LLM APIs.**

Define a REST API spec. specllm turns it into a live service where every response is powered by an LLM — validated, cached, and retried automatically. The caller never knows an LLM is involved.

```
┌──────────┐       ┌─────────────────────────────────────┐       ┌─────────┐
│ Your App │──────▶│ specllm                             │──────▶│   LLM   │
│          │◀──────│ (validates, retries, caches, serves)│◀──────│(any LLM)│
└──────────┘ JSON  └─────────────────────────────────────┘ JSON  └─────────┘
```

```python
from specllm import SpecLLM

app = SpecLLM.from_openapi("./api.yaml", provider=my_provider)
app.serve(port=8080)
```

## Install

```bash
pip install specllm
pip install specllm[yaml]  # optional, for YAML specs
```

## How It Works

```
Request → [Input Validation] → [Resolve Constraints] → [Cache Check] → [Build Prompt] → [Call LLM]
                                                                                              ↓
Response ← [Cache Store] ← [Output Validation] ←──── Valid? ←──── [LLM Response]
                                                       ↓ No
                                              [Retry with feedback] (up to 3x)
                                                       ↓ Still invalid
                                              [422 structured error]
```

Bad input → instant 400 (zero LLM cost). Invalid output → retry with error feedback. Still invalid → structured 422. Valid → cache and return.

## Features

### Your OpenAPI spec is the implementation

Write a spec with request/response schemas. specllm generates prompts, validates LLM output, retries on failure. No glue code.

### Provider-Agnostic

Implement one method. Swap models without changing your API contract:

```python
class MyProvider(LLMProvider):
    def call(self, prompt: str, system_prompt=None) -> dict:
        ...
```

### Dynamic Response Constraints (`x-constrain-from`)

Let the caller control what the LLM can return — per request:

```yaml
intent:
  type: string
  x-constrain-from: "intents"    # enum pulled from request body field "intents"

score:
  type: integer
  x-constrain-from:
    minimum: "min_score"          # range pulled from request body
    maximum: "max_score"
```

If the LLM picks outside the caller's options, specllm retries with feedback automatically.

### Configuration

```python
app = SpecLLM.from_openapi("./api.yaml", provider=my_provider, config={
    "timeout_seconds": 30,
    "fallback_provider": backup_provider,
    "endpoint_models": {"/v1/simple": "haiku", "/v1/complex": "sonnet"},
    "cost_limit_daily": 50.0,
    "cache_ttl": 3600,
})
```

### Custom Validation & Prompts

```python
@app.validate("/v1/route-ticket")
def only_enterprise(body):
    if body.get("customer_tier") != "enterprise":
        return "Only enterprise tickets accepted"

@app.prompt("/v1/moderate")
def moderate_prompt(body):
    return f"Evaluate against community guidelines:\n{body['text']}"
```

### Observability

Every response includes headers: `X-SpecLLM-Request-Id`, `X-SpecLLM-Latency-Ms`, `X-SpecLLM-Tokens-Used`, `X-SpecLLM-Retries`, `X-SpecLLM-Cache-Hit`.

### Testing (no LLM credentials needed)

```python
from specllm.testing.record_replay import RecordReplayProvider

# Record locally, replay in CI:
provider = RecordReplayProvider(provider=real_provider, cassette="tests/tape.json")
```

## Error Handling

| Scenario | HTTP | Code |
|----------|------|------|
| Bad input | 400 | `INPUT_VALIDATION_FAILED` |
| Output failed after retries | 422 | `OUTPUT_SCHEMA_VIOLATION` |
| Provider down/timeout | 503/504 | `PROVIDER_UNAVAILABLE` / `PROVIDER_TIMEOUT` |
| Cost limit hit | 503 | `COST_LIMIT_REACHED` |

All errors return structured JSON with `code`, `status`, `message`, `request_id`, `timestamp`.

## Architecture

```
specllm (zero required dependencies)

├── spec/         → OpenAPI parser + $ref resolution + JSON Schema validator
├── pipeline/     → Request orchestration, cache, retry, constraints, cost tracking
├── llm/          → Provider ABC + MockProvider
├── server/       → ThreadingHTTPServer + async server
├── prompts/      → Auto-prompt generation from endpoint specs
├── errors/       → Structured error codes + HTTP status mapping
├── observability/ → Response headers
└── testing/      → Contract tests + record/replay
```

## Roadmap

- ✅ OpenAPI parsing (JSON + YAML) with $ref resolution
- ✅ Schema validation + retry with error feedback
- ✅ Custom validation, prompts, provider fallback, per-endpoint models
- ✅ Caching, timeout enforcement, cost limits, async server, webhooks
- ✅ Dynamic response constraints (`x-constrain-from`)
- ✅ Record/replay testing
- ⬜ Built-in providers (Anthropic, OpenAI, Google, Ollama)
- ⬜ Redis cache backend
- ⬜ Prometheus metrics + OpenTelemetry tracing

## License

Apache License 2.0
