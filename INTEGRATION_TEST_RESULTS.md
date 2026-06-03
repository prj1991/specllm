# specllm Integration Test Results & Fix Plan

## Test Execution Summary

| Suite | Result |
|-------|--------|
| Unit tests (existing) | ✅ 130/130 passed |
| Integration test (real LLM, TestClient) | ✅ 35/35 passed |
| HTTP server integration (real LLM) | ✅ 5/5 passed |

**LLM Used:** Claude Haiku 4.5 via AWS Bedrock  
**Total tokens consumed:** ~2,500 across all integration tests  
**All tests pass** — the core pipeline works correctly end-to-end.

---

## Faults Found (verified)

### CRITICAL

| # | Issue | Impact | Fix |
|---|-------|--------|-----|
| 1 | **String provider silently accepted** — `from_openapi(provider="anthropic")` stores the string, creates a pipeline, but crashes with `AttributeError: 'str' object has no attribute 'call'` on first request | Users following README pattern get runtime crash | Add provider factory registry, or raise `TypeError` immediately if provider is not an `LLMProvider` instance |
| 2 | **No thread-safety in cache** — `OrderedDict` with no locking, used with `ThreadingHTTPServer` | Data corruption under concurrent load | Add `threading.Lock` around `get`/`set` operations |

### HIGH

| # | Issue | Impact | Fix |
|---|-------|--------|-----|
| 3 | **No timeout on LLM calls** — `PROVIDER_TIMEOUT` error code defined but never raised | Slow providers block the server thread forever | Add configurable timeout in `RequestPipeline.handle()`, catch timeout and return `PROVIDER_TIMEOUT` error |
| 4 | **Non-dict provider return not handled** — If `provider.call()` returns a string or None, the validator still tries to validate it, burns all retries, and returns a generic 422 | Misleading error; users can't distinguish "LLM returned garbage" from "LLM returned wrong schema" | Check return type before validation; if not dict, format specific feedback for retry: "You must respond with a JSON object" |

### MEDIUM

| # | Issue | Impact | Fix |
|---|-------|--------|-----|
| 5 | **Cache key doesn't include provider/model** | Switching providers mid-session serves stale responses from previous model | Include provider class name + model in cache key, or clear cache on provider change |
| 6 | **TestClient has no status code** | Can't write assertions on HTTP status in test_client mode | Return a response object with `.status_code` and `.json()` instead of raw dict |
| 7 | **`@app.prompt()` API mismatch with README** | README shows `@app.prompt("/v1/moderate")` (1 arg), code requires `(path, method)` | Make `method` default to `"post"` since that's the dominant pattern |
| 8 | **`datetime.utcnow()` deprecated** | Python 3.12+ deprecation warning in error responses | Use `datetime.now(datetime.UTC)` |

### LOW

| # | Issue | Impact | Fix |
|---|-------|--------|-----|
| 9 | **contract_test has no retry** | Contract tests fail on first LLM flake, even though production retries | Option: add `with_retries=True` parameter |
| 10 | **No response schema = zero validation** | Endpoints without response schemas pass anything through | Log a warning at startup for endpoints without response_schema |

---

## Recommended Fix Priority

1. **Thread-safe cache** (critical for production HTTP server)
2. **Provider type validation** (prevents silent crash)
3. **Timeout support** (prevents thread starvation)
4. **prompt() decorator default method** (API consistency with docs)
5. **TestClient response object** (developer experience)
6. **Non-dict return handling** (better error feedback)

---

## What Works Well

- ✅ Full end-to-end pipeline: spec → validate → prompt → LLM → validate → serve
- ✅ Input validation catches missing fields, bad types, enum violations
- ✅ Output validation catches type mismatches, range violations, missing fields
- ✅ Retry logic correctly feeds back validation errors and recovers
- ✅ Caching works correctly (deterministic keys, TTL expiry)
- ✅ $ref resolution in OpenAPI specs
- ✅ HTTP server with proper status codes and observability headers
- ✅ Custom prompt decorator works
- ✅ Structured error responses with consistent format
- ✅ LLM successfully produces schema-valid JSON in 100% of integration test cases
