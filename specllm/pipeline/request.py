"""Request pipeline - orchestrates the full request flow."""

import datetime
import json
import uuid
import concurrent.futures
from typing import Callable, Dict, Optional

from specllm.pipeline.cache import Cache
from specllm.pipeline.constraints import resolve_constraints
from specllm.spec.parser import Endpoint
from specllm.spec.validator import validate_schema, ValidationError

# --- Error codes ---

_STATUS = {
    "INPUT_VALIDATION_FAILED": 400,
    "OUTPUT_SCHEMA_VIOLATION": 422,
    "PROVIDER_TIMEOUT": 504,
    "PROVIDER_UNAVAILABLE": 503,
    "COST_LIMIT_REACHED": 503,
}


def _error(code: str, message: str, request_id: str) -> dict:
    return {"error": {
        "code": code,
        "status": _STATUS.get(code, 500),
        "message": message,
        "request_id": request_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }}


# --- Prompt generation ---

def _build_prompt(endpoint: Endpoint, request_body: dict, response_schema: Optional[dict]) -> str:
    parts = []
    if endpoint.description:
        parts.append(f"Task: {endpoint.description}\n")
    parts.append("You must respond with valid JSON matching the following schema:\n")
    schema = response_schema or endpoint.response_schema
    if schema:
        parts.append(json.dumps(schema, indent=2))
    if request_body:
        parts.append(f"\nRequest body:\n{json.dumps(request_body, indent=2)}")
    return "\n".join(parts)


# --- Retry with feedback ---

class _MaxRetries(Exception):
    pass


def _retry(call_fn: Callable, validate_fn: Callable, max_retries: int) -> dict:
    result = call_fn(None)
    errors = validate_fn(result)
    if not errors:
        return result
    for _ in range(max_retries):
        feedback = "Your previous response failed schema validation:\n"
        feedback += "\n".join(f"- {e.path}: {e.message}" for e in errors)
        feedback += "\nPlease respond with valid JSON."
        result = call_fn(feedback)
        errors = validate_fn(result)
        if not errors:
            return result
    raise _MaxRetries()


# --- Pipeline ---

class RequestPipeline:
    """Orchestrates: validate → constrain → cache → prompt → LLM → validate → retry."""

    def __init__(
        self,
        provider: object,
        fallback_provider: Optional[object] = None,
        max_retries: int = 3,
        cache_ttl: int = 3600,
        timeout_seconds: int = 30,
        custom_prompts: Optional[Dict[tuple, Callable]] = None,
        custom_validators: Optional[Dict[tuple, Callable]] = None,
        cost_tracker: Optional[object] = None,
        endpoint_models: Optional[Dict[str, str]] = None,
    ) -> None:
        self.provider = provider
        self.fallback_provider = fallback_provider
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.cache = Cache(default_ttl=cache_ttl)
        self.custom_prompts = custom_prompts if custom_prompts is not None else {}
        self.custom_validators = custom_validators if custom_validators is not None else {}
        self.cost_tracker = cost_tracker
        self.endpoint_models = endpoint_models or {}
        self.last_metadata: dict = {}

    def _call_provider(self, provider: object, prompt: str) -> dict:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            try:
                result = ex.submit(provider.call, prompt).result(timeout=self.timeout_seconds)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"LLM call timed out after {self.timeout_seconds}s")
        if not isinstance(result, dict):
            return {"__specllm_invalid__": True}
        return result

    def _try_fallback(self, prompt: str, validate_fn: Callable, cache_key: str) -> Optional[dict]:
        if not self.fallback_provider:
            return None
        try:
            result = self._call_provider(self.fallback_provider, prompt)
            if not validate_fn(result):
                self.cache.set(cache_key, result)
                return result
        except Exception:
            pass
        return None

    def _fail(self, code, message, prompt, validate_fn, cache_key, call_count, request_id):
        fb = self._try_fallback(prompt, validate_fn, cache_key)
        if fb:
            return fb
        self.last_metadata["retries"] = max(0, call_count - 1)
        return _error(code, message, request_id)

    def handle(self, endpoint: Endpoint, request_body: dict) -> dict:
        request_id = str(uuid.uuid4())
        self.last_metadata = {"retries": 0, "cache_hit": False, "tokens_used": 0}

        # Cost limit
        if self.cost_tracker and not self.cost_tracker.check_limit():
            return _error("COST_LIMIT_REACHED", "Daily cost limit reached.", request_id)

        # Input validation (schema)
        if endpoint.request_schema:
            errors = validate_schema(request_body, endpoint.request_schema)
            if errors:
                return _error("INPUT_VALIDATION_FAILED", "; ".join(e.message for e in errors), request_id)

        # Input validation (custom)
        validator = self.custom_validators.get((endpoint.path, endpoint.method))
        if validator:
            rejection = validator(request_body)
            if rejection:
                return _error("INPUT_VALIDATION_FAILED", str(rejection), request_id)

        # Resolve dynamic constraints
        response_schema = resolve_constraints(endpoint.response_schema, request_body)

        # Cache
        cache_key = self.cache.generate_key(endpoint.path, endpoint.method, request_body)
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.last_metadata["cache_hit"] = True
            return cached

        # Prompt
        prompt_fn = self.custom_prompts.get((endpoint.path, endpoint.method))
        prompt = prompt_fn(request_body) if prompt_fn else _build_prompt(endpoint, request_body, response_schema)

        # Provider selection
        active_provider = self.provider
        model = self.endpoint_models.get(endpoint.path)
        if model and hasattr(self.provider, "with_model"):
            active_provider = self.provider.with_model(model)

        # Call LLM with retry
        call_count = 0

        def call_fn(feedback: Optional[str] = None) -> dict:
            nonlocal call_count
            call_count += 1
            p = prompt + "\n\n" + feedback if feedback else prompt
            return self._call_provider(active_provider, p)

        def validate_fn(result: dict) -> list:
            if result.get("__specllm_invalid__"):
                return [ValidationError(path=".", message="Non-JSON output from provider.")]
            return validate_schema(result, response_schema) if response_schema else []

        try:
            result = _retry(call_fn, validate_fn, self.max_retries)
        except _MaxRetries:
            return self._fail("OUTPUT_SCHEMA_VIOLATION", "LLM output failed validation after retries",
                             prompt, validate_fn, cache_key, call_count, request_id)
        except TimeoutError as e:
            return self._fail("PROVIDER_TIMEOUT", str(e), prompt, validate_fn, cache_key, call_count, request_id)
        except Exception as e:
            return self._fail("PROVIDER_UNAVAILABLE", f"LLM provider error: {e}",
                             prompt, validate_fn, cache_key, call_count, request_id)

        self.last_metadata["retries"] = max(0, call_count - 1)
        if self.cost_tracker:
            tokens = (len(prompt) + len(json.dumps(result))) // 4
            self.cost_tracker.record(tokens)
            self.last_metadata["tokens_used"] = tokens

        self.cache.set(cache_key, result)
        return result
