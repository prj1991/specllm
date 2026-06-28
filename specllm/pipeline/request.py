"""Request pipeline - orchestrates the full request flow."""

import json
import uuid
import concurrent.futures
from typing import Callable, Dict, Optional

from specllm.pipeline.cache import Cache
from specllm.spec.validator import validate_schema, ValidationError
from specllm.spec.parser import Endpoint
from specllm.prompts.generator import generate_prompt
from specllm.pipeline.retry import RetryHandler, MaxRetriesExceeded
from specllm.pipeline.constraints import resolve_constraints
from specllm.errors.codes import build_error_response, ErrorCode


class ProviderError(Exception):
    """Raised when an LLM provider call fails."""

    pass


class RequestPipeline:
    """Orchestrates request handling: validate, cache, call LLM, validate output."""

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
        self.retry_handler = RetryHandler(max_retries=max_retries)
        self.custom_prompts: Dict[tuple, Callable] = custom_prompts if custom_prompts is not None else {}
        self.custom_validators: Dict[tuple, Callable] = custom_validators if custom_validators is not None else {}
        self.cost_tracker = cost_tracker
        self.endpoint_models: Dict[str, str] = endpoint_models or {}
        self.last_metadata: dict = {"retries": 0, "cache_hit": False, "tokens_used": 0}

    def _call_provider(self, provider: object, prompt: str) -> dict:
        """Call a provider with timeout enforcement."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(provider.call, prompt)
            try:
                result = future.result(timeout=self.timeout_seconds)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"LLM call timed out after {self.timeout_seconds}s")
        if not isinstance(result, dict):
            return {"__specllm_invalid__": True, "__raw__": str(result)[:500]}
        return result

    def _handle_failure(
        self, code: ErrorCode, message: str,
        prompt: str, validate_fn: Callable, cache_key: str, call_count: int, request_id: str,
    ) -> dict:
        """Try fallback, otherwise return structured error."""
        if self.fallback_provider:
            try:
                result = self._call_provider(self.fallback_provider, prompt)
                if not validate_fn(result):
                    self.last_metadata["retries"] = call_count
                    self.cache.set(cache_key, result)
                    return result
            except Exception:
                pass
        self.last_metadata["retries"] = max(0, call_count - 1)
        return build_error_response(code, message=message, request_id=request_id)

    def handle(self, endpoint: Endpoint, request_body: dict) -> dict:
        """Handle a request through the full pipeline."""
        request_id = str(uuid.uuid4())
        self.last_metadata = {"retries": 0, "cache_hit": False, "tokens_used": 0}

        # Cost limit check
        if self.cost_tracker and not self.cost_tracker.check_limit():
            return build_error_response(
                ErrorCode.COST_LIMIT_REACHED,
                message="Daily cost limit reached. Requests are paused until the next day.",
                request_id=request_id,
            )

        # Validate input (schema)
        if endpoint.request_schema:
            input_errors = validate_schema(request_body, endpoint.request_schema)
            if input_errors:
                return build_error_response(
                    ErrorCode.INPUT_VALIDATION_FAILED,
                    message="; ".join(e.message for e in input_errors),
                    request_id=request_id,
                )

        # Validate input (custom business rules)
        custom_validator = self.custom_validators.get((endpoint.path, endpoint.method))
        if custom_validator:
            rejection = custom_validator(request_body)
            if rejection:
                return build_error_response(
                    ErrorCode.INPUT_VALIDATION_FAILED,
                    message=str(rejection),
                    request_id=request_id,
                )

        # Apply dynamic constraints from request body (x-constrain-from in spec)
        constrained_response_schema = resolve_constraints(endpoint.response_schema, request_body)

        # Check cache
        cache_key = self.cache.generate_key(endpoint.path, endpoint.method, request_body)
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.last_metadata["cache_hit"] = True
            return cached

        # Build prompt
        custom_prompt_fn = self.custom_prompts.get((endpoint.path, endpoint.method))
        if custom_prompt_fn:
            prompt = custom_prompt_fn(request_body)
        else:
            prompt = generate_prompt(endpoint, request_body, constrained_response_schema)

        # Determine which provider to use (per-endpoint model override)
        active_provider = self.provider
        endpoint_model = self.endpoint_models.get(endpoint.path)
        if endpoint_model and hasattr(self.provider, "with_model"):
            active_provider = self.provider.with_model(endpoint_model)

        call_count = 0

        def call_fn(feedback: Optional[str] = None) -> dict:
            nonlocal call_count
            call_count += 1
            actual_prompt = prompt + "\n\n" + feedback if feedback else prompt
            return self._call_provider(active_provider, actual_prompt)

        def validate_fn(result: dict) -> list:
            if result.get("__specllm_invalid__"):
                return [ValidationError(
                    path=".", message="Response must be a JSON object. Got non-JSON output from provider."
                )]
            if constrained_response_schema:
                return validate_schema(result, constrained_response_schema)
            return []

        try:
            result = self.retry_handler.execute(call_fn, validate_fn)
        except MaxRetriesExceeded:
            return self._handle_failure(
                ErrorCode.OUTPUT_SCHEMA_VIOLATION,
                "LLM output failed schema validation after all retries",
                prompt, validate_fn, cache_key, call_count, request_id,
            )
        except TimeoutError as e:
            return self._handle_failure(
                ErrorCode.PROVIDER_TIMEOUT, str(e),
                prompt, validate_fn, cache_key, call_count, request_id,
            )
        except Exception as e:
            return self._handle_failure(
                ErrorCode.PROVIDER_UNAVAILABLE, f"LLM provider error: {e}",
                prompt, validate_fn, cache_key, call_count, request_id,
            )

        self.last_metadata["retries"] = max(0, call_count - 1)

        # Track cost
        if self.cost_tracker:
            # Estimate tokens from prompt + response (rough: 4 chars ≈ 1 token)
            estimated_tokens = (len(prompt) + len(json.dumps(result))) // 4
            self.cost_tracker.record(estimated_tokens)
            self.last_metadata["tokens_used"] = estimated_tokens

        self.cache.set(cache_key, result)
        return result
