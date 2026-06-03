"""specllm - The production framework for spec-first LLM APIs."""

import json
from typing import Any, Callable, Dict, List, Optional

from specllm.spec.parser import parse_openapi_spec, Endpoint
from specllm.pipeline.request import RequestPipeline
from specllm.prompts.generator import generate_prompt


class SpecLLM:
    """Main entry point for the specllm framework.

    Parses an OpenAPI spec, connects an LLM provider, and orchestrates
    request handling through validation, caching, and retry pipelines.
    """

    def __init__(
        self,
        spec: dict,
        provider: Optional[Any] = None,
        model: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.spec = spec
        self.model = model
        self.config = config or {}
        self.endpoints: List[Endpoint] = parse_openapi_spec(spec)
        self._custom_prompts: Dict[tuple, Callable] = {}
        self._custom_validators: Dict[tuple, Callable] = {}
        self._pipeline: Optional[RequestPipeline] = None

        # Validate provider type
        if provider is not None and isinstance(provider, str):
            raise TypeError(
                f"provider must be an LLMProvider instance, got string '{provider}'. "
                f"Use a provider class like MockProvider or implement LLMProvider."
            )
        self.provider = provider

        # Fallback provider
        self.fallback_provider = self.config.get("fallback_provider")

        # Cost tracking
        self._cost_tracker = CostTracker(
            daily_limit=self.config.get("cost_limit_daily"),
            cost_per_token=self.config.get("cost_per_token", 0.000001),
        )

        if provider:
            self._pipeline = RequestPipeline(
                provider=provider,
                fallback_provider=self.fallback_provider,
                custom_prompts=self._custom_prompts,
                custom_validators=self._custom_validators,
                timeout_seconds=self.config.get("timeout_seconds", 30),
                cost_tracker=self._cost_tracker,
                endpoint_models=self.config.get("endpoint_models"),
            )

    @classmethod
    def from_openapi(
        cls,
        spec: Any,
        provider: Optional[Any] = None,
        model: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> "SpecLLM":
        """Create a SpecLLM instance from an OpenAPI spec dict, JSON, or YAML file path."""
        if isinstance(spec, str):
            with open(spec, "r") as f:
                content = f.read()
            if spec.endswith(".json"):
                spec = json.loads(content)
            elif spec.endswith((".yaml", ".yml")):
                try:
                    import yaml
                except ImportError:
                    raise ImportError("PyYAML is required for YAML specs: pip install pyyaml")
                spec = yaml.safe_load(content)
            else:
                # Try JSON first, then YAML
                try:
                    spec = json.loads(content)
                except json.JSONDecodeError:
                    try:
                        import yaml

                        spec = yaml.safe_load(content)
                    except ImportError:
                        raise ValueError("Could not parse spec. Install pyyaml for YAML support: pip install pyyaml")
        return cls(spec=spec, provider=provider, model=model, config=config)

    def prompt(self, path: str, method: str = "post") -> Callable:
        """Decorator to register a custom prompt function for an endpoint."""

        def decorator(func: Callable) -> Callable:
            self._custom_prompts[(path, method)] = func
            return func

        return decorator

    def validate(self, path: str, method: str = "post") -> Callable:
        """Decorator to register a custom input validator for an endpoint.

        The validator function receives the request body and should return
        None (or a falsy value) if valid, or a string error message if invalid.
        Invalid requests are rejected with 400 INPUT_VALIDATION_FAILED before
        the LLM is called (zero cost).
        """

        def decorator(func: Callable) -> Callable:
            self._custom_validators[(path, method)] = func
            return func

        return decorator

    def test_client(self) -> "TestClient":
        """Return a test client for this app."""
        return TestClient(self)

    def contract_test(
        self, samples: List[dict], path: Optional[str] = None, method: Optional[str] = None
    ) -> Optional[Any]:
        """Run contract tests with the given samples."""
        from specllm.testing.contract import ContractTestRunner

        runner = ContractTestRunner(provider=self.provider)
        if path and method:
            for ep in self.endpoints:
                if ep.path == path and ep.method.lower() == method.lower():
                    return runner.run(ep, samples)
            return None
        results = []
        for ep in self.endpoints:
            results.append(runner.run(ep, samples))
        if not results:
            return None
        return results[0] if len(results) == 1 else results

    def serve(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        """Start the HTTP server."""
        from specllm.server.app import SpecLLMServer

        server = SpecLLMServer(spec=self.spec, provider=self.provider, host=host, port=port)
        server.serve()

    def serve_async(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        """Start the async HTTP server (supports concurrent LLM calls)."""
        from specllm.server.async_app import AsyncSpecLLMServer

        server = AsyncSpecLLMServer(app=self, host=host, port=port)
        server.serve()


class CostTracker:
    """Tracks token usage and enforces daily cost limits."""

    def __init__(self, daily_limit: Optional[float] = None, cost_per_token: float = 0.000001) -> None:
        self.daily_limit = daily_limit
        self.cost_per_token = cost_per_token
        self._total_tokens: int = 0
        self._daily_tokens: int = 0
        self._current_day: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def daily_cost(self) -> float:
        return self._daily_tokens * self.cost_per_token

    def record(self, tokens: int) -> None:
        """Record token usage."""
        import datetime

        today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        if self._current_day != today:
            self._current_day = today
            self._daily_tokens = 0
        self._total_tokens += tokens
        self._daily_tokens += tokens

    def check_limit(self) -> bool:
        """Return True if within limit, False if exceeded."""
        if self.daily_limit is None:
            return True
        return self.daily_cost < self.daily_limit


class TestClient:
    """Test client for making requests without starting a server."""

    def __init__(self, app: SpecLLM) -> None:
        self.app = app

    def post(self, path: str, json_body: Optional[dict] = None) -> dict:
        """Make a POST request."""
        return self._request("post", path, json_body)

    def get(self, path: str, json_body: Optional[dict] = None) -> dict:
        """Make a GET request."""
        return self._request("get", path, json_body)

    def put(self, path: str, json_body: Optional[dict] = None) -> dict:
        """Make a PUT request."""
        return self._request("put", path, json_body)

    def delete(self, path: str, json_body: Optional[dict] = None) -> dict:
        """Make a DELETE request."""
        return self._request("delete", path, json_body)

    def _request(self, method: str, path: str, json_body: Optional[dict] = None) -> dict:
        """Process a request through the pipeline."""
        for ep in self.app.endpoints:
            if ep.path == path and ep.method.lower() == method.lower():
                if self.app._pipeline:
                    return self.app._pipeline.handle(ep, json_body or {})
                return {
                    "error": {
                        "code": "PROVIDER_NOT_CONFIGURED",
                        "message": "No LLM provider configured. Pass a provider to SpecLLM.",
                    }
                }
        return {"error": {"code": "ENDPOINT_NOT_FOUND", "message": f"No endpoint: {method.upper()} {path}"}}
