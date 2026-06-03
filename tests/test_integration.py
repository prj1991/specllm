"""End-to-end integration tests using MockProvider.

Tests the FULL pipeline without mocking internals:
  spec parsing → input validation → prompt generation → LLM call (mocked) →
  output validation → retry with feedback → caching → error responses

No network calls, no patching - just MockProvider returning controlled responses.
"""
import json
import pytest
import threading
import time
import urllib.request
import urllib.error

from specllm import SpecLLM, TestClient
from specllm.llm.providers import MockProvider
from specllm.server.app import SpecLLMServer


SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Test API", "version": "1.0.0"},
    "components": {
        "schemas": {
            "Priority": {"type": "integer", "minimum": 1, "maximum": 5},
        }
    },
    "paths": {
        "/v1/route-ticket": {
            "post": {
                "description": "Route a support ticket to the right team.",
                "requestBody": {"content": {"application/json": {"schema": {
                    "type": "object",
                    "required": ["title", "body"],
                    "properties": {
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                        "customer_tier": {"type": "string", "enum": ["free", "pro", "enterprise"]},
                    },
                }}}},
                "responses": {"200": {"content": {"application/json": {"schema": {
                    "type": "object",
                    "required": ["team", "priority"],
                    "properties": {
                        "team": {"type": "string"},
                        "priority": {"$ref": "#/components/schemas/Priority"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                }}}}},
            }
        },
        "/v1/classify": {
            "post": {
                "description": "Classify text sentiment.",
                "requestBody": {"content": {"application/json": {"schema": {
                    "type": "object",
                    "required": ["text"],
                    "properties": {"text": {"type": "string"}},
                }}}},
                "responses": {"200": {"content": {"application/json": {"schema": {
                    "type": "object",
                    "required": ["label", "score"],
                    "properties": {
                        "label": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                        "score": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                }}}}},
            }
        },
    },
}


class TestEndToEndHappyPath:
    """Full pipeline: valid input → valid LLM output → 200."""

    def test_valid_request_returns_llm_response(self):
        provider = MockProvider(responses=[{"team": "billing", "priority": 3, "tags": ["payment"]}])
        app = SpecLLM(spec=SPEC, provider=provider)
        result = app.test_client().post("/v1/route-ticket", json_body={"title": "Help", "body": "Payment issue"})

        assert result == {"team": "billing", "priority": 3, "tags": ["payment"]}
        assert provider._call_count == 1

    def test_prompt_contains_request_body_and_schema(self):
        provider = MockProvider(responses=[{"label": "positive", "score": 0.9}])
        app = SpecLLM(spec=SPEC, provider=provider)
        app.test_client().post("/v1/classify", json_body={"text": "Great product"})

        prompt = provider._calls[0]["prompt"]
        assert "Great product" in prompt
        assert "label" in prompt
        assert "score" in prompt

    def test_multiple_endpoints_route_correctly(self):
        provider = MockProvider(responses=[
            {"team": "infra", "priority": 1},
            {"label": "neutral", "score": 0.5},
        ])
        app = SpecLLM(spec=SPEC, provider=provider)
        client = app.test_client()

        r1 = client.post("/v1/route-ticket", json_body={"title": "t", "body": "b"})
        r2 = client.post("/v1/classify", json_body={"text": "ok"})

        assert r1["team"] == "infra"
        assert r2["label"] == "neutral"


class TestEndToEndInputValidation:
    """Input validation rejects bad requests before calling the LLM."""

    def test_missing_required_field(self):
        provider = MockProvider(responses=[{"team": "x", "priority": 1}])
        app = SpecLLM(spec=SPEC, provider=provider)
        result = app.test_client().post("/v1/route-ticket", json_body={"title": "Only title"})

        assert result["error"]["code"] == "INPUT_VALIDATION_FAILED"
        assert "body" in result["error"]["message"]
        assert provider._call_count == 0  # LLM never called

    def test_invalid_enum_value(self):
        provider = MockProvider(responses=[{"team": "x", "priority": 1}])
        app = SpecLLM(spec=SPEC, provider=provider)
        result = app.test_client().post("/v1/route-ticket", json_body={
            "title": "t", "body": "b", "customer_tier": "platinum"
        })

        assert result["error"]["code"] == "INPUT_VALIDATION_FAILED"
        assert provider._call_count == 0

    def test_wrong_type_in_input(self):
        provider = MockProvider(responses=[{"label": "positive", "score": 0.5}])
        app = SpecLLM(spec=SPEC, provider=provider)
        result = app.test_client().post("/v1/classify", json_body={"text": 12345})

        assert result["error"]["code"] == "INPUT_VALIDATION_FAILED"
        assert provider._call_count == 0


class TestEndToEndOutputValidation:
    """Output validation catches bad LLM responses and triggers retries."""

    def test_wrong_type_triggers_retry_then_succeeds(self):
        provider = MockProvider(responses=[
            {"team": "billing", "priority": "HIGH"},  # bad: string not int
            {"team": "billing", "priority": 2},       # good
        ])
        app = SpecLLM(spec=SPEC, provider=provider)
        result = app.test_client().post("/v1/route-ticket", json_body={"title": "t", "body": "b"})

        assert result == {"team": "billing", "priority": 2}
        assert provider._call_count == 2

    def test_out_of_range_triggers_retry(self):
        provider = MockProvider(responses=[
            {"team": "x", "priority": 99},  # bad: > maximum 5
            {"team": "x", "priority": 5},   # good
        ])
        app = SpecLLM(spec=SPEC, provider=provider)
        result = app.test_client().post("/v1/route-ticket", json_body={"title": "t", "body": "b"})

        assert result["priority"] == 5
        assert provider._call_count == 2

    def test_invalid_enum_output_triggers_retry(self):
        provider = MockProvider(responses=[
            {"label": "amazing", "score": 0.9},  # bad enum
            {"label": "positive", "score": 0.9},  # good
        ])
        app = SpecLLM(spec=SPEC, provider=provider)
        result = app.test_client().post("/v1/classify", json_body={"text": "hi"})

        assert result["label"] == "positive"

    def test_missing_required_output_field_triggers_retry(self):
        provider = MockProvider(responses=[
            {"team": "billing"},               # missing priority
            {"team": "billing", "priority": 3},  # complete
        ])
        app = SpecLLM(spec=SPEC, provider=provider)
        result = app.test_client().post("/v1/route-ticket", json_body={"title": "t", "body": "b"})

        assert result["priority"] == 3

    def test_all_retries_exhausted_returns_422(self):
        provider = MockProvider(responses=[{"team": "x", "priority": "always_bad"}])
        app = SpecLLM(spec=SPEC, provider=provider)
        result = app.test_client().post("/v1/route-ticket", json_body={"title": "t", "body": "b"})

        assert result["error"]["code"] == "OUTPUT_SCHEMA_VIOLATION"
        assert result["error"]["status"] == 422
        assert provider._call_count == 4  # 1 initial + 3 retries

    def test_retry_feedback_includes_error_details(self):
        provider = MockProvider(responses=[
            {"team": "x", "priority": "bad"},
            {"team": "x", "priority": 2},
        ])
        app = SpecLLM(spec=SPEC, provider=provider)
        app.test_client().post("/v1/route-ticket", json_body={"title": "t", "body": "b"})

        # Second call should have retry feedback in the prompt
        second_prompt = provider._calls[1]["prompt"]
        assert "schema validation" in second_prompt.lower() or "validation" in second_prompt.lower()
        assert "priority" in second_prompt


class TestEndToEndCaching:
    """Cache returns stored responses without calling LLM again."""

    def test_identical_request_uses_cache(self):
        provider = MockProvider(responses=[{"label": "positive", "score": 0.8}])
        app = SpecLLM(spec=SPEC, provider=provider)
        client = app.test_client()

        r1 = client.post("/v1/classify", json_body={"text": "great"})
        r2 = client.post("/v1/classify", json_body={"text": "great"})

        assert r1 == r2
        assert provider._call_count == 1  # only called once
        assert app._pipeline.last_metadata["cache_hit"] is True

    def test_different_request_body_not_cached(self):
        provider = MockProvider(responses=[
            {"label": "positive", "score": 0.9},
            {"label": "negative", "score": 0.7},
        ])
        app = SpecLLM(spec=SPEC, provider=provider)
        client = app.test_client()

        r1 = client.post("/v1/classify", json_body={"text": "love it"})
        r2 = client.post("/v1/classify", json_body={"text": "hate it"})

        assert r1["label"] == "positive"
        assert r2["label"] == "negative"
        assert provider._call_count == 2


class TestEndToEndCustomPrompts:
    """Custom prompt decorators override auto-generated prompts."""

    def test_custom_prompt_used(self):
        provider = MockProvider(responses=[{"label": "positive", "score": 0.99}])
        app = SpecLLM(spec=SPEC, provider=provider)

        @app.prompt("/v1/classify")
        def my_prompt(body):
            return f"CUSTOM: classify '{body['text']}'"

        app.test_client().post("/v1/classify", json_body={"text": "hello"})

        assert provider._calls[0]["prompt"] == "CUSTOM: classify 'hello'"


class TestEndToEndCustomValidators:
    """Custom validators reject requests based on business rules."""

    def test_validator_rejects_request(self):
        provider = MockProvider(responses=[{"team": "x", "priority": 1}])
        app = SpecLLM(spec=SPEC, provider=provider)

        @app.validate("/v1/route-ticket")
        def only_enterprise(body):
            if body.get("customer_tier") != "enterprise":
                return "Only enterprise tickets are accepted"
            return None

        result = app.test_client().post("/v1/route-ticket", json_body={
            "title": "t", "body": "b", "customer_tier": "free"
        })

        assert result["error"]["code"] == "INPUT_VALIDATION_FAILED"
        assert "enterprise" in result["error"]["message"]
        assert provider._call_count == 0  # LLM never called

    def test_validator_passes_valid_request(self):
        provider = MockProvider(responses=[{"team": "vip", "priority": 1}])
        app = SpecLLM(spec=SPEC, provider=provider)

        @app.validate("/v1/route-ticket")
        def only_enterprise(body):
            if body.get("customer_tier") != "enterprise":
                return "Only enterprise tickets accepted"

        result = app.test_client().post("/v1/route-ticket", json_body={
            "title": "t", "body": "b", "customer_tier": "enterprise"
        })

        assert result == {"team": "vip", "priority": 1}
        assert provider._call_count == 1

    def test_validator_runs_after_schema_validation(self):
        """Schema validation runs first; custom validator only sees valid data."""
        provider = MockProvider(responses=[{"team": "x", "priority": 1}])
        app = SpecLLM(spec=SPEC, provider=provider)
        validator_called = []

        @app.validate("/v1/route-ticket")
        def track(body):
            validator_called.append(True)

        # Missing required field — schema rejects before custom validator runs
        app.test_client().post("/v1/route-ticket", json_body={"title": "only title"})
        assert len(validator_called) == 0


class TestEndToEndRefResolution:
    """$ref in schemas resolved and validated correctly."""

    def test_ref_minimum_maximum_enforced(self):
        # Priority is $ref to {"type": "integer", "minimum": 1, "maximum": 5}
        provider = MockProvider(responses=[
            {"team": "x", "priority": 0},   # below minimum
            {"team": "x", "priority": 3},   # valid
        ])
        app = SpecLLM(spec=SPEC, provider=provider)
        result = app.test_client().post("/v1/route-ticket", json_body={"title": "t", "body": "b"})

        assert result["priority"] == 3
        assert provider._call_count == 2  # first rejected


class TestEndToEndErrorCases:
    """Error handling for routing and provider failures."""

    def test_unknown_endpoint_returns_not_found(self):
        app = SpecLLM(spec=SPEC, provider=MockProvider())
        result = app.test_client().post("/v1/nonexistent", json_body={})

        assert result["error"]["code"] == "ENDPOINT_NOT_FOUND"

    def test_no_provider_returns_error(self):
        app = SpecLLM(spec=SPEC)
        result = app.test_client().post("/v1/classify", json_body={"text": "hi"})

        assert result["error"]["code"] == "PROVIDER_NOT_CONFIGURED"

    def test_provider_exception_returns_503(self):
        class CrashingProvider:
            def call(self, prompt, system_prompt=None):
                raise ConnectionError("Connection refused")

        app = SpecLLM(spec=SPEC, provider=CrashingProvider())
        result = app.test_client().post("/v1/classify", json_body={"text": "hi"})

        assert result["error"]["code"] == "PROVIDER_UNAVAILABLE"
        assert result["error"]["status"] == 503

    def test_provider_timeout_returns_504(self):
        import time

        class SlowProvider:
            def call(self, prompt, system_prompt=None):
                time.sleep(5)
                return {"label": "positive", "score": 0.9}

        app = SpecLLM(spec=SPEC, provider=SlowProvider(), config={"timeout_seconds": 1})
        result = app.test_client().post("/v1/classify", json_body={"text": "hi"})

        assert result["error"]["code"] == "PROVIDER_TIMEOUT"
        assert result["error"]["status"] == 504

    def test_timeout_default_is_30_seconds(self):
        app = SpecLLM(spec=SPEC, provider=MockProvider(responses=[{"label": "positive", "score": 0.9}]))
        assert app._pipeline.timeout_seconds == 30

    def test_timeout_configurable_via_config(self):
        app = SpecLLM(spec=SPEC, provider=MockProvider(), config={"timeout_seconds": 60})
        assert app._pipeline.timeout_seconds == 60


class TestEndToEndHTTPServer:
    """Full HTTP server integration with MockProvider."""

    @pytest.fixture(autouse=True)
    def setup_server(self):
        provider = MockProvider(responses=[
            {"team": "support", "priority": 2, "tags": ["account"]},
        ])
        self.server = SpecLLMServer(spec=SPEC, provider=provider, host="127.0.0.1", port=0)
        self.port = self.server.port
        self.thread = threading.Thread(target=self.server.serve, daemon=True)
        self.thread.start()
        time.sleep(0.1)
        yield
        self.server.shutdown()

    def _post(self, path, body):
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=data,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read()), dict(resp.headers)
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read()), dict(e.headers)

    def test_full_http_request_returns_200(self):
        status, body, headers = self._post("/v1/route-ticket", {"title": "t", "body": "b"})

        assert status == 200
        assert body["team"] == "support"
        assert body["priority"] == 2

    def test_observability_headers_present(self):
        _, _, headers = self._post("/v1/route-ticket", {"title": "t", "body": "b"})

        assert "X-SpecLLM-Request-Id" in headers
        assert "X-SpecLLM-Latency-Ms" in headers
        assert "X-SpecLLM-Retries" in headers
        assert "X-SpecLLM-Cache-Hit" in headers
        assert headers["X-SpecLLM-Cache-Hit"] == "false"

    def test_input_validation_returns_400(self):
        status, body, _ = self._post("/v1/route-ticket", {"title": "missing body"})

        assert status == 400
        assert body["error"]["code"] == "INPUT_VALIDATION_FAILED"

    def test_unknown_path_returns_404(self):
        status, body, _ = self._post("/v1/nope", {"x": 1})

        assert status == 404


class TestEndToEndYAMLSupport:
    """YAML spec loading."""

    def test_yaml_spec_loads_and_works(self, tmp_path):
        yaml_content = """
openapi: '3.0.0'
info:
  title: YAML Test
  version: '1'
paths:
  /v1/hello:
    post:
      description: Say hello
      requestBody:
        content:
          application/json:
            schema:
              type: object
              required: [name]
              properties:
                name:
                  type: string
      responses:
        '200':
          content:
            application/json:
              schema:
                type: object
                required: [greeting]
                properties:
                  greeting:
                    type: string
"""
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(yaml_content)

        app = SpecLLM.from_openapi(str(spec_file), provider=MockProvider(responses=[{"greeting": "Hi Alice"}]))
        result = app.test_client().post("/v1/hello", json_body={"name": "Alice"})
        assert result == {"greeting": "Hi Alice"}


class TestEndToEndFallbackProvider:
    """Fallback provider activates when primary fails."""

    def test_fallback_used_when_primary_crashes(self):
        class CrashProvider:
            def call(self, prompt, system_prompt=None):
                raise ConnectionError("down")

        fallback = MockProvider(responses=[{"team": "fallback", "priority": 1}])
        app = SpecLLM(spec=SPEC, provider=CrashProvider(), config={"fallback_provider": fallback})
        result = app.test_client().post("/v1/route-ticket", json_body={"title": "t", "body": "b"})

        assert result == {"team": "fallback", "priority": 1}

    def test_fallback_used_when_primary_times_out(self):
        import time as _time

        class SlowProvider:
            def call(self, prompt, system_prompt=None):
                _time.sleep(5)

        fallback = MockProvider(responses=[{"team": "fast", "priority": 2}])
        app = SpecLLM(spec=SPEC, provider=SlowProvider(), config={"timeout_seconds": 1, "fallback_provider": fallback})
        result = app.test_client().post("/v1/route-ticket", json_body={"title": "t", "body": "b"})

        assert result == {"team": "fast", "priority": 2}

    def test_no_fallback_returns_error(self):
        class CrashProvider:
            def call(self, prompt, system_prompt=None):
                raise ConnectionError("down")

        app = SpecLLM(spec=SPEC, provider=CrashProvider())
        result = app.test_client().post("/v1/route-ticket", json_body={"title": "t", "body": "b"})
        assert result["error"]["code"] == "PROVIDER_UNAVAILABLE"


class TestEndToEndCostLimit:
    """Cost limit enforcement."""

    def test_cost_limit_blocks_requests(self):
        provider = MockProvider(responses=[{"label": "positive", "score": 0.9}])
        app = SpecLLM(spec=SPEC, provider=provider, config={"cost_limit_daily": 0.0000001, "cost_per_token": 1.0})
        client = app.test_client()

        # First request succeeds
        r1 = client.post("/v1/classify", json_body={"text": "hi"})
        assert "error" not in r1

        # Second request blocked (limit exceeded after first)
        r2 = client.post("/v1/classify", json_body={"text": "bye"})
        assert r2["error"]["code"] == "COST_LIMIT_REACHED"
        assert r2["error"]["status"] == 503


class TestEndToEndRecordReplay:
    """Record/replay provider for credential-free testing."""

    def test_record_and_replay(self, tmp_path):
        from specllm.testing.record_replay import RecordReplayProvider

        cassette_file = str(tmp_path / "cassette.json")
        real_provider = MockProvider(responses=[{"label": "positive", "score": 0.8}])

        # Record
        recorder = RecordReplayProvider(provider=real_provider, cassette=cassette_file)
        result1 = recorder.call("classify: great product")
        assert result1 == {"label": "positive", "score": 0.8}

        # Replay (no real provider)
        replayer = RecordReplayProvider(cassette=cassette_file)
        result2 = replayer.call("classify: great product")
        assert result2 == {"label": "positive", "score": 0.8}

    def test_replay_without_recording_raises(self, tmp_path):
        import pytest
        from specllm.testing.record_replay import RecordReplayProvider

        replayer = RecordReplayProvider(cassette=str(tmp_path / "empty.json"))
        with pytest.raises(RuntimeError, match="No recording found"):
            replayer.call("unknown prompt")


class TestEndToEndWebhook:
    """Webhook manager for async operations."""

    def test_submit_job_returns_id_and_completes(self):
        import time
        from specllm.pipeline.webhook import WebhookManager

        mgr = WebhookManager()
        job_id = mgr.submit(lambda: {"result": "done"})

        assert job_id is not None
        time.sleep(0.1)
        status = mgr.get_status(job_id)
        assert status["status"] == "completed"
        assert status["result"] == {"result": "done"}

    def test_failed_job_reports_error(self):
        import time
        from specllm.pipeline.webhook import WebhookManager

        mgr = WebhookManager()
        job_id = mgr.submit(lambda: 1 / 0)

        time.sleep(0.1)
        status = mgr.get_status(job_id)
        assert status["status"] == "failed"
