"""HTTP server using stdlib http.server."""

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from typing import Any, Dict, Set

from specllm.spec.parser import parse_openapi_spec, Endpoint
from specllm.pipeline.request import RequestPipeline
from specllm.observability.headers import build_headers


class SpecLLMServer:
    """HTTP server that routes requests through the pipeline."""

    def __init__(
        self,
        spec: dict,
        provider: object,
        host: str = "127.0.0.1",
        port: int = 8080,
        provider_name: str = "default",
        model_name: str = "default",
    ) -> None:
        self.spec = spec
        self.provider = provider
        self.host = host
        self.port = port
        self.provider_name = provider_name
        self.model_name = model_name

        self.endpoints = parse_openapi_spec(spec)
        self._routes: Dict[tuple, Endpoint] = {}
        for ep in self.endpoints:
            self._routes[(ep.method.lower(), ep.path)] = ep

        self._path_methods: Dict[str, Set[str]] = {}
        for ep in self.endpoints:
            self._path_methods.setdefault(ep.path, set()).add(ep.method.lower())

        self.pipeline = RequestPipeline(provider=provider)

        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                server_ref._handle_request(self, "post")

            def do_GET(self) -> None:
                server_ref._handle_request(self, "get")

            def do_PUT(self) -> None:
                server_ref._handle_request(self, "put")

            def do_DELETE(self) -> None:
                server_ref._handle_request(self, "delete")

            def do_PATCH(self) -> None:
                server_ref._handle_request(self, "patch")

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self.port = self._httpd.server_address[1]

    def _handle_request(self, handler: BaseHTTPRequestHandler, method: str) -> None:
        """Route and process an incoming HTTP request."""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        path = handler.path

        if path not in self._path_methods:
            self._send_error(
                handler,
                404,
                {
                    "error": {
                        "code": "ENDPOINT_NOT_FOUND",
                        "message": f"No endpoint found for path: {path}",
                        "request_id": request_id,
                    }
                },
                request_id,
                start_time,
            )
            return

        if method not in self._path_methods[path]:
            handler.send_response(405)
            handler.send_header("Content-Type", "application/json")
            handler.end_headers()
            body = json.dumps(
                {"error": {"code": "METHOD_NOT_ALLOWED", "message": f"Method {method.upper()} not allowed"}}
            )
            handler.wfile.write(body.encode("utf-8"))
            return

        content_length = int(handler.headers.get("Content-Length", 0))
        if content_length > 0:
            raw_body = handler.rfile.read(content_length)
            try:
                request_body = json.loads(raw_body)
            except (json.JSONDecodeError, ValueError):
                self._send_error(
                    handler,
                    400,
                    {
                        "error": {
                            "code": "MALFORMED_REQUEST",
                            "message": "Invalid JSON in request body",
                            "request_id": request_id,
                        }
                    },
                    request_id,
                    start_time,
                )
                return
        else:
            request_body = {}

        endpoint = self._routes[(method, path)]
        result = self.pipeline.handle(endpoint, request_body)
        latency_ms = int((time.time() - start_time) * 1000)
        metadata = self.pipeline.last_metadata

        obs_headers = build_headers(
            request_id=request_id,
            provider=self.provider_name,
            model=self.model_name,
            latency_ms=latency_ms,
            tokens_used=metadata.get("tokens_used", 0),
            retries=metadata.get("retries", 0),
            cache_hit=metadata.get("cache_hit", False),
        )

        if "error" in result:
            error_obj = result["error"]
            if "status" in error_obj:
                status = error_obj["status"]
            else:
                error_code = error_obj.get("code", "")
                if error_code == "INPUT_VALIDATION_FAILED":
                    status = 400
                elif error_code == "OUTPUT_SCHEMA_VIOLATION":
                    status = 422
                elif error_code == "PROVIDER_UNAVAILABLE":
                    status = 503
                else:
                    status = 500
            handler.send_response(status)
        else:
            handler.send_response(200)

        handler.send_header("Content-Type", "application/json")
        for key, value in obs_headers.items():
            handler.send_header(key, value)
        handler.end_headers()
        handler.wfile.write(json.dumps(result).encode("utf-8"))

    def _send_error(
        self, handler: BaseHTTPRequestHandler, status: int, error_body: dict, request_id: str, start_time: float
    ) -> None:
        """Send an error response with observability headers."""
        latency_ms = int((time.time() - start_time) * 1000)
        obs_headers = build_headers(
            request_id=request_id,
            provider=self.provider_name,
            model=self.model_name,
            latency_ms=latency_ms,
            tokens_used=0,
            retries=0,
            cache_hit=False,
        )
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        for key, value in obs_headers.items():
            handler.send_header(key, value)
        handler.end_headers()
        handler.wfile.write(json.dumps(error_body).encode("utf-8"))

    def serve(self) -> None:
        """Start serving (blocking)."""
        self._httpd.serve_forever()

    def shutdown(self) -> None:
        """Shutdown the server."""
        self._httpd.shutdown()
