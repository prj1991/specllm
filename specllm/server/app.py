"""HTTP server using stdlib http.server."""

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Set


def _build_headers(request_id, latency_ms, tokens_used, retries, cache_hit):
    return {
        "X-SpecLLM-Request-Id": str(request_id),
        "X-SpecLLM-Latency-Ms": str(int(latency_ms)),
        "X-SpecLLM-Tokens-Used": str(int(tokens_used)),
        "X-SpecLLM-Retries": str(int(retries)),
        "X-SpecLLM-Cache-Hit": "true" if cache_hit else "false",
    }


class SpecLLMServer:
    """HTTP server that routes requests through the app's pipeline."""

    def __init__(self, app, host: str = "127.0.0.1", port: int = 8080) -> None:
        self.app = app
        self.host = host
        self.port = port

        self._routes: Dict[tuple, Any] = {}
        self._path_methods: Dict[str, Set[str]] = {}
        for ep in app.endpoints:
            self._routes[(ep.method.lower(), ep.path)] = ep
            self._path_methods.setdefault(ep.path, set()).add(ep.method.lower())

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

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self.port = self._httpd.server_address[1]

    def _handle_request(self, handler: BaseHTTPRequestHandler, method: str) -> None:
        request_id = str(uuid.uuid4())
        start_time = time.time()
        path = handler.path

        if path not in self._path_methods:
            self._respond(handler, 404, {"error": {
                "code": "ENDPOINT_NOT_FOUND", "message": f"No endpoint: {path}", "request_id": request_id,
            }}, request_id, start_time)
            return

        if method not in self._path_methods[path]:
            self._respond(handler, 405, {"error": {
                "code": "METHOD_NOT_ALLOWED", "message": f"{method.upper()} not allowed",
            }}, request_id, start_time)
            return

        content_length = int(handler.headers.get("Content-Length", 0))
        if content_length > 0:
            try:
                request_body = json.loads(handler.rfile.read(content_length))
            except (json.JSONDecodeError, ValueError):
                self._respond(handler, 400, {"error": {
                    "code": "MALFORMED_REQUEST", "message": "Invalid JSON", "request_id": request_id,
                }}, request_id, start_time)
                return
        else:
            request_body = {}

        endpoint = self._routes[(method, path)]
        result = self.app._pipeline.handle(endpoint, request_body)

        status = result["error"]["status"] if "error" in result else 200
        self._respond(handler, status, result, request_id, start_time)

    def _respond(self, handler, status, body, request_id, start_time):
        metadata = getattr(self.app._pipeline, "last_metadata", {})
        headers = _build_headers(
            request_id=request_id,
            latency_ms=int((time.time() - start_time) * 1000),
            tokens_used=metadata.get("tokens_used", 0),
            retries=metadata.get("retries", 0),
            cache_hit=metadata.get("cache_hit", False),
        )
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        for k, v in headers.items():
            handler.send_header(k, v)
        handler.end_headers()
        handler.wfile.write(json.dumps(body).encode())

    def serve(self) -> None:
        self._httpd.serve_forever()

    def shutdown(self) -> None:
        self._httpd.shutdown()
