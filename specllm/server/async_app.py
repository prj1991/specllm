"""Async HTTP server for specllm using asyncio."""

import asyncio
import json
import time
import uuid
from typing import Any

from specllm.observability.headers import build_headers


class AsyncSpecLLMServer:
    """Async HTTP server that handles concurrent LLM requests."""

    def __init__(self, app: Any, host: str = "127.0.0.1", port: int = 8080) -> None:
        self.app = app
        self.host = host
        self.port = port

    async def _handle_request(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle a single HTTP request."""
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Read HTTP request
        request_line = await reader.readline()
        if not request_line:
            writer.close()
            return

        parts = request_line.decode().strip().split(" ")
        if len(parts) < 2:
            writer.close()
            return

        method = parts[0].lower()
        path = parts[1]

        # Read headers
        content_length = 0
        while True:
            line = await reader.readline()
            if line == b"\r\n" or line == b"\n" or not line:
                break
            header = line.decode().strip().lower()
            if header.startswith("content-length:"):
                content_length = int(header.split(":")[1].strip())

        # Read body
        request_body = {}
        if content_length > 0:
            raw = await reader.readexactly(content_length)
            try:
                request_body = json.loads(raw)
            except json.JSONDecodeError:
                self._send_response(writer, 400, {"error": {"code": "MALFORMED_REQUEST", "message": "Invalid JSON"}})
                return

        # Route through pipeline
        endpoint = None
        for ep in self.app.endpoints:
            if ep.path == path and ep.method.lower() == method:
                endpoint = ep
                break

        if not endpoint:
            self._send_response(writer, 404, {"error": {"code": "ENDPOINT_NOT_FOUND", "message": f"No endpoint: {path}"}})
            return

        # Run pipeline in executor to not block event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.app._pipeline.handle, endpoint, request_body)

        latency_ms = int((time.time() - start_time) * 1000)
        metadata = self.app._pipeline.last_metadata

        status = 200
        if "error" in result:
            status = result["error"].get("status", 500)

        headers = build_headers(
            request_id=request_id,
            provider="default",
            model=self.app.model or "default",
            latency_ms=latency_ms,
            tokens_used=metadata.get("tokens_used", 0),
            retries=metadata.get("retries", 0),
            cache_hit=metadata.get("cache_hit", False),
        )

        self._send_response(writer, status, result, headers)

    def _send_response(self, writer: asyncio.StreamWriter, status: int, body: dict, extra_headers: dict = None) -> None:
        """Write an HTTP response."""
        body_bytes = json.dumps(body).encode()
        status_text = {200: "OK", 400: "Bad Request", 404: "Not Found", 422: "Unprocessable Entity", 503: "Service Unavailable", 504: "Gateway Timeout"}.get(status, "Error")

        lines = [f"HTTP/1.1 {status} {status_text}"]
        lines.append(f"Content-Type: application/json")
        lines.append(f"Content-Length: {len(body_bytes)}")
        if extra_headers:
            for k, v in extra_headers.items():
                lines.append(f"{k}: {v}")
        lines.append("")
        lines.append("")

        writer.write("\r\n".join(lines).encode() + body_bytes)
        writer.close()

    def serve(self) -> None:
        """Start the async server (blocking)."""
        asyncio.run(self._run())

    async def _run(self) -> None:
        server = await asyncio.start_server(self._handle_request, self.host, self.port)
        async with server:
            await server.serve_forever()
