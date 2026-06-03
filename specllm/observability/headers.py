"""Observability headers for specllm responses."""

from typing import Dict


def build_headers(
    request_id: str,
    provider: str,
    model: str,
    latency_ms: int,
    tokens_used: int,
    retries: int,
    cache_hit: bool,
) -> Dict[str, str]:
    """Build X-SpecLLM-* observability headers dict."""
    return {
        "X-SpecLLM-Request-Id": str(request_id),
        "X-SpecLLM-Provider": str(provider),
        "X-SpecLLM-Model": str(model),
        "X-SpecLLM-Latency-Ms": str(int(latency_ms)),
        "X-SpecLLM-Tokens-Used": str(int(tokens_used)),
        "X-SpecLLM-Retries": str(int(retries)),
        "X-SpecLLM-Cache-Hit": "true" if cache_hit else "false",
    }
