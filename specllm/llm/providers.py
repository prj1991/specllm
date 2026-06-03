"""LLM provider interface and implementations."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def call(self, prompt: str, system_prompt: Optional[str] = None) -> dict:
        """Call the LLM with a prompt and optional system prompt."""
        pass

    def with_model(self, model: str) -> "LLMProvider":
        """Return a provider instance configured for a specific model.

        Override this in subclasses to support per-endpoint model selection.
        Default implementation returns self (same model for all endpoints).
        """
        return self


class MockProvider(LLMProvider):
    """Mock provider that returns configurable responses for testing."""

    def __init__(self, responses: Optional[List[dict]] = None) -> None:
        self._responses: List[dict] = responses or []
        self._call_count: int = 0
        self._calls: List[Dict[str, Optional[str]]] = []

    def call(self, prompt: str, system_prompt: Optional[str] = None) -> dict:
        """Return next configured response or empty dict."""
        self._calls.append({"prompt": prompt, "system_prompt": system_prompt})
        if self._responses:
            response = self._responses[self._call_count % len(self._responses)]
        else:
            response = {}
        self._call_count += 1
        return response
