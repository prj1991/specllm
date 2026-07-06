"""LLM provider interface and implementations."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def call(self, prompt: str, system_prompt: Optional[str] = None) -> dict:
        """Call the LLM. May return dict or str (pipeline handles JSON extraction)."""
        pass

    def with_model(self, model: str) -> "LLMProvider":
        """Return a provider configured for a specific model."""
        return self


class MockProvider(LLMProvider):
    """Returns configurable responses for testing."""

    def __init__(self, responses: Optional[List[dict]] = None) -> None:
        self._responses: List[dict] = responses or []
        self._call_count: int = 0
        self._calls: List[Dict[str, Optional[str]]] = []

    def call(self, prompt: str, system_prompt: Optional[str] = None) -> dict:
        self._calls.append({"prompt": prompt, "system_prompt": system_prompt})
        response = self._responses[self._call_count % len(self._responses)] if self._responses else {}
        self._call_count += 1
        return response


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider. Requires: pip install anthropic"""

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: Optional[str] = None) -> None:
        try:
            import anthropic
        except ImportError:
            raise ImportError("Anthropic provider requires: pip install specllm[anthropic]")
        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def call(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        kwargs = {"model": self.model, "max_tokens": 4096, "messages": [{"role": "user", "content": prompt}]}
        if system_prompt:
            kwargs["system"] = system_prompt
        response = self._client.messages.create(**kwargs)
        return response.content[0].text

    def with_model(self, model: str) -> "AnthropicProvider":
        return AnthropicProvider(model=model, api_key=self._client.api_key)


class OpenAIProvider(LLMProvider):
    """OpenAI provider. Requires: pip install openai"""

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError("OpenAI provider requires: pip install specllm[openai]")
        self.model = model
        self._client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()

    def call(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(
            model=self.model, messages=messages,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def with_model(self, model: str) -> "OpenAIProvider":
        return OpenAIProvider(model=model, api_key=self._client.api_key)


def create_provider(model: str) -> LLMProvider:
    """Infer and create the right provider from a model name."""
    if model.startswith(("claude", "anthropic")):
        return AnthropicProvider(model=model)
    if model.startswith(("gpt", "o1", "o3", "openai")):
        return OpenAIProvider(model=model)
    raise ValueError(
        f"Cannot infer provider for model '{model}'. "
        f"Use a model name starting with 'claude' (Anthropic) or 'gpt'/'o1'/'o3' (OpenAI), "
        f"or pass provider= explicitly."
    )
