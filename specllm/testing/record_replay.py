"""Record/replay provider for testing without LLM credentials.

Records real LLM responses to a JSON file, then replays them in CI.

Usage:
    # Record mode (needs credentials):
    provider = RecordReplayProvider(real_provider, cassette="tests/cassettes/my_test.json")
    app = SpecLLM(spec=spec, provider=provider)
    # ... run your tests, responses are saved to cassette file

    # Replay mode (no credentials needed):
    provider = RecordReplayProvider(cassette="tests/cassettes/my_test.json")
    app = SpecLLM(spec=spec, provider=provider)
    # ... responses served from cassette file
"""

import hashlib
import json
import os
from typing import Optional

from specllm.llm.providers import LLMProvider


class RecordReplayProvider(LLMProvider):
    """Provider that records real responses or replays from a cassette file.

    If a real provider is given, it records. If not, it replays.
    """

    def __init__(self, provider: Optional[LLMProvider] = None, cassette: str = "cassette.json") -> None:
        self.provider = provider
        self.cassette = cassette
        self._recordings: dict = {}
        self._load_cassette()

    def _load_cassette(self) -> None:
        """Load existing recordings from cassette file."""
        if os.path.exists(self.cassette):
            with open(self.cassette, "r") as f:
                self._recordings = json.load(f)

    def _save_cassette(self) -> None:
        """Save recordings to cassette file."""
        os.makedirs(os.path.dirname(self.cassette) or ".", exist_ok=True)
        with open(self.cassette, "w") as f:
            json.dump(self._recordings, f, indent=2)

    def _key(self, prompt: str) -> str:
        """Generate a deterministic key from prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]

    def call(self, prompt: str, system_prompt: Optional[str] = None) -> dict:
        """Call real provider and record, or replay from cassette."""
        key = self._key(prompt)

        # Replay mode
        if key in self._recordings:
            return self._recordings[key]

        # Record mode
        if self.provider is None:
            raise RuntimeError(
                f"No recording found for prompt (key={key}) and no real provider configured. "
                f"Run in record mode first (pass a real provider)."
            )

        result = self.provider.call(prompt, system_prompt)
        self._recordings[key] = result
        self._save_cassette()
        return result
