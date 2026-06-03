"""Retry logic for LLM calls with validation feedback."""

from typing import Callable, List, Optional


class MaxRetriesExceeded(Exception):
    """Raised when all retry attempts are exhausted."""

    pass


def _format_feedback(errors: list) -> str:
    """Format validation errors into feedback for the LLM."""
    lines = ["Your previous response failed schema validation with the following errors:"]
    for err in errors:
        lines.append(f"- Field '{err.path}': {err.message}")
    lines.append("Please correct these issues and respond with valid JSON.")
    return "\n".join(lines)


class RetryHandler:
    """Handles retrying LLM calls on validation failures."""

    def __init__(self, max_retries: int = 3) -> None:
        self.max_retries = max_retries

    def execute(self, call_fn: Callable, validate_fn: Callable[[dict], list]) -> dict:
        """Execute call_fn and retry if validate_fn returns errors.

        Total calls = 1 (initial) + max_retries. On retries, call_fn
        receives a feedback string describing the validation errors.
        """
        result = call_fn(None)
        errors = validate_fn(result)

        if not errors:
            return result

        for _ in range(self.max_retries):
            feedback = _format_feedback(errors)
            result = call_fn(feedback)
            errors = validate_fn(result)
            if not errors:
                return result

        raise MaxRetriesExceeded(f"Max retries ({self.max_retries}) exceeded")
