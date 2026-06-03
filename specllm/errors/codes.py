"""Error codes and structured error responses."""

import datetime
from enum import Enum
from typing import Dict, Optional


class ErrorCode(Enum):
    """Standard error codes for specllm."""

    INPUT_VALIDATION_FAILED = "INPUT_VALIDATION_FAILED"
    OUTPUT_SCHEMA_VIOLATION = "OUTPUT_SCHEMA_VIOLATION"
    PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    RATE_LIMITED = "RATE_LIMITED"
    COST_LIMIT_REACHED = "COST_LIMIT_REACHED"
    ENDPOINT_NOT_FOUND = "ENDPOINT_NOT_FOUND"
    ALL_PROVIDERS_FAILED = "ALL_PROVIDERS_FAILED"
    PROVIDER_RATE_LIMITED = "PROVIDER_RATE_LIMITED"
    ENDPOINT_DISABLED = "ENDPOINT_DISABLED"
    OUTPUT_SAFETY_VIOLATION = "OUTPUT_SAFETY_VIOLATION"


HTTP_STATUS_MAP: Dict[ErrorCode, int] = {
    ErrorCode.INPUT_VALIDATION_FAILED: 400,
    ErrorCode.OUTPUT_SCHEMA_VIOLATION: 422,
    ErrorCode.PROVIDER_TIMEOUT: 504,
    ErrorCode.PROVIDER_UNAVAILABLE: 503,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.COST_LIMIT_REACHED: 503,
    ErrorCode.ENDPOINT_NOT_FOUND: 404,
    ErrorCode.ALL_PROVIDERS_FAILED: 503,
    ErrorCode.PROVIDER_RATE_LIMITED: 503,
    ErrorCode.ENDPOINT_DISABLED: 503,
    ErrorCode.OUTPUT_SAFETY_VIOLATION: 500,
}


def build_error_response(
    code: ErrorCode,
    message: str,
    request_id: str,
    details: Optional[dict] = None,
) -> dict:
    """Build a structured error response dict."""
    error: dict = {
        "code": code.name,
        "status": HTTP_STATUS_MAP.get(code, 500),
        "message": message,
        "request_id": request_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if details:
        error["details"] = details
    return {"error": error}
