"""Webhook support for long-running LLM operations.

Enables async processing: accept request → return 202 + job_id →
process in background → POST result to callback URL.
"""

import json
import threading
import time
import urllib.request
import uuid
from typing import Any, Callable, Dict, Optional


class WebhookManager:
    """Manages async job execution and webhook callbacks."""

    def __init__(self) -> None:
        self._jobs: Dict[str, dict] = {}

    def submit(self, job_fn: Callable, callback_url: Optional[str] = None) -> str:
        """Submit a job for background execution. Returns job_id."""
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {"status": "pending", "result": None, "created_at": time.time()}

        def run():
            try:
                result = job_fn()
                self._jobs[job_id]["status"] = "completed"
                self._jobs[job_id]["result"] = result
            except Exception as e:
                self._jobs[job_id]["status"] = "failed"
                self._jobs[job_id]["result"] = {"error": str(e)}

            # Fire webhook callback if configured
            if callback_url:
                self._fire_webhook(callback_url, job_id)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return job_id

    def get_status(self, job_id: str) -> Optional[dict]:
        """Get job status and result."""
        return self._jobs.get(job_id)

    def _fire_webhook(self, callback_url: str, job_id: str) -> None:
        """POST job result to callback URL."""
        job = self._jobs.get(job_id)
        if not job:
            return
        payload = json.dumps({"job_id": job_id, "status": job["status"], "result": job["result"]}).encode()
        req = urllib.request.Request(callback_url, data=payload, headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass  # Best-effort delivery
