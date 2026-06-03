"""In-memory cache with TTL support."""

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any, Optional


class Cache:
    """Thread-safe in-memory cache with TTL expiration and max size eviction."""

    def __init__(self, default_ttl: int = 3600, max_size: int = 1000) -> None:
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._store: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache, or None if missing/expired."""
        with self._lock:
            if key not in self._store:
                return None
            value, expires_at = self._store[key]
            if time.time() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in cache. Evicts oldest entry if at max_size."""
        if ttl is None:
            ttl = self.default_ttl
        expires_at = time.time() + ttl
        with self._lock:
            if key in self._store:
                del self._store[key]
            elif len(self._store) >= self.max_size:
                self._store.popitem(last=False)
            self._store[key] = (value, expires_at)

    def generate_key(self, endpoint_path: str, method: str, body: Optional[Any] = None) -> str:
        """Generate a deterministic SHA-256 cache key from endpoint info."""
        key_data = json.dumps(
            {
                "path": endpoint_path,
                "method": method,
                "body": body,
            },
            sort_keys=True,
        )
        return hashlib.sha256(key_data.encode()).hexdigest()
