"""Tests for specllm.pipeline.cache module - in-memory caching with TTL."""
import unittest
from unittest.mock import patch

from specllm.pipeline.cache import Cache

class TestCache(unittest.TestCase):
    """Test in-memory cache with TTL."""

    def setUp(self):
        """Set up cache with default TTL."""
        self.cache = Cache(default_ttl=3600)

    def test_cache_miss_returns_none(self):
        """Cache miss returns None."""
        result = self.cache.get("nonexistent_key")
        self.assertIsNone(result)

    def test_cache_set_then_get_returns_value(self):
        """Cache set then get returns the stored value."""
        self.cache.set("key1", {"id": 1, "name": "Alice"})
        result = self.cache.get("key1")
        self.assertEqual(result, {"id": 1, "name": "Alice"})

    @patch("time.time")
    def test_ttl_expiry(self, mock_time):
        """After TTL expires, get returns None."""
        mock_time.return_value = 1000.0
        self.cache.set("key1", {"data": "value"}, ttl=60)

        # Before expiry
        mock_time.return_value = 1050.0
        self.assertEqual(self.cache.get("key1"), {"data": "value"})

        # After expiry
        mock_time.return_value = 1061.0
        self.assertIsNone(self.cache.get("key1"))

    def test_different_keys_are_independent(self):
        """Different keys store and retrieve independently."""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        self.assertEqual(self.cache.get("key1"), "value1")
        self.assertEqual(self.cache.get("key2"), "value2")

    def test_cache_key_generation_deterministic(self):
        """Same input produces same cache key."""
        key1 = self.cache.generate_key("/users", "post", {"name": "Alice"})
        key2 = self.cache.generate_key("/users", "post", {"name": "Alice"})
        self.assertEqual(key1, key2)

    def test_cache_key_different_inputs(self):
        """Different inputs produce different cache keys."""
        key1 = self.cache.generate_key("/users", "post", {"name": "Alice"})
        key2 = self.cache.generate_key("/users", "post", {"name": "Bob"})
        self.assertNotEqual(key1, key2)

    def test_cache_key_is_string(self):
        """Generated cache key is a string."""
        key = self.cache.generate_key("/users", "get", None)
        self.assertIsInstance(key, str)

    def test_default_ttl(self):
        """Cache uses default_ttl when no ttl specified in set."""
        cache = Cache(default_ttl=3600)
        self.assertEqual(cache.default_ttl, 3600)

if __name__ == "__main__":
    unittest.main()
