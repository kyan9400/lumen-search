"""Result and embedding cache: Redis when configured, otherwise an in-process LRU with expiry."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Protocol

log = logging.getLogger(__name__)


class Cache(Protocol):
    backend: str

    def get(self, key: str) -> bytes | None: ...

    def set(self, key: str, value: bytes, ttl: int) -> None: ...

    def ping(self) -> bool: ...


class MemoryCache:
    backend = "memory"

    def __init__(self, max_items: int = 4096) -> None:
        self.max_items = max_items
        self._items: OrderedDict[str, tuple[float, bytes]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires, value = item
            if expires < time.monotonic():
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return value

    def set(self, key: str, value: bytes, ttl: int) -> None:
        with self._lock:
            self._items[key] = (time.monotonic() + ttl, value)
            self._items.move_to_end(key)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    def ping(self) -> bool:
        return True


class NullCache:
    """Caches nothing: used when measuring latency."""

    backend = "none"

    def get(self, key: str) -> bytes | None:
        return None

    def set(self, key: str, value: bytes, ttl: int) -> None:
        return None

    def ping(self) -> bool:
        return True


class RedisCache:
    """Redis with short timeouts. A cache must never take search down: errors count as misses, and
    after a failure Redis is left alone for ``cooldown`` seconds."""

    backend = "redis"

    def __init__(self, url: str, *, prefix: str = "lumen:", cooldown: float = 30.0) -> None:
        import redis

        self._redis = redis.Redis.from_url(url, socket_timeout=0.5, socket_connect_timeout=0.5)
        self._errors: tuple[type[Exception], ...] = (redis.RedisError, OSError)
        self.prefix = prefix
        self.cooldown = cooldown
        self._down_until = 0.0

    def _usable(self) -> bool:
        return time.monotonic() >= self._down_until

    def _failed(self, exc: Exception) -> None:
        log.warning("Redis unavailable, caching paused for %.0f s: %s", self.cooldown, exc)
        self._down_until = time.monotonic() + self.cooldown

    def get(self, key: str) -> bytes | None:
        if not self._usable():
            return None
        try:
            value = self._redis.get(self.prefix + key)
        except self._errors as exc:
            self._failed(exc)
            return None
        return value if isinstance(value, bytes) else None

    def set(self, key: str, value: bytes, ttl: int) -> None:
        if not self._usable():
            return
        try:
            self._redis.set(self.prefix + key, value, ex=ttl)
        except self._errors as exc:
            self._failed(exc)

    def ping(self) -> bool:
        try:
            return bool(self._redis.ping())
        except self._errors:
            return False


def make_cache(redis_url: str | None) -> Cache:
    return RedisCache(redis_url) if redis_url else MemoryCache()
