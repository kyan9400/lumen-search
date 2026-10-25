import time

import fakeredis
import pytest
import redis

from lumen.cache import MemoryCache, NullCache, RedisCache, make_cache


def test_memory_cache_expires_and_evicts(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    cache = MemoryCache(max_items=2)
    cache.set("a", b"1", ttl=10)
    cache.set("b", b"2", ttl=10)
    assert cache.get("a") == b"1"  # "a" is now the most recently used
    cache.set("c", b"3", ttl=10)
    assert cache.get("b") is None  # least recently used went first
    assert cache.get("a") == b"1"
    now[0] += 11
    assert cache.get("a") is None


def test_null_cache_stores_nothing() -> None:
    cache = NullCache()
    cache.set("a", b"1", ttl=10)
    assert cache.get("a") is None


def test_redis_cache_round_trip() -> None:
    cache = RedisCache("redis://localhost:6379/0")
    cache._redis = fakeredis.FakeRedis()
    cache.set("k", b"value", ttl=30)
    assert cache.get("k") == b"value"
    assert cache._redis.ttl("lumen:k") > 0
    assert cache.ping()


class _BrokenRedis:
    calls = 0

    def get(self, key: str) -> bytes:
        _BrokenRedis.calls += 1
        raise redis.ConnectionError("down")

    def set(self, *args: object, **kwargs: object) -> None:
        _BrokenRedis.calls += 1
        raise redis.ConnectionError("down")

    def ping(self) -> bool:
        raise redis.ConnectionError("down")


def test_redis_errors_are_misses_and_pause_the_cache() -> None:
    cache = RedisCache("redis://localhost:6379/0", cooldown=60)
    cache._redis = _BrokenRedis()  # type: ignore[assignment]
    assert cache.get("k") is None
    cache.set("k", b"v", ttl=5)
    assert cache.get("k") is None
    assert _BrokenRedis.calls == 1  # after the first failure Redis is not called again during the cooldown
    assert cache.ping() is False


def test_make_cache_chooses_the_backend() -> None:
    assert make_cache(None).backend == "memory"
    assert make_cache("redis://localhost:6379/0").backend == "redis"
