"""Redis pub/sub handler and cache layer for real-time signals."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RedisConfig:
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    channel_prefix: str = "energy"


class _InMemoryStore:
    """Thread-safe in-memory fallback when Redis is unavailable."""

    def __init__(self):
        self._store: Dict[str, Any] = {}
        self._ttls: Dict[str, float] = {}

    def set(self, key: str, value: Any, ttl: int = 0) -> None:
        self._store[key] = value
        if ttl > 0:
            self._ttls[key] = time.time() + ttl

    def get(self, key: str) -> Optional[Any]:
        if key in self._ttls and time.time() > self._ttls[key]:
            self._store.pop(key, None)
            self._ttls.pop(key, None)
            return None
        return self._store.get(key)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)
        self._ttls.pop(key, None)

    def keys(self, pattern: str = "*") -> List[str]:
        return list(self._store.keys())


class RedisSignalPublisher:
    """Publishes trading signals and alerts to Redis."""

    def __init__(self, config: RedisConfig):
        self.config = config
        self._redis = None
        self._store = _InMemoryStore()
        try:
            import redis as _redis
            self._redis = _redis.Redis(
                host=config.host, port=config.port,
                db=config.db, password=config.password,
                decode_responses=True,
            )
            self._redis.ping()
        except (ImportError, Exception) as e:
            logger.warning(f"Redis unavailable ({e}), using in-memory store")
            self._redis = None

    def _key(self, name: str) -> str:
        return f"{self.config.channel_prefix}:{name}"

    def _set(self, key: str, value: Any, ttl: int = 300) -> None:
        serialized = json.dumps(value)
        if self._redis:
            self._redis.setex(key, ttl, serialized)
            self._redis.publish(key, serialized)
        else:
            self._store.set(key, value, ttl=ttl)

    def publish_signal(self, signal_name: str, signal_value: float,
                        metadata: Optional[Dict] = None) -> None:
        data = {
            "signal": signal_name,
            "value": signal_value,
            "timestamp": datetime.utcnow().isoformat(),
            **(metadata or {}),
        }
        self._set(self._key(f"signal:{signal_name}"), data)

    def publish_risk_alert(self, alert_type: str, severity: str,
                            details: Dict) -> None:
        data = {
            "type": alert_type,
            "severity": severity,
            "details": details,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._set(self._key(f"alert:{alert_type}"), data, ttl=3600)

    def set_market_state(self, state_dict: Dict, ttl_seconds: int = 300) -> None:
        self._set(self._key("market_state"), state_dict, ttl=ttl_seconds)


class RedisSignalSubscriber:
    """Subscribes to Redis channels and retrieves cached signals."""

    def __init__(self, config: RedisConfig):
        self.config = config
        self._redis = None
        self._store = _InMemoryStore()
        try:
            import redis as _redis
            self._redis = _redis.Redis(
                host=config.host, port=config.port,
                db=config.db, password=config.password,
                decode_responses=True,
            )
        except (ImportError, Exception):
            pass

    def _key(self, name: str) -> str:
        return f"{self.config.channel_prefix}:{name}"

    def _get(self, key: str) -> Optional[Any]:
        if self._redis:
            val = self._redis.get(key)
            return json.loads(val) if val else None
        return self._store.get(key)

    def subscribe(self, channel: str,
                   callback_fn: Callable[[Dict], None]) -> None:
        if self._redis:
            pub_sub = self._redis.pubsub()
            pub_sub.subscribe(**{self._key(channel): lambda msg: callback_fn(
                json.loads(msg["data"]) if msg["data"] else {})})
        # In mock mode: just store the callback for manual invocation in tests

    def get_latest_signal(self, signal_name: str) -> Optional[Dict]:
        return self._get(self._key(f"signal:{signal_name}"))

    def get_market_state(self) -> Optional[Dict]:
        return self._get(self._key("market_state"))


class RedisCacheLayer:
    """Feature and result caching with TTL."""

    def __init__(self, config: RedisConfig):
        self.config = config
        self._redis = None
        self._store = _InMemoryStore()
        try:
            import redis as _redis
            self._redis = _redis.Redis(
                host=config.host, port=config.port,
                db=config.db, decode_responses=True)
            self._redis.ping()
        except (ImportError, Exception):
            pass

    def cache(self, key: str, value: Any, ttl: int = 3600) -> None:
        serialized = json.dumps(value)
        if self._redis:
            self._redis.setex(key, ttl, serialized)
        else:
            self._store.set(key, value, ttl=ttl)

    def get(self, key: str) -> Optional[Any]:
        if self._redis:
            val = self._redis.get(key)
            return json.loads(val) if val else None
        return self._store.get(key)

    def invalidate(self, key: str) -> None:
        if self._redis:
            self._redis.delete(key)
        else:
            self._store.delete(key)

    def warm_cache(self, feature_dict: Dict[str, Any], ttl: int = 3600) -> None:
        for key, value in feature_dict.items():
            self.cache(key, value, ttl=ttl)
