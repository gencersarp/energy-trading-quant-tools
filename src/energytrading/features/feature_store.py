"""Feature store: registry, versioning, caching, and materialization."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureDefinition:
    name: str
    version: str
    compute_fn: Callable
    dependencies: List[str] = field(default_factory=list)
    description: str = ""
    tags: List[str] = field(default_factory=list)
    ttl_seconds: Optional[int] = None  # None = no expiry


@dataclass
class FeatureRecord:
    name: str
    version: str
    data: Any
    computed_at: float = field(default_factory=time.time)
    ttl_seconds: Optional[int] = None

    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        return (time.time() - self.computed_at) > self.ttl_seconds


class FeatureStore:
    """
    Lightweight feature store with registry, versioning, and in-memory caching.
    Supports both offline (batch) and online (single-row) feature retrieval.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self._registry: Dict[str, FeatureDefinition] = {}
        self._cache: Dict[str, FeatureRecord] = {}
        self._cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Registry
    # ------------------------------------------------------------------ #

    def register(self, defn: FeatureDefinition) -> "FeatureStore":
        key = f"{defn.name}:{defn.version}"
        self._registry[key] = defn
        return self

    def register_fn(self, name: str, version: str = "1.0",
                    dependencies: Optional[List[str]] = None,
                    description: str = "", tags: Optional[List[str]] = None,
                    ttl_seconds: Optional[int] = None):
        """Decorator to register a function as a feature."""
        def decorator(fn: Callable) -> Callable:
            defn = FeatureDefinition(
                name=name, version=version, compute_fn=fn,
                dependencies=dependencies or [],
                description=description, tags=tags or [],
                ttl_seconds=ttl_seconds,
            )
            self.register(defn)
            return fn
        return decorator

    def list_features(self, tag: Optional[str] = None) -> List[Dict]:
        out = []
        for key, defn in self._registry.items():
            if tag and tag not in defn.tags:
                continue
            out.append({
                "name": defn.name, "version": defn.version,
                "description": defn.description, "tags": defn.tags,
                "dependencies": defn.dependencies,
            })
        return out

    def get_definition(self, name: str, version: str = "latest") -> FeatureDefinition:
        if version == "latest":
            matches = [(k, v) for k, v in self._registry.items()
                       if v.name == name]
            if not matches:
                raise KeyError(f"Feature '{name}' not registered")
            key = sorted(matches, key=lambda x: x[1].version)[-1][0]
        else:
            key = f"{name}:{version}"
        if key not in self._registry:
            raise KeyError(f"Feature '{key}' not registered")
        return self._registry[key]

    # ------------------------------------------------------------------ #
    # Compute & cache
    # ------------------------------------------------------------------ #

    def compute(self, name: str, *args, version: str = "latest",
                force: bool = False, **kwargs) -> Any:
        defn = self.get_definition(name, version)
        cache_key = self._cache_key(name, defn.version, args, kwargs)

        if not force:
            record = self._cache.get(cache_key)
            if record and not record.is_expired():
                return record.data
            # Try disk cache
            disk_data = self._load_from_disk(cache_key)
            if disk_data is not None:
                self._cache[cache_key] = FeatureRecord(
                    name=name, version=defn.version, data=disk_data,
                    ttl_seconds=defn.ttl_seconds,
                )
                return disk_data

        result = defn.compute_fn(*args, **kwargs)
        record = FeatureRecord(name=name, version=defn.version, data=result,
                               ttl_seconds=defn.ttl_seconds)
        self._cache[cache_key] = record
        self._save_to_disk(cache_key, result)
        return result

    def compute_many(self, names: List[str], *args,
                     version: str = "latest", **kwargs) -> Dict[str, Any]:
        return {n: self.compute(n, *args, version=version, **kwargs) for n in names}

    def materialize(self, df: pd.DataFrame, feature_names: List[str],
                    version: str = "latest") -> pd.DataFrame:
        """Compute and attach features to a DataFrame."""
        result = df.copy()
        for name in feature_names:
            defn = self.get_definition(name, version)
            result[name] = defn.compute_fn(df)
        return result

    # ------------------------------------------------------------------ #
    # Point-in-time / entity lookup
    # ------------------------------------------------------------------ #

    def get_online(self, name: str, entity_id: str,
                   version: str = "latest") -> Optional[Any]:
        """Retrieve latest cached value for an entity (online serving)."""
        defn = self.get_definition(name, version)
        cache_key = self._cache_key(name, defn.version, (entity_id,), {})
        record = self._cache.get(cache_key)
        if record and not record.is_expired():
            return record.data
        return None

    def set_online(self, name: str, entity_id: str, value: Any,
                   version: str = "latest") -> None:
        defn = self.get_definition(name, version)
        cache_key = self._cache_key(name, defn.version, (entity_id,), {})
        self._cache[cache_key] = FeatureRecord(
            name=name, version=defn.version, data=value,
            ttl_seconds=defn.ttl_seconds,
        )

    # ------------------------------------------------------------------ #
    # Dependency resolution
    # ------------------------------------------------------------------ #

    def resolve_order(self, names: List[str],
                      version: str = "latest") -> List[str]:
        """Topological sort of features based on dependencies."""
        order: List[str] = []
        visited: set = set()

        def visit(n: str):
            if n in visited:
                return
            visited.add(n)
            defn = self.get_definition(n, version)
            for dep in defn.dependencies:
                visit(dep)
            order.append(n)

        for n in names:
            visit(n)
        return order

    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #

    def _cache_key(self, name: str, version: str, args, kwargs) -> str:
        try:
            sig = json.dumps({"n": name, "v": version,
                              "a": str(args)[:200], "k": str(kwargs)[:200]},
                             sort_keys=True)
        except Exception:
            sig = f"{name}:{version}"
        return hashlib.md5(sig.encode()).hexdigest()

    def _disk_path(self, key: str) -> str:
        if not self._cache_dir:
            return ""
        return os.path.join(self._cache_dir, f"{key}.pkl")

    def _save_to_disk(self, key: str, data: Any) -> None:
        if not self._cache_dir:
            return
        try:
            import pickle
            with open(self._disk_path(key), "wb") as f:
                pickle.dump(data, f)
        except Exception:
            pass

    def _load_from_disk(self, key: str) -> Optional[Any]:
        if not self._cache_dir:
            return None
        path = self._disk_path(key)
        if not os.path.exists(path):
            return None
        try:
            import pickle
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def clear_cache(self, expired_only: bool = False) -> int:
        if expired_only:
            to_delete = [k for k, v in self._cache.items() if v.is_expired()]
        else:
            to_delete = list(self._cache.keys())
        for k in to_delete:
            del self._cache[k]
        return len(to_delete)

    def cache_stats(self) -> Dict[str, int]:
        total = len(self._cache)
        expired = sum(1 for v in self._cache.values() if v.is_expired())
        return {"total": total, "live": total - expired, "expired": expired}
