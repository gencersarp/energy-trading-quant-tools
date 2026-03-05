"""Production model registry with versioning, staging, and disk persistence."""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


@dataclass
class ModelMetadata:
    model_id: str
    name: str
    version: str
    model_type: str
    created_at: str
    metrics: Dict[str, float]
    tags: List[str]
    artifact_path: str
    description: str
    stage: str = "development"  # development | staging | production | archived


class ModelRegistry:
    """Tracks, versions, and promotes ML models."""

    def __init__(self, base_dir: str = "model_registry"):
        self._base_dir = base_dir
        self._models: Dict[str, Any] = {}
        self._metadata: Dict[str, ModelMetadata] = {}

    def register(self, model: Any, name: str, version: str,
                 metrics: Dict[str, float] | None = None,
                 tags: List[str] | None = None,
                 description: str = "",
                 model_type: str = "generic") -> str:
        model_id = str(uuid.uuid4())[:8]
        artifact_path = os.path.join(self._base_dir, f"{name}_{version}_{model_id}.pkl")
        meta = ModelMetadata(
            model_id=model_id,
            name=name,
            version=version,
            model_type=model_type,
            created_at=datetime.utcnow().isoformat(),
            metrics=metrics or {},
            tags=tags or [],
            artifact_path=artifact_path,
            description=description,
        )
        self._models[model_id] = model
        self._metadata[model_id] = meta
        return model_id

    def get(self, model_id: str) -> Tuple[Any, ModelMetadata]:
        if model_id not in self._models:
            raise KeyError(f"Model {model_id} not found")
        return self._models[model_id], self._metadata[model_id]

    def get_latest(self, name: str) -> Tuple[Any, ModelMetadata]:
        matches = [(mid, m) for mid, m in self._metadata.items()
                   if m.name == name]
        if not matches:
            raise KeyError(f"No model named '{name}'")
        latest_id = max(matches, key=lambda x: x[1].created_at)[0]
        return self.get(latest_id)

    def list_models(self, name: Optional[str] = None,
                    tag: Optional[str] = None) -> List[ModelMetadata]:
        result = list(self._metadata.values())
        if name:
            result = [m for m in result if m.name == name]
        if tag:
            result = [m for m in result if tag in m.tags]
        return sorted(result, key=lambda m: m.created_at, reverse=True)

    def promote(self, model_id: str, stage: str = "production") -> None:
        valid = {"development", "staging", "production", "archived"}
        if stage not in valid:
            raise ValueError(f"stage must be one of {valid}")
        # Demote any existing production model with same name
        if stage == "production":
            name = self._metadata[model_id].name
            for mid, meta in self._metadata.items():
                if meta.name == name and meta.stage == "production" and mid != model_id:
                    self._metadata[mid].stage = "archived"
        self._metadata[model_id].stage = stage

    def compare(self, model_ids: List[str]) -> pd.DataFrame:
        rows = []
        for mid in model_ids:
            meta = self._metadata[mid]
            row = {"model_id": mid, "name": meta.name, "version": meta.version,
                   "stage": meta.stage, **meta.metrics}
            rows.append(row)
        return pd.DataFrame(rows)

    def delete(self, model_id: str) -> None:
        self._models.pop(model_id, None)
        self._metadata.pop(model_id, None)

    def save_to_disk(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        for mid, model in self._models.items():
            try:
                joblib.dump(model, os.path.join(path, f"{mid}.pkl"))
            except Exception:
                pass
        metadata_dict = {mid: asdict(m) for mid, m in self._metadata.items()}
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata_dict, f, indent=2)

    def load_from_disk(self, path: str) -> None:
        meta_path = os.path.join(path, "metadata.json")
        if not os.path.exists(meta_path):
            return
        with open(meta_path) as f:
            metadata_dict = json.load(f)
        for mid, m_dict in metadata_dict.items():
            self._metadata[mid] = ModelMetadata(**m_dict)
            pkl_path = os.path.join(path, f"{mid}.pkl")
            if os.path.exists(pkl_path):
                try:
                    self._models[mid] = joblib.load(pkl_path)
                except Exception:
                    pass
