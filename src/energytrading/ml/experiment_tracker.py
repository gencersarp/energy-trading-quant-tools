"""Experiment tracking system (MLflow-compatible API, no MLflow dependency)."""
from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional

import pandas as pd


@dataclass
class Run:
    run_id: str
    experiment_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    status: str = "RUNNING"  # RUNNING | FINISHED | FAILED


class ExperimentTracker:
    """Lightweight experiment tracker compatible with MLflow's API surface."""

    def __init__(self, storage_path: str = ".experiments"):
        self._storage_path = storage_path
        self._runs: Dict[str, Run] = {}
        self._active_run_id: Optional[str] = None

    def start_run(self, experiment_name: str = "default",
                  tags: Optional[Dict[str, str]] = None) -> str:
        run_id = str(uuid.uuid4())[:8]
        run = Run(
            run_id=run_id,
            experiment_name=experiment_name,
            tags=tags or {},
            start_time=datetime.utcnow().isoformat(),
        )
        self._runs[run_id] = run
        self._active_run_id = run_id
        return run_id

    @contextmanager
    def run_context(self, experiment_name: str = "default",
                    tags: Optional[Dict] = None) -> Generator[str, None, None]:
        run_id = self.start_run(experiment_name, tags)
        try:
            yield run_id
            self.end_run(status="FINISHED")
        except Exception:
            self.end_run(status="FAILED")
            raise

    def log_param(self, key: str, value: Any) -> None:
        if self._active_run_id:
            self._runs[self._active_run_id].params[key] = value

    def log_params(self, params: Dict[str, Any]) -> None:
        for k, v in params.items():
            self.log_param(k, v)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        if self._active_run_id:
            run = self._runs[self._active_run_id]
            if key not in run.metrics:
                run.metrics[key] = []
            run.metrics[key].append(float(value))

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for k, v in metrics.items():
            self.log_metric(k, v, step)

    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        if self._active_run_id:
            self._runs[self._active_run_id].artifacts.append(name or path)

    def end_run(self, status: str = "FINISHED") -> None:
        if self._active_run_id:
            run = self._runs[self._active_run_id]
            run.end_time = datetime.utcnow().isoformat()
            run.status = status
            self._active_run_id = None

    def get_run(self, run_id: str) -> Run:
        if run_id not in self._runs:
            raise KeyError(f"Run {run_id} not found")
        return self._runs[run_id]

    def search_runs(self, experiment_name: Optional[str] = None,
                    filter_dict: Optional[Dict] = None) -> pd.DataFrame:
        rows = []
        for run in self._runs.values():
            if experiment_name and run.experiment_name != experiment_name:
                continue
            if filter_dict:
                match = all(run.params.get(k) == v for k, v in filter_dict.items())
                if not match:
                    continue
            row = {
                "run_id": run.run_id,
                "experiment": run.experiment_name,
                "status": run.status,
                "start_time": run.start_time,
                **{f"param.{k}": v for k, v in run.params.items()},
                **{f"metric.{k}": vals[-1] if vals else None
                   for k, vals in run.metrics.items()},
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def best_run(self, experiment_name: str, metric: str,
                 mode: str = "max") -> Run:
        candidates = [r for r in self._runs.values()
                      if r.experiment_name == experiment_name
                      and metric in r.metrics and r.metrics[metric]]
        if not candidates:
            raise ValueError(f"No runs with metric '{metric}' in '{experiment_name}'")
        key = lambda r: r.metrics[metric][-1]
        if mode == "max":
            return max(candidates, key=key)
        return min(candidates, key=key)

    def plot_metric_history(self, run_id: str, metric: str) -> Dict:
        run = self.get_run(run_id)
        values = run.metrics.get(metric, [])
        return {"metric": metric, "steps": list(range(len(values))),
                "values": values, "run_id": run_id}

    def save(self, path: Optional[str] = None) -> None:
        path = path or self._storage_path
        os.makedirs(path, exist_ok=True)
        data = {rid: asdict(r) for rid, r in self._runs.items()}
        with open(os.path.join(path, "runs.json"), "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Optional[str] = None) -> None:
        path = path or self._storage_path
        runs_path = os.path.join(path, "runs.json")
        if not os.path.exists(runs_path):
            return
        with open(runs_path) as f:
            data = json.load(f)
        for rid, r_dict in data.items():
            self._runs[rid] = Run(**r_dict)
