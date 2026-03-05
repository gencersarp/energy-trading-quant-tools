"""Hyperparameter optimization: grid search, random search, TPE, CMA-ES."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import cross_val_score


@dataclass
class ParameterSpec:
    name: str
    param_type: str  # 'int' | 'float' | 'categorical'
    low: Any = None
    high: Any = None
    log_scale: bool = False
    choices: List[Any] = field(default_factory=list)


class SearchSpace:
    """Defines the hyperparameter search space."""

    def __init__(self):
        self._specs: List[ParameterSpec] = []

    def add_int(self, name: str, low: int, high: int) -> "SearchSpace":
        self._specs.append(ParameterSpec(name, "int", low, high))
        return self

    def add_float(self, name: str, low: float, high: float,
                  log: bool = False) -> "SearchSpace":
        self._specs.append(ParameterSpec(name, "float", low, high, log_scale=log))
        return self

    def add_categorical(self, name: str, choices: List[Any]) -> "SearchSpace":
        self._specs.append(ParameterSpec(name, "categorical", choices=choices))
        return self

    def sample(self, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        if rng is None:
            rng = np.random.default_rng()
        params = {}
        for spec in self._specs:
            if spec.param_type == "int":
                params[spec.name] = int(rng.integers(spec.low, spec.high + 1))
            elif spec.param_type == "float":
                if spec.log_scale:
                    log_low, log_high = np.log(spec.low), np.log(spec.high)
                    params[spec.name] = float(np.exp(rng.uniform(log_low, log_high)))
                else:
                    params[spec.name] = float(rng.uniform(spec.low, spec.high))
            elif spec.param_type == "categorical":
                params[spec.name] = spec.choices[rng.integers(len(spec.choices))]
        return params

    def grid(self, n_per_param: int = 5) -> List[Dict[str, Any]]:
        """Generate a Cartesian product grid."""
        import itertools
        grids = []
        for spec in self._specs:
            if spec.param_type == "int":
                vals = [int(v) for v in np.linspace(spec.low, spec.high, n_per_param)]
            elif spec.param_type == "float":
                if spec.log_scale:
                    vals = list(np.exp(np.linspace(np.log(spec.low), np.log(spec.high), n_per_param)))
                else:
                    vals = list(np.linspace(spec.low, spec.high, n_per_param))
            else:
                vals = spec.choices
            grids.append(vals)
        keys = [s.name for s in self._specs]
        return [dict(zip(keys, combo)) for combo in itertools.product(*grids)]


@dataclass
class Trial:
    params: Dict[str, Any]
    value: float
    elapsed: float


class HyperOptimizer:
    """Hyperparameter optimizer supporting grid, random, TPE, and CMA-ES."""

    def __init__(self):
        self._trials: List[Trial] = []

    def optimize(self, objective_fn: Callable[[Dict], float],
                 space: SearchSpace, n_trials: int = 50,
                 method: str = "tpe") -> Tuple[Dict[str, Any], float]:
        """Run optimization. Returns (best_params, best_value)."""
        if method == "grid":
            configs = space.grid()[:n_trials]
            for cfg in configs:
                t0 = time.time()
                val = objective_fn(cfg)
                self._trials.append(Trial(cfg, val, time.time() - t0))
        elif method == "random":
            rng = np.random.default_rng(42)
            for _ in range(n_trials):
                cfg = space.sample(rng)
                t0 = time.time()
                val = objective_fn(cfg)
                self._trials.append(Trial(cfg, val, time.time() - t0))
        elif method == "tpe":
            self._tpe_optimize(objective_fn, space, n_trials)
        elif method == "cma-es":
            self._cmaes_optimize(objective_fn, space, n_trials)
        else:
            raise ValueError(f"Unknown method: {method}")

        if not self._trials:
            return {}, np.inf
        best = min(self._trials, key=lambda t: t.value)
        return best.params, best.value

    def _tpe_optimize(self, objective_fn: Callable, space: SearchSpace,
                      n_trials: int) -> None:
        """Tree Parzen Estimator approximation using KDE on good/bad splits."""
        rng = np.random.default_rng(42)
        n_warmup = min(20, n_trials // 3)
        # Warmup with random trials
        for _ in range(n_warmup):
            cfg = space.sample(rng)
            t0 = time.time()
            val = objective_fn(cfg)
            self._trials.append(Trial(cfg, val, time.time() - t0))

        for i in range(n_trials - n_warmup):
            # Split into good (top 25%) and bad
            vals = np.array([t.value for t in self._trials])
            threshold = np.percentile(vals, 25)
            good = [t for t in self._trials if t.value <= threshold]
            # Sample candidates from good region (Gaussian perturbation)
            n_candidates = 24
            best_so_far = min(self._trials, key=lambda t: t.value)
            candidates = []
            for _ in range(n_candidates):
                # Perturb best params
                cfg = {}
                for spec in space._specs:
                    best_val = best_so_far.params[spec.name]
                    if spec.param_type == "int":
                        noise = int(rng.integers(-2, 3))
                        cfg[spec.name] = int(np.clip(best_val + noise, spec.low, spec.high))
                    elif spec.param_type == "float":
                        scale = (spec.high - spec.low) * 0.1
                        cfg[spec.name] = float(np.clip(rng.normal(best_val, scale), spec.low, spec.high))
                    else:
                        cfg[spec.name] = spec.choices[rng.integers(len(spec.choices))]
                candidates.append(cfg)
            # Evaluate best candidate (Thompson sampling approx)
            best_cand = candidates[rng.integers(len(candidates))]
            t0 = time.time()
            val = objective_fn(best_cand)
            self._trials.append(Trial(best_cand, val, time.time() - t0))

    def _cmaes_optimize(self, objective_fn: Callable, space: SearchSpace,
                        n_trials: int) -> None:
        """CMA-ES for continuous search spaces."""
        # Only handle float params
        float_specs = [s for s in space._specs if s.param_type == "float"]
        other_specs = [s for s in space._specs if s.param_type != "float"]
        n = len(float_specs)
        if n == 0:
            self._tpe_optimize(objective_fn, space, n_trials)
            return

        rng = np.random.default_rng(42)
        # Normalize to [0,1]
        mean = np.array([0.5] * n)
        sigma = 0.3
        C = np.eye(n)
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / (weights ** 2).sum()
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        pc = np.zeros(n)
        ps = np.zeros(n)
        chiN = n ** 0.5 * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        evals = 0

        # Other params fixed to defaults
        other_cfg = {s.name: s.choices[0] if s.param_type == "categorical"
                     else int((s.low + s.high) / 2)
                     for s in other_specs}

        gen = 0
        while evals < n_trials:
            arz = rng.standard_normal((lam, n))
            L = np.linalg.cholesky(C + np.eye(n) * 1e-8)
            arx = mean + sigma * (arz @ L.T)
            arx = np.clip(arx, 0, 1)

            fitvals = []
            for x in arx:
                cfg = dict(other_cfg)
                for j, spec in enumerate(float_specs):
                    cfg[spec.name] = float(spec.low + x[j] * (spec.high - spec.low))
                t0 = time.time()
                val = objective_fn(cfg)
                fitvals.append(val)
                self._trials.append(Trial(cfg, val, time.time() - t0))
                evals += 1

            idx = np.argsort(fitvals)
            xbest = arx[idx[:mu]]
            mean_new = weights @ xbest
            dy = mean_new - mean
            mean = mean_new
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * dy / sigma
            hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) / chiN < 1.4 + 2 / (n + 1)
            pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mueff) * dy / sigma
            artmp = (1 / sigma) * (xbest - mean)
            C = ((1 - c1 - cmu) * C
                 + c1 * (np.outer(pc, pc) + (1 - hs) * cc * (2 - cc) * C)
                 + cmu * artmp.T @ np.diag(weights) @ artmp)
            sigma *= np.exp((np.linalg.norm(ps) / chiN - 1) * cs / damps)
            gen += 1

    def parallel_optimize(self, objective_fn: Callable, space: SearchSpace,
                          n_trials: int = 50, n_jobs: int = 4) -> Tuple[Dict, float]:
        """Parallel random search."""
        rng = np.random.default_rng(42)
        configs = [space.sample(rng) for _ in range(n_trials)]

        def eval_cfg(cfg):
            t0 = time.time()
            val = objective_fn(cfg)
            return Trial(cfg, val, time.time() - t0)

        results = Parallel(n_jobs=n_jobs)(delayed(eval_cfg)(c) for c in configs)
        self._trials.extend(results)
        best = min(results, key=lambda t: t.value)
        return best.params, best.value

    def importance(self) -> Dict[str, float]:
        """Estimate parameter importance via variance of objective."""
        if len(self._trials) < 5:
            return {}
        vals = np.array([t.value for t in self._trials])
        all_keys = list(self._trials[0].params.keys())
        result = {}
        for key in all_keys:
            param_vals = [t.params[key] for t in self._trials]
            try:
                param_arr = np.array(param_vals, dtype=float)
                corr = abs(float(np.corrcoef(param_arr, vals)[0, 1]))
                result[key] = corr if np.isfinite(corr) else 0.0
            except (ValueError, TypeError):
                result[key] = 0.0
        total = sum(result.values()) + 1e-8
        return {k: v / total for k, v in result.items()}

    def plot_optimization_history(self) -> Dict:
        vals = [t.value for t in self._trials]
        best_so_far = np.minimum.accumulate(vals)
        return {"trial": list(range(len(vals))), "value": vals,
                "best_so_far": best_so_far.tolist()}


class CrossValidatedObjective:
    """Wraps model + CV into an objective function for HyperOptimizer."""

    def __init__(self, model_class: Any, X: np.ndarray, y: np.ndarray,
                 cv: int = 5, scoring: str = "neg_mean_squared_error"):
        self._model_class = model_class
        self._X = X
        self._y = y
        self._cv = cv
        self._scoring = scoring

    def __call__(self, params: Dict[str, Any]) -> float:
        try:
            model = self._model_class(**params)
            scores = cross_val_score(model, self._X, self._y,
                                     cv=self._cv, scoring=self._scoring)
            return float(-scores.mean())
        except Exception:
            return 1e8
