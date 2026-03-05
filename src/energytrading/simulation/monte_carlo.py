"""Comprehensive Monte Carlo simulation engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc as _qmc


@dataclass
class SimulationConfig:
    n_paths: int = 1000
    n_steps: int = 252
    dt: float = 1 / 252
    seed: int = 42
    use_antithetic: bool = True
    use_qmc: bool = False


class MonteCarloEngine:
    """Unified MC engine for all stochastic models."""

    def _get_normals(self, cfg: SimulationConfig,
                     n_dims: int = 1) -> np.ndarray:
        """Draw standard normals, optionally with variance reduction."""
        n = cfg.n_paths // 2 if cfg.use_antithetic else cfg.n_paths
        if cfg.use_qmc:
            sampler = _qmc.Sobol(d=n_dims, scramble=True, seed=cfg.seed)
            u = sampler.random(n * cfg.n_steps)
            Z = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
            Z = Z.reshape(n, cfg.n_steps, n_dims)
        else:
            rng = np.random.default_rng(cfg.seed)
            Z = rng.standard_normal((n, cfg.n_steps, n_dims))
        if cfg.use_antithetic:
            Z = np.concatenate([Z, -Z], axis=0)
        return Z

    def simulate_gbm(self, S0: float, mu: float, sigma: float,
                     cfg: SimulationConfig) -> np.ndarray:
        """Geometric Brownian Motion. Returns (n_steps+1, n_paths)."""
        Z = self._get_normals(cfg)[:, :, 0].T  # (n_steps, n_paths)
        paths = np.zeros((cfg.n_steps + 1, cfg.n_paths))
        paths[0] = S0
        for t in range(cfg.n_steps):
            paths[t + 1] = paths[t] * np.exp(
                (mu - 0.5 * sigma ** 2) * cfg.dt
                + sigma * np.sqrt(cfg.dt) * Z[t]
            )
        return paths

    def simulate_ou(self, x0: float, theta: float, mu: float, sigma: float,
                    cfg: SimulationConfig) -> np.ndarray:
        """Ornstein-Uhlenbeck mean-reverting process."""
        Z = self._get_normals(cfg)[:, :, 0].T
        paths = np.zeros((cfg.n_steps + 1, cfg.n_paths))
        paths[0] = x0
        a = np.exp(-theta * cfg.dt)
        var = sigma ** 2 * (1 - np.exp(-2 * theta * cfg.dt)) / (2 * theta)
        std = np.sqrt(max(var, 0))
        for t in range(cfg.n_steps):
            paths[t + 1] = mu + (paths[t] - mu) * a + std * Z[t]
        return paths

    def simulate_merton_jump(self, S0: float, mu: float, sigma: float,
                              lambda_j: float, mu_j: float, sigma_j: float,
                              cfg: SimulationConfig) -> np.ndarray:
        """Merton jump-diffusion process."""
        rng = np.random.default_rng(cfg.seed)
        Z = self._get_normals(cfg)[:, :, 0].T
        paths = np.zeros((cfg.n_steps + 1, cfg.n_paths))
        paths[0] = S0
        k = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1
        drift = (mu - 0.5 * sigma ** 2 - lambda_j * k) * cfg.dt
        for t in range(cfg.n_steps):
            n_jumps = rng.poisson(lambda_j * cfg.dt, cfg.n_paths)
            jumps = np.array([
                np.sum(rng.normal(mu_j, sigma_j, nj)) if nj > 0 else 0.0
                for nj in n_jumps
            ])
            paths[t + 1] = paths[t] * np.exp(
                drift + sigma * np.sqrt(cfg.dt) * Z[t] + jumps
            )
        return paths

    def simulate_heston(self, S0: float, v0: float, mu: float,
                         kappa: float, theta: float, sigma: float, rho: float,
                         cfg: SimulationConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Heston stochastic volatility. Returns (price_paths, var_paths)."""
        Z = self._get_normals(cfg, n_dims=2)  # (n_paths, n_steps, 2)
        zv = Z[:, :, 0].T
        zs = (rho * Z[:, :, 0] + np.sqrt(1 - rho ** 2) * Z[:, :, 1]).T
        S = np.zeros((cfg.n_steps + 1, cfg.n_paths))
        V = np.zeros((cfg.n_steps + 1, cfg.n_paths))
        S[0], V[0] = S0, v0
        for t in range(cfg.n_steps):
            V_pos = np.maximum(V[t], 0.0)
            V[t + 1] = np.maximum(
                V_pos + kappa * (theta - V_pos) * cfg.dt
                + sigma * np.sqrt(V_pos * cfg.dt) * zv[t], 0.0)
            S[t + 1] = S[t] * np.exp(
                (mu - 0.5 * V_pos) * cfg.dt
                + np.sqrt(V_pos * cfg.dt) * zs[t])
        return S, V

    def simulate_correlated(self, S0_vec: np.ndarray,
                             params_list: List[Dict],
                             corr_matrix: np.ndarray,
                             cfg: SimulationConfig) -> np.ndarray:
        """Correlated multi-asset GBM. Returns (n_assets, n_steps+1, n_paths)."""
        n_assets = len(S0_vec)
        rng = np.random.default_rng(cfg.seed)
        L = np.linalg.cholesky(corr_matrix + np.eye(n_assets) * 1e-8)
        Z_raw = rng.standard_normal((cfg.n_steps, n_assets, cfg.n_paths))
        Z_corr = np.einsum("ij,tjp->tip", L, Z_raw)  # correlated
        paths = np.zeros((n_assets, cfg.n_steps + 1, cfg.n_paths))
        paths[:, 0, :] = S0_vec[:, None]
        for t in range(cfg.n_steps):
            for i, p in enumerate(params_list):
                mu, sigma = p.get("mu", 0.0), p.get("sigma", 0.2)
                paths[i, t + 1, :] = paths[i, t, :] * np.exp(
                    (mu - 0.5 * sigma ** 2) * cfg.dt
                    + sigma * np.sqrt(cfg.dt) * Z_corr[t, i, :])
        return paths

    def price_option_mc(self, paths: np.ndarray, K: float, r: float, T: float,
                         option_type: str = "call") -> Tuple[float, float, Tuple]:
        """Price European option from simulated paths."""
        final = paths[-1] if paths.ndim == 2 else paths[-1, :]
        if option_type == "call":
            payoffs = np.maximum(final - K, 0)
        else:
            payoffs = np.maximum(K - final, 0)
        discount = np.exp(-r * T)
        price = float(discount * payoffs.mean())
        se = float(discount * payoffs.std() / np.sqrt(len(payoffs)))
        ci = (price - 1.96 * se, price + 1.96 * se)
        return price, se, ci

    def compute_path_statistics(self, paths: np.ndarray) -> Dict:
        """Compute percentiles and moments across paths."""
        final = paths[-1] if paths.ndim == 2 else paths[-1, :]
        return {
            "mean": float(final.mean()),
            "std": float(final.std()),
            "p5": float(np.percentile(final, 5)),
            "p25": float(np.percentile(final, 25)),
            "p50": float(np.percentile(final, 50)),
            "p75": float(np.percentile(final, 75)),
            "p95": float(np.percentile(final, 95)),
            "skew": float(((final - final.mean()) ** 3).mean() / (final.std() ** 3 + 1e-8)),
            "kurt": float(((final - final.mean()) ** 4).mean() / (final.std() ** 4 + 1e-8) - 3),
        }
