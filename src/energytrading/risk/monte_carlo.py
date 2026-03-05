"""Monte Carlo risk engine for energy portfolios."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import qmc


class CorrelatedPathSimulator:
    """Multi-asset correlated path simulator (GBM + optional Merton jumps)."""

    def simulate(self, S0: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                 corr: np.ndarray, T: float, n_steps: int, n_paths: int,
                 seed: int = 42) -> np.ndarray:
        """Returns paths of shape (n_assets, n_steps+1, n_paths)."""
        rng = np.random.default_rng(seed)
        n_assets = len(S0)
        dt = T / n_steps
        L = np.linalg.cholesky(corr + np.eye(n_assets) * 1e-8)
        paths = np.zeros((n_assets, n_steps + 1, n_paths))
        paths[:, 0, :] = S0[:, None]
        for t in range(n_steps):
            Z = rng.standard_normal((n_assets, n_paths))
            W = L @ Z  # correlated Brownians
            for i in range(n_assets):
                paths[i, t + 1, :] = paths[i, t, :] * np.exp(
                    (mu[i] - 0.5 * sigma[i] ** 2) * dt
                    + sigma[i] * np.sqrt(dt) * W[i]
                )
        return paths

    def with_jumps(self, paths: np.ndarray, lambda_vec: np.ndarray,
                   mu_j_vec: np.ndarray, sigma_j_vec: np.ndarray,
                   dt: float, seed: int = 42) -> np.ndarray:
        """Superimpose Merton jumps on existing GBM paths."""
        rng = np.random.default_rng(seed)
        n_assets, n_steps_plus1, n_paths = paths.shape
        out = paths.copy()
        for i in range(n_assets):
            k = np.exp(mu_j_vec[i] + 0.5 * sigma_j_vec[i] ** 2) - 1
            for t in range(1, n_steps_plus1):
                n_jumps = rng.poisson(lambda_vec[i] * dt, n_paths)
                for p in range(n_paths):
                    if n_jumps[p] > 0:
                        j = np.sum(rng.normal(mu_j_vec[i], sigma_j_vec[i], n_jumps[p]))
                        out[i, t, p] *= np.exp(j - lambda_vec[i] * k * dt)
        return out


class MCRiskEngine:
    """Full Monte Carlo VaR/CVaR/ES risk engine with decomposition."""

    def __init__(self, n_simulations: int = 10_000, seed: int = 42):
        self.n_simulations = n_simulations
        self.seed = seed
        self._simulator = CorrelatedPathSimulator()

    def simulate_portfolio(self, returns_history: pd.DataFrame,
                           positions: Dict[str, float],
                           horizon: int = 1) -> np.ndarray:
        """Simulate portfolio P&L over horizon days via historical bootstrap + parametric."""
        rng = np.random.default_rng(self.seed)
        assets = list(positions.keys())
        pos_vec = np.array([positions[a] for a in assets])
        rets = returns_history[assets].dropna().values if isinstance(returns_history, pd.DataFrame) else returns_history

        mu = rets.mean(axis=0)
        cov = np.cov(rets.T) + np.eye(len(assets)) * 1e-8
        L = np.linalg.cholesky(cov)
        Z = rng.standard_normal((self.n_simulations, len(assets)))
        sim_returns = mu * horizon + (Z @ L.T) * np.sqrt(horizon)
        pnl = sim_returns @ pos_vec
        return pnl

    def var(self, simulations: np.ndarray, alpha: float = 0.05) -> float:
        return float(np.percentile(simulations, alpha * 100))

    def cvar(self, simulations: np.ndarray, alpha: float = 0.05) -> float:
        threshold = self.var(simulations, alpha)
        tail = simulations[simulations <= threshold]
        return float(tail.mean()) if len(tail) > 0 else threshold

    def expected_shortfall_decomposition(self, simulations: np.ndarray,
                                          asset_simulations: np.ndarray,
                                          positions: np.ndarray,
                                          alpha: float = 0.05) -> np.ndarray:
        """Component ES per asset (Euler allocation)."""
        threshold = np.percentile(simulations, alpha * 100)
        tail_mask = simulations <= threshold
        if tail_mask.sum() == 0:
            return np.zeros(len(positions))
        component_es = asset_simulations[tail_mask].mean(axis=0) * positions
        return component_es

    def marginal_var(self, returns_history: np.ndarray,
                     positions: np.ndarray,
                     alpha: float = 0.05,
                     delta: float = 0.01) -> np.ndarray:
        """Numerical marginal VaR: dVaR/d_position_i."""
        n_assets = len(positions)
        base_pnl = returns_history @ positions
        base_var = float(np.percentile(base_pnl, alpha * 100))
        marginal = np.zeros(n_assets)
        for i in range(n_assets):
            bumped = positions.copy()
            bumped[i] *= (1 + delta)
            bumped_pnl = returns_history @ bumped
            marginal[i] = (float(np.percentile(bumped_pnl, alpha * 100)) - base_var) / (positions[i] * delta + 1e-8)
        return marginal

    def diversification_ratio(self, positions: np.ndarray,
                               covariance: np.ndarray) -> float:
        """Diversification Ratio = weighted avg vol / portfolio vol."""
        vols = np.sqrt(np.diag(covariance))
        portfolio_vol = np.sqrt(positions @ covariance @ positions)
        weighted_avg_vol = float(positions @ vols)
        return float(weighted_avg_vol / (portfolio_vol + 1e-8))

    def liquidity_adjusted_var(self, simulations: np.ndarray,
                                positions: Dict[str, float],
                                liquidity_horizons: Dict[str, int],
                                alpha: float = 0.05) -> float:
        """LVAR: scale position VaR by sqrt(liquidity horizon)."""
        base_var = abs(self.var(simulations, alpha))
        lvar = 0.0
        total_pos = sum(abs(v) for v in positions.values()) + 1e-8
        for asset, pos in positions.items():
            h = liquidity_horizons.get(asset, 1)
            lvar += abs(pos) / total_pos * base_var * np.sqrt(h)
        return float(lvar)


class VarianceReduction:
    """Variance reduction techniques for Monte Carlo simulation."""

    @staticmethod
    def antithetic_variates(Z: np.ndarray) -> np.ndarray:
        """Concatenate Z with -Z for antithetic pairs."""
        return np.concatenate([Z, -Z], axis=0)

    @staticmethod
    def control_variates(simulated_payoffs: np.ndarray,
                          control_payoffs: np.ndarray,
                          control_true: float) -> float:
        """Adjusted MC estimate using control variate."""
        cov_matrix = np.cov(simulated_payoffs, control_payoffs)
        c = -cov_matrix[0, 1] / (cov_matrix[1, 1] + 1e-8)
        adjusted = simulated_payoffs + c * (control_payoffs - control_true)
        return float(adjusted.mean())

    @staticmethod
    def importance_sampling(paths: np.ndarray,
                             mu_shift: float) -> Tuple[np.ndarray, np.ndarray]:
        """Importance sampling: shift drift by mu_shift, return (paths, IS_weights)."""
        n = paths.shape[-1] if paths.ndim > 1 else len(paths)
        log_weights = -mu_shift * paths[-1] if paths.ndim > 1 else -mu_shift * paths
        log_weights -= log_weights.max()
        weights = np.exp(log_weights)
        weights /= weights.sum()
        return paths, weights

    @staticmethod
    def quasi_monte_carlo(n_paths: int, n_dims: int,
                           seed: int = 42) -> np.ndarray:
        """Sobol quasi-random sequences for low-discrepancy sampling."""
        sampler = qmc.Sobol(d=n_dims, scramble=True, seed=seed)
        u = sampler.random(n_paths)
        from scipy.stats import norm
        return norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
