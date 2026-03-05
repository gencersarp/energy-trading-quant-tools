"""Portfolio optimization: MVO, risk parity, CVaR, Kelly, dynamic allocation."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import pulp


# ---------------------------------------------------------------------------
# Mean-Variance Optimizer (Markowitz)
# ---------------------------------------------------------------------------

class MeanVarianceOptimizer:
    """Markowitz mean-variance portfolio optimization."""

    def efficient_frontier(self, mu: np.ndarray, Sigma: np.ndarray,
                            n_points: int = 100) -> Tuple[List, np.ndarray, np.ndarray]:
        mu_min = float(self.min_variance(mu, Sigma) @ mu)
        mu_max = float(np.max(mu))
        targets = np.linspace(mu_min, mu_max, n_points)
        weights_list, rets, vols = [], [], []
        for target in targets:
            try:
                w = self.target_return(mu, Sigma, target)
                weights_list.append(w)
                rets.append(float(w @ mu))
                vols.append(float(np.sqrt(w @ Sigma @ w)))
            except Exception:
                pass
        return weights_list, np.array(rets), np.array(vols)

    def max_sharpe(self, mu: np.ndarray, Sigma: np.ndarray,
                   rf: float = 0.0) -> np.ndarray:
        n = len(mu)
        excess = mu - rf

        def neg_sharpe(w):
            w = np.array(w)
            port_ret = float(w @ excess)
            port_vol = float(np.sqrt(w @ Sigma @ w + 1e-10))
            return -port_ret / port_vol

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0.0, 1.0)] * n
        x0 = np.ones(n) / n
        result = minimize(neg_sharpe, x0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"ftol": 1e-9})
        return result.x / (result.x.sum() + 1e-10)

    def min_variance(self, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        n = len(mu)
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0.0, 1.0)] * n

        def portfolio_var(w):
            return float(np.array(w) @ Sigma @ np.array(w))

        result = minimize(portfolio_var, np.ones(n) / n, method="SLSQP",
                          bounds=bounds, constraints=constraints)
        return result.x

    def target_return(self, mu: np.ndarray, Sigma: np.ndarray,
                      target: float) -> np.ndarray:
        n = len(mu)
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w: float(np.array(w) @ mu) - target},
        ]
        bounds = [(0.0, 1.0)] * n
        result = minimize(lambda w: float(np.array(w) @ Sigma @ np.array(w)),
                          np.ones(n) / n, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"ftol": 1e-9})
        return result.x


# ---------------------------------------------------------------------------
# Risk Parity
# ---------------------------------------------------------------------------

class RiskParityOptimizer:
    """Equal Risk Contribution portfolio optimization."""

    def fit(self, Sigma: np.ndarray, max_iter: int = 500,
            tol: float = 1e-8) -> np.ndarray:
        """Cyclical coordinate descent for ERC."""
        n = Sigma.shape[0]
        w = np.ones(n) / n
        for _ in range(max_iter):
            w_prev = w.copy()
            for i in range(n):
                # Closed-form update (Bruder-Roncalli)
                a = float(Sigma[i, i])
                b = float(2 * (Sigma[:, i] @ w) - Sigma[i, i] * w[i])
                c = float(-1.0 / n)
                discriminant = b ** 2 - 4 * a * c
                if discriminant < 0:
                    discriminant = 0.0
                w[i] = (-b + np.sqrt(discriminant)) / (2 * a)
            w = np.maximum(w, 1e-8)
            w /= w.sum()
            if np.max(np.abs(w - w_prev)) < tol:
                break
        return w

    def risk_contributions(self, w: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        """Marginal risk contribution per asset."""
        port_var = float(w @ Sigma @ w)
        return (Sigma @ w * w) / (port_var + 1e-10)


# ---------------------------------------------------------------------------
# CVaR Portfolio Optimization (Rockafellar-Uryasev)
# ---------------------------------------------------------------------------

class CVaROptimizer:
    """CVaR portfolio optimization via linear programming."""

    def fit(self, returns_scenarios: np.ndarray, alpha: float = 0.05,
            allow_short: bool = False) -> np.ndarray:
        """
        Minimize CVaR subject to sum(w) = 1.
        returns_scenarios: (n_scenarios, n_assets)
        """
        S, n = returns_scenarios.shape
        losses = -returns_scenarios  # minimize losses

        prob = pulp.LpProblem("CVaR_Opt", pulp.LpMinimize)
        w = [pulp.LpVariable(f"w_{i}", lowBound=0 if not allow_short else None,
                             upBound=1) for i in range(n)]
        z_vars = [pulp.LpVariable(f"z_{s}", lowBound=0) for s in range(S)]
        gamma = pulp.LpVariable("gamma")

        # Objective: gamma + 1/(alpha*S) * sum(z_s)
        prob += gamma + pulp.lpSum(z_vars) / (alpha * S)

        # Constraints: z_s >= sum_i(loss_i,s * w_i) - gamma
        for s in range(S):
            prob += z_vars[s] >= pulp.lpSum(losses[s, i] * w[i] for i in range(n)) - gamma

        # Budget constraint
        prob += pulp.lpSum(w) == 1

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        weights = np.array([pulp.value(w[i]) for i in range(n)])
        weights = np.maximum(weights, 0)
        total = weights.sum()
        return weights / total if total > 0 else np.ones(n) / n


# ---------------------------------------------------------------------------
# Kelly Criterion
# ---------------------------------------------------------------------------

class KellyCriterion:
    """Kelly optimal position sizing."""

    def full_kelly(self, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        """f* = Sigma^{-1} * mu (unconstrained Kelly)."""
        return np.linalg.solve(Sigma + np.eye(len(mu)) * 1e-8, mu)

    def fractional_kelly(self, mu: np.ndarray, Sigma: np.ndarray,
                          fraction: float = 0.5) -> np.ndarray:
        return fraction * self.full_kelly(mu, Sigma)

    def dynamic_leverage(self, mu: np.ndarray, Sigma: np.ndarray,
                          target_vol: float = 0.10) -> float:
        """Scale positions to hit target annual volatility."""
        kelly = self.full_kelly(mu, Sigma)
        kelly = np.maximum(kelly, 0)
        kelly /= kelly.sum() + 1e-8
        port_vol = float(np.sqrt(kelly @ Sigma @ kelly))
        return float(target_vol / (port_vol + 1e-8))


# ---------------------------------------------------------------------------
# Dynamic / Regime-Conditioned Allocation
# ---------------------------------------------------------------------------

class DynamicAllocation:
    """Regime-conditioned portfolio allocation."""

    def __init__(self):
        self._regime_weights: Dict[int, np.ndarray] = {}
        self._mvo = MeanVarianceOptimizer()

    def fit(self, returns: pd.DataFrame, regimes: pd.Series) -> "DynamicAllocation":
        for regime in regimes.unique():
            mask = regimes == regime
            r = returns[mask].dropna()
            if len(r) < 10:
                continue
            mu = r.mean().values
            Sigma = r.cov().values + np.eye(len(mu)) * 1e-6
            try:
                w = self._mvo.max_sharpe(mu, Sigma)
            except Exception:
                w = np.ones(len(mu)) / len(mu)
            self._regime_weights[int(regime)] = w
        return self

    def allocate(self, current_regime: int) -> Optional[np.ndarray]:
        return self._regime_weights.get(current_regime)
