"""Market impact models: linear, square-root, Almgren-Chriss, adverse selection."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class LinearImpactModel:
    """Linear permanent price impact: impact = eta * volume / ADV."""

    def __init__(self):
        self._eta: float = 0.1

    def fit(self, volumes: np.ndarray, price_changes: np.ndarray) -> "LinearImpactModel":
        v = np.asarray(volumes)
        dp = np.asarray(price_changes)
        self._eta = float(np.dot(v, dp) / (np.dot(v, v) + 1e-8))
        return self

    def estimate(self, volume: float, adv: float) -> float:
        return float(self._eta * volume / (adv + 1e-8))

    def temporary_impact(self, volume: float, adv: float) -> float:
        """Temporary impact decays after trade."""
        return float(0.5 * self._eta * volume / (adv + 1e-8))

    def permanent_impact(self, volume: float, adv: float) -> float:
        return float(0.5 * self._eta * volume / (adv + 1e-8))


class SquareRootImpactModel:
    """Almgren-Chriss square-root impact: impact = sigma * sqrt(volume/ADV)."""

    def __init__(self):
        self._gamma: float = 1.0  # scaling constant

    def fit(self, volumes: np.ndarray, price_changes: np.ndarray,
            sigma: float = 0.01, adv: float = 1000.0) -> "SquareRootImpactModel":
        v = np.asarray(volumes)
        dp = np.asarray(price_changes)
        predicted = sigma * np.sqrt(np.abs(v) / adv)
        self._gamma = float(np.dot(predicted, np.abs(dp)) /
                           (np.dot(predicted, predicted) + 1e-8))
        return self

    def estimate(self, volume: float, adv: float, sigma: float = 0.01) -> float:
        return float(self._gamma * sigma * np.sqrt(abs(volume) / (adv + 1e-8)))


class AlmgrenChrissOptimizer:
    """Optimal VWAP execution via Almgren-Chriss model."""

    def __init__(self, S0: float, V: float, sigma: float,
                 eta: float, gamma: float, tau: float):
        self.S0 = S0
        self.V = V          # total volume to execute
        self.sigma = sigma  # price volatility
        self.eta = eta      # temporary impact coefficient
        self.gamma = gamma  # permanent impact coefficient
        self.tau = tau      # time horizon

    def optimal_trajectory(self, n_steps: int,
                            risk_aversion: float = 0.01) -> np.ndarray:
        """Almgren-Chriss optimal execution schedule."""
        T = self.tau
        dt = T / n_steps
        lam = risk_aversion
        kappa = np.sqrt(lam * self.sigma ** 2 / (self.eta + 1e-8))
        sinh_kappa_T = np.sinh(kappa * T)
        schedule = np.zeros(n_steps)
        for j in range(n_steps):
            t_j = j * dt
            t_next = (j + 1) * dt
            # Trajectory: x(t) = V * sinh(kappa*(T-t)) / sinh(kappa*T)
            x_t = self.V * np.sinh(kappa * (T - t_j)) / (sinh_kappa_T + 1e-8)
            x_next = self.V * np.sinh(kappa * (T - t_next)) / (sinh_kappa_T + 1e-8)
            schedule[j] = x_t - x_next
        return np.maximum(schedule, 0)

    def implementation_shortfall(self, trajectory: np.ndarray,
                                  prices: np.ndarray) -> float:
        """Expected implementation shortfall vs decision price."""
        IS = 0.0
        S = self.S0
        for j, (qty, p) in enumerate(zip(trajectory, prices[:len(trajectory)])):
            perm_impact = self.gamma * qty
            temp_impact = self.eta * qty
            IS += qty * (p - self.S0) + qty * (perm_impact + temp_impact)
            S += perm_impact
        return float(IS)

    def efficient_frontier(self, risk_aversions: List[float],
                            n_steps: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Risk-adjusted cost frontier for different risk aversions."""
        costs, risks = [], []
        for lam in risk_aversions:
            schedule = self.optimal_trajectory(n_steps, lam)
            # Expected cost (permanent + temporary)
            E_cost = (0.5 * self.gamma * self.V ** 2 / self.S0
                      + self.eta * np.sum(schedule ** 2))
            # Risk (variance of execution cost)
            E_risk = self.sigma ** 2 * self.tau * self.V ** 2 / 3
            costs.append(E_cost)
            risks.append(np.sqrt(E_risk))
        return np.array(costs), np.array(risks)


class AdverseSelectionEstimator:
    """Estimates informed order flow and adverse selection costs."""

    def roll_spread(self, returns: np.ndarray) -> float:
        """Roll (1984) effective spread estimator: 2*sqrt(-Cov(r_t, r_{t-1}))."""
        r = np.asarray(returns)
        cov = float(np.cov(r[:-1], r[1:])[0, 1])
        if cov >= 0:
            return 0.0
        return float(2 * np.sqrt(-cov))

    def estimate_informed_order_fraction(self, order_flow: np.ndarray,
                                          price_changes: np.ndarray,
                                          n_bins: int = 5) -> float:
        """
        Simplified PIN (Probability of Informed Trading) proxy.
        High correlation of signed order flow with subsequent price changes → high PIN.
        """
        flow = np.asarray(order_flow)
        dp = np.asarray(price_changes)
        # Lead-lag correlation: informed traders move prices
        if len(flow) < 3:
            return 0.0
        corr = float(np.corrcoef(flow[:-1], dp[1:])[0, 1])
        return float(np.clip((corr + 1) / 2, 0, 1))

    def hasbrouck_gpdv(self, trades: pd.Series,
                        quotes: pd.Series) -> Dict[str, float]:
        """Simplified Hasbrouck (1991) information share decomposition."""
        trade_direction = np.sign(np.diff(trades))
        mid_changes = np.diff(quotes)
        if len(trade_direction) < 5:
            return {"info_share": 0.0, "trade_impact": 0.0}
        corr = float(np.corrcoef(trade_direction, mid_changes[:len(trade_direction)])[0, 1])
        return {"info_share": float(np.clip(corr, 0, 1)),
                "trade_impact": float(np.abs(np.cov(trade_direction, mid_changes[:len(trade_direction)])[0, 1]))}
