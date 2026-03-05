"""Cross-commodity hedging for energy portfolios."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


class MinimumVarianceHedge:
    """OLS hedge ratio for spot-futures hedging."""

    def __init__(self):
        self._beta: float = 0.0
        self._effectiveness: float = 0.0

    def fit(self, spot_returns: np.ndarray,
            futures_returns: np.ndarray) -> "MinimumVarianceHedge":
        s = np.asarray(spot_returns, float)
        f = np.asarray(futures_returns, float)
        X = np.column_stack([f, np.ones(len(f))])
        coefs, *_ = np.linalg.lstsq(X, s, rcond=None)
        self._beta = float(coefs[0])
        ss_res = float(np.sum((s - X @ coefs) ** 2))
        ss_tot = float(np.sum((s - s.mean()) ** 2))
        self._effectiveness = float(1 - ss_res / (ss_tot + 1e-8))
        return self

    @property
    def hedge_ratio(self) -> float:
        return self._beta

    @property
    def effectiveness(self) -> float:
        return self._effectiveness

    def hedge(self, position_size: float) -> float:
        return -self._beta * position_size


class CrossCommodityHedger:
    """Multi-asset cross-commodity hedging (power, gas, carbon, coal)."""

    def __init__(self):
        self._hedge_ratios: Dict[str, float] = {}

    def compute_hedge_ratios(self, returns_df: pd.DataFrame,
                              target_asset: str) -> Dict[str, float]:
        y = returns_df[target_asset].values
        hedge_assets = [c for c in returns_df.columns if c != target_asset]
        X = np.column_stack([returns_df[a].values for a in hedge_assets] +
                            [np.ones(len(y))])
        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        self._hedge_ratios = {a: float(coefs[i]) for i, a in enumerate(hedge_assets)}
        return self._hedge_ratios

    def compute_portfolio_hedge(self, positions_dict: Dict[str, float],
                                 returns_df: pd.DataFrame) -> Dict[str, float]:
        hedged = dict(positions_dict)
        for target, pos in positions_dict.items():
            if target not in returns_df.columns:
                continue
            ratios = self.compute_hedge_ratios(returns_df, target)
            for hedge_asset, ratio in ratios.items():
                hedged[hedge_asset] = hedged.get(hedge_asset, 0.0) - pos * ratio
        return hedged


class SparkSpreadHedger:
    """Hedge gas-fired power plant (spark spread) exposure."""

    def __init__(self, efficiency: float = 0.50):
        self.efficiency = efficiency

    def compute_delta(self, power_pos: float, gas_pos: float) -> float:
        """Net exposure in power equivalent MW."""
        return power_pos - gas_pos / self.efficiency

    def hedge_with_options(self, power_price: float, gas_price: float,
                            power_vol: float, gas_vol: float,
                            rho: float = 0.7) -> Dict[str, float]:
        """Approximate spark spread option notional via Margrabe formula."""
        S1, S2 = power_price, gas_price / self.efficiency
        sigma = np.sqrt(power_vol ** 2 + gas_vol ** 2 - 2 * rho * power_vol * gas_vol)
        d1 = (np.log(S1 / S2) + 0.5 * sigma ** 2) / (sigma + 1e-8)
        from scipy.stats import norm
        delta_power = float(norm.cdf(d1))
        delta_gas = -float(norm.cdf(d1 - sigma))
        return {"power_delta": delta_power, "gas_delta": delta_gas,
                "spread_vol": float(sigma)}


class WeatherHedger:
    """Temperature-based energy demand hedging via degree days."""

    def __init__(self):
        self._hdd_beta: float = 0.0
        self._cdd_beta: float = 0.0

    def compute_degree_day_exposure(self, revenue: np.ndarray,
                                     hdd: np.ndarray,
                                     cdd: np.ndarray) -> Tuple[float, float]:
        X = np.column_stack([hdd, cdd, np.ones(len(revenue))])
        coefs, *_ = np.linalg.lstsq(X, np.asarray(revenue), rcond=None)
        self._hdd_beta, self._cdd_beta = float(coefs[0]), float(coefs[1])
        return self._hdd_beta, self._cdd_beta

    def optimal_weather_derivative_position(self, exposure: float,
                                             hdd_beta: float,
                                             cdd_beta: float) -> Dict[str, float]:
        return {
            "hdd_swap_notional": -exposure * hdd_beta,
            "cdd_swap_notional": -exposure * cdd_beta,
        }


class DynamicHedgeRebalancer:
    """Tracks and rebalances hedges as correlations shift."""

    def fit(self, spot: pd.Series, hedge: pd.Series,
            lookback: int = 60) -> pd.Series:
        """Rolling OLS hedge ratio."""
        ratios = pd.Series(index=spot.index, dtype=float)
        for i in range(lookback, len(spot)):
            y = spot.iloc[i - lookback: i].values
            x = hedge.iloc[i - lookback: i].values
            X = np.column_stack([x, np.ones(lookback)])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            ratios.iloc[i] = float(coef[0])
        return ratios.fillna(method="bfill")

    def rebalance_signal(self, current_ratio: float, target_ratio: float,
                          tolerance: float = 0.05) -> bool:
        return abs(current_ratio - target_ratio) > tolerance
