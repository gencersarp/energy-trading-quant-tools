"""TWAP, VWAP, POV, and IS execution algorithms for backtesting."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ExecutionResult:
    algorithm: str
    total_qty: float
    avg_execution_price: float
    benchmark_price: float
    slippage_bps: float
    implementation_shortfall: float
    schedule: pd.Series        # executed qty per slice
    prices: pd.Series          # execution price per slice
    timing: pd.DatetimeIndex


class TWAPExecutor:
    """
    Time-Weighted Average Price execution.
    Splits a total order equally across N time slices.
    """

    def __init__(self, n_slices: int = 10, add_noise: bool = True,
                 noise_bps: float = 2.0):
        self.n_slices = n_slices
        self.add_noise = add_noise
        self.noise_bps = noise_bps

    def execute(self, total_qty: float, prices: pd.Series,
                decision_price: Optional[float] = None) -> ExecutionResult:
        """
        prices: pd.Series of market prices (length >= n_slices)
        decision_price: VWAP/arrival price benchmark; defaults to first price.
        """
        n = min(self.n_slices, len(prices))
        indices = np.linspace(0, len(prices) - 1, n, dtype=int)
        slice_prices = prices.iloc[indices].values
        slice_qty = np.full(n, total_qty / n)

        if self.add_noise:
            noise = slice_prices * self.noise_bps / 10_000 * np.random.randn(n)
            slice_prices = slice_prices + noise

        benchmark = float(decision_price or prices.iloc[0])
        avg_price = float(np.dot(slice_qty, slice_prices) /
                          (np.sum(slice_qty) + 1e-8))
        slippage_bps = (avg_price - benchmark) / (benchmark + 1e-8) * 10_000
        is_cost = (avg_price - benchmark) * total_qty

        return ExecutionResult(
            algorithm="TWAP",
            total_qty=total_qty,
            avg_execution_price=avg_price,
            benchmark_price=benchmark,
            slippage_bps=float(slippage_bps),
            implementation_shortfall=float(is_cost),
            schedule=pd.Series(slice_qty, index=prices.iloc[indices].index),
            prices=pd.Series(slice_prices, index=prices.iloc[indices].index),
            timing=prices.iloc[indices].index,
        )


class VWAPExecutor:
    """
    Volume-Weighted Average Price execution.
    Uses a volume profile to weight execution across time slices.
    """

    def __init__(self, volume_profile: Optional[np.ndarray] = None,
                 n_slices: int = 10, add_noise: bool = True,
                 noise_bps: float = 2.0):
        self.volume_profile = volume_profile
        self.n_slices = n_slices
        self.add_noise = add_noise
        self.noise_bps = noise_bps

    def _default_profile(self, n: int) -> np.ndarray:
        """U-shaped intraday volume profile (higher at open/close)."""
        t = np.linspace(0, np.pi, n)
        profile = 1.5 - np.cos(2 * t)  # U-shape
        return profile / profile.sum()

    def execute(self, total_qty: float, prices: pd.Series,
                volumes: Optional[pd.Series] = None,
                decision_price: Optional[float] = None) -> ExecutionResult:
        n = min(self.n_slices, len(prices))
        indices = np.linspace(0, len(prices) - 1, n, dtype=int)
        slice_prices = prices.iloc[indices].values

        if volumes is not None and len(volumes) >= len(indices):
            raw_vols = volumes.iloc[indices].values.astype(float)
            profile = raw_vols / (raw_vols.sum() + 1e-8)
        elif self.volume_profile is not None:
            profile = np.interp(np.linspace(0, 1, n),
                                np.linspace(0, 1, len(self.volume_profile)),
                                self.volume_profile)
            profile = profile / (profile.sum() + 1e-8)
        else:
            profile = self._default_profile(n)

        slice_qty = total_qty * profile

        if self.add_noise:
            noise = slice_prices * self.noise_bps / 10_000 * np.random.randn(n)
            slice_prices = slice_prices + noise

        benchmark = float(decision_price or
                          float(np.dot(profile, slice_prices)))
        avg_price = float(np.dot(slice_qty, slice_prices) /
                          (np.sum(slice_qty) + 1e-8))
        slippage_bps = (avg_price - benchmark) / (benchmark + 1e-8) * 10_000
        is_cost = (avg_price - benchmark) * total_qty

        return ExecutionResult(
            algorithm="VWAP",
            total_qty=total_qty,
            avg_execution_price=avg_price,
            benchmark_price=benchmark,
            slippage_bps=float(slippage_bps),
            implementation_shortfall=float(is_cost),
            schedule=pd.Series(slice_qty, index=prices.iloc[indices].index),
            prices=pd.Series(slice_prices, index=prices.iloc[indices].index),
            timing=prices.iloc[indices].index,
        )


class POVExecutor:
    """
    Percentage of Volume (POV / participation rate) execution.
    Trades at a fixed fraction of market volume in each period.
    """

    def __init__(self, participation_rate: float = 0.05,
                 max_slippage_bps: float = 10.0):
        self.participation_rate = participation_rate
        self.max_slippage_bps = max_slippage_bps

    def execute(self, total_qty: float, prices: pd.Series,
                volumes: pd.Series,
                decision_price: Optional[float] = None) -> ExecutionResult:
        remaining = total_qty
        exec_qtys: List[float] = []
        exec_prices: List[float] = []
        exec_idx: List = []

        for i in range(len(prices)):
            if remaining <= 0:
                break
            market_vol = float(volumes.iloc[i])
            slice_qty = min(market_vol * self.participation_rate, remaining)
            impact = prices.iloc[i] * slice_qty / (market_vol + 1) * 0.1
            exec_price = float(prices.iloc[i]) + float(impact)

            exec_qtys.append(slice_qty)
            exec_prices.append(exec_price)
            exec_idx.append(prices.index[i])
            remaining -= slice_qty

        if not exec_qtys:
            return ExecutionResult("POV", 0, 0, 0, 0, 0,
                                   pd.Series(dtype=float),
                                   pd.Series(dtype=float),
                                   pd.DatetimeIndex([]))

        eq = np.array(exec_qtys)
        ep = np.array(exec_prices)
        benchmark = float(decision_price or prices.iloc[0])
        avg_price = float(np.dot(eq, ep) / (eq.sum() + 1e-8))
        slippage_bps = (avg_price - benchmark) / (benchmark + 1e-8) * 10_000
        is_cost = (avg_price - benchmark) * total_qty

        idx = pd.DatetimeIndex(exec_idx)
        return ExecutionResult(
            algorithm="POV",
            total_qty=float(eq.sum()),
            avg_execution_price=avg_price,
            benchmark_price=benchmark,
            slippage_bps=float(slippage_bps),
            implementation_shortfall=float(is_cost),
            schedule=pd.Series(eq, index=idx),
            prices=pd.Series(ep, index=idx),
            timing=idx,
        )


class IS_Optimizer:
    """
    Implementation Shortfall (Arrival Price) optimal execution.
    Trades off market impact vs timing risk using Almgren-Chriss framework.
    """

    def __init__(self, eta: float = 0.1, gamma: float = 0.05,
                 sigma: float = 0.01, risk_aversion: float = 0.01):
        self.eta = eta           # temporary impact
        self.gamma = gamma       # permanent impact
        self.sigma = sigma       # price volatility
        self.risk_aversion = risk_aversion

    def optimal_schedule(self, total_qty: float,
                          n_slices: int = 10,
                          tau: float = 1.0) -> np.ndarray:
        """Compute the Almgren-Chriss optimal execution schedule."""
        kappa = np.sqrt(self.risk_aversion * self.sigma ** 2 /
                        (self.eta + 1e-8))
        dt = tau / n_slices
        T = tau
        schedule = np.zeros(n_slices)
        sinh_kT = np.sinh(kappa * T)

        for j in range(n_slices):
            t0 = j * dt
            t1 = (j + 1) * dt
            x0 = total_qty * np.sinh(kappa * (T - t0)) / (sinh_kT + 1e-8)
            x1 = total_qty * np.sinh(kappa * (T - t1)) / (sinh_kT + 1e-8)
            schedule[j] = max(x0 - x1, 0)

        # Normalize to ensure total qty is exactly met
        s = schedule.sum()
        if s > 0:
            schedule *= total_qty / s
        return schedule

    def execute(self, total_qty: float, prices: pd.Series,
                n_slices: Optional[int] = None,
                decision_price: Optional[float] = None) -> ExecutionResult:
        n = n_slices or min(10, len(prices))
        tau = n / max(len(prices), 1)
        schedule = self.optimal_schedule(total_qty, n, tau)

        indices = np.linspace(0, len(prices) - 1, n, dtype=int)
        slice_prices = prices.iloc[indices].values.copy()

        # Apply temporary price impact
        for j in range(n):
            impact = self.eta * schedule[j]
            slice_prices[j] += impact

        benchmark = float(decision_price or prices.iloc[0])
        avg_price = float(np.dot(schedule, slice_prices) /
                          (np.sum(schedule) + 1e-8))
        slippage_bps = (avg_price - benchmark) / (benchmark + 1e-8) * 10_000
        is_cost = (avg_price - benchmark) * total_qty

        idx = prices.iloc[indices].index
        return ExecutionResult(
            algorithm="IS",
            total_qty=float(np.sum(schedule)),
            avg_execution_price=avg_price,
            benchmark_price=benchmark,
            slippage_bps=float(slippage_bps),
            implementation_shortfall=float(is_cost),
            schedule=pd.Series(schedule, index=idx),
            prices=pd.Series(slice_prices, index=idx),
            timing=idx,
        )


class SmartOrderRouter:
    """
    Route orders across multiple venues/exchanges minimizing total cost.
    Considers venue fees, liquidity, and impact.
    """

    def __init__(self, venue_fees: Dict[str, float] = None,
                 venue_impact: Dict[str, float] = None):
        self.venue_fees = venue_fees or {}
        self.venue_impact = venue_impact or {}

    def route(self, total_qty: float,
              venue_prices: Dict[str, float],
              venue_liquidity: Dict[str, float]) -> Dict[str, float]:
        """
        Allocate order across venues to minimize cost.
        Returns {venue_name: qty_to_send}.
        """
        venues = list(venue_prices.keys())
        if not venues:
            return {}

        # Adjusted cost = price + fee + impact
        adjusted_costs = {}
        for v in venues:
            price = venue_prices[v]
            fee = self.venue_fees.get(v, 0.0005)  # 5bps default
            impact_coeff = self.venue_impact.get(v, 0.001)
            liq = venue_liquidity.get(v, 1.0)
            # Simple proxy: lower liquidity → higher impact
            impact = impact_coeff / (liq + 1e-8)
            adjusted_costs[v] = price * (1 + fee) + impact

        # Sort venues by adjusted cost (ascending)
        sorted_venues = sorted(adjusted_costs.items(), key=lambda x: x[1])
        allocation: Dict[str, float] = {}
        remaining = total_qty

        for venue, _ in sorted_venues:
            if remaining <= 0:
                break
            available = venue_liquidity.get(venue, float("inf"))
            sent = min(available, remaining)
            allocation[venue] = sent
            remaining -= sent

        if remaining > 0:
            # Route any residual to the best venue
            best_venue = sorted_venues[0][0]
            allocation[best_venue] = allocation.get(best_venue, 0) + remaining

        return allocation

    def expected_cost(self, allocation: Dict[str, float],
                      venue_prices: Dict[str, float]) -> float:
        total = sum(qty * venue_prices.get(v, 0)
                    for v, qty in allocation.items())
        fees = sum(qty * self.venue_fees.get(v, 0.0005)
                   for v, qty in allocation.items())
        return float(total + fees)
