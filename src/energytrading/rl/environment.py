"""Reinforcement learning trading environments (gym-compatible, no gym required)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SpaceSpec:
    shape: Tuple[int, ...]
    low: float
    high: float
    dtype: str = "float32"


class EnergyMarketEnv:
    """Energy market trading environment."""

    def __init__(self, price_series: np.ndarray, capacity_mw: float = 100.0,
                 initial_position: float = 0.0, max_position: float = 500.0,
                 transaction_cost: float = 0.5, n_lags: int = 12):
        self._prices = np.asarray(price_series, float)
        self.capacity = capacity_mw
        self.max_position = max_position
        self.transaction_cost = transaction_cost
        self.n_lags = n_lags

        self._state_dim = n_lags + 4  # lags + position + hour + rolling_vol + zscore
        self.observation_space = SpaceSpec((self._state_dim,), -np.inf, np.inf)
        self.action_space = SpaceSpec((1,), -1.0, 1.0)

        self._t = 0
        self._position = initial_position
        self._pnl = 0.0
        self._done = False

    def reset(self) -> np.ndarray:
        self._t = self.n_lags
        self._position = 0.0
        self._pnl = 0.0
        self._done = False
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        t = self._t
        prices = self._prices[max(0, t - self.n_lags): t]
        if len(prices) < self.n_lags:
            prices = np.pad(prices, (self.n_lags - len(prices), 0), mode="edge")
        # Normalize prices by rolling mean
        mu = prices.mean()
        sigma = prices.std() + 1e-8
        lag_features = (prices - mu) / sigma
        roll_vol = sigma / (mu + 1e-8)
        zscore = (self._prices[t - 1] - mu) / sigma
        hour_of_day = (t % 24) / 24.0
        pos_norm = self._position / (self.max_position + 1e-8)
        return np.concatenate([lag_features, [pos_norm, hour_of_day, roll_vol, zscore]])

    def step(self, action: float | np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        if self._done:
            raise RuntimeError("Episode is done, call reset()")
        action = float(np.clip(action, -1.0, 1.0))
        delta_pos = action * self.capacity  # position change in MW
        new_position = np.clip(self._position + delta_pos, -self.max_position, self.max_position)
        actual_delta = new_position - self._position

        price_now = self._prices[self._t]
        price_prev = self._prices[self._t - 1]
        price_change = price_now - price_prev

        # PnL: position * price change - transaction cost
        reward = self._position * price_change - abs(actual_delta) * self.transaction_cost
        self._position = new_position
        self._pnl += reward
        self._t += 1

        self._done = self._t >= len(self._prices) - 1
        next_state = self._get_state() if not self._done else np.zeros(self._state_dim)
        return next_state, float(reward), self._done, {
            "position": self._position, "pnl": self._pnl, "price": price_now
        }

    def render(self) -> Dict:
        return {"t": self._t, "position": self._position,
                "pnl": self._pnl, "price": float(self._prices[min(self._t, len(self._prices)-1)])}


class BatteryStorageEnv:
    """Battery storage arbitrage RL environment."""

    def __init__(self, price_series: np.ndarray, capacity_mwh: float = 100.0,
                 max_power_mw: float = 25.0, efficiency: float = 0.90,
                 degradation_cost: float = 0.5, n_lags: int = 24):
        self._prices = np.asarray(price_series, float)
        self.capacity = capacity_mwh
        self.max_power = max_power_mw
        self.efficiency = efficiency
        self.degradation_cost = degradation_cost
        self.n_lags = n_lags
        self._state_dim = n_lags + 3  # price lags + soc + hour + forecast_error

        self.observation_space = SpaceSpec((self._state_dim,), -np.inf, np.inf)
        self.action_space = SpaceSpec((1,), -1.0, 1.0)  # -1=discharge, +1=charge

        self._soc = 0.5
        self._t = n_lags
        self._done = False
        self._total_revenue = 0.0

    def reset(self) -> np.ndarray:
        self._soc = 0.5
        self._t = self.n_lags
        self._done = False
        self._total_revenue = 0.0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        t = self._t
        prices = self._prices[max(0, t - self.n_lags): t]
        if len(prices) < self.n_lags:
            prices = np.pad(prices, (self.n_lags - len(prices), 0), mode="edge")
        mu, sigma = prices.mean(), prices.std() + 1e-8
        lag_feats = (prices - mu) / sigma
        hour_norm = (t % 24) / 24.0
        forecast_err = float(np.random.normal(0, 0.05))  # simulated forecast noise
        return np.concatenate([lag_feats, [self._soc, hour_norm, forecast_err]])

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        action = float(np.clip(action, -1.0, 1.0))
        price = self._prices[self._t]

        # action > 0: charge, action < 0: discharge
        power = action * self.max_power
        if power > 0:  # charging
            energy = min(power, self.capacity * (1 - self._soc))
            self._soc += energy * self.efficiency / self.capacity
            revenue = -price * energy / 1000  # cost of charging
        else:  # discharging
            energy = min(-power, self.capacity * self._soc)
            self._soc -= energy / (self.efficiency * self.capacity)
            revenue = price * energy / 1000  # revenue from discharge
        self._soc = np.clip(self._soc, 0.0, 1.0)
        degradation = self.degradation_cost * abs(power / self.max_power)
        reward = float(revenue - degradation)
        self._total_revenue += revenue
        self._t += 1
        self._done = self._t >= len(self._prices) - 1
        state = self._get_state() if not self._done else np.zeros(self._state_dim)
        return state, reward, self._done, {"soc": self._soc, "revenue": revenue}


class SpreadTradingEnv:
    """Spread / pairs trading environment."""

    def __init__(self, spread_series: np.ndarray, max_position: float = 1.0,
                 transaction_cost: float = 0.01, n_lags: int = 20):
        self._spread = np.asarray(spread_series, float)
        self.max_pos = max_position
        self.tc = transaction_cost
        self.n_lags = n_lags
        self._state_dim = n_lags + 3

        self.observation_space = SpaceSpec((self._state_dim,), -np.inf, np.inf)
        self.action_space = SpaceSpec((1,), -1.0, 1.0)

        self._pos = 0.0
        self._t = n_lags
        self._pnl = 0.0
        self._done = False

    def reset(self) -> np.ndarray:
        self._pos = 0.0
        self._t = self.n_lags
        self._pnl = 0.0
        self._done = False
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        t = self._t
        hist = self._spread[max(0, t - self.n_lags): t]
        if len(hist) < self.n_lags:
            hist = np.pad(hist, (self.n_lags - len(hist), 0), mode="edge")
        mu, sigma = hist.mean(), hist.std() + 1e-8
        zscore = (hist[-1] - mu) / sigma
        regime = float(abs(zscore) > 2.0)
        half_life = self.n_lags / 2  # approximation
        return np.concatenate([hist, [zscore, self._pos, regime]])

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        action = float(np.clip(action, -1.0, 1.0))
        new_pos = action * self.max_pos
        delta_pos = new_pos - self._pos
        tc = abs(delta_pos) * self.tc
        spread_return = (self._spread[self._t] - self._spread[self._t - 1]) * self._pos
        reward = float(spread_return - tc)
        self._pos = new_pos
        self._pnl += reward
        self._t += 1
        self._done = self._t >= len(self._spread) - 1
        state = self._get_state() if not self._done else np.zeros(self._state_dim)
        return state, reward, self._done, {"position": self._pos, "pnl": self._pnl}
