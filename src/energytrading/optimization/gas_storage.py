"""Gas storage optimization: Dynamic Programming and LP formulations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pulp


@dataclass
class GasStorageParams:
    max_volume: float         # MWh (or bcm)
    min_volume: float         # MWh cushion gas
    max_injection_rate: float # MWh/day
    max_withdrawal_rate: float
    injection_cost: float     # EUR/MWh
    withdrawal_cost: float
    cushion_gas_fraction: float = 0.10


class GasStorageDP:
    """Backward induction Dynamic Programming optimizer for gas storage."""

    def __init__(self, params: GasStorageParams, n_volume_states: int = 50):
        self.params = params
        self.n_vol = n_volume_states
        p = params
        self._vol_grid = np.linspace(p.min_volume, p.max_volume, n_vol)

    def optimize(self, prices: np.ndarray, horizon_days: int | None = None,
                 discount: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward induction. Returns (dispatch_schedule, value_matrix, optimal_path)."""
        p = self.params
        T = len(prices) if horizon_days is None else min(len(prices), horizon_days)
        prices = np.asarray(prices[:T], float)
        vol = self._vol_grid
        n = len(vol)

        # Terminal value: 0
        V = np.zeros((T + 1, n))

        # Precompute feasible actions for each state
        # Action: injection (+) or withdrawal (-)
        # Backward pass
        policy = np.zeros((T, n))  # optimal action index
        for t in range(T - 1, -1, -1):
            for s_idx, s in enumerate(vol):
                best_val = -np.inf
                best_action = 0.0
                # Discretize actions
                inj_max = min(p.max_injection_rate, p.max_volume - s)
                wdr_max = min(p.max_withdrawal_rate, s - p.min_volume)
                actions = np.linspace(-wdr_max, inj_max, 21)
                for a in actions:
                    s_next = s + a
                    if s_next < p.min_volume - 1e-6 or s_next > p.max_volume + 1e-6:
                        continue
                    cost = (p.injection_cost * max(a, 0)
                            + p.withdrawal_cost * max(-a, 0))
                    revenue = prices[t] * (-a)  # positive if withdrawing
                    immediate = revenue - cost
                    # Interpolate next state value
                    v_next = float(np.interp(s_next, vol, V[t + 1]))
                    total = immediate + discount * v_next
                    if total > best_val:
                        best_val = total
                        best_action = a
                V[t, s_idx] = best_val
                policy[t, s_idx] = best_action

        # Forward pass to extract optimal path starting from 0 volume
        s = float(p.min_volume)
        path = [s]
        schedule = []
        for t in range(T):
            action = float(np.interp(s, vol, policy[t]))
            schedule.append(action)
            s = np.clip(s + action, p.min_volume, p.max_volume)
            path.append(s)

        return np.array(schedule), V, np.array(path)

    def intrinsic_value(self, prices: np.ndarray) -> float:
        """Value from perfect-foresight dispatch using LP."""
        lp = GasStorageLP(self.params)
        schedule = lp.optimize(prices)
        p = self.params
        value = 0.0
        for t, a in enumerate(schedule):
            cost = p.injection_cost * max(a, 0) + p.withdrawal_cost * max(-a, 0)
            value += prices[t] * (-a) - cost
        return float(value)

    def extrinsic_value(self, prices: np.ndarray, n_paths: int = 500,
                         vol: float = 0.3, seed: int = 42) -> float:
        """Extrinsic (option) value via Monte Carlo re-optimization."""
        rng = np.random.default_rng(seed)
        intrinsic = self.intrinsic_value(prices)
        scenario_values = []
        for _ in range(n_paths):
            noise = rng.lognormal(0, vol * np.sqrt(1 / len(prices)), len(prices))
            scenario_prices = prices * noise
            val = self.intrinsic_value(scenario_prices)
            scenario_values.append(val)
        return float(np.maximum(np.mean(scenario_values) - intrinsic, 0))


class GasStorageLP:
    """LP formulation for deterministic gas storage dispatch."""

    def __init__(self, params: GasStorageParams):
        self.params = params

    def optimize(self, prices: np.ndarray) -> np.ndarray:
        p = self.params
        T = len(prices)
        prob = pulp.LpProblem("GasStorageLP", pulp.LpMaximize)

        inject = [pulp.LpVariable(f"inj_{t}", 0, p.max_injection_rate) for t in range(T)]
        withdraw = [pulp.LpVariable(f"wdr_{t}", 0, p.max_withdrawal_rate) for t in range(T)]
        soc = [pulp.LpVariable(f"soc_{t}", p.min_volume, p.max_volume) for t in range(T + 1)]

        prob += pulp.lpSum(
            prices[t] * withdraw[t] - prices[t] * inject[t]
            - p.withdrawal_cost * withdraw[t] - p.injection_cost * inject[t]
            for t in range(T)
        )

        prob += (soc[0] == p.min_volume)
        for t in range(T):
            prob += (soc[t + 1] == soc[t] + inject[t] - withdraw[t])
        prob += (soc[T] >= p.min_volume)

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        return np.array([pulp.value(inject[t]) - pulp.value(withdraw[t]) for t in range(T)])
