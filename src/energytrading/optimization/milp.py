"""MILP unit commitment and economic dispatch for thermal plants."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pulp


@dataclass
class ThermalUnit:
    name: str
    capacity_min: float  # MW
    capacity_max: float  # MW
    ramp_up: float       # MW/h
    ramp_down: float
    start_cost: float    # EUR
    no_load_cost: float  # EUR/h when online
    marginal_cost: float # EUR/MWh
    min_up_time: int     # hours
    min_down_time: int


class UnitCommitmentOptimizer:
    """Mixed-Integer Linear Programming unit commitment."""

    def __init__(self):
        self._units: List[ThermalUnit] = []

    def add_unit(self, unit: ThermalUnit) -> None:
        self._units.append(unit)

    def optimize(self, prices: np.ndarray, demand: np.ndarray,
                 horizon: int = 24) -> Dict:
        T = min(len(prices), len(demand), horizon)
        N = len(self._units)
        prices = prices[:T]
        demand = demand[:T]

        prob = pulp.LpProblem("UnitCommitment", pulp.LpMinimize)

        # Decision variables
        u = [[pulp.LpVariable(f"u_{n}_{t}", cat="Binary")
              for t in range(T)] for n in range(N)]
        p = [[pulp.LpVariable(f"p_{n}_{t}", 0)
              for t in range(T)] for n in range(N)]
        startup = [[pulp.LpVariable(f"su_{n}_{t}", cat="Binary")
                    for t in range(T)] for n in range(N)]

        # Objective: minimize total cost
        prob += pulp.lpSum(
            self._units[n].marginal_cost * p[n][t]
            + self._units[n].no_load_cost * u[n][t]
            + self._units[n].start_cost * startup[n][t]
            for n in range(N) for t in range(T)
        )

        for t in range(T):
            # Demand balance
            prob += pulp.lpSum(p[n][t] for n in range(N)) >= demand[t]
            for n in range(N):
                un = self._units[n]
                # Capacity limits
                prob += p[n][t] >= un.capacity_min * u[n][t]
                prob += p[n][t] <= un.capacity_max * u[n][t]
                # Startup detection
                if t > 0:
                    prob += startup[n][t] >= u[n][t] - u[n][t - 1]
                else:
                    prob += startup[n][t] == 0
                # Ramp constraints
                if t > 0:
                    prob += p[n][t] - p[n][t - 1] <= un.ramp_up * u[n][t - 1] + un.capacity_max * startup[n][t]
                    prob += p[n][t - 1] - p[n][t] <= un.ramp_down

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        commitment = np.array([[pulp.value(u[n][t]) or 0 for t in range(T)]
                                for n in range(N)])
        dispatch = np.array([[pulp.value(p[n][t]) or 0 for t in range(T)]
                              for n in range(N)])
        total_cost = float(pulp.value(prob.objective) or 0)

        return {
            "commitment": commitment,
            "dispatch": dispatch,
            "total_cost": total_cost,
            "status": pulp.LpStatus[prob.status],
        }

    def economic_dispatch_only(self, prices: np.ndarray, demand: np.ndarray,
                                committed_units: List[int]) -> np.ndarray:
        """LP economic dispatch for pre-committed units."""
        T = len(demand)
        N_committed = len(committed_units)
        if N_committed == 0:
            return np.zeros((0, T))

        prob = pulp.LpProblem("EconomicDispatch", pulp.LpMinimize)
        units = [self._units[i] for i in committed_units]
        p = [[pulp.LpVariable(f"p_{n}_{t}", u.capacity_min, u.capacity_max)
              for t in range(T)] for n, u in enumerate(units)]

        prob += pulp.lpSum(units[n].marginal_cost * p[n][t]
                           for n in range(N_committed) for t in range(T))
        for t in range(T):
            prob += pulp.lpSum(p[n][t] for n in range(N_committed)) >= demand[t]

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        return np.array([[pulp.value(p[n][t]) or 0 for t in range(T)]
                          for n in range(N_committed)])


class StochasticUnitCommitment:
    """Two-stage stochastic unit commitment."""

    def __init__(self, uc: UnitCommitmentOptimizer):
        self._uc = uc

    def optimize_two_stage(self, price_scenarios: np.ndarray,
                            demand_scenarios: np.ndarray,
                            n_scenarios: int = 10) -> Dict:
        """
        Stage 1: commitment decisions (here-and-now).
        Stage 2: dispatch for each scenario.
        SAA: average cost over scenarios.
        """
        n = min(n_scenarios, price_scenarios.shape[0])
        T = price_scenarios.shape[1] if price_scenarios.ndim > 1 else len(price_scenarios)
        avg_prices = price_scenarios[:n].mean(axis=0)
        avg_demand = demand_scenarios[:n].mean(axis=0)

        # Stage 1: commit based on average scenario
        stage1 = self._uc.optimize(avg_prices, avg_demand)
        commitment = stage1["commitment"]

        # Stage 2: dispatch per scenario
        total_costs = []
        for i in range(n):
            prices = price_scenarios[i] if price_scenarios.ndim > 1 else price_scenarios
            demand = demand_scenarios[i] if demand_scenarios.ndim > 1 else demand_scenarios
            committed = [n for n in range(len(self._uc._units))
                         if commitment[n].mean() > 0.5]
            if committed:
                dispatch = self._uc.economic_dispatch_only(prices, demand, committed)
                cost = sum(self._uc._units[committed[n]].marginal_cost * dispatch[n].sum()
                           for n in range(len(committed)))
                total_costs.append(cost)

        return {
            "first_stage_commitment": commitment,
            "expected_total_cost": float(np.mean(total_costs)) if total_costs else 0.0,
            "cost_std": float(np.std(total_costs)) if total_costs else 0.0,
        }
