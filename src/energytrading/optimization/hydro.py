"""Hydro reservoir optimization: LP and stochastic scheduling."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pulp


@dataclass
class HydroParams:
    max_reservoir: float      # MWh equivalent
    min_reservoir: float
    max_turbine_capacity: float  # MW
    min_flow: float           # MW (environmental constraint)
    efficiency: float         # MWh / m³
    spill_cost: float = 1.0   # EUR/MWh spilled
    natural_inflow_profile: Optional[np.ndarray] = None


class HydroScheduler:
    """LP-based hydro reservoir weekly/daily scheduler."""

    def __init__(self, params: HydroParams):
        self.params = params

    def optimize_lp(self, prices: np.ndarray, inflows: np.ndarray,
                    initial_level: Optional[float] = None,
                    horizon: int = 168) -> Dict:
        p = self.params
        T = min(len(prices), len(inflows), horizon)
        prices = np.asarray(prices[:T], float)
        inflows = np.asarray(inflows[:T], float)
        init = initial_level if initial_level is not None else p.max_reservoir * 0.5

        prob = pulp.LpProblem("HydroLP", pulp.LpMaximize)
        gen = [pulp.LpVariable(f"gen_{t}", p.min_flow, p.max_turbine_capacity)
               for t in range(T)]
        spill = [pulp.LpVariable(f"spill_{t}", 0) for t in range(T)]
        res = [pulp.LpVariable(f"res_{t}", p.min_reservoir, p.max_reservoir)
               for t in range(T + 1)]

        prob += pulp.lpSum(prices[t] * gen[t] - p.spill_cost * spill[t]
                           for t in range(T))

        prob += (res[0] == init)
        for t in range(T):
            prob += (res[t + 1] == res[t] + inflows[t] - gen[t] / p.efficiency - spill[t])

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        return {
            "generation": np.array([pulp.value(gen[t]) for t in range(T)]),
            "spill": np.array([pulp.value(spill[t]) for t in range(T)]),
            "reservoir": np.array([pulp.value(res[t]) for t in range(T + 1)]),
            "revenue": float(pulp.value(prob.objective)),
        }

    def value_of_water(self, prices: np.ndarray, inflows: np.ndarray) -> np.ndarray:
        """Shadow price (dual variable) of reservoir level — approximated numerically."""
        base = self.optimize_lp(prices, inflows)["revenue"]
        delta = self.params.max_reservoir * 0.01
        shadow_prices = []
        for t in range(len(prices)):
            inflows_bumped = inflows.copy()
            inflows_bumped[t] += delta
            bumped_rev = self.optimize_lp(prices, inflows_bumped)["revenue"]
            shadow_prices.append((bumped_rev - base) / delta)
        return np.array(shadow_prices)

    def optimize_stochastic(self, price_scenarios: np.ndarray,
                             inflow_scenarios: np.ndarray,
                             n_scenarios: int = 10) -> Dict:
        """Average schedule across scenarios (sample average approximation)."""
        n = min(n_scenarios, price_scenarios.shape[0], inflow_scenarios.shape[0])
        schedules = []
        for i in range(n):
            result = self.optimize_lp(price_scenarios[i], inflow_scenarios[i])
            schedules.append(result["generation"])
        avg_schedule = np.mean(schedules, axis=0)
        return {"mean_generation": avg_schedule,
                "std_generation": np.std(schedules, axis=0)}

    def seasonal_allocation(self, annual_prices: np.ndarray,
                             monthly_inflows: np.ndarray) -> np.ndarray:
        """Monthly water allocation targets based on value."""
        T = len(monthly_inflows)
        w = np.maximum(annual_prices[:T] * monthly_inflows, 0)
        return w / (w.sum() + 1e-8) * self.params.max_reservoir


# ---------------------------------------------------------------------------
# Cascade Hydro System
# ---------------------------------------------------------------------------

@dataclass
class CascadeNode:
    name: str
    params: HydroParams
    upstream: Optional[str] = None


class CascadeHydroSystem:
    """Multi-reservoir cascade optimization."""

    def __init__(self):
        self._nodes: Dict[str, CascadeNode] = {}

    def add_reservoir(self, name: str, params: HydroParams,
                      upstream: Optional[str] = None) -> None:
        self._nodes[name] = CascadeNode(name, params, upstream)

    def _topological_order(self) -> List[str]:
        """Sort reservoirs from upstream to downstream."""
        in_degree = {n: 0 for n in self._nodes}
        children = {n: [] for n in self._nodes}
        for name, node in self._nodes.items():
            if node.upstream and node.upstream in self._nodes:
                in_degree[name] += 1
                children[node.upstream].append(name)
        order = []
        queue = [n for n, d in in_degree.items() if d == 0]
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in children[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return order

    def optimize(self, prices: np.ndarray, horizon: int = 24) -> Dict[str, Dict]:
        """Sequential optimization through cascade."""
        order = self._topological_order()
        schedules = {}
        spillover = {}
        for name in order:
            node = self._nodes[name]
            upstream_name = node.upstream
            inflows = spillover.get(upstream_name, np.zeros(horizon))
            natural = (node.params.natural_inflow_profile[:horizon]
                       if node.params.natural_inflow_profile is not None
                       else np.ones(horizon) * 10)
            total_inflows = inflows + natural
            scheduler = HydroScheduler(node.params)
            result = scheduler.optimize_lp(prices, total_inflows, horizon=horizon)
            schedules[name] = result
            spillover[name] = result["spill"]
        return schedules
