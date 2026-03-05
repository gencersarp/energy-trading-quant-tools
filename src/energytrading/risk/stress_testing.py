"""Stress testing framework for energy portfolios."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class StressScenario:
    name: str
    description: str
    shocks: Dict[str, float]           # asset -> % shock
    correlation_adjustment: float = 0.0
    probability: float = 0.01


@dataclass
class StressResult:
    scenario_name: str
    portfolio_pnl: float
    asset_pnl: Dict[str, float]
    worst_asset: str
    probability: float
    severity: str  # 'mild' | 'severe' | 'extreme'


class StressTestSuite:
    """Run historical, hypothetical, and reverse stress tests."""

    def __init__(self):
        self._scenarios: List[StressScenario] = []

    def add_scenario(self, scenario: StressScenario) -> None:
        self._scenarios.append(scenario)

    def add_historical_scenario(self, name: str, start_date: str,
                                 end_date: str,
                                 returns_data: pd.DataFrame) -> None:
        try:
            segment = returns_data.loc[start_date:end_date]
            cumulative = (1 + segment).prod() - 1
            shocks = {col: float(cumulative[col]) for col in cumulative.index}
        except Exception:
            shocks = {}
        self._scenarios.append(StressScenario(
            name=name,
            description=f"Historical: {start_date} to {end_date}",
            shocks=shocks,
            probability=0.05,
        ))

    def run_scenario(self, scenario: StressScenario,
                     portfolio_positions: Dict[str, float]) -> StressResult:
        asset_pnl = {}
        total_pnl = 0.0
        for asset, position in portfolio_positions.items():
            shock = scenario.shocks.get(asset, 0.0)
            pnl = position * shock
            asset_pnl[asset] = pnl
            total_pnl += pnl

        worst = min(asset_pnl, key=asset_pnl.get) if asset_pnl else ""
        severity = "mild" if total_pnl > -0.05 else ("severe" if total_pnl > -0.20 else "extreme")
        return StressResult(
            scenario_name=scenario.name,
            portfolio_pnl=total_pnl,
            asset_pnl=asset_pnl,
            worst_asset=worst,
            probability=scenario.probability,
            severity=severity,
        )

    def run_all(self, portfolio_positions: Dict[str, float]) -> pd.DataFrame:
        rows = []
        for scenario in self._scenarios:
            result = self.run_scenario(scenario, portfolio_positions)
            rows.append({
                "scenario": result.scenario_name,
                "portfolio_pnl": result.portfolio_pnl,
                "severity": result.severity,
                "probability": result.probability,
                **{f"pnl_{k}": v for k, v in result.asset_pnl.items()},
            })
        return pd.DataFrame(rows).sort_values("portfolio_pnl")

    def reverse_stress_test(self, portfolio_positions: Dict[str, float],
                             loss_threshold: float = -0.10) -> List[StressScenario]:
        """Return scenarios causing portfolio loss >= loss_threshold."""
        return [
            s for s in self._scenarios
            if self.run_scenario(s, portfolio_positions).portfolio_pnl <= loss_threshold
        ]


# ---------------------------------------------------------------------------
# Predefined Energy Scenarios
# ---------------------------------------------------------------------------

def energy_price_spike() -> StressScenario:
    return StressScenario(
        name="Energy Price Spike",
        description="3-sigma spike in power prices, gas up, renewables collapse",
        shocks={"power": 2.50, "gas": 0.80, "coal": 0.30, "carbon": 0.50,
                "renewables": -0.30},
        probability=0.02,
    )


def cold_snap() -> StressScenario:
    return StressScenario(
        name="Cold Snap / Polar Vortex",
        description="Demand surge, gas shortage, extreme power prices",
        shocks={"power": 4.00, "gas": 1.50, "coal": 0.60, "carbon": 0.20,
                "renewables": -0.10, "storage": 0.50},
        correlation_adjustment=0.5,
        probability=0.01,
    )


def renewable_collapse() -> StressScenario:
    return StressScenario(
        name="Renewable Collapse",
        description="Sudden wind/solar generation loss, thermal plants needed",
        shocks={"power": 1.20, "gas": 0.40, "coal": 0.25, "carbon": 0.10,
                "renewables": -0.50},
        probability=0.03,
    )


def carbon_shock() -> StressScenario:
    return StressScenario(
        name="Carbon Price Surge",
        description="Regulatory carbon price shock, dark/spark spread compression",
        shocks={"power": 0.30, "gas": 0.05, "coal": -0.10, "carbon": 1.50,
                "css": -0.30, "cds": -0.50},
        probability=0.05,
    )


def grid_failure() -> StressScenario:
    return StressScenario(
        name="Grid Failure / Transmission Outage",
        description="Major interconnector failure, extreme nodal price divergence",
        shocks={"power_node_a": 3.00, "power_node_b": -0.50, "gas": 0.20,
                "interconnector": -1.0},
        probability=0.005,
    )


def gas_supply_disruption() -> StressScenario:
    return StressScenario(
        name="Gas Supply Disruption",
        description="Pipeline/LNG supply cut, gas and power price surge",
        shocks={"power": 1.80, "gas": 2.50, "coal": 0.40, "carbon": 0.15,
                "css": -0.80, "cds": 0.30},
        probability=0.008,
    )


def negative_price_event() -> StressScenario:
    return StressScenario(
        name="Negative Price Event",
        description="High renewables + low demand: sustained negative power prices",
        shocks={"power": -0.80, "gas": -0.20, "renewables": 0.15,
                "battery": 0.40, "css": -0.50},
        probability=0.04,
    )


class ScenarioGenerator:
    """Generates correlated multi-asset scenario shocks."""

    def generate(self, assets: List[str], mu: np.ndarray,
                 corr: np.ndarray, vol: np.ndarray,
                 n_scenarios: int = 1000,
                 horizon: float = 1.0,
                 seed: int = 42) -> pd.DataFrame:
        """Generate n_scenarios × n_assets correlated returns."""
        rng = np.random.default_rng(seed)
        sigma = np.diag(vol) @ corr @ np.diag(vol)
        L = np.linalg.cholesky(sigma * horizon + np.eye(len(assets)) * 1e-10)
        Z = rng.standard_normal((n_scenarios, len(assets)))
        shocks = mu * horizon + Z @ L.T
        return pd.DataFrame(shocks, columns=assets)
