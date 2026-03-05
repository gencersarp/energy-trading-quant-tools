"""Energy market scenario library and agent-based market simulator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .monte_carlo import MonteCarloEngine, SimulationConfig


@dataclass
class EnergyScenario:
    name: str
    description: str
    horizon_hours: int
    power_path: np.ndarray
    gas_path: np.ndarray
    wind_path: np.ndarray
    solar_path: np.ndarray
    temperature_path: np.ndarray
    metadata: Dict = field(default_factory=dict)


class ScenarioLibrary:
    """Catalog of named energy market scenarios."""

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._mc = MonteCarloEngine()

    def _base_cfg(self, n_paths: int = 1000, n_hours: int = 168) -> SimulationConfig:
        return SimulationConfig(n_paths=n_paths, n_steps=n_hours,
                                dt=1/8760, seed=self._seed, use_antithetic=False)

    def polar_vortex(self, n_paths: int = 1000) -> EnergyScenario:
        """Cold snap: gas shortage + extreme power price spike."""
        cfg = self._base_cfg(n_paths)
        power = self._mc.simulate_merton_jump(
            S0=50, mu=0.5, sigma=0.8, lambda_j=2.0, mu_j=0.5, sigma_j=0.4, cfg=cfg)
        gas = self._mc.simulate_gbm(S0=30, mu=0.3, sigma=0.4, cfg=cfg)
        wind = self._mc.simulate_ou(x0=500, theta=2.0, mu=200, sigma=100, cfg=cfg)
        solar = np.zeros_like(wind)  # winter
        temp = self._mc.simulate_ou(x0=-15, theta=0.5, mu=-20, sigma=5, cfg=cfg)
        return EnergyScenario("Polar Vortex", "Cold snap with gas shortage",
                              cfg.n_steps, power, gas, np.maximum(wind, 0),
                              solar, temp, {"risk": "extreme"})

    def summer_heatwave(self, n_paths: int = 1000) -> EnergyScenario:
        cfg = self._base_cfg(n_paths)
        power = self._mc.simulate_merton_jump(
            S0=60, mu=0.3, sigma=0.5, lambda_j=1.5, mu_j=0.3, sigma_j=0.3, cfg=cfg)
        gas = self._mc.simulate_gbm(S0=25, mu=0.1, sigma=0.3, cfg=cfg)
        solar = self._mc.simulate_ou(x0=3000, theta=1.0, mu=3500, sigma=400, cfg=cfg)
        wind = self._mc.simulate_ou(x0=800, theta=1.5, mu=600, sigma=200, cfg=cfg)
        temp = self._mc.simulate_ou(x0=38, theta=0.3, mu=35, sigma=3, cfg=cfg)
        return EnergyScenario("Summer Heatwave", "Demand surge, solar strong",
                              cfg.n_steps, power, gas, np.maximum(wind, 0),
                              np.maximum(solar, 0), temp, {"risk": "severe"})

    def wind_drought(self, n_paths: int = 1000) -> EnergyScenario:
        cfg = self._base_cfg(n_paths)
        power = self._mc.simulate_gbm(S0=55, mu=0.2, sigma=0.35, cfg=cfg)
        gas = self._mc.simulate_gbm(S0=28, mu=0.1, sigma=0.25, cfg=cfg)
        wind = self._mc.simulate_ou(x0=100, theta=3.0, mu=50, sigma=30, cfg=cfg)
        solar = self._mc.simulate_ou(x0=1500, theta=1.0, mu=1200, sigma=300, cfg=cfg)
        temp = self._mc.simulate_ou(x0=10, theta=0.5, mu=12, sigma=2, cfg=cfg)
        return EnergyScenario("Wind Drought", "Sustained low wind, thermal dispatch",
                              cfg.n_steps, power, gas, np.maximum(wind, 0),
                              np.maximum(solar, 0), temp, {"risk": "moderate"})

    def negative_price_event(self, n_paths: int = 1000) -> EnergyScenario:
        cfg = self._base_cfg(n_paths)
        power = self._mc.simulate_ou(x0=-20, theta=4.0, mu=-15, sigma=20, cfg=cfg)
        gas = self._mc.simulate_gbm(S0=22, mu=-0.1, sigma=0.2, cfg=cfg)
        wind = self._mc.simulate_ou(x0=8000, theta=0.5, mu=7000, sigma=500, cfg=cfg)
        solar = self._mc.simulate_ou(x0=6000, theta=0.5, mu=5500, sigma=600, cfg=cfg)
        temp = self._mc.simulate_ou(x0=20, theta=0.5, mu=18, sigma=2, cfg=cfg)
        return EnergyScenario("Negative Price Event", "Renewables glut, negative prices",
                              cfg.n_steps, power, gas, np.maximum(wind, 0),
                              np.maximum(solar, 0), temp, {"risk": "structural"})

    def black_swan(self, n_paths: int = 1000, severity: float = 5.0) -> EnergyScenario:
        cfg = self._base_cfg(n_paths)
        power = self._mc.simulate_merton_jump(
            S0=50, mu=0.0, sigma=1.5 * severity / 5,
            lambda_j=5.0, mu_j=severity * 0.3, sigma_j=severity * 0.2, cfg=cfg)
        gas = self._mc.simulate_gbm(S0=30, mu=0.1, sigma=0.8, cfg=cfg)
        wind = self._mc.simulate_ou(x0=500, theta=2.0, mu=100, sigma=200, cfg=cfg)
        solar = np.maximum(self._mc.simulate_ou(x0=1000, theta=1.0, mu=500, sigma=300, cfg=cfg), 0)
        temp = self._mc.simulate_ou(x0=5, theta=1.0, mu=0, sigma=10, cfg=cfg)
        return EnergyScenario("Black Swan", f"Extreme tail event severity={severity}",
                              cfg.n_steps, power, gas, np.maximum(wind, 0),
                              solar, temp, {"severity": severity, "risk": "extreme"})


# ---------------------------------------------------------------------------
# Merit Order / Agent-Based Market Simulator
# ---------------------------------------------------------------------------

@dataclass
class Generator:
    name: str
    capacity: float      # MW
    marginal_cost: float # EUR/MWh
    ramp_rate: float     # MW/h


@dataclass
class Consumer:
    name: str
    base_demand: float   # MW
    elasticity: float = -0.1


class AgentBasedMarketSimulator:
    """Simple merit order market clearing with generator and consumer agents."""

    def __init__(self):
        self._generators: List[Generator] = []
        self._consumers: List[Consumer] = []

    def add_generator(self, name: str, capacity: float,
                      marginal_cost: float, ramp_rate: float = 999.0) -> None:
        self._generators.append(Generator(name, capacity, marginal_cost, ramp_rate))

    def add_consumer(self, name: str, demand: float, elasticity: float = -0.1) -> None:
        self._consumers.append(Consumer(name, demand, elasticity))

    def merit_order_dispatch(self, total_demand: float) -> Dict:
        """Dispatch generators in merit order to meet demand."""
        sorted_gen = sorted(self._generators, key=lambda g: g.marginal_cost)
        dispatch = {}
        remaining = total_demand
        mcp = 0.0  # market clearing price
        for gen in sorted_gen:
            d = min(gen.capacity, remaining)
            dispatch[gen.name] = d
            if d > 0:
                mcp = gen.marginal_cost
            remaining -= d
            if remaining <= 0:
                break
        return {"dispatch": dispatch, "mcp": mcp, "unmet_demand": max(remaining, 0)}

    def simulate(self, hours: int = 24, demand_noise: float = 0.05,
                 seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows = []
        base_demand = sum(c.base_demand for c in self._consumers)
        for h in range(hours):
            hour_of_day = h % 24
            # Add hourly seasonality
            seasonal = 1.0 + 0.2 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi)
            demand = base_demand * seasonal * (1 + rng.normal(0, demand_noise))
            result = self.merit_order_dispatch(demand)
            rows.append({
                "hour": h,
                "demand": demand,
                "mcp": result["mcp"],
                "unmet": result["unmet_demand"],
                **{f"dispatch_{k}": v for k, v in result["dispatch"].items()},
            })
        return pd.DataFrame(rows)


class WeatherScenarioGenerator:
    """Generates correlated temperature + wind + solar scenarios."""

    def fit(self, historical_weather: pd.DataFrame) -> "WeatherScenarioGenerator":
        self._mu = historical_weather.mean().values
        self._cov = historical_weather.cov().values + np.eye(len(self._mu)) * 1e-6
        self._cols = list(historical_weather.columns)
        return self

    def sample(self, n_scenarios: int, horizon_days: int = 30,
               seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        L = np.linalg.cholesky(self._cov)
        Z = rng.standard_normal((n_scenarios * horizon_days, len(self._mu)))
        samples = self._mu + Z @ L.T
        df = pd.DataFrame(samples, columns=self._cols)
        df["scenario"] = np.repeat(np.arange(n_scenarios), horizon_days)
        df["day"] = np.tile(np.arange(horizon_days), n_scenarios)
        return df
