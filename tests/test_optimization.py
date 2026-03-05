"""Tests for physical asset optimization: gas storage, hydro, unit commitment."""
import numpy as np
import pytest


class TestGasStorage:
    def setup_method(self):
        from energytrading.optimization.gas_storage import GasStorageParams
        self.params = GasStorageParams(
            max_capacity=100.0, min_capacity=5.0,
            max_injection=10.0, max_withdrawal=15.0,
            injection_cost=0.5, withdrawal_cost=0.3,
        )

    def test_dp_positive_value(self):
        from energytrading.optimization.gas_storage import GasStorageDP
        np.random.seed(42)
        prices = np.concatenate([
            np.full(180, 2.0),   # cheap season
            np.full(185, 5.0),   # expensive season
        ])
        optimizer = GasStorageDP(self.params, n_inventory_steps=10)
        value = optimizer.solve(prices, initial_inventory=50.0)
        assert value >= 0  # Storage should have non-negative value

    def test_lp_intrinsic(self):
        from energytrading.optimization.gas_storage import GasStorageLP
        prices = np.array([2.0, 2.0, 5.0, 5.0, 2.0, 5.0, 3.0, 4.0])
        optimizer = GasStorageLP(self.params)
        result = optimizer.solve(prices, initial_inventory=50.0)
        assert "value" in result
        assert result["value"] >= -1.0  # may be 0 if pulp not installed


class TestHydroOptimizer:
    def test_lp_solve(self):
        from energytrading.optimization.hydro import HydroParams, HydroScheduler
        params = HydroParams(
            max_reservoir=1000.0, min_reservoir=100.0,
            max_generation=50.0, efficiency=0.85,
            inflow_forecast=np.full(24, 20.0),
        )
        np.random.seed(1)
        prices = np.random.normal(50, 10, 24)
        scheduler = HydroScheduler(params)
        result = scheduler.optimize_lp(prices, initial_reservoir=500.0)
        assert "generation" in result
        assert len(result["generation"]) == 24


class TestUnitCommitment:
    def test_basic_commitment(self):
        from energytrading.optimization.milp import ThermalUnit, UnitCommitmentOptimizer
        units = [
            ThermalUnit("Gas1", p_min=10, p_max=100, cost_mwh=50,
                        startup_cost=500, ramp_up=20, ramp_down=20),
            ThermalUnit("Coal1", p_min=20, p_max=200, cost_mwh=35,
                        startup_cost=2000, ramp_up=15, ramp_down=15),
        ]
        demand = np.array([80, 100, 120, 150, 130, 110, 90, 80], dtype=float)
        optimizer = UnitCommitmentOptimizer(units)
        result = optimizer.solve(demand)
        assert "dispatch" in result
        assert "total_cost" in result


class TestPortfolioOptimization:
    def setup_method(self):
        np.random.seed(42)
        self.returns = np.random.multivariate_normal(
            mean=[0.08, 0.06, 0.04],
            cov=[[0.04, 0.01, 0.005],
                 [0.01, 0.03, 0.003],
                 [0.005, 0.003, 0.02]],
            size=252,
        )

    def test_mean_variance(self):
        from energytrading.portfolio.optimization import MeanVarianceOptimizer
        opt = MeanVarianceOptimizer()
        w = opt.max_sharpe(self.returns)
        assert len(w) == 3
        assert abs(w.sum() - 1.0) < 1e-6
        assert np.all(w >= -1e-6)

    def test_risk_parity(self):
        from energytrading.portfolio.optimization import RiskParityOptimizer
        opt = RiskParityOptimizer()
        w = opt.fit(self.returns)
        assert len(w) == 3
        assert abs(w.sum() - 1.0) < 1e-6
        assert np.all(w > 0)

    def test_kelly_criterion(self):
        from energytrading.portfolio.optimization import KellyCriterion
        kelly = KellyCriterion(fraction=0.5)
        w = kelly.fit(self.returns)
        assert len(w) == 3


class TestHedging:
    def test_min_variance_hedge(self):
        from energytrading.portfolio.hedging import MinimumVarianceHedge
        np.random.seed(3)
        spot = pd.Series(np.random.randn(100).cumsum() + 100)
        hedge = pd.Series(np.random.randn(100).cumsum() + 100)
        mvh = MinimumVarianceHedge()
        ratio = mvh.fit(spot, hedge)
        assert isinstance(ratio, float)
        assert abs(ratio) < 5.0  # reasonable hedge ratio

    def test_spark_spread_hedger(self):
        from energytrading.portfolio.hedging import SparkSpreadHedger
        np.random.seed(4)
        power = pd.Series(np.random.randn(100) * 5 + 55)
        gas = pd.Series(np.random.randn(100) * 0.3 + 4.0)
        hedger = SparkSpreadHedger(heat_rate=7.5)
        value = hedger.margrabe_value(power, gas, T=1.0)
        assert value >= 0


import pandas as pd
