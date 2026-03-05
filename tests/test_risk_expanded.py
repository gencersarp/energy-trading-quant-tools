"""Tests for expanded risk modules: copula, stress testing, Monte Carlo."""
import numpy as np
import pandas as pd
import pytest


class TestCopula:
    def setup_method(self):
        np.random.seed(42)
        self.u = np.column_stack([
            np.random.uniform(0, 1, 200),
            np.random.uniform(0, 1, 200),
        ])
        # Correlated data
        cov = np.array([[1.0, 0.7], [0.7, 1.0]])
        raw = np.random.multivariate_normal([0, 0], cov, 200)
        from scipy.stats import norm
        self.u_corr = norm.cdf(raw)

    def test_gaussian_copula(self):
        from energytrading.risk.copula import GaussianCopula
        cop = GaussianCopula()
        cop.fit(self.u_corr)
        assert cop._rho is not None
        samples = cop.sample(100)
        assert samples.shape == (100, 2)
        assert np.all(samples >= 0) and np.all(samples <= 1)

    def test_clayton_copula(self):
        from energytrading.risk.copula import ClaytonCopula
        cop = ClaytonCopula()
        cop.fit(self.u_corr)
        assert cop._theta > 0
        samples = cop.sample(50)
        assert samples.shape == (50, 2)

    def test_copula_selector(self):
        from energytrading.risk.copula import CopulaSelector
        selector = CopulaSelector()
        best = selector.select(self.u_corr)
        assert best is not None


class TestStressTesting:
    def test_predefined_scenario(self):
        from energytrading.risk.stress_testing import StressTestSuite
        np.random.seed(0)
        portfolio = {"power": 100.0, "gas": -50.0, "renewables": 30.0}
        suite = StressTestSuite(portfolio)
        results = suite.run_all()
        assert len(results) > 0
        for r in results:
            assert hasattr(r, "scenario_name")
            assert hasattr(r, "portfolio_pnl")

    def test_reverse_stress_test(self):
        from energytrading.risk.stress_testing import StressTestSuite
        portfolio = {"power": 100.0, "gas": -50.0}
        suite = StressTestSuite(portfolio)
        scenarios = suite.reverse_stress_test(loss_threshold=-20.0)
        assert isinstance(scenarios, list)


class TestMonteCarlo:
    def test_correlated_simulator(self):
        from energytrading.risk.monte_carlo import CorrelatedPathSimulator
        sim = CorrelatedPathSimulator(
            mu=np.array([0.05, 0.03]),
            sigma=np.array([0.2, 0.15]),
            corr=np.array([[1.0, 0.6], [0.6, 1.0]]),
        )
        paths = sim.simulate(S0=np.array([100.0, 80.0]),
                             T=1.0, n_steps=52, n_paths=500)
        assert paths.shape == (500, 53, 2)
        assert np.all(paths >= 0)

    def test_mc_risk_engine(self):
        from energytrading.risk.monte_carlo import CorrelatedPathSimulator, MCRiskEngine
        sim = CorrelatedPathSimulator(
            mu=np.array([0.0, 0.0]),
            sigma=np.array([0.2, 0.15]),
            corr=np.eye(2),
        )
        engine = MCRiskEngine(sim, n_simulations=1000)
        paths = engine.simulate_portfolio(
            weights=np.array([0.5, 0.5]),
            S0=np.array([100.0, 100.0]),
            T=1.0, n_steps=10,
        )
        var = engine.var(paths, confidence=0.95)
        cvar = engine.cvar(paths, confidence=0.95)
        assert var < 0
        assert cvar <= var  # CVaR is worse loss than VaR
