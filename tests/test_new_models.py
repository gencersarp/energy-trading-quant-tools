"""Tests for new stochastic models: ARIMA, Kalman, Cointegration, Heston."""
import numpy as np
import pandas as pd
import pytest


# ------------------------------------------------------------------ #
# ARIMA / GARCH
# ------------------------------------------------------------------ #

class TestARIMAModel:
    def test_ar_fit_predict(self):
        from energytrading.models.arima import ARIMAModel
        np.random.seed(42)
        series = np.cumsum(np.random.randn(200))
        model = ARIMAModel(p=2, d=0, q=0)
        model.fit(series)
        pred = model.predict(10)
        assert len(pred) == 10
        assert not np.any(np.isnan(pred))

    def test_garch_fit(self):
        from energytrading.models.arima import GARCHModel
        np.random.seed(0)
        returns = np.random.randn(300) * 0.02
        gm = GARCHModel(p=1, q=1)
        gm.fit(returns)
        vols = gm.conditional_volatility(returns)
        assert len(vols) == len(returns)
        assert np.all(vols >= 0)

    def test_auto_arima(self):
        from energytrading.models.arima import fit_arima_auto
        np.random.seed(1)
        series = np.random.randn(100)
        model = fit_arima_auto(series, max_p=2, max_d=1, max_q=1)
        assert model is not None


# ------------------------------------------------------------------ #
# Kalman Filter
# ------------------------------------------------------------------ #

class TestKalmanFilter:
    def test_filter_basic(self):
        from energytrading.models.kalman import KalmanFilter
        np.random.seed(5)
        obs = np.random.randn(50)
        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.F = np.array([[1.0]])
        kf.H = np.array([[1.0]])
        kf.Q = np.array([[0.01]])
        kf.R = np.array([[1.0]])
        kf.x = np.array([0.0])
        kf.P = np.array([[1.0]])
        means, covs = kf.filter(obs)
        assert len(means) == len(obs)
        assert not np.any(np.isnan(means))

    def test_kalman_regression_tracker(self):
        from energytrading.models.kalman import KalmanRegressionTracker
        np.random.seed(7)
        x = np.random.randn(100)
        y = 2.0 * x + np.random.randn(100) * 0.1
        tracker = KalmanRegressionTracker()
        betas = tracker.fit(x, y)
        assert len(betas) == len(x)
        # Betas should converge toward ~2.0
        assert abs(betas[-1] - 2.0) < 0.5


# ------------------------------------------------------------------ #
# Cointegration
# ------------------------------------------------------------------ #

class TestCointegration:
    def setup_method(self):
        np.random.seed(42)
        common = np.cumsum(np.random.randn(200))
        self.x = pd.Series(common + np.random.randn(200) * 0.1)
        self.y = pd.Series(2.0 * common + np.random.randn(200) * 0.1)

    def test_engle_granger(self):
        from energytrading.models.cointegration import CointegrationAnalyzer
        ca = CointegrationAnalyzer()
        result = ca.engle_granger_test(self.y, self.x)
        assert "adf_stat" in result
        assert "pvalue" in result

    def test_spread_model(self):
        from energytrading.models.cointegration import SpreadTradingModel
        stm = SpreadTradingModel(lookback=30, entry_z=1.5)
        stm.fit(self.y, self.x)
        signal = stm.entry_signal(self.y, self.x)
        assert len(signal) == len(self.y)
        assert signal.isin([-1, 0, 1]).all()


# ------------------------------------------------------------------ #
# Heston
# ------------------------------------------------------------------ #

class TestHestonModel:
    def test_simulate(self):
        from energytrading.models.heston import HestonParams, HestonModel
        params = HestonParams(S0=100, V0=0.04, kappa=2.0,
                              theta=0.04, sigma_v=0.3, rho=-0.7, r=0.02)
        model = HestonModel(params)
        paths = model.simulate(T=1.0, n_steps=50, n_paths=100)
        assert paths.shape == (100, 51)
        assert np.all(paths >= 0)

    def test_call_price(self):
        from energytrading.models.heston import HestonParams, HestonModel
        params = HestonParams(S0=100, V0=0.04, kappa=2.0,
                              theta=0.04, sigma_v=0.3, rho=-0.7, r=0.02)
        model = HestonModel(params)
        price = model.call_price(K=100, T=1.0)
        assert price > 0
        assert price < 50  # sanity: less than stock price


# ------------------------------------------------------------------ #
# Deep Learning
# ------------------------------------------------------------------ #

class TestDeepLearning:
    def test_lstm_fit_predict(self):
        from energytrading.models.deep_learning import LSTMForecaster
        np.random.seed(1)
        X = np.random.randn(100, 10, 5)
        y = np.random.randn(100)
        model = LSTMForecaster(input_size=5, hidden_size=16, n_epochs=2)
        model.fit(X, y)
        pred = model.predict(X[:5])
        assert len(pred) == 5

    def test_ensemble_forecaster(self):
        from energytrading.models.deep_learning import EnsembleDeepForecaster
        np.random.seed(2)
        X = np.random.randn(80, 5, 3)
        y = np.random.randn(80)
        model = EnsembleDeepForecaster()
        model.fit(X, y)
        pred = model.predict(X[:10])
        assert len(pred) == 10
