"""Tests for ML infrastructure, RL, streaming, weather, regimes, simulation."""
import numpy as np
import pandas as pd
import pytest


# ------------------------------------------------------------------ #
# ML Infrastructure
# ------------------------------------------------------------------ #

class TestModelRegistry:
    def test_register_and_get(self, tmp_path):
        pytest.importorskip("joblib")
        from energytrading.ml.model_registry import ModelRegistry, ModelMetadata
        registry = ModelRegistry(storage_path=str(tmp_path))
        class DummyModel:
            def predict(self, x): return x * 2
        meta = ModelMetadata(name="m", version="1.0", stage="dev",
                             metrics={"rmse": 0.05}, params={})
        registry.register(DummyModel(), meta)
        retrieved = registry.get("m", "1.0")
        assert retrieved is not None

    def test_promote(self, tmp_path):
        pytest.importorskip("joblib")
        from energytrading.ml.model_registry import ModelRegistry, ModelMetadata
        registry = ModelRegistry(storage_path=str(tmp_path))
        registry.register(object(), ModelMetadata("m", "1.0", "dev", {}, {}))
        registry.promote("m", "1.0", "staging")
        assert registry.get_metadata("m", "1.0").stage == "staging"


class TestExperimentTracker:
    def test_run_context(self, tmp_path):
        pytest.importorskip("joblib")
        from energytrading.ml.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker(experiment_name="test", storage_path=str(tmp_path))
        with tracker.start_run("run1") as run:
            run.log_param("lr", 0.01)
            run.log_metric("rmse", 0.05)
        assert len(tracker.search_runs()) >= 1


class TestHyperOptimizer:
    def test_random_search(self):
        pytest.importorskip("joblib")
        from energytrading.ml.hyperopt import SearchSpace, HyperOptimizer
        space = SearchSpace()
        space.add_float("lr", 1e-4, 1e-1, log_scale=True)
        space.add_int("n_layers", 1, 5)
        def objective(params): return params["lr"] * 10 + params["n_layers"]
        opt = HyperOptimizer(space, objective, method="random", n_trials=5)
        best_params, best_score = opt.optimize()
        assert "lr" in best_params


# ------------------------------------------------------------------ #
# RL Environments & Agents
# ------------------------------------------------------------------ #

class TestRLEnvironments:
    def test_energy_market_env_step(self):
        from energytrading.rl.environment import EnergyMarketEnv
        np.random.seed(0)
        env = EnergyMarketEnv(np.random.normal(50, 10, 200))
        state = env.reset()
        assert len(state) > 0
        action = np.array([0.1])
        next_state, reward, done, info = env.step(action)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_battery_env(self):
        from energytrading.rl.environment import BatteryStorageEnv
        np.random.seed(1)
        env = BatteryStorageEnv(np.random.normal(50, 15, 200))
        state = env.reset()
        assert state is not None
        _, reward, done, _ = env.step(np.array([0.5]))
        assert isinstance(reward, float)

    def test_random_agent_act(self):
        from energytrading.rl.environment import EnergyMarketEnv
        from energytrading.rl.agents import RandomAgent
        np.random.seed(2)
        env = EnergyMarketEnv(np.random.normal(50, 10, 100))
        agent = RandomAgent()
        state = env.reset()
        action = agent.act(state)
        assert action is not None


# ------------------------------------------------------------------ #
# Weather
# ------------------------------------------------------------------ #

class TestWeather:
    def test_degree_days(self):
        pytest.importorskip("sklearn")
        from energytrading.weather.forecasting import DegreeDay
        dd = DegreeDay(base_temp=65.0)
        temps = pd.Series([60.0, 70.0, 55.0, 80.0, 65.0])
        hdd = dd.hdd(temps)
        cdd = dd.cdd(temps)
        assert float(hdd.iloc[0]) == 5.0
        assert float(cdd.iloc[1]) == 5.0
        assert float(hdd.iloc[4]) == 0.0

    def test_wind_power_curve(self):
        pytest.importorskip("sklearn")
        from energytrading.weather.forecasting import WindPowerForecaster
        forecaster = WindPowerForecaster(rated_power=2.0)
        power = forecaster.power_curve(pd.Series([0.0, 3.0, 7.0, 10.0, 25.0]))
        assert power.iloc[0] == 0.0
        assert power.iloc[-1] == 0.0

    def test_weather_option(self):
        from energytrading.weather.derivatives import DegreeDayOption, WeatherDerivativesPricer
        opt = DegreeDayOption(
            option_type="hdd", strike_dd=3000.0, tick_size=2500.0,
            period_start=pd.Timestamp("2024-11-01"),
            period_end=pd.Timestamp("2024-03-31"),
        )
        pricer = WeatherDerivativesPricer()
        np.random.seed(5)
        temps = np.random.normal(55, 10, 90)
        result = pricer.price_hdd_option(opt, temps, base_temp=65.0)
        price = result[0] if isinstance(result, tuple) else result
        assert price >= 0


# ------------------------------------------------------------------ #
# Regime Detection
# ------------------------------------------------------------------ #

class TestRegimeDetection:
    def setup_method(self):
        np.random.seed(42)
        self.returns = pd.Series(np.random.randn(300) * 0.01)
        self.returns.iloc[100:150] = np.random.randn(50) * 0.05

    def test_threshold_regimes(self):
        from energytrading.regimes.detector import RegimeDetector
        regimes = RegimeDetector().threshold_regimes(self.returns)
        assert len(regimes) == len(self.returns)

    def test_hmm_regimes(self):
        pytest.importorskip("hmmlearn")
        from energytrading.regimes.detector import RegimeDetector
        regimes = RegimeDetector().hmm_regimes(self.returns, n_states=2)
        assert len(regimes) == len(self.returns)
        assert set(regimes.unique()).issubset({0, 1})

    def test_turbulence_index(self):
        pytest.importorskip("sklearn")
        from energytrading.regimes.turbulence import TurbulenceIndex
        returns = pd.DataFrame(
            np.random.multivariate_normal([0, 0], [[0.01, 0.005], [0.005, 0.01]], 200),
            columns=["a", "b"],
        )
        scores = TurbulenceIndex(lookback=50).compute(returns)
        assert len(scores) == len(returns) and scores.min() >= 0


# ------------------------------------------------------------------ #
# Streaming (in-memory fallback)
# ------------------------------------------------------------------ #

class TestStreamingFallback:
    def test_kafka_producer_fallback(self):
        from energytrading.streaming.kafka_handler import EnergyPriceProducer, KafkaConfig
        config = KafkaConfig(bootstrap_servers=["nonexistent:9999"], topic="test")
        producer = EnergyPriceProducer(config)
        producer.publish_price("HH", 3.5, 100.0)  # must not raise

    def test_redis_cache_fallback(self):
        from energytrading.streaming.redis_handler import RedisCacheLayer, RedisConfig
        config = RedisConfig(host="nonexistent_host", port=6380)
        cache = RedisCacheLayer(config)
        cache.cache("key", {"v": 42})
        result = cache.get("key")
        assert result == {"v": 42}


# ------------------------------------------------------------------ #
# Simulation
# ------------------------------------------------------------------ #

class TestSimulation:
    def test_gbm_paths_shape(self):
        from energytrading.simulation.monte_carlo import MonteCarloEngine, SimulationConfig
        cfg = SimulationConfig(n_paths=100, n_steps=50, dt=1/252)
        paths = MonteCarloEngine().simulate_gbm(S0=100.0, mu=0.05, sigma=0.2, cfg=cfg)
        # Shape is (n_steps+1, n_paths)
        assert paths.shape[1] == cfg.n_paths
        assert np.all(paths >= 0)

    def test_ou_paths_shape(self):
        from energytrading.simulation.monte_carlo import MonteCarloEngine, SimulationConfig
        cfg = SimulationConfig(n_paths=50, n_steps=30, dt=1/252)
        paths = MonteCarloEngine().simulate_ou(x0=50.0, theta=2.0, mu=50.0, sigma=5.0, cfg=cfg)
        assert paths.shape[1] == cfg.n_paths

    def test_scenario_library_methods(self):
        from energytrading.simulation.scenarios import ScenarioLibrary
        lib = ScenarioLibrary()
        assert hasattr(lib, "polar_vortex")
        assert hasattr(lib, "summer_heatwave")
        assert hasattr(lib, "wind_drought")
