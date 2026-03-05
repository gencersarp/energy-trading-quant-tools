"""Tests for features, data clients, transforms, and normalizer."""
import numpy as np
import pandas as pd
import pytest


# ------------------------------------------------------------------ #
# Feature store
# ------------------------------------------------------------------ #

class TestFeatureStore:
    def test_register_and_compute(self):
        from energytrading.features.feature_store import FeatureStore, FeatureDefinition

        store = FeatureStore()

        def my_feature(x):
            return x * 2

        defn = FeatureDefinition(name="double", version="1.0",
                                 compute_fn=my_feature)
        store.register(defn)
        result = store.compute("double", 5.0)
        assert result == 10.0

    def test_decorator_registration(self):
        from energytrading.features.feature_store import FeatureStore

        store = FeatureStore()

        @store.register_fn(name="square", version="1.0")
        def square(x):
            return x ** 2

        assert store.compute("square", 3.0) == 9.0

    def test_cache_hit(self):
        from energytrading.features.feature_store import FeatureStore, FeatureDefinition

        store = FeatureStore()
        call_count = [0]

        def expensive(x):
            call_count[0] += 1
            return x + 1

        store.register(FeatureDefinition("add_one", "1.0", expensive))
        store.compute("add_one", 10)
        store.compute("add_one", 10)  # should be cached
        assert call_count[0] == 1

    def test_materialize(self):
        from energytrading.features.feature_store import FeatureStore, FeatureDefinition

        store = FeatureStore()
        store.register(FeatureDefinition("price_x2", "1.0",
                                         lambda df: df["price"] * 2))
        df = pd.DataFrame({"price": [10.0, 20.0, 30.0]})
        result = store.materialize(df, ["price_x2"])
        assert "price_x2" in result.columns
        assert list(result["price_x2"]) == [20.0, 40.0, 60.0]


# ------------------------------------------------------------------ #
# Energy features
# ------------------------------------------------------------------ #

class TestEnergyFeatures:
    def setup_method(self):
        n = 100
        idx = pd.date_range("2023-01-01", periods=n, freq="H")
        self.power = pd.Series(np.random.normal(55, 10, n), index=idx)
        self.gas = pd.Series(np.random.normal(4.0, 0.3, n), index=idx)
        self.coal = pd.Series(np.random.normal(100, 5, n), index=idx)

    def test_spark_spread(self):
        from energytrading.features.energy_features import EnergyFeatureEngineer
        spark = EnergyFeatureEngineer.spark_spread(self.power, self.gas)
        assert len(spark) == 100
        # spark ≈ power - gas * 7.5
        expected = self.power - self.gas * 7.5
        np.testing.assert_allclose(spark.values, expected.values, rtol=1e-10)

    def test_dark_spread(self):
        from energytrading.features.energy_features import EnergyFeatureEngineer
        dark = EnergyFeatureEngineer.dark_spread(self.power, self.coal)
        assert len(dark) == 100

    def test_realized_vol(self):
        from energytrading.features.energy_features import EnergyFeatureEngineer
        vol = EnergyFeatureEngineer.realized_vol(self.power)
        assert vol.isna().sum() < 30  # most values should be non-NaN

    def test_build_feature_matrix(self):
        from energytrading.features.energy_features import EnergyFeatureEngineer
        df = pd.DataFrame({
            "power_price": self.power,
            "gas_price": self.gas,
        })
        result = EnergyFeatureEngineer.build_feature_matrix(df)
        assert "spark_spread" in result.columns
        assert "realized_vol_21" in result.columns

    def test_calendar_features(self):
        from energytrading.features.energy_features import CalendarFeatures
        idx = pd.date_range("2023-01-01", periods=100, freq="H")
        df = pd.DataFrame({"price": np.random.randn(100)}, index=idx)
        result = CalendarFeatures.add_all(df)
        assert "hour" in result.columns
        assert "is_heating" in result.columns
        assert "sin_daily" in result.columns


# ------------------------------------------------------------------ #
# Transforms
# ------------------------------------------------------------------ #

class TestTransforms:
    def test_fourier_features(self):
        from energytrading.features.transforms import FourierFeatures
        n = 256
        t = np.arange(n)
        signal = np.sin(2 * np.pi * t / 24) + 0.5 * np.sin(2 * np.pi * t / 168)
        ff = FourierFeatures(n_harmonics=5)
        ff.fit(signal)
        approx = ff.transform(n)
        assert len(approx) == n
        freqs = ff.dominant_frequencies()
        assert len(freqs) == 5

    def test_wavelet_decompose(self):
        from energytrading.features.transforms import WaveletFeatures
        signal = np.random.randn(128)
        components = WaveletFeatures.haar_decompose(signal, levels=3)
        assert len(components) == 4  # 1 approx + 3 details

    def test_wavelet_energy(self):
        from energytrading.features.transforms import WaveletFeatures
        signal = np.random.randn(64)
        energy = WaveletFeatures.energy_by_level(signal, levels=3)
        assert len(energy) == 4
        assert np.all(energy >= 0)

    def test_rolling_features(self):
        from energytrading.features.transforms import RollingFeatureExtractor
        series = pd.Series(np.random.randn(100))
        extractor = RollingFeatureExtractor(windows=[5, 10])
        result = extractor.transform(series, features=["mean", "std", "zscore"])
        assert "mean_5" in result.columns
        assert "zscore_10" in result.columns
        assert len(result) == 100

    def test_spike_detector_zscore(self):
        from energytrading.features.transforms import SpikeDetector
        series = pd.Series(np.random.randn(100))
        series.iloc[50] = 50.0  # inject spike
        det = SpikeDetector(method="zscore", threshold=3.0)
        spikes = det.detect(series)
        assert spikes.iloc[50] == True  # spike should be detected

    def test_spike_clip(self):
        from energytrading.features.transforms import SpikeDetector
        series = pd.Series(np.random.randn(100))
        series.iloc[50] = 100.0
        det = SpikeDetector(method="zscore", threshold=3.0)
        clipped = det.clip_spikes(series)
        assert abs(clipped.iloc[50]) < abs(series.iloc[50])


# ------------------------------------------------------------------ #
# Data normalizer
# ------------------------------------------------------------------ #

class TestDataNormalizer:
    def setup_method(self):
        np.random.seed(10)
        self.df = pd.DataFrame({
            "price": np.random.normal(50, 10, 100),
            "vol": np.random.normal(0.2, 0.05, 100),
        })

    def test_zscore(self):
        from energytrading.data.normalizer import DataNormalizer
        norm = DataNormalizer(method="zscore")
        normalized = norm.fit_transform(self.df)
        assert abs(normalized["price"].mean()) < 0.1
        assert abs(normalized["price"].std() - 1.0) < 0.1

    def test_inverse_transform(self):
        from energytrading.data.normalizer import DataNormalizer
        norm = DataNormalizer(method="zscore")
        normalized = norm.fit_transform(self.df)
        recovered = norm.inverse_transform(normalized)
        np.testing.assert_allclose(
            recovered["price"].values, self.df["price"].values, rtol=1e-5)

    def test_minmax(self):
        from energytrading.data.normalizer import DataNormalizer
        norm = DataNormalizer(method="minmax")
        normalized = norm.fit_transform(self.df)
        assert normalized["price"].min() >= 0.0 - 1e-10
        assert normalized["price"].max() <= 1.0 + 1e-10


# ------------------------------------------------------------------ #
# EIA Client (synthetic fallback)
# ------------------------------------------------------------------ #

class TestEIAClient:
    def test_gas_price_synthetic(self):
        from energytrading.data.eia_client import EIAClient
        client = EIAClient(api_key=None)  # no key → synthetic
        series = client.get_gas_spot_price("2023-01-01", "2023-06-01")
        assert len(series) > 0
        assert series.min() > 0

    def test_crude_price_synthetic(self):
        from energytrading.data.eia_client import EIAClient
        client = EIAClient()
        series = client.get_crude_spot_price("2023-01-01", "2023-03-01")
        assert len(series) > 0
        assert series.min() > 0

    def test_generation_mix_synthetic(self):
        from energytrading.data.eia_client import EIAClient
        client = EIAClient()
        df = client.get_electricity_generation("2023-01-01", "2023-12-01")
        assert not df.empty
        assert "natural_gas" in df.columns


# ------------------------------------------------------------------ #
# ENTSO-E Client (synthetic fallback)
# ------------------------------------------------------------------ #

class TestENTSOEClient:
    def test_da_prices_synthetic(self):
        from energytrading.data.entsoe_client import ENTSOEClient
        client = ENTSOEClient(security_token=None)
        prices = client.get_day_ahead_prices("DE", "2024-01-01", "2024-01-07")
        assert len(prices) > 0

    def test_load_synthetic(self):
        from energytrading.data.entsoe_client import ENTSOEClient
        client = ENTSOEClient()
        load = client.get_actual_load("DE", "2024-01-01", "2024-01-03")
        assert len(load) > 0
        assert load.min() > 0

    def test_generation_synthetic(self):
        from energytrading.data.entsoe_client import ENTSOEClient
        client = ENTSOEClient()
        gen = client.get_generation_by_source("DE", "2024-01-01", "2024-01-03")
        assert not gen.empty
