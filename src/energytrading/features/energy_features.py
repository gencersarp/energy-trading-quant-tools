"""Domain-specific energy market feature engineering."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class EnergyFeatureEngineer:
    """
    Compute energy-market-specific features:
    spark/dark spreads, heat rates, reserve margins, duck curves,
    capture rates, basis differentials, and more.
    """

    # ------------------------------------------------------------------ #
    # Spread features
    # ------------------------------------------------------------------ #

    @staticmethod
    def spark_spread(power_price: pd.Series, gas_price: pd.Series,
                     heat_rate: float = 7.5) -> pd.Series:
        """Spark spread = power - gas * heat_rate ($/MWh)."""
        return power_price - gas_price * heat_rate

    @staticmethod
    def dark_spread(power_price: pd.Series, coal_price: pd.Series,
                    heat_rate: float = 10.5) -> pd.Series:
        """Dark spread = power - coal * heat_rate."""
        return power_price - coal_price * heat_rate

    @staticmethod
    def clean_spark_spread(power_price: pd.Series, gas_price: pd.Series,
                           co2_price: pd.Series, heat_rate: float = 7.5,
                           emission_factor: float = 0.2) -> pd.Series:
        """Clean spark spread accounting for carbon cost."""
        return power_price - gas_price * heat_rate - co2_price * emission_factor * heat_rate

    @staticmethod
    def crack_spread_321(crude: pd.Series, gasoline: pd.Series,
                         distillate: pd.Series) -> pd.Series:
        """3-2-1 crack spread: 2*gasoline + 1*distillate - 3*crude (all in $/bbl)."""
        return (2 * gasoline + distillate - 3 * crude) / 3

    @staticmethod
    def basis_differential(spot: pd.Series, hub_price: pd.Series) -> pd.Series:
        """Local basis = spot - hub (pipeline / congestion premium)."""
        return spot - hub_price

    # ------------------------------------------------------------------ #
    # Heat rate features
    # ------------------------------------------------------------------ #

    @staticmethod
    def implied_heat_rate(power_price: pd.Series,
                          gas_price: pd.Series) -> pd.Series:
        """Implied heat rate = power / gas (MMBtu/MWh)."""
        return power_price / (gas_price + 1e-8)

    @staticmethod
    def heat_rate_z_score(power_price: pd.Series, gas_price: pd.Series,
                          window: int = 30) -> pd.Series:
        hr = power_price / (gas_price + 1e-8)
        roll_mean = hr.rolling(window).mean()
        roll_std = hr.rolling(window).std()
        return (hr - roll_mean) / (roll_std + 1e-8)

    # ------------------------------------------------------------------ #
    # Reserve margin and load features
    # ------------------------------------------------------------------ #

    @staticmethod
    def reserve_margin(capacity: pd.Series, peak_load: pd.Series) -> pd.Series:
        """Reserve margin = (capacity - peak_load) / peak_load."""
        return (capacity - peak_load) / (peak_load + 1e-8)

    @staticmethod
    def load_factor(energy_consumed: pd.Series, peak_load: pd.Series,
                    hours: int = 8760) -> pd.Series:
        """Load factor = energy / (peak * hours)."""
        return energy_consumed / (peak_load * hours + 1e-8)

    @staticmethod
    def duck_curve_ramp(net_load: pd.Series,
                        hour_col: Optional[pd.Series] = None) -> pd.Series:
        """
        Duck curve ramp metric: steepness of late-afternoon ramp.
        Returns hour-over-hour change in net load.
        """
        return net_load.diff().fillna(0)

    @staticmethod
    def net_load(total_load: pd.Series, renewables: pd.Series) -> pd.Series:
        return total_load - renewables

    # ------------------------------------------------------------------ #
    # Renewable capture rate
    # ------------------------------------------------------------------ #

    @staticmethod
    def capture_rate(generation: pd.Series, spot_price: pd.Series,
                     reference_price: Optional[pd.Series] = None) -> pd.Series:
        """
        Capture rate = weighted average price received by generator /
                       average spot price.
        """
        weighted = generation * spot_price
        ref = reference_price if reference_price is not None else spot_price
        window_gen = generation.rolling(24, min_periods=1).sum()
        window_rev = weighted.rolling(24, min_periods=1).sum()
        avg_ref = ref.rolling(24, min_periods=1).mean()
        avg_captured = window_rev / (window_gen + 1e-8)
        return avg_captured / (avg_ref + 1e-8)

    # ------------------------------------------------------------------ #
    # Price shape / temporal features
    # ------------------------------------------------------------------ #

    @staticmethod
    def on_off_peak_ratio(price: pd.Series, hour: pd.Series) -> pd.Series:
        """On-peak (7am-11pm) vs off-peak ratio by day."""
        on_mask = hour.between(7, 22)
        on_avg = price.where(on_mask).rolling(24, min_periods=1).mean()
        off_avg = price.where(~on_mask).rolling(24, min_periods=1).mean()
        return on_avg / (off_avg + 1e-8)

    @staticmethod
    def price_shape_index(price: pd.Series, window: int = 24) -> pd.Series:
        """How far peak price deviates from average (intraday shape)."""
        rolling_max = price.rolling(window).max()
        rolling_mean = price.rolling(window).mean()
        return rolling_max / (rolling_mean + 1e-8)

    @staticmethod
    def contango_backwardation(front_month: pd.Series,
                               next_month: pd.Series) -> pd.Series:
        """Positive = contango (next > front). Negative = backwardation."""
        return next_month - front_month

    # ------------------------------------------------------------------ #
    # Volatility features
    # ------------------------------------------------------------------ #

    @staticmethod
    def realized_vol(price: pd.Series, window: int = 21,
                     annualize: int = 252) -> pd.Series:
        log_ret = np.log(price / price.shift(1))
        return log_ret.rolling(window).std() * np.sqrt(annualize)

    @staticmethod
    def parkinson_vol(high: pd.Series, low: pd.Series,
                      window: int = 21) -> pd.Series:
        """Parkinson (1980) range-based volatility estimator."""
        hl = np.log(high / (low + 1e-8)) ** 2
        return np.sqrt(hl.rolling(window).mean() / (4 * np.log(2)))

    @staticmethod
    def vol_regime_indicator(price: pd.Series, short_window: int = 5,
                             long_window: int = 21) -> pd.Series:
        """1 if short-term vol > long-term vol (high-vol regime)."""
        short_vol = np.log(price / price.shift(1)).rolling(short_window).std()
        long_vol = np.log(price / price.shift(1)).rolling(long_window).std()
        return (short_vol > long_vol).astype(float)

    # ------------------------------------------------------------------ #
    # Storage / inventory features
    # ------------------------------------------------------------------ #

    @staticmethod
    def storage_deficit(inventory: pd.Series,
                        seasonal_norm: pd.Series) -> pd.Series:
        """How far storage is below seasonal norm (positive = deficit)."""
        return seasonal_norm - inventory

    @staticmethod
    def injection_withdrawal_rate(inventory: pd.Series,
                                  window: int = 7) -> pd.Series:
        return inventory.diff(window) / window

    # ------------------------------------------------------------------ #
    # Convenience: build all features in one call
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_feature_matrix(df: pd.DataFrame,
                             config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Given a DataFrame with columns like power_price, gas_price, load, etc.,
        compute a standard set of features and return an augmented DataFrame.
        """
        result = df.copy()
        cfg = config or {}

        if "power_price" in df and "gas_price" in df:
            hr = cfg.get("heat_rate", 7.5)
            result["spark_spread"] = EnergyFeatureEngineer.spark_spread(
                df["power_price"], df["gas_price"], hr)
            result["implied_heat_rate"] = EnergyFeatureEngineer.implied_heat_rate(
                df["power_price"], df["gas_price"])
            result["heat_rate_zscore"] = EnergyFeatureEngineer.heat_rate_z_score(
                df["power_price"], df["gas_price"])

        if "power_price" in df and "coal_price" in df:
            result["dark_spread"] = EnergyFeatureEngineer.dark_spread(
                df["power_price"], df["coal_price"])

        if "power_price" in df:
            result["realized_vol_21"] = EnergyFeatureEngineer.realized_vol(
                df["power_price"], window=21)

        if "total_load" in df and "renewables" in df:
            result["net_load"] = EnergyFeatureEngineer.net_load(
                df["total_load"], df["renewables"])
            result["net_load_ramp"] = EnergyFeatureEngineer.duck_curve_ramp(
                result["net_load"])

        if "capacity" in df and "peak_load" in df:
            result["reserve_margin"] = EnergyFeatureEngineer.reserve_margin(
                df["capacity"], df["peak_load"])

        if "inventory" in df and "seasonal_norm" in df:
            result["storage_deficit"] = EnergyFeatureEngineer.storage_deficit(
                df["inventory"], df["seasonal_norm"])

        return result


class CalendarFeatures:
    """Time-based features for energy markets."""

    @staticmethod
    def hour_of_day(dt_index: pd.DatetimeIndex) -> np.ndarray:
        return dt_index.hour.values

    @staticmethod
    def day_of_week(dt_index: pd.DatetimeIndex) -> np.ndarray:
        return dt_index.dayofweek.values

    @staticmethod
    def is_weekend(dt_index: pd.DatetimeIndex) -> np.ndarray:
        return (dt_index.dayofweek >= 5).astype(int)

    @staticmethod
    def month(dt_index: pd.DatetimeIndex) -> np.ndarray:
        return dt_index.month.values

    @staticmethod
    def quarter(dt_index: pd.DatetimeIndex) -> np.ndarray:
        return dt_index.quarter.values

    @staticmethod
    def is_heating_season(dt_index: pd.DatetimeIndex) -> np.ndarray:
        """Nov–Mar = heating season."""
        return dt_index.month.isin([11, 12, 1, 2, 3]).astype(int)

    @staticmethod
    def is_cooling_season(dt_index: pd.DatetimeIndex) -> np.ndarray:
        """Jun–Sep = cooling season."""
        return dt_index.month.isin([6, 7, 8, 9]).astype(int)

    @staticmethod
    def fourier_time_features(dt_index: pd.DatetimeIndex,
                               periods: List[Tuple[str, float]] = None) -> pd.DataFrame:
        """
        Sine/cosine encoding for periodicity.
        periods: list of (name, period_in_hours)
        """
        if periods is None:
            periods = [("daily", 24), ("weekly", 168), ("annual", 8760)]

        t = np.arange(len(dt_index), dtype=float)
        cols: Dict = {}
        for name, period in periods:
            cols[f"sin_{name}"] = np.sin(2 * np.pi * t / period)
            cols[f"cos_{name}"] = np.cos(2 * np.pi * t / period)
        return pd.DataFrame(cols, index=dt_index)

    @staticmethod
    def add_all(df: pd.DataFrame) -> pd.DataFrame:
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            return df
        result = df.copy()
        result["hour"] = CalendarFeatures.hour_of_day(idx)
        result["dow"] = CalendarFeatures.day_of_week(idx)
        result["is_weekend"] = CalendarFeatures.is_weekend(idx)
        result["month"] = CalendarFeatures.month(idx)
        result["is_heating"] = CalendarFeatures.is_heating_season(idx)
        result["is_cooling"] = CalendarFeatures.is_cooling_season(idx)
        fourier = CalendarFeatures.fourier_time_features(idx)
        return pd.concat([result, fourier], axis=1)
