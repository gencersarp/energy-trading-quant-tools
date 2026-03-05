"""Weather-driven load and renewable generation forecasting."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


class DegreeDay:
    """Heating and cooling degree day calculations."""

    @staticmethod
    def heating_degree_days(temp: pd.Series, base: float = 18.0) -> pd.Series:
        return np.maximum(base - temp, 0).rename("HDD")

    @staticmethod
    def cooling_degree_days(temp: pd.Series, base: float = 18.0) -> pd.Series:
        return np.maximum(temp - base, 0).rename("CDD")

    @staticmethod
    def effective_degree_days(temp: pd.Series, base: float = 18.0,
                               w_heat: float = 1.0, w_cool: float = 0.8) -> pd.Series:
        hdd = np.maximum(base - temp, 0)
        cdd = np.maximum(temp - base, 0)
        return (w_heat * hdd + w_cool * cdd).rename("EHDD")


class WeatherLoadModel:
    """Models electricity load as a function of temperature + calendar features."""

    def __init__(self):
        self._model: Optional[Pipeline] = None
        self._base_load_model: Optional[Ridge] = None

    def _build_features(self, temperature: pd.Series,
                         include_time: bool = True) -> pd.DataFrame:
        temp = np.asarray(temperature)
        hdd = np.maximum(18 - temp, 0)
        cdd = np.maximum(temp - 18, 0)
        features = {"temp": temp, "hdd": hdd, "cdd": cdd,
                    "temp2": temp ** 2}
        if include_time and hasattr(temperature.index, "hour"):
            features["hour"] = temperature.index.hour
            features["day_of_week"] = temperature.index.dayofweek
            features["month"] = temperature.index.month
            features["is_weekend"] = (temperature.index.dayofweek >= 5).astype(int)
        return pd.DataFrame(features, index=temperature.index)

    def fit(self, temperature: pd.Series, load: pd.Series,
            include_seasonality: bool = True) -> "WeatherLoadModel":
        X = self._build_features(temperature, include_time=include_seasonality)
        y = np.asarray(load)
        self._model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)),
        ])
        self._model.fit(X, y)
        return self

    def predict(self, temperature_forecast: pd.Series) -> pd.Series:
        if self._model is None:
            raise ValueError("Fit model first")
        X = self._build_features(temperature_forecast)
        preds = self._model.predict(X)
        return pd.Series(preds, index=temperature_forecast.index, name="load_forecast")

    def temperature_sensitivity(self, temperature: pd.Series,
                                  load: pd.Series) -> pd.Series:
        """Sensitivity (MW/°C) estimated by local polynomial regression."""
        temp_vals = np.asarray(temperature)
        load_vals = np.asarray(load)
        # Sort by temperature
        idx = np.argsort(temp_vals)
        temp_sorted = temp_vals[idx]
        load_sorted = load_vals[idx]
        # Rolling derivative
        window = max(20, len(temp_sorted) // 20)
        sensitivity = pd.Series(index=temperature.index, dtype=float)
        for i in range(len(temp_sorted)):
            lo = max(0, i - window // 2)
            hi = min(len(temp_sorted), i + window // 2)
            if hi - lo < 3:
                continue
            t_w = temp_sorted[lo:hi]
            l_w = load_sorted[lo:hi]
            slope = np.polyfit(t_w, l_w, 1)[0]
            sensitivity.iloc[i] = slope
        return sensitivity.fillna(method="bfill").fillna(method="ffill")


class WindPowerForecaster:
    """Wind power forecasting from wind speed using empirical power curve."""

    def __init__(self):
        self._cut_in: float = 3.0   # m/s
        self._rated: float = 12.0   # m/s
        self._cut_out: float = 25.0 # m/s
        self._rated_power: float = 1.0  # MW (normalized)
        self._poly_coefs: Optional[np.ndarray] = None

    def power_curve_fitting(self, wind_speed: np.ndarray,
                             power: np.ndarray) -> "WindPowerForecaster":
        """Fit piecewise cubic power curve."""
        ws = np.asarray(wind_speed)
        pw = np.asarray(power)
        mask = (ws >= self._cut_in) & (ws <= self._cut_out)
        if mask.sum() > 4:
            self._poly_coefs = np.polyfit(ws[mask], pw[mask], deg=3)
        self._rated_power = float(pw.max())
        return self

    def fit(self, wind_speed: pd.Series, wind_power: pd.Series,
            turbine_curve: Optional[dict] = None) -> "WindPowerForecaster":
        self.power_curve_fitting(np.asarray(wind_speed), np.asarray(wind_power))
        return self

    def predict(self, wind_speed_forecast: pd.Series) -> pd.Series:
        ws = np.asarray(wind_speed_forecast)
        if self._poly_coefs is not None:
            power = np.polyval(self._poly_coefs, ws)
        else:
            # Default cubic power curve
            power = np.where(ws < self._cut_in, 0,
                    np.where(ws >= self._rated, self._rated_power,
                    np.where(ws >= self._cut_out, 0,
                    self._rated_power * ((ws - self._cut_in) /
                                          (self._rated - self._cut_in)) ** 3)))
        power = np.clip(power, 0, self._rated_power)
        return pd.Series(power, index=wind_speed_forecast.index, name="wind_power_mw")

    def capacity_factor(self, wind_speed_forecast: pd.Series) -> float:
        pwr = self.predict(wind_speed_forecast)
        return float(pwr.mean() / (self._rated_power + 1e-8))


class SolarPowerForecaster:
    """Solar power forecasting from irradiance and temperature."""

    def __init__(self, panel_efficiency: float = 0.20,
                 temp_coefficient: float = -0.004):
        self.efficiency = panel_efficiency
        self.temp_coef = temp_coefficient
        self._scaler = StandardScaler()
        self._model = Ridge(alpha=1.0)
        self._fitted = False

    def fit(self, irradiance: pd.Series, solar_power: pd.Series,
            temperature: Optional[pd.Series] = None) -> "SolarPowerForecaster":
        X = self._build_features(irradiance, temperature)
        self._scaler.fit(X)
        self._model.fit(self._scaler.transform(X), np.asarray(solar_power))
        self._fitted = True
        return self

    def _build_features(self, irradiance: pd.Series,
                          temperature: Optional[pd.Series] = None) -> np.ndarray:
        ghi = np.asarray(irradiance).reshape(-1, 1)
        if temperature is not None:
            temp_correction = (1 + self.temp_coef * (np.asarray(temperature) - 25)).reshape(-1, 1)
        else:
            temp_correction = np.ones_like(ghi)
        return np.hstack([ghi, ghi ** 2, temp_correction, ghi * temp_correction])

    def predict(self, irradiance_forecast: pd.Series,
                temperature_forecast: Optional[pd.Series] = None) -> pd.Series:
        if not self._fitted:
            # Simple physical model
            power = np.asarray(irradiance_forecast) * self.efficiency / 1000
            return pd.Series(np.maximum(power, 0), index=irradiance_forecast.index)
        X = self._build_features(irradiance_forecast, temperature_forecast)
        preds = np.maximum(self._model.predict(self._scaler.transform(X)), 0)
        return pd.Series(preds, index=irradiance_forecast.index, name="solar_power_mw")

    def compute_irradiance_from_position(self, lat: float, lon: float,
                                          timestamps: pd.DatetimeIndex) -> pd.Series:
        """Estimate clear-sky GHI from solar position (simplified)."""
        if hasattr(timestamps, "tz_localize") and timestamps.tz is None:
            timestamps = timestamps.tz_localize("UTC")
        doy = timestamps.dayofyear.values
        hour = timestamps.hour.values + timestamps.minute.values / 60
        # Solar declination (degrees)
        decl = 23.45 * np.sin(np.radians(360 * (doy - 81) / 365))
        # Hour angle
        ha = (hour - 12) * 15
        # Solar altitude
        lat_r = np.radians(lat)
        decl_r = np.radians(decl)
        ha_r = np.radians(ha)
        sin_alt = (np.sin(lat_r) * np.sin(decl_r)
                   + np.cos(lat_r) * np.cos(decl_r) * np.cos(ha_r))
        altitude = np.degrees(np.arcsin(np.clip(sin_alt, -1, 1)))
        ghi = 1000 * np.maximum(np.sin(np.radians(altitude)), 0)
        return pd.Series(ghi, index=timestamps, name="GHI")
