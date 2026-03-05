"""Weather data ingestion from open APIs and synthetic fallbacks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class WeatherStation:
    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float = 0.0
    country: str = ""


class OpenMeteoClient:
    """Free weather API client (open-meteo.com, no API key required)."""

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    def get_historical(self, lat: float, lon: float,
                        start_date: str, end_date: str,
                        variables: Optional[List[str]] = None) -> pd.DataFrame:
        if variables is None:
            variables = ["temperature_2m", "wind_speed_10m",
                         "surface_pressure", "shortwave_radiation"]
        try:
            import requests
            params = {
                "latitude": lat, "longitude": lon,
                "start_date": start_date, "end_date": end_date,
                "hourly": ",".join(variables),
                "timezone": "UTC",
            }
            resp = requests.get(self.BASE_URL, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                df = pd.DataFrame(data["hourly"])
                df["time"] = pd.to_datetime(df["time"])
                df.set_index("time", inplace=True)
                return df
        except Exception:
            pass
        return self._synthetic_weather(start_date, end_date, variables)

    def get_forecast(self, lat: float, lon: float,
                      horizon_days: int = 7) -> pd.DataFrame:
        try:
            import requests
            params = {
                "latitude": lat, "longitude": lon,
                "hourly": "temperature_2m,wind_speed_10m,shortwave_radiation",
                "forecast_days": horizon_days, "timezone": "UTC",
            }
            resp = requests.get(self.FORECAST_URL, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                df = pd.DataFrame(data["hourly"])
                df["time"] = pd.to_datetime(df["time"])
                df.set_index("time", inplace=True)
                return df
        except Exception:
            pass
        start = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        end = (pd.Timestamp.utcnow() + pd.Timedelta(days=horizon_days)).strftime("%Y-%m-%d")
        return self._synthetic_weather(start, end,
                                        ["temperature_2m", "wind_speed_10m", "shortwave_radiation"])

    def _synthetic_weather(self, start_date: str, end_date: str,
                             variables: List[str]) -> pd.DataFrame:
        """Synthetic weather data with realistic seasonality."""
        idx = pd.date_range(start_date, end_date, freq="h", tz="UTC")
        n = len(idx)
        rng = np.random.default_rng(42)
        data = {}
        t = np.arange(n) / 8760  # fraction of year
        for var in variables:
            if "temperature" in var:
                base = 10 + 15 * np.sin(2 * np.pi * t - np.pi / 2)
                data[var] = base + rng.normal(0, 3, n)
            elif "wind" in var:
                data[var] = np.abs(rng.normal(8, 4, n))
            elif "radiation" in var:
                solar = np.maximum(0, 500 * np.sin(2 * np.pi * t) *
                                   np.sin(np.pi * (idx.hour.values / 24)))
                data[var] = solar + rng.normal(0, 20, n).clip(min=0)
            else:
                data[var] = rng.normal(0, 1, n)
        return pd.DataFrame(data, index=idx)


class WeatherIngestionPipeline:
    """Orchestrates weather data fetch and alignment for multiple stations."""

    def __init__(self):
        self._stations: List[WeatherStation] = []
        self._client = OpenMeteoClient()

    def add_station(self, station: WeatherStation) -> None:
        self._stations.append(station)

    def fetch_all(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        result = {}
        for station in self._stations:
            df = self._client.get_historical(
                station.latitude, station.longitude, start_date, end_date)
            result[station.station_id] = df
        return result

    def merge_grid_data(self, stations_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Simple spatial average across all stations."""
        dfs = list(stations_data.values())
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs).groupby(level=0).mean()

    def compute_degree_days(self, temp_series: pd.Series,
                             base_temp: float = 18.0) -> pd.DataFrame:
        daily_temp = temp_series.resample("D").mean() if hasattr(temp_series.index, "freq") else temp_series
        hdd = np.maximum(base_temp - daily_temp, 0)
        cdd = np.maximum(daily_temp - base_temp, 0)
        return pd.DataFrame({"HDD": hdd, "CDD": cdd, "temperature": daily_temp})
