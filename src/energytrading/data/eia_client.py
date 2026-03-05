"""EIA (U.S. Energy Information Administration) API client."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


class EIAClient:
    """
    Client for the EIA Open Data API v2.
    Falls back to synthetic data when the API key is not set or requests fail.

    API docs: https://www.eia.gov/opendata/
    """

    BASE_URL = "https://api.eia.gov/v2"

    # Common dataset paths
    DATASETS = {
        # Natural gas
        "gas_spot":         "natural-gas/pri/sum/data",
        "gas_storage":      "natural-gas/stor/wkly/data",
        "gas_production":   "natural-gas/prod/sum/data",
        "gas_consumption":  "natural-gas/cons/sum/data",
        # Electricity
        "electricity_price": "electricity/retail-sales/data",
        "electricity_gen":   "electricity/electric-power-operational-data/data",
        "electricity_load":  "electricity/rto/region-data/data",
        # Petroleum
        "crude_spot":       "petroleum/pri/spt/data",
        "crude_inventory":  "petroleum/stoc/wstk/data",
        "refinery_utilization": "petroleum/pnp/refp/data",
        # Coal
        "coal_price":       "coal/shipments/mines-by-util/data",
    }

    def __init__(self, api_key: Optional[str] = None,
                 timeout: int = 30, max_retries: int = 3):
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = None

    def _get_session(self):
        if not _HAS_REQUESTS:
            return None
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    def _request(self, path: str, params: Dict) -> Optional[Dict]:
        session = self._get_session()
        if session is None or not self.api_key:
            return None

        url = f"{self.BASE_URL}/{path}"
        params["api_key"] = self.api_key

        for attempt in range(self.max_retries):
            try:
                resp = session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"EIA API request failed: {e}")
                    return None
                time.sleep(2 ** attempt)
        return None

    def _parse_response(self, response: Optional[Dict]) -> pd.DataFrame:
        if response is None:
            return pd.DataFrame()
        try:
            data = response.get("response", {}).get("data", [])
            if not data:
                return pd.DataFrame()
            df = pd.DataFrame(data)
            if "period" in df.columns:
                df["period"] = pd.to_datetime(df["period"], errors="coerce")
                df = df.dropna(subset=["period"]).set_index("period").sort_index()
            return df
        except Exception:
            return pd.DataFrame()

    # ------------------------------------------------------------------ #
    # Natural gas
    # ------------------------------------------------------------------ #

    def get_gas_spot_price(self, start: str, end: str,
                           hub: str = "Henry Hub") -> pd.Series:
        """Henry Hub or other hub spot prices ($/MMBtu)."""
        params = {
            "frequency": "daily",
            "data[]": "value",
            "facets[series][]": "RNGWHHD",  # Henry Hub Daily
            "start": start, "end": end,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": 0, "length": 5000,
        }
        response = self._request(self.DATASETS["gas_spot"], params)
        df = self._parse_response(response)

        if df.empty:
            return self._synthetic_gas_price(start, end)

        if "value" in df.columns:
            return pd.to_numeric(df["value"], errors="coerce").dropna()
        return self._synthetic_gas_price(start, end)

    def get_gas_storage(self, start: str, end: str,
                        region: str = "National") -> pd.Series:
        """Weekly working gas in storage (Bcf)."""
        params = {
            "frequency": "weekly",
            "data[]": "value",
            "facets[series][]": "NW2_EPG0_SWO_R48_BCF",
            "start": start, "end": end,
            "offset": 0, "length": 2000,
        }
        response = self._request(self.DATASETS["gas_storage"], params)
        df = self._parse_response(response)

        if df.empty:
            return self._synthetic_storage(start, end)

        if "value" in df.columns:
            return pd.to_numeric(df["value"], errors="coerce").dropna()
        return self._synthetic_storage(start, end)

    # ------------------------------------------------------------------ #
    # Electricity
    # ------------------------------------------------------------------ #

    def get_electricity_price(self, start: str, end: str,
                              sector: str = "commercial",
                              state: str = "US") -> pd.Series:
        """Retail electricity price (cents/kWh)."""
        params = {
            "frequency": "monthly",
            "data[]": "price",
            "facets[stateid][]": state,
            "facets[sectorid][]": sector,
            "start": start, "end": end,
            "offset": 0, "length": 1000,
        }
        response = self._request(self.DATASETS["electricity_price"], params)
        df = self._parse_response(response)

        if df.empty:
            return self._synthetic_electricity_price(start, end)

        if "price" in df.columns:
            return pd.to_numeric(df["price"], errors="coerce").dropna()
        return self._synthetic_electricity_price(start, end)

    def get_electricity_generation(self, start: str, end: str,
                                   fuel_type: str = "all",
                                   state: str = "US") -> pd.DataFrame:
        """Electricity generation by fuel type (MWh)."""
        params = {
            "frequency": "monthly",
            "data[]": "generation",
            "facets[location][]": state,
            "start": start, "end": end,
            "offset": 0, "length": 2000,
        }
        response = self._request(self.DATASETS["electricity_gen"], params)
        df = self._parse_response(response)

        if df.empty:
            return self._synthetic_generation_mix(start, end)
        return df

    def get_rto_load(self, start: str, end: str,
                     rto: str = "ERCO") -> pd.Series:
        """Regional transmission organization hourly load (MWh)."""
        params = {
            "frequency": "hourly",
            "data[]": "value",
            "facets[respondent][]": rto,
            "facets[type][]": "D",  # demand
            "start": start, "end": end,
            "offset": 0, "length": 5000,
        }
        response = self._request(self.DATASETS["electricity_load"], params)
        df = self._parse_response(response)

        if df.empty:
            return self._synthetic_load(start, end)

        if "value" in df.columns:
            return pd.to_numeric(df["value"], errors="coerce").dropna()
        return self._synthetic_load(start, end)

    # ------------------------------------------------------------------ #
    # Petroleum
    # ------------------------------------------------------------------ #

    def get_crude_spot_price(self, start: str, end: str,
                             crude_type: str = "WTI") -> pd.Series:
        """WTI or Brent spot price ($/bbl)."""
        series_id = "RWTC" if crude_type == "WTI" else "RBRTE"
        params = {
            "frequency": "daily",
            "data[]": "value",
            "facets[series][]": series_id,
            "start": start, "end": end,
            "offset": 0, "length": 5000,
        }
        response = self._request(self.DATASETS["crude_spot"], params)
        df = self._parse_response(response)

        if df.empty:
            return self._synthetic_crude_price(start, end, crude_type)

        if "value" in df.columns:
            return pd.to_numeric(df["value"], errors="coerce").dropna()
        return self._synthetic_crude_price(start, end, crude_type)

    def get_crude_inventory(self, start: str, end: str) -> pd.Series:
        """Weekly crude oil stocks (Mbbl)."""
        params = {
            "frequency": "weekly",
            "data[]": "value",
            "facets[series][]": "WCRSTUS1",
            "start": start, "end": end,
            "offset": 0, "length": 2000,
        }
        response = self._request(self.DATASETS["crude_inventory"], params)
        df = self._parse_response(response)

        if df.empty:
            return self._synthetic_inventory(start, end, level=430_000)

        if "value" in df.columns:
            return pd.to_numeric(df["value"], errors="coerce").dropna()
        return self._synthetic_inventory(start, end, level=430_000)

    # ------------------------------------------------------------------ #
    # Synthetic fallbacks
    # ------------------------------------------------------------------ #

    @staticmethod
    def _synthetic_gas_price(start: str, end: str) -> pd.Series:
        idx = pd.date_range(start, end, freq="B")
        np.random.seed(42)
        prices = np.exp(np.random.normal(np.log(3.0), 0.02, len(idx)).cumsum() * 0 +
                        np.log(3.0) + np.random.normal(0, 0.05, len(idx)))
        prices = np.maximum(prices, 1.0)
        return pd.Series(prices, index=idx, name="gas_spot_henry_hub")

    @staticmethod
    def _synthetic_storage(start: str, end: str) -> pd.Series:
        idx = pd.date_range(start, end, freq="W")
        np.random.seed(43)
        t = np.arange(len(idx))
        storage = 2500 + 500 * np.sin(2 * np.pi * t / 52) + \
                  np.random.normal(0, 30, len(idx)).cumsum() * 0.1
        return pd.Series(np.maximum(storage, 500), index=idx, name="gas_storage_bcf")

    @staticmethod
    def _synthetic_electricity_price(start: str, end: str) -> pd.Series:
        idx = pd.date_range(start, end, freq="MS")
        np.random.seed(44)
        prices = 11 + np.random.normal(0, 0.5, len(idx))
        return pd.Series(prices, index=idx, name="electricity_price_cents_kwh")

    @staticmethod
    def _synthetic_generation_mix(start: str, end: str) -> pd.DataFrame:
        idx = pd.date_range(start, end, freq="MS")
        np.random.seed(45)
        n = len(idx)
        return pd.DataFrame({
            "natural_gas": np.random.normal(150_000, 10_000, n),
            "coal": np.random.normal(80_000, 5_000, n),
            "nuclear": np.random.normal(65_000, 1_000, n),
            "wind": np.random.normal(30_000, 5_000, n),
            "solar": np.maximum(np.random.normal(20_000, 8_000, n), 0),
            "hydro": np.random.normal(25_000, 3_000, n),
        }, index=idx)

    @staticmethod
    def _synthetic_load(start: str, end: str) -> pd.Series:
        idx = pd.date_range(start, end, freq="H")
        np.random.seed(46)
        t = np.arange(len(idx))
        load = (40_000 + 8_000 * np.sin(2 * np.pi * t / 8760) +   # seasonal
                5_000 * np.sin(2 * np.pi * t / 24) +                # daily
                np.random.normal(0, 1_000, len(idx)))
        return pd.Series(np.maximum(load, 10_000), index=idx, name="rto_load_mwh")

    @staticmethod
    def _synthetic_crude_price(start: str, end: str,
                                crude_type: str = "WTI") -> pd.Series:
        idx = pd.date_range(start, end, freq="B")
        np.random.seed(47)
        base = 70.0 if crude_type == "WTI" else 75.0
        log_prices = np.log(base) + np.cumsum(np.random.normal(0, 0.015, len(idx)))
        return pd.Series(np.exp(log_prices), index=idx,
                         name=f"crude_{crude_type.lower()}_spot")

    @staticmethod
    def _synthetic_inventory(start: str, end: str,
                              level: float = 430_000) -> pd.Series:
        idx = pd.date_range(start, end, freq="W")
        np.random.seed(48)
        inv = level + np.random.normal(0, 5_000, len(idx)).cumsum() * 0.05
        return pd.Series(np.maximum(inv, 200_000), index=idx,
                         name="crude_inventory_mbbl")
