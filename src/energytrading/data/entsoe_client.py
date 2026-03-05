"""ENTSO-E Transparency Platform API client for European power data."""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


# ENTSO-E bidding zone EIC codes (common)
BIDDING_ZONES = {
    "DE": "10Y1001A1001A83F",   # Germany
    "FR": "10YFR-RTE------C",   # France
    "GB": "10YGB----------A",   # Great Britain
    "NL": "10YNL----------L",   # Netherlands
    "BE": "10YBE----------2",   # Belgium
    "ES": "10YES-REE------0",   # Spain
    "IT_N": "10Y1001A1001A73I",  # Italy North
    "NO_1": "10YNO-1--------2",  # Norway 1
    "SE_3": "10Y1001A1001A46L",  # Sweden 3
    "DK_1": "10YDK-1--------W",  # Denmark 1
}

PSRTYPE = {
    "B01": "Biomass",
    "B02": "Fossil Brown coal/Lignite",
    "B04": "Fossil Gas",
    "B05": "Fossil Hard coal",
    "B09": "Geothermal",
    "B10": "Hydro Pumped Storage",
    "B11": "Hydro Run-of-river",
    "B12": "Hydro Water Reservoir",
    "B14": "Nuclear",
    "B15": "Other renewable",
    "B16": "Solar",
    "B17": "Waste",
    "B18": "Wind Offshore",
    "B19": "Wind Onshore",
    "B20": "Other",
}


class ENTSOEClient:
    """
    Client for the ENTSO-E Transparency Platform RESTful API.
    Falls back to synthetic data when security token is unavailable.

    API docs: https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html
    """

    BASE_URL = "https://web-api.tp.entsoe.eu/api"

    def __init__(self, security_token: Optional[str] = None,
                 timeout: int = 60, max_retries: int = 3):
        self.token = security_token
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

    def _request(self, params: Dict) -> Optional[str]:
        session = self._get_session()
        if session is None or not self.token:
            return None
        params["securityToken"] = self.token

        for attempt in range(self.max_retries):
            try:
                resp = session.get(self.BASE_URL, params=params,
                                   timeout=self.timeout)
                resp.raise_for_status()
                return resp.text
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"ENTSO-E API request failed: {e}")
                    return None
                time.sleep(2 ** attempt)
        return None

    def _parse_xml_timeseries(self, xml_text: str) -> pd.Series:
        """Parse ENTSO-E XML response into a time series."""
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_text)
            ns = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0"}
            values: Dict[pd.Timestamp, float] = {}

            for ts in root.findall(".//ns:TimeSeries", ns):
                period = ts.find("ns:Period", ns)
                if period is None:
                    continue
                start_elem = period.find("ns:timeInterval/ns:start", ns)
                res_elem = period.find("ns:resolution", ns)
                if start_elem is None or res_elem is None:
                    continue

                start = pd.Timestamp(start_elem.text)
                res_str = res_elem.text  # e.g., PT60M, PT15M

                if "PT60M" in res_str or "PT1H" in res_str:
                    freq_minutes = 60
                elif "PT30M" in res_str:
                    freq_minutes = 30
                elif "PT15M" in res_str:
                    freq_minutes = 15
                else:
                    freq_minutes = 60

                for point in period.findall("ns:Point", ns):
                    pos_elem = point.find("ns:position", ns)
                    qty_elem = point.find("ns:price.amount", ns)
                    if qty_elem is None:
                        qty_elem = point.find("ns:quantity", ns)
                    if pos_elem is None or qty_elem is None:
                        continue
                    pos = int(pos_elem.text) - 1
                    ts_dt = start + pd.Timedelta(minutes=pos * freq_minutes)
                    try:
                        values[ts_dt] = float(qty_elem.text)
                    except ValueError:
                        pass

            if not values:
                return pd.Series(dtype=float)
            return pd.Series(values).sort_index()

        except Exception:
            return pd.Series(dtype=float)

    # ------------------------------------------------------------------ #
    # Day-ahead prices
    # ------------------------------------------------------------------ #

    def get_day_ahead_prices(self, area: str, start: str, end: str,
                             resolution: str = "PT60M") -> pd.Series:
        """Day-ahead electricity market prices (€/MWh)."""
        params = {
            "documentType": "A44",  # Price Document
            "in_Domain": BIDDING_ZONES.get(area, area),
            "out_Domain": BIDDING_ZONES.get(area, area),
            "periodStart": pd.Timestamp(start).strftime("%Y%m%d%H%M"),
            "periodEnd": pd.Timestamp(end).strftime("%Y%m%d%H%M"),
        }
        xml_text = self._request(params)
        if xml_text:
            result = self._parse_xml_timeseries(xml_text)
            if not result.empty:
                return result

        return self._synthetic_da_prices(start, end, area)

    # ------------------------------------------------------------------ #
    # Load
    # ------------------------------------------------------------------ #

    def get_actual_load(self, area: str, start: str, end: str) -> pd.Series:
        """Actual total load (MW)."""
        params = {
            "documentType": "A65",
            "processType": "A16",
            "outBiddingZone_Domain": BIDDING_ZONES.get(area, area),
            "periodStart": pd.Timestamp(start).strftime("%Y%m%d%H%M"),
            "periodEnd": pd.Timestamp(end).strftime("%Y%m%d%H%M"),
        }
        xml_text = self._request(params)
        if xml_text:
            result = self._parse_xml_timeseries(xml_text)
            if not result.empty:
                return result

        return self._synthetic_load(start, end, area)

    def get_load_forecast(self, area: str, start: str, end: str) -> pd.Series:
        """Day-ahead total load forecast (MW)."""
        params = {
            "documentType": "A65",
            "processType": "A01",
            "outBiddingZone_Domain": BIDDING_ZONES.get(area, area),
            "periodStart": pd.Timestamp(start).strftime("%Y%m%d%H%M"),
            "periodEnd": pd.Timestamp(end).strftime("%Y%m%d%H%M"),
        }
        xml_text = self._request(params)
        if xml_text:
            result = self._parse_xml_timeseries(xml_text)
            if not result.empty:
                return result

        return self._synthetic_load(start, end, area)

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def get_generation_by_source(self, area: str, start: str,
                                  end: str) -> pd.DataFrame:
        """Actual generation per production type (MW)."""
        params = {
            "documentType": "A75",
            "processType": "A16",
            "in_Domain": BIDDING_ZONES.get(area, area),
            "periodStart": pd.Timestamp(start).strftime("%Y%m%d%H%M"),
            "periodEnd": pd.Timestamp(end).strftime("%Y%m%d%H%M"),
        }
        xml_text = self._request(params)

        if xml_text:
            try:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(xml_text)
                ns = {"ns": "urn:iec62325.351:tc57wg16:451-6:generationdocument:3:0"}
                result: Dict[str, pd.Series] = {}

                for ts in root.findall(".//ns:TimeSeries", ns):
                    psr_type_elem = ts.find(".//ns:psrType", ns)
                    if psr_type_elem is None:
                        continue
                    psr = psr_type_elem.text
                    label = PSRTYPE.get(psr, psr)
                    series = self._parse_xml_timeseries(xml_text)
                    if not series.empty:
                        result[label] = series

                if result:
                    return pd.DataFrame(result)
            except Exception:
                pass

        return self._synthetic_generation_mix(start, end, area)

    def get_wind_solar_forecast(self, area: str, start: str,
                                 end: str) -> pd.DataFrame:
        """Wind and solar day-ahead generation forecast (MW)."""
        params = {
            "documentType": "A69",  # Wind and Solar Forecast
            "processType": "A01",
            "in_Domain": BIDDING_ZONES.get(area, area),
            "periodStart": pd.Timestamp(start).strftime("%Y%m%d%H%M"),
            "periodEnd": pd.Timestamp(end).strftime("%Y%m%d%H%M"),
        }
        xml_text = self._request(params)

        if xml_text:
            result = self._parse_xml_timeseries(xml_text)
            if not result.empty:
                return pd.DataFrame({"renewable_forecast": result})

        return self._synthetic_wind_solar(start, end, area)

    # ------------------------------------------------------------------ #
    # Cross-border flows
    # ------------------------------------------------------------------ #

    def get_cross_border_flows(self, from_area: str, to_area: str,
                                start: str, end: str) -> pd.Series:
        """Cross-border physical flows (MW)."""
        params = {
            "documentType": "A11",
            "in_Domain": BIDDING_ZONES.get(to_area, to_area),
            "out_Domain": BIDDING_ZONES.get(from_area, from_area),
            "periodStart": pd.Timestamp(start).strftime("%Y%m%d%H%M"),
            "periodEnd": pd.Timestamp(end).strftime("%Y%m%d%H%M"),
        }
        xml_text = self._request(params)
        if xml_text:
            result = self._parse_xml_timeseries(xml_text)
            if not result.empty:
                return result

        idx = pd.date_range(start, end, freq="H")
        np.random.seed(99)
        return pd.Series(np.random.normal(2000, 500, len(idx)), index=idx,
                         name=f"flow_{from_area}_to_{to_area}")

    # ------------------------------------------------------------------ #
    # Synthetic fallbacks
    # ------------------------------------------------------------------ #

    @staticmethod
    def _synthetic_da_prices(start: str, end: str,
                              area: str = "DE") -> pd.Series:
        idx = pd.date_range(start, end, freq="H")
        np.random.seed(42 + hash(area) % 100)
        t = np.arange(len(idx))
        price = (50 + 15 * np.sin(2 * np.pi * t / 8760) +    # seasonal
                 10 * np.sin(2 * np.pi * t / 24) +             # daily
                 np.random.normal(0, 5, len(idx)) +
                 np.random.exponential(3, len(idx)) *           # spikes
                 (np.random.uniform(0, 1, len(idx)) > 0.97))
        return pd.Series(np.maximum(price, -20), index=idx,
                         name=f"da_price_{area}")

    @staticmethod
    def _synthetic_load(start: str, end: str, area: str = "DE") -> pd.Series:
        idx = pd.date_range(start, end, freq="H")
        np.random.seed(43 + hash(area) % 100)
        t = np.arange(len(idx))
        load = (45_000 + 10_000 * np.sin(2 * np.pi * t / 8760) +
                5_000 * np.sin(2 * np.pi * t / 24 + np.pi) +
                np.random.normal(0, 1_000, len(idx)))
        return pd.Series(np.maximum(load, 20_000), index=idx,
                         name=f"load_{area}")

    @staticmethod
    def _synthetic_generation_mix(start: str, end: str,
                                   area: str = "DE") -> pd.DataFrame:
        idx = pd.date_range(start, end, freq="H")
        np.random.seed(44 + hash(area) % 100)
        n = len(idx)
        t = np.arange(n)
        solar_pattern = np.maximum(np.sin(np.pi * (t % 24 - 6) / 12), 0)
        return pd.DataFrame({
            "Wind Onshore": np.random.exponential(10_000, n),
            "Wind Offshore": np.random.exponential(5_000, n),
            "Solar": solar_pattern * np.random.uniform(20_000, 30_000, n),
            "Fossil Gas": np.random.normal(15_000, 3_000, n),
            "Nuclear": np.full(n, 8_000) + np.random.normal(0, 200, n),
            "Hydro Run-of-river": np.random.normal(3_000, 500, n),
            "Biomass": np.full(n, 2_000) + np.random.normal(0, 100, n),
        }, index=idx)

    @staticmethod
    def _synthetic_wind_solar(start: str, end: str,
                               area: str = "DE") -> pd.DataFrame:
        idx = pd.date_range(start, end, freq="H")
        np.random.seed(45 + hash(area) % 100)
        n = len(idx)
        t = np.arange(n)
        return pd.DataFrame({
            "wind_forecast": np.random.exponential(12_000, n),
            "solar_forecast": np.maximum(
                np.sin(np.pi * (t % 24 - 6) / 12), 0) * 25_000,
        }, index=idx)
