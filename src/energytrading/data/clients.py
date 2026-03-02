import requests
import pandas as pd


class EIAClient:
    """Client for US Energy Information Administration (EIA) API."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.eia.gov/v2"

    def get_electricity_prices(self, frequency: str = "monthly") -> pd.DataFrame:
        """Fetches electricity retail sales and prices."""
        url = f"{self.base_url}/electricity/retail-sales/data/"
        params = {
            "api_key": self.api_key,
            "frequency": frequency,
            "data[0]": "price"
        }
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json().get("response", {}).get("data", [])
            return pd.DataFrame(data)
        
        raise ConnectionError(f"EIA API Error: {response.text}")


class ENTSOEClientStub:
    """Stub for European Network of Transmission System Operators for Electricity."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def get_day_ahead_prices(self, in_domain: str, start: str, end: str) -> pd.DataFrame:
        """
        Returns mock data as placeholder for real XML parsing.
        Use 'entsoe-py' in production for robust ENTSO-E parsing.
        """
        dates = pd.date_range(start, end, freq="h")
        return pd.DataFrame({"Price": 50.0}, index=dates)