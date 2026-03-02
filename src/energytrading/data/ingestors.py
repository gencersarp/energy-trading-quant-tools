import pandas as pd
import numpy as np

def fetch_yfinance_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetches data from Yahoo Finance. Returns mock data if yfinance is not installed."""
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False)
        return df
    except ImportError:
        dates = pd.date_range(start, end, freq='B')
        df = pd.DataFrame(index=dates, data={'Close': np.random.lognormal(0, 0.02, len(dates)).cumprod() * 100})
        return df

def align_time_series(df_hourly: pd.DataFrame, df_daily: pd.DataFrame) -> pd.DataFrame:
    """Aligns daily gas prices to hourly power prices via forward fill."""
    df_daily_resampled = df_daily.reindex(df_hourly.index, method='ffill')
    return pd.concat([df_hourly, df_daily_resampled], axis=1)