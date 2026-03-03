import numpy as np
import pandas as pd


def order_book_imbalance(bid_vol: pd.Series, ask_vol: pd.Series) -> pd.Series:
    """
    Calculates the Order Book Imbalance (OBI). 
    Range is [-1, 1]. A highly positive OBI predicts upward short-term price pressure.
    
    Formula: (Bid Vol - Ask Vol) / (Bid Vol + Ask Vol)
    """
    total_vol = bid_vol + ask_vol
    # Replace 0 with NaN to avoid division by zero, then fill with 0.0
    obi = (bid_vol - ask_vol) / total_vol.replace(0, np.nan)
    return obi.fillna(0.0)


def calculate_vwap(prices: np.ndarray, volumes: np.ndarray) -> float:
    """
    Calculates Volume Weighted Average Price (VWAP) for a given intraday trading window.
    Crucial for benchmarking execution algorithms in EPEX/Nord Pool intraday markets.
    """
    if np.sum(volumes) == 0:
        return 0.0
    return float(np.average(prices, weights=volumes))


def tick_bid_ask_bounce(price_ticks: pd.Series) -> pd.Series:
    """
    Roll method for estimating effective bid-ask spread from tick data alone.
    Covariance of consecutive price changes.
    """
    dp = price_ticks.diff().dropna()
    dp_lag = dp.shift(1).dropna()
    cov = np.cov(dp[1:], dp_lag)[0, 1]
    
    if cov >= 0:
        return pd.Series(0.0, index=price_ticks.index)
        
    spread = 2 * np.sqrt(-cov)
    return pd.Series(spread, index=price_ticks.index)