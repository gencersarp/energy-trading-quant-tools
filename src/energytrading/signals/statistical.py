import pandas as pd


def volatility_breakout(prices: pd.Series, window: int = 20, multiplier: float = 2.0) -> pd.Series:
    """Generates a breakout signal when current return exceeds rolling vol."""
    returns = prices.pct_change()
    roll_vol = returns.rolling(window).std()
    
    signal = pd.Series(0, index=prices.index)
    signal[returns > roll_vol * multiplier] = 1
    signal[returns < -roll_vol * multiplier] = -1
    return signal


def weather_deviation_signal(actual_temp: pd.Series, normal_temp: pd.Series, threshold: float = 5.0) -> pd.Series:
    """Energy demand proxy: buy power if extreme weather (heatwave/freeze)."""
    deviation = actual_temp - normal_temp
    signal = pd.Series(0, index=actual_temp.index)
    signal[deviation > threshold] = 1
    signal[deviation < -threshold] = 1
    return signal