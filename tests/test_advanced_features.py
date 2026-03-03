import pandas as pd

from energytrading.signals.statistical import volatility_breakout, weather_deviation_signal
from energytrading.backtest.execution_realism import apply_bid_ask_slippage
from energytrading.risk.scenarios import extreme_weather_stress_test


def test_volatility_breakout():
    prices = pd.Series([100.0, 102.0, 101.0, 100.0, 105.0, 120.0, 125.0, 120.0])
    signal = volatility_breakout(prices, window=3, multiplier=1.0)
    assert len(signal) == len(prices)


def test_weather_signal():
    actual = pd.Series([70.0, 72.0, 85.0, 90.0])
    normal = pd.Series([70.0, 70.0, 70.0, 70.0])
    signal = weather_deviation_signal(actual, normal, threshold=10.0)
    assert signal.iloc[2] == 1


def test_slippage():
    trades = pd.Series([10.0, -10.0, 0.0])
    prices = pd.Series([50.0, 50.0, 50.0])
    cost = apply_bid_ask_slippage(trades, prices, spread_bps=0.01)
    assert cost.iloc[0] == 2.5


def test_stress_test():
    returns = pd.Series([-0.01, -0.05, 0.02])
    shock = extreme_weather_stress_test(returns)
    assert shock < -0.1