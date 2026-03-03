import pandas as pd


def apply_bid_ask_slippage(trades: pd.Series, prices: pd.Series, spread_bps: float = 0.005) -> pd.Series:
    """
    Applies bid/ask slippage to trade executions.
    spread_bps: Bid-Ask spread in basis points. Half spread applied per trade.
    """
    slippage_cost = trades.abs() * prices * (spread_bps / 2.0)
    return slippage_cost