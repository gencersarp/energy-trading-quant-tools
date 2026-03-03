import pandas as pd


class VolatilityBreakout:
    """Generates signals based on Donchian Channels / Volatility Breakouts."""
    def __init__(self, window: int = 24, z_multiplier: float = 2.0):
        self.window = window
        self.z_multiplier = z_multiplier

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        """
        1: Long (Price breaks above rolling mean + Z * rolling std)
        -1: Short (Price breaks below rolling mean - Z * rolling std)
        0: Flat
        """
        roll_mean = prices.rolling(window=self.window).mean()
        roll_std = prices.rolling(window=self.window).std()
        
        upper_band = roll_mean + self.z_multiplier * roll_std
        lower_band = roll_mean - self.z_multiplier * roll_std
        
        signals = pd.Series(0, index=prices.index)
        signals[prices > upper_band] = 1
        signals[prices < lower_band] = -1
        
        return signals