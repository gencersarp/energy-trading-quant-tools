import pandas as pd
from .engine import Strategy

class MeanReversionStrategy(Strategy):
    """Statistical arbitrage strategy based on Bollinger Bands."""
    def __init__(self, window: int = 24, z_score_threshold: float = 2.0):
        self.window = window
        self.threshold = z_score_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        roll_mean = data['close'].rolling(self.window).mean()
        roll_std = data['close'].rolling(self.window).std()
        
        # Avoid division by zero
        z_score = (data['close'] - roll_mean) / roll_std.replace(0, 1e-9)

        signals = pd.Series(0, index=data.index)
        signals[z_score < -self.threshold] = 1   # Long
        signals[z_score > self.threshold] = -1   # Short
        
        # Exit rules (reversion to mean)
        signals[(z_score > -0.5) & (z_score < 0.5)] = 0 
        
        return signals.ffill()