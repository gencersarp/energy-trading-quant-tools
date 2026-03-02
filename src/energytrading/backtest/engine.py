import numpy as np
import pandas as pd
from typing import Protocol

class Strategy(Protocol):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        ...

class BacktestEngine:
    """Vectorized backtesting engine for quantitative energy strategies."""
    def __init__(self, data: pd.DataFrame, strategy: Strategy, initial_capital: float = 1e6):
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        
    def run(self) -> pd.DataFrame:
        signals = self.strategy.generate_signals(self.data)
        
        # Calculate log returns instead of simple returns for robustness
        returns = np.log(self.data['close'] / self.data['close'].shift(1)).fillna(0)
        
        pnl = signals.shift(1).fillna(0) * returns
        equity = self.initial_capital * np.exp(pnl.cumsum())
        
        return pd.DataFrame({
            'signal': signals,
            'returns': returns,
            'log_pnl': pnl,
            'equity': equity
        })