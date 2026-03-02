import pandas as pd
import numpy as np
from typing import Protocol, Dict

class Strategy(Protocol):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        ...

class BacktestEngine:
    """Vectorized backtesting engine for quantitative energy strategies."""
    def __init__(self, data: pd.DataFrame, strategy: Strategy, initial_capital: float = 1e6, transaction_cost: float = 0.001):
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
    def run(self) -> pd.DataFrame:
        signals = self.strategy.generate_signals(self.data)
        
        # Calculate log returns instead of simple returns for robustness
        returns = np.log(self.data['close'] / self.data['close'].shift(1)).fillna(0)
        
        # Shift signal by 1 to prevent lookahead bias
        position = signals.shift(1).fillna(0)
        
        # Transaction costs: applied when position changes
        trades = position.diff().fillna(0).abs()
        tc = trades * self.transaction_cost
        
        pnl = position * returns - tc
        equity = self.initial_capital * np.exp(pnl.cumsum())
        
        return pd.DataFrame({
            'signal': signals,
            'position': position,
            'returns': returns,
            'tc': tc,
            'log_pnl': pnl,
            'equity': equity
        })
        
    def get_metrics(self, results: pd.DataFrame) -> Dict[str, float]:
        """Calculates key performance metrics."""
        pnl = results['log_pnl']
        returns = np.exp(pnl) - 1
        
        # Sharpe ratio (annualized, assuming daily data)
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Max Drawdown
        cum_ret = results['equity']
        running_max = cum_ret.cummax()
        drawdown = (cum_ret - running_max) / running_max
        max_dd = drawdown.min()
        
        # Win rate
        winning_trades = (pnl > 0).sum()
        total_trades = (pnl != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            "Total_Return": float((results['equity'].iloc[-1] / self.initial_capital) - 1),
            "Sharpe_Ratio": float(sharpe),
            "Max_Drawdown": float(max_dd),
            "Win_Rate": float(win_rate)
        }