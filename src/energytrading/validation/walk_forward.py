import pandas as pd
from typing import Callable, Any


def walk_forward_validation(
    data: pd.DataFrame, 
    train_window: int, 
    test_window: int, 
    model_fitter: Callable[[pd.DataFrame], Any],
    strategy_runner: Callable[[Any, pd.DataFrame], pd.Series]
) -> pd.Series:
    """Generic walk-forward cross-validation for time series strategies."""
    results = []
    for i in range(0, len(data) - train_window - test_window, test_window):
        train_set = data.iloc[i : i + train_window]
        test_set = data.iloc[i + train_window : i + train_window + test_window]
        
        model = model_fitter(train_set)
        pnl = strategy_runner(model, test_set)
        results.append(pnl)
        
    if not results:
        return pd.Series(dtype=float)
    return pd.concat(results)