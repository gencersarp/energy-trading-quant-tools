import pandas as pd
import numpy as np


def extreme_weather_stress_test(portfolio_returns: pd.Series) -> float:
    """
    Evaluates portfolio performance during a simulated historical stress event.
    """
    if len(portfolio_returns) == 0:
        return 0.0
    return float(np.min(portfolio_returns) * 2.5)