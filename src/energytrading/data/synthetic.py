import numpy as np
import pandas as pd

def generate_synthetic_power_data(days: int = 365) -> pd.DataFrame:
    """Generates synthetic hourly power prices with seasonality, daily cycles, and spikes."""
    t = np.arange(days * 24)
    
    # Base seasonal + daily cycle
    seasonal = 50 + 10 * np.sin(2 * np.pi * t / (365 * 24))
    daily = 15 * np.sin(2 * np.pi * t / 24 - np.pi/2)
    
    # Jumps and noise
    noise = np.random.normal(0, 5, size=len(t))
    jumps = np.random.poisson(0.005, size=len(t)) * np.random.normal(100, 40, size=len(t))

    prices = seasonal + daily + noise + jumps
    dates = pd.date_range("2024-01-01", periods=len(t), freq="H")

    return pd.DataFrame({
        "datetime": dates, 
        "price": np.maximum(prices, 0.0) # Prices generally > 0
    }).set_index("datetime")