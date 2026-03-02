import numpy as np
from scipy.stats import norm


def margrabe_spark_spread(
    f_power: float, 
    f_gas: float, 
    vol_power: float, 
    vol_gas: float, 
    rho: float, 
    tau: float, 
    heat_rate: float, 
    r: float = 0.0
) -> float:
    """
    Prices a spark spread option using Margrabe's formula.
    Payoff: Max(Power - HeatRate * Gas, 0)
    """
    if tau <= 0:
        return max(f_power - heat_rate * f_gas, 0.0)
        
    f2 = heat_rate * f_gas
    vol = np.sqrt(vol_power**2 + vol_gas**2 - 2 * rho * vol_power * vol_gas)
    
    if vol <= 0:
        return max(f_power - f2, 0.0)
    
    d1 = (np.log(f_power / f2) + 0.5 * vol**2 * tau) / (vol * np.sqrt(tau))
    d2 = d1 - vol * np.sqrt(tau)
    
    call = np.exp(-r * tau) * (f_power * norm.cdf(d1) - f2 * norm.cdf(d2))
    return call