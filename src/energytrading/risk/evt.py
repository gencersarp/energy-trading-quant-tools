import numpy as np
from scipy.stats import genpareto


def fit_peaks_over_threshold(returns: np.ndarray, threshold: float) -> dict:
    """
    Fits Extreme Value Theory (EVT) Peaks-Over-Threshold (POT) using a 
    Generalized Pareto Distribution (GPD) on the left tail (losses).
    """
    losses = -returns
    exceedances = losses[losses > threshold] - threshold
    
    if len(exceedances) < 5:
        raise ValueError("Insufficient tail data to fit GPD stably.")
        
    shape, loc, scale = genpareto.fit(exceedances, floc=0)
    return {
        "shape": shape, 
        "scale": scale, 
        "threshold": threshold, 
        "n_exceed": len(exceedances),
        "total_n": len(returns)
    }


def evt_cvar(fit_dict: dict, alpha: float) -> float:
    """
    Computes the EVT-adjusted Conditional Value at Risk (Expected Shortfall) 
    at confidence level alpha (e.g., 0.99 for 99% CVaR).
    """
    xi = fit_dict['shape']
    beta = fit_dict['scale']
    u = fit_dict['threshold']
    Nu = fit_dict['n_exceed']
    n = fit_dict['total_n']
    
    # Probability of exceedance
    pu = Nu / n
    
    # Check if alpha is deep enough into the tail
    if (1 - alpha) > pu:
        raise ValueError("Alpha is too low. Must be inside the POT tail region.")
        
    # EVT VaR formulation
    var_alpha = u + (beta / xi) * (((n / Nu) * (1 - alpha))**(-xi) - 1)
    
    # EVT CVaR formulation
    cvar_alpha = (var_alpha + beta - xi * u) / (1 - xi)
    
    return float(cvar_alpha)