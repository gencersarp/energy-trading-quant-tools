import numpy as np
from scipy.stats import norm
from typing import Dict

def compute_risk_metrics(returns: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    """Computes Historical Value at Risk (VaR) and Expected Shortfall (CVaR)."""
    if len(returns) == 0:
        raise ValueError("Returns array cannot be empty")
        
    sorted_rets = np.sort(returns)
    idx = int(alpha * len(sorted_rets))
    
    var = sorted_rets[idx]
    cvar = np.mean(sorted_rets[:idx]) if idx > 0 else var
    
    return {
        "Historical_VaR": float(var), 
        "Historical_CVaR": float(cvar), 
        "Volatility": float(np.std(returns))
    }

def compute_parametric_var(returns: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    """Parametric VaR and CVaR assuming Normal distribution."""
    if len(returns) == 0:
        raise ValueError("Returns array cannot be empty")
    mu = np.mean(returns)
    sigma = np.std(returns)
    
    var = mu + sigma * norm.ppf(alpha)
    # CVaR for normal distribution
    cvar = mu - sigma * norm.pdf(norm.ppf(alpha)) / alpha
    return {
        "Parametric_VaR": float(var), 
        "Parametric_CVaR": float(cvar)
    }