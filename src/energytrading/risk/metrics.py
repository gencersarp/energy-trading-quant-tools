import numpy as np
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
        "VaR": float(var), 
        "CVaR": float(cvar), 
        "Volatility": float(np.std(returns))
    }