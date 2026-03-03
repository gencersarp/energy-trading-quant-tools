import numpy as np


def regime_weighted_kelly(
    mu_base: float, var_base: float,
    mu_spike: float, var_spike: float,
    prob_base: float, prob_spike: float
) -> float:
    """
    Computes the optimal regime-conditional Kelly fraction using a second-order 
    Taylor expansion of the expected log-growth rate under state uncertainty.
    
    f* = E[mu] / E[variance] (adjusted for state mixture)
    """
    expected_mu = (prob_base * mu_base) + (prob_spike * mu_spike)
    expected_second_moment = (prob_base * (var_base + mu_base**2)) + (prob_spike * (var_spike + mu_spike**2))
    expected_var = expected_second_moment - expected_mu**2
    
    if expected_var <= 0:
        return 0.0
        
    return float(expected_mu / expected_var)


def regime_volatility_target(
    vol_base: float, vol_spike: float, 
    prob_base: float, prob_spike: float, 
    target_vol: float
) -> float:
    """
    Scales position size to maintain a constant volatility target, 
    conditional on the current HMM state probabilities.
    """
    expected_variance = (prob_base * vol_base**2) + (prob_spike * vol_spike**2)
    expected_vol = np.sqrt(expected_variance)
    
    if expected_vol <= 0:
        return 0.0
        
    return float(target_vol / expected_vol)