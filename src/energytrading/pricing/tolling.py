import numpy as np

def tolling_agreement_value(power_prices: np.ndarray, gas_prices: np.ndarray, 
                            heat_rate: float, vome: float, capacity: float) -> float:
    """
    Calculates the extrinsic value of a tolling agreement (virtual power plant).
    Assumes perfect foresight daily dispatch.
    
    vome: Variable Operation & Maintenance Expense
    """
    spreads = power_prices - (gas_prices * heat_rate) - vome
    
    # Plant is dispatched only when spark spread is positive
    dispatch = np.maximum(spreads, 0)
    
    return float(np.sum(dispatch) * capacity)