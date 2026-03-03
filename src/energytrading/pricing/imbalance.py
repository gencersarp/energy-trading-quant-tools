def dual_pricing_cashout(
    position_mwh: float,
    market_price: float,
    system_state: str,
    penalty_markup: float = 0.20
) -> float:
    """
    Simulates the penalizing cash-out mechanism in European power markets 
    (e.g., EPEX/Elexion) when a trader holds an open position past Gate Closure.
    
    If your open position exacerbates the grid's overall imbalance, you are heavily penalized.
    
    Args:
        position_mwh: + (long/shortage of physical power), - (short/excess generation)
        market_price: The underlying spot reference price
        system_state: "SHORT" (grid needs power) or "LONG" (grid has too much power)
        penalty_markup: The percentage penalty applied to the balancing price
        
    Returns:
        float: Total cash settlement value (negative means trader pays, positive means trader receives)
    """
    # Note: Long financial position means you BOUGHT power. 
    # If you bought power and don't physically consume it, you are physically LONG.
    
    if position_mwh > 0:  # We have excess power to dump on the grid
        if system_state == "LONG":
            # Grid has too much power, and we are dumping more. Punitive sell price.
            return position_mwh * market_price * (1 - penalty_markup)
        else:
            # Grid needs power, we help the grid. Normal or slight premium price.
            return position_mwh * market_price

    elif position_mwh < 0:  # We are short power (we owe power to the grid)
        if system_state == "SHORT":
            # Grid is short power, and we are withdrawing/defaulting. Punitive buy price.
            return position_mwh * market_price * (1 + penalty_markup)
        else:
            # Grid has excess power, our shortage actually helps. Normal price.
            return position_mwh * market_price
            
    return 0.0