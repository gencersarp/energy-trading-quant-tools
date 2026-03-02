import numpy as np
import pulp


class BatteryStorageOptimizer:
    """Production-grade battery storage optimization using Linear Programming."""
    def __init__(self, capacity: float, max_charge: float, max_discharge: float, 
                 efficiency: float = 0.9, degradation_cost: float = 0.5):
        self.capacity = capacity
        self.max_charge = max_charge
        self.max_discharge = max_discharge
        self.efficiency = efficiency
        self.degradation_cost = degradation_cost

    def optimize_schedule(self, prices: np.ndarray) -> np.ndarray:
        """
        Optimizes battery dispatch over a price horizon using LP.
        Objective: Maximize Arbitrage Revenue - Degradation Costs.
        """
        t_len = len(prices)
        prob = pulp.LpProblem("Battery_Optimization", pulp.LpMaximize)
        
        charge = pulp.LpVariable.dicts("Charge", range(t_len), lowBound=0, upBound=self.max_charge)
        discharge = pulp.LpVariable.dicts("Discharge", range(t_len), lowBound=0, upBound=self.max_discharge)
        soc = pulp.LpVariable.dicts("SoC", range(t_len + 1), lowBound=0, upBound=self.capacity)
        
        # Objective: Profit = (Discharge * Price) - (Charge * Price) - Degradation
        prob += pulp.lpSum([
            prices[t] * discharge[t] - 
            prices[t] * charge[t] - 
            self.degradation_cost * (charge[t] + discharge[t]) 
            for t in range(t_len)
        ])
        
        # Constraints
        prob += (soc[0] == 0.0) # Start empty
        for t in range(t_len):
            prob += (soc[t + 1] == soc[t] + charge[t] * self.efficiency - discharge[t] / self.efficiency)
            
        # Ensure we end empty to realize full PnL (optional, but good practice)
        prob += (soc[t_len] == 0.0)
            
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        return np.array([charge[t].varValue - discharge[t].varValue for t in range(t_len)])