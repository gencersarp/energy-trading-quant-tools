import numpy as np

class BatteryStorageOptimizer:
    """Evaluates battery storage dispatch optimization."""
    def __init__(self, capacity: float, max_charge: float, max_discharge: float, efficiency: float = 0.9):
        self.capacity = capacity
        self.charge_rate = max_charge
        self.discharge_rate = max_discharge
        self.efficiency = efficiency

    def optimize_schedule(self, prices: np.ndarray) -> np.ndarray:
        """
        Simple greedy dispatch under perfect foresight for backtesting.
        (Production uses LP/MIP solvers like PuLP/Gurobi)
        """
        schedule = np.zeros_like(prices)
        soc = 0.0
        mean_price = np.mean(prices)
        
        for i, p in enumerate(prices):
            if p > mean_price and soc > 0:
                discharge = min(self.discharge_rate, soc)
                schedule[i] = -discharge
                soc -= discharge
            elif p < mean_price and soc < self.capacity:
                charge = min(self.charge_rate, self.capacity - soc)
                schedule[i] = charge
                soc += charge * self.efficiency
        return schedule