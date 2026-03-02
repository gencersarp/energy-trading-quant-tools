import numpy as np

class SchwartzSmithModel:
    """Two-factor Schwartz-Smith model: short-term deviations + long-term equilibrium."""
    def __init__(self):
        self.kappa = 0.0 # reversion speed of short-term factor
        self.mu = 0.0 # drift of long-term factor
        self.sigma_chi = 0.0 # vol of short-term
        self.sigma_xi = 0.0 # vol of long-term
        self.rho = 0.0 # correlation
        
    def fit_kalman(self, prices: np.ndarray, dt: float = 1.0):
        """Placeholder for Kalman filter calibration.
        
        A real implementation would use pykalman or manual EKF.
        For now, simplistic moments matching to keep dependencies light.
        """
        returns = np.diff(np.log(prices))
        self.mu = np.mean(returns) / dt
        self.sigma_xi = np.std(returns) / np.sqrt(dt)
        self.sigma_chi = self.sigma_xi * 1.5
        self.kappa = 1.0
        self.rho = 0.3
        
    def simulate(self, x0: float, n_steps: int, n_paths: int = 1, dt: float = 1.0) -> np.ndarray:
        """Simulates paths using the two-factor model."""
        rng = np.random.default_rng()
        chi = np.zeros((n_steps + 1, n_paths))
        xi = np.zeros((n_steps + 1, n_paths))
        xi[0] = np.log(x0)
        
        for t in range(n_steps):
            z1 = rng.standard_normal(n_paths)
            z2 = rng.standard_normal(n_paths)
            dz_chi = z1
            dz_xi = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2
            
            chi[t+1] = chi[t] * np.exp(-self.kappa * dt) + self.sigma_chi * np.sqrt(dt) * dz_chi
            xi[t+1] = xi[t] + self.mu * dt + self.sigma_xi * np.sqrt(dt) * dz_xi
            
        return np.exp(chi + xi)