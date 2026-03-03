import numpy as np


class MertonJumpDiffusion:
    """
    Simulates Merton's Jump Diffusion model. 
    Crucial for benchmarking against HMM in power markets.
    """
    def __init__(self, mu: float, sigma: float, lambda_j: float, mu_j: float, sigma_j: float):
        self.mu = mu
        self.sigma = sigma
        self.lambda_j = lambda_j  # Jump intensity (Poisson)
        self.mu_j = mu_j          # Mean of log-jump size
        self.sigma_j = sigma_j    # Vol of log-jump size
        
    def simulate(self, s0: float, t_horizon: float, dt: float, n_paths: int = 1) -> np.ndarray:
        """Generates paths simulating base diffusion + Poisson jumps."""
        n_steps = int(t_horizon / dt)
        paths = np.zeros((n_steps + 1, n_paths))
        paths[0] = s0
        
        # Drift correction for the jump compensator
        k = np.exp(self.mu_j + 0.5 * self.sigma_j**2) - 1
        drift = self.mu - 0.5 * self.sigma**2 - self.lambda_j * k
        
        for t in range(1, n_steps + 1):
            z = np.random.standard_normal(n_paths)
            n_jumps = np.random.poisson(self.lambda_j * dt, n_paths)
            
            jumps = np.zeros(n_paths)
            for i in range(n_paths):
                if n_jumps[i] > 0:
                    jumps[i] = np.sum(np.random.normal(self.mu_j, self.sigma_j, n_jumps[i]))
                    
            paths[t] = paths[t-1] * np.exp(drift * dt + self.sigma * np.sqrt(dt) * z + jumps)
            
        return paths