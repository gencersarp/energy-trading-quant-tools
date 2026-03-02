import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class JumpDiffusionParams:
    mu: float
    sigma: float
    jump_intensity: float
    jump_mean: float
    jump_std: float
    dt: float = 1.0

class MertonJumpDiffusion:
    """Merton Jump Diffusion model for energy price spikes."""
    def __init__(self, params: JumpDiffusionParams):
        self.params = params

    def fit(self, returns: np.ndarray) -> None:
        """Calibrates continuous diffusion parameters (simplified matching)."""
        self.params.mu = float(np.mean(returns) / self.params.dt)
        self.params.sigma = float(np.std(returns) / np.sqrt(self.params.dt))
        logger.info("Fitted base diffusion. Jump params require advanced MCMC/MLE.")

    def simulate(self, s0: float, n_steps: int, n_paths: int = 1, seed: int | None = None) -> np.ndarray:
        """Simulates paths using Geometric Brownian Motion with Poisson jumps."""
        rng = np.random.default_rng(seed)
        paths = np.zeros((n_steps + 1, n_paths))
        paths[0] = s0

        dt = self.params.dt
        drift = self.params.mu - 0.5 * self.params.sigma**2
        
        for t in range(n_steps):
            z = rng.standard_normal(n_paths)
            n_jumps = rng.poisson(self.params.jump_intensity * dt, n_paths)
            jump_sizes = rng.normal(self.params.jump_mean, self.params.jump_std, n_paths) * n_jumps
            
            paths[t+1] = paths[t] * np.exp(drift * dt + self.params.sigma * np.sqrt(dt) * z + jump_sizes)
            
        return paths