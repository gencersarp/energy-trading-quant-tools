import numpy as np
from scipy.stats import norm, weibull_min


class SpatialWindCopula:
    """
    Models spatially correlated wind speeds across multiple wind farms using a Gaussian Copula.
    Wind speeds are converted from uniform marginals to Weibull distributions (industry standard).
    """
    def __init__(self, correlation_matrix: np.ndarray, shape_params: list[float], scale_params: list[float]):
        """
        correlation_matrix: NxN matrix of spatial correlations between sites.
        shape_params: Weibull shape parameter (k) for each site (typically ~2.0).
        scale_params: Weibull scale parameter (c) for each site.
        """
        self.corr = np.asarray(correlation_matrix)
        self.shape_params = shape_params
        self.scale_params = scale_params
        
        # Cholesky decomposition for generating correlated normals
        self.L = np.linalg.cholesky(self.corr)
        self.n_sites = len(shape_params)

    def simulate_wind_speeds(self, n_samples: int) -> np.ndarray:
        """Simulates correlated wind speeds (m/s) across all sites."""
        # 1. Generate independent standard normals
        z = np.random.standard_normal((self.n_sites, n_samples))
        
        # 2. Induce spatial correlation
        z_corr = self.L @ z
        
        # 3. Transform to uniform marginals via Normal CDF
        u = norm.cdf(z_corr)
        
        # 4. Transform to Weibull distributed wind speeds using PPF (Inverse CDF)
        wind_speeds = np.zeros_like(u)
        for i in range(self.n_sites):
            wind_speeds[i, :] = weibull_min.ppf(u[i, :], c=self.shape_params[i], scale=self.scale_params[i])
            
        return wind_speeds.T