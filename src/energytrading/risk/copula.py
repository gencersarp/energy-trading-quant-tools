"""Copula models for energy portfolio tail dependence."""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar, minimize


class CopulaType(str, Enum):
    GAUSSIAN = "gaussian"
    STUDENT_T = "student_t"
    CLAYTON = "clayton"
    GUMBEL = "gumbel"
    FRANK = "frank"


def _to_pseudo_obs(data: np.ndarray) -> np.ndarray:
    """Convert data to pseudo-observations in (0,1)."""
    n = data.shape[0]
    u = np.zeros_like(data, dtype=float)
    for j in range(data.shape[1]):
        ranks = stats.rankdata(data[:, j])
        u[:, j] = ranks / (n + 1)
    return u


class GaussianCopula:
    """Gaussian copula: linear correlation structure."""

    def __init__(self):
        self._rho: Optional[np.ndarray] = None
        self._n: int = 2

    def fit(self, data: np.ndarray) -> "GaussianCopula":
        u = _to_pseudo_obs(np.asarray(data, dtype=float))
        z = stats.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
        self._rho = np.corrcoef(z.T)
        self._n = data.shape[1]
        return self

    def sample(self, n: int, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        L = np.linalg.cholesky(self._rho + np.eye(self._n) * 1e-8)
        Z = rng.standard_normal((n, self._n)) @ L.T
        return stats.norm.cdf(Z)

    def tail_dependence(self) -> Tuple[float, float]:
        """Gaussian copula has zero tail dependence."""
        return 0.0, 0.0

    def log_likelihood(self, u: np.ndarray) -> float:
        u = np.clip(np.asarray(u), 1e-6, 1 - 1e-6)
        z = stats.norm.ppf(u)
        rho_inv = np.linalg.inv(self._rho)
        n = u.shape[0]
        sign, logdet = np.linalg.slogdet(self._rho)
        ll = -0.5 * n * logdet - 0.5 * np.sum(z @ (rho_inv - np.eye(self._n)) * z)
        return float(ll)


class StudentTCopula:
    """Student-t copula with tail dependence."""

    def __init__(self):
        self._rho: Optional[np.ndarray] = None
        self._nu: float = 4.0
        self._n: int = 2

    def fit(self, data: np.ndarray) -> "StudentTCopula":
        u = _to_pseudo_obs(np.asarray(data, dtype=float))
        self._n = data.shape[1]
        t_data = stats.t.ppf(np.clip(u, 1e-6, 1 - 1e-6), df=self._nu)
        self._rho = np.corrcoef(t_data.T)

        def neg_ll(nu):
            if nu <= 2:
                return 1e10
            t_z = stats.t.ppf(np.clip(u, 1e-6, 1 - 1e-6), df=nu)
            rho_inv = np.linalg.inv(self._rho + np.eye(self._n) * 1e-8)
            _, logdet = np.linalg.slogdet(self._rho)
            d = self._n
            ll = (-len(u) * 0.5 * logdet
                  + len(u) * (np.log(stats.gamma((nu + d) / 2))
                               - np.log(stats.gamma(nu / 2))
                               - (d / 2) * np.log(np.pi * nu))
                  + np.sum([
                      -(nu + d) / 2 * np.log(1 + np.dot(z, rho_inv @ z) / nu)
                      + np.sum([(nu + 1) / 2 * np.log(1 + zi ** 2 / nu)
                                for zi in z])
                      for z in t_z
                  ]))
            return -ll

        result = minimize_scalar(neg_ll, bounds=(2.1, 50), method="bounded")
        self._nu = float(result.x)
        return self

    def sample(self, n: int, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        L = np.linalg.cholesky(self._rho + np.eye(self._n) * 1e-8)
        Z = rng.standard_normal((n, self._n)) @ L.T
        chi2 = rng.chisquare(self._nu, n)
        T = Z / np.sqrt(chi2[:, None] / self._nu)
        return stats.t.cdf(T, df=self._nu)

    def tail_dependence(self) -> Tuple[float, float]:
        """Lower and upper tail dependence (equal for t-copula)."""
        if self._rho is None or self._n < 2:
            return 0.0, 0.0
        # Use first off-diagonal element
        rho = float(self._rho[0, 1])
        nu = self._nu
        td = 2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)),
                              df=nu + 1)
        return float(td), float(td)


class ClaytonCopula:
    """Clayton copula: strong lower tail dependence."""

    def __init__(self):
        self._theta: float = 1.0

    def fit(self, data: np.ndarray) -> "ClaytonCopula":
        u = _to_pseudo_obs(np.asarray(data, dtype=float))
        # Method of moments: theta = 2*tau / (1 - tau)
        tau = float(stats.kendalltau(u[:, 0], u[:, 1])[0])
        tau = np.clip(tau, 0.01, 0.99)
        self._theta = float(2 * tau / (1 - tau))
        return self

    def sample(self, n: int, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        u = rng.uniform(0, 1, n)
        t = rng.gamma(1 / self._theta, 1, n)
        v = -np.log(rng.uniform(0, 1, (n, 2))) / t[:, None]
        return (1 + v) ** (-1 / self._theta)

    def cdf(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(np.asarray(u), 1e-6, 1 - 1e-6)
        return (np.sum(u ** (-self._theta), axis=1) - u.shape[1] + 1) ** (-1 / self._theta)

    def tail_dependence_lower(self) -> float:
        return float(2 ** (-1 / self._theta))


class GumbelCopula:
    """Gumbel copula: strong upper tail dependence."""

    def __init__(self):
        self._theta: float = 2.0

    def fit(self, data: np.ndarray) -> "GumbelCopula":
        u = _to_pseudo_obs(np.asarray(data, dtype=float))
        tau = float(stats.kendalltau(u[:, 0], u[:, 1])[0])
        tau = np.clip(tau, 0.01, 0.99)
        self._theta = float(1 / (1 - tau))
        return self

    def sample(self, n: int, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        # Stable distribution method
        u1 = rng.uniform(np.pi * 0.001, np.pi * 0.999, n)
        w = rng.exponential(1, n)
        S = (np.sin((self._theta - 1) * u1) / (np.sin(u1) ** (1 / (self._theta - 1 + 1e-8)))
             ) * (np.sin(u1 / self._theta) / w) ** ((1 - self._theta) / self._theta + 1e-8)
        S = np.clip(S, 1e-8, None)
        u2 = rng.uniform(0, 1, (n, 2))
        return np.exp(-(-np.log(np.clip(u2, 1e-8, 1 - 1e-8))) ** (1 / self._theta) / S[:, None] ** (1 / self._theta))

    def tail_dependence_upper(self) -> float:
        return float(2 - 2 ** (1 / self._theta))


class CopulaSelector:
    """Fits multiple copulas and selects best by AIC."""

    def __init__(self):
        self._fitted: Dict[str, object] = {}
        self._aic: Dict[str, float] = {}
        self._best: Optional[str] = None

    def fit(self, data: np.ndarray) -> "CopulaSelector":
        data = np.asarray(data, dtype=float)
        u = _to_pseudo_obs(data)
        copulas = {
            CopulaType.GAUSSIAN: GaussianCopula(),
            CopulaType.STUDENT_T: StudentTCopula(),
            CopulaType.CLAYTON: ClaytonCopula(),
            CopulaType.GUMBEL: GumbelCopula(),
        }
        for name, cop in copulas.items():
            try:
                cop.fit(data)
                self._fitted[name] = cop
                if hasattr(cop, "log_likelihood"):
                    ll = cop.log_likelihood(u)
                    k = 2 if name != CopulaType.STUDENT_T else 3
                    self._aic[name] = float(2 * k - 2 * ll)
                else:
                    self._aic[name] = 0.0
            except Exception:
                self._aic[name] = np.inf
        self._best = min(self._aic, key=self._aic.get)
        return self

    def best_fit(self) -> str:
        return str(self._best)

    def plot_fit_diagnostics(self) -> Dict:
        return {"aic": {str(k): v for k, v in self._aic.items()},
                "best": str(self._best)}


class PortfolioDependenceAnalyzer:
    """Analyzes tail dependence between energy assets."""

    def __init__(self):
        self._copula: Optional[StudentTCopula] = None

    def fit(self, returns_df: pd.DataFrame) -> "PortfolioDependenceAnalyzer":
        self._copula = StudentTCopula()
        self._copula.fit(returns_df.values)
        self._returns = returns_df
        return self

    def compute_joint_exceedance_prob(self, threshold: float = 0.05) -> float:
        """P(both assets below threshold quantile simultaneously)."""
        if self._copula is None:
            raise ValueError("Fit first")
        u = _to_pseudo_obs(self._returns.values)
        joint = np.mean(np.all(u < threshold, axis=1))
        return float(joint)

    def conditional_var(self, asset_idx: int, given_idx: int,
                        alpha: float = 0.05) -> float:
        """CVaR of asset_idx given given_idx is in loss tail."""
        r = self._returns.values
        threshold = np.percentile(r[:, given_idx], alpha * 100)
        mask = r[:, given_idx] <= threshold
        if mask.sum() < 5:
            return float(np.percentile(r[:, asset_idx], alpha * 100))
        conditional_losses = -r[:, asset_idx][mask]
        return float(np.mean(conditional_losses))
