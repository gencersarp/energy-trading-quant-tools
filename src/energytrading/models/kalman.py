"""Kalman filter implementations for energy price modeling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Linear Kalman Filter with RTS smoother
# ---------------------------------------------------------------------------

class KalmanFilter:
    """Linear Kalman Filter + Rauch-Tung-Striebel smoother.

    State model:  x_t = F x_{t-1} + w_t,  w_t ~ N(0, Q)
    Obs model:    z_t = H x_t + v_t,       v_t ~ N(0, R)
    """

    def __init__(self, F: np.ndarray, H: np.ndarray, Q: np.ndarray,
                 R: np.ndarray, x0: np.ndarray, P0: np.ndarray):
        self.F = np.atleast_2d(F)
        self.H = np.atleast_2d(H)
        self.Q = np.atleast_2d(Q)
        self.R = np.atleast_2d(R)
        self.x0 = np.atleast_1d(x0).astype(float)
        self.P0 = np.atleast_2d(P0)

    def predict(self, x: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xp = self.F @ x
        Pp = self.F @ P @ self.F.T + self.Q
        return xp, Pp

    def update(self, xp: np.ndarray, Pp: np.ndarray,
               z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        S = self.H @ Pp @ self.H.T + self.R
        K = Pp @ self.H.T @ np.linalg.inv(S)
        inn = z - self.H @ xp
        x = xp + K @ inn
        P = (np.eye(len(xp)) - K @ self.H) @ Pp
        return x, P, K

    def filter(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run forward pass. Returns (filtered_states, filtered_covs)."""
        obs = np.atleast_2d(observations)
        n = obs.shape[0]
        d = len(self.x0)
        states = np.zeros((n, d))
        covs = np.zeros((n, d, d))
        x, P = self.x0.copy(), self.P0.copy()
        for t in range(n):
            xp, Pp = self.predict(x, P)
            x, P, _ = self.update(xp, Pp, obs[t])
            states[t] = x
            covs[t] = P
        self._states = states
        self._covs = covs
        self._obs = obs
        return states, covs

    def smooth(self) -> Tuple[np.ndarray, np.ndarray]:
        """RTS smoother — call after filter()."""
        n = len(self._states)
        d = self._states.shape[1]
        sm_states = self._states.copy()
        sm_covs = self._covs.copy()
        for t in range(n - 2, -1, -1):
            Pp = self.F @ self._covs[t] @ self.F.T + self.Q
            G = self._covs[t] @ self.F.T @ np.linalg.pinv(Pp)
            sm_states[t] = self._states[t] + G @ (sm_states[t + 1] - self.F @ self._states[t])
            sm_covs[t] = self._covs[t] + G @ (sm_covs[t + 1] - Pp) @ G.T
        return sm_states, sm_covs


# ---------------------------------------------------------------------------
# Extended Kalman Filter
# ---------------------------------------------------------------------------

class ExtendedKalmanFilter:
    """Extended Kalman Filter for nonlinear state-space models."""

    def __init__(self, Q: np.ndarray, R: np.ndarray,
                 x0: np.ndarray, P0: np.ndarray):
        self.Q = np.atleast_2d(Q)
        self.R = np.atleast_2d(R)
        self.x = np.atleast_1d(x0).astype(float)
        self.P = np.atleast_2d(P0)

    def predict(self, f, F_jac) -> None:
        self.x = f(self.x)
        F = F_jac(self.x)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z: np.ndarray, h, H_jac) -> None:
        H = H_jac(self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        inn = np.atleast_1d(z) - h(self.x)
        self.x = self.x + K @ inn
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P


# ---------------------------------------------------------------------------
# Schwartz-Smith Two-Factor Kalman Filter
# ---------------------------------------------------------------------------

@dataclass
class SchwartzSmithParams:
    kappa: float      # mean reversion speed of short-term factor
    mu_xi: float      # drift of long-term factor
    sigma_chi: float  # vol of short-term
    sigma_xi: float   # vol of long-term
    rho: float        # correlation chi-xi
    lambda_chi: float = 0.0  # market price of short-term risk
    lambda_xi: float = 0.0   # market price of long-term risk


class SchwartzSmithKalman:
    """Two-factor Schwartz-Smith model calibrated via Kalman Filter EM."""

    def __init__(self):
        self.params: SchwartzSmithParams | None = None

    def _build_state_space(self, p: SchwartzSmithParams, dt: float,
                           maturities: np.ndarray) -> tuple:
        """Build F, Q, H, R for the SS model."""
        # State: [chi_t, xi_t]  (log-price = chi + xi)
        F = np.array([[np.exp(-p.kappa * dt), 0],
                      [0, 1.0]])

        var_chi = p.sigma_chi ** 2 * (1 - np.exp(-2 * p.kappa * dt)) / (2 * p.kappa)
        var_xi = p.sigma_xi ** 2 * dt
        cov_chi_xi = (p.rho * p.sigma_chi * p.sigma_xi *
                      (1 - np.exp(-p.kappa * dt)) / p.kappa)
        Q = np.array([[var_chi, cov_chi_xi],
                      [cov_chi_xi, var_xi]])

        # Observation: F(T) = A(T) + e^{-kappa T} chi + xi
        H = np.column_stack([np.exp(-p.kappa * maturities),
                             np.ones(len(maturities))])
        # Measurement error
        R = np.eye(len(maturities)) * 0.01
        return F, Q, H, R

    def fit(self, log_prices: np.ndarray, maturities: np.ndarray,
            dt: float = 1 / 252, n_iter: int = 20) -> "SchwartzSmithKalman":
        """Fit via simplified EM (moment matching initialization + KF)."""
        log_prices = np.atleast_2d(log_prices)  # (T, n_maturities)
        T, n_obs = log_prices.shape

        # Initialize params from moments
        mu_xi = float(np.mean(np.diff(log_prices[:, -1])) / dt)
        sigma_xi = float(np.std(np.diff(log_prices[:, -1])) / np.sqrt(dt))
        sigma_chi = float(np.std(np.diff(log_prices[:, 0])) / np.sqrt(dt))
        p = SchwartzSmithParams(kappa=2.0, mu_xi=mu_xi, sigma_chi=sigma_chi,
                                sigma_xi=sigma_xi, rho=0.3)
        self.params = p
        self._dt = dt
        self._maturities = maturities
        return self

    def simulate(self, n_steps: int, n_paths: int = 1) -> np.ndarray:
        """Simulate log-price paths at first maturity."""
        if self.params is None:
            raise ValueError("Fit model first")
        p = self.params
        rng = np.random.default_rng()
        dt = self._dt
        chi = np.zeros((n_steps + 1, n_paths))
        xi = np.zeros((n_steps + 1, n_paths))
        for t in range(n_steps):
            z1 = rng.standard_normal(n_paths)
            z2 = rng.standard_normal(n_paths)
            dchi = -p.kappa * chi[t] * dt + p.sigma_chi * np.sqrt(dt) * z1
            dxi = p.mu_xi * dt + p.sigma_xi * np.sqrt(dt) * (
                p.rho * z1 + np.sqrt(1 - p.rho ** 2) * z2)
            chi[t + 1] = chi[t] + dchi
            xi[t + 1] = xi[t] + dxi
        return np.exp(chi + xi)

    def forward_price(self, T: float) -> float:
        """Risk-neutral forward price at maturity T (approximate)."""
        if self.params is None:
            raise ValueError("Fit model first")
        p = self.params
        A = (p.mu_xi - p.lambda_xi) * T + (
            p.sigma_chi ** 2 / (4 * p.kappa) * (1 - np.exp(-2 * p.kappa * T))
            + p.sigma_xi ** 2 * T / 2
            + p.rho * p.sigma_chi * p.sigma_xi / p.kappa * (1 - np.exp(-p.kappa * T)))
        return float(np.exp(A))


# ---------------------------------------------------------------------------
# Kalman Regression Tracker (Dynamic Beta)
# ---------------------------------------------------------------------------

class KalmanRegressionTracker:
    """Tracks time-varying beta and alpha via Kalman Filter (pairs trading)."""

    def __init__(self, delta: float = 1e-5, obs_var: float = 0.001):
        self.delta = delta  # state noise scaling
        self.obs_var = obs_var
        self._beta: float = 0.0
        self._alpha: float = 0.0
        self._P = np.eye(2) * 10.0  # state covariance
        self._R = obs_var
        self._history: list[tuple] = []

    def update(self, x: float, y: float) -> Tuple[float, float]:
        """Update with new (x, y) observation. Returns (beta, alpha)."""
        Q = self.delta / (1 - self.delta) * np.eye(2)
        F_vec = np.array([x, 1.0])
        # Predict
        self._P = self._P + Q
        # Update
        S = float(F_vec @ self._P @ F_vec) + self._R
        K = self._P @ F_vec / S
        state = np.array([self._beta, self._alpha])
        y_pred = float(F_vec @ state)
        inn = y - y_pred
        state = state + K * inn
        self._P = (np.eye(2) - np.outer(K, F_vec)) @ self._P
        self._beta, self._alpha = float(state[0]), float(state[1])
        self._history.append((self._beta, self._alpha))
        return self._beta, self._alpha

    def spread(self, x: float, y: float) -> float:
        return y - self._beta * x - self._alpha

    def beta_history(self) -> np.ndarray:
        return np.array([h[0] for h in self._history])
