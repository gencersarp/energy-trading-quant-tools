from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OUParams:
    """Ornstein-Uhlenbeck (mean-reverting) process parameters.

    Continuous-time SDE:
        dX_t = theta * (mu - X_t) dt + sigma dW_t

    Args:
        mu: Long-run mean level.
        theta: Mean reversion speed (per unit time). Must be > 0.
        sigma: Volatility parameter (per sqrt(time)). Must be >= 0.
        dt: Time step size used for discretization.
    """

    mu: float
    theta: float
    sigma: float
    dt: float = 1.0

    def validate(self) -> None:
        if not np.isfinite(self.mu):
            raise ValueError("mu must be finite")
        if not (np.isfinite(self.theta) and self.theta > 0):
            raise ValueError("theta must be finite and > 0")
        if not (np.isfinite(self.sigma) and self.sigma >= 0):
            raise ValueError("sigma must be finite and >= 0")
        if not (np.isfinite(self.dt) and self.dt > 0):
            raise ValueError("dt must be finite and > 0")


def simulate_ou(
    x0: float,
    n_steps: int,
    params: OUParams,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate an OU path with exact discretization.

    Transition:
        X_{t+dt} = mu + (X_t - mu) * exp(-theta dt) + eps
        eps ~ N(0, var), var = (sigma^2 / (2 theta)) * (1 - exp(-2 theta dt))
    """
    params.validate()
    if n_steps <= 0:
        raise ValueError("n_steps must be > 0")
    if not np.isfinite(x0):
        raise ValueError("x0 must be finite")

    rng = np.random.default_rng(seed)
    x = np.empty(n_steps + 1, dtype=float)
    x[0] = float(x0)

    a = np.exp(-params.theta * params.dt)
    var = (params.sigma ** 2) * (1.0 - np.exp(-2.0 * params.theta * params.dt)) / (
        2.0 * params.theta
    )
    std = float(np.sqrt(max(var, 0.0)))

    for t in range(n_steps):
        x[t + 1] = params.mu + (x[t] - params.mu) * a + rng.normal(0.0, std)

    return x


@dataclass(frozen=True)
class OUFit:
    """Fitted OU parameters from a univariate time series."""

    params: OUParams
    residual_std: float


def fit_ou_mle(x: np.ndarray, dt: float = 1.0) -> OUFit:
    """Fit OU parameters via Gaussian MLE using AR(1) representation.

    For equally spaced observations, OU implies:
        X_{t+1} = c + b X_t + e_t
    where b = exp(-theta dt), c = mu (1 - b).

    Estimates (c, b) via OLS and maps back to (mu, theta, sigma).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 3:
        raise ValueError("x must be a 1D array with at least 3 points")
    if not (np.isfinite(dt) and dt > 0):
        raise ValueError("dt must be finite and > 0")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain only finite values")

    x_t = x[:-1]
    x_next = x[1:]

    X = np.column_stack([np.ones_like(x_t), x_t])
    beta, *_ = np.linalg.lstsq(X, x_next, rcond=None)
    c_hat, b_hat = float(beta[0]), float(beta[1])

    b_hat = float(np.clip(b_hat, 1e-6, 1 - 1e-6))
    theta_hat = -np.log(b_hat) / dt
    mu_hat = c_hat / (1.0 - b_hat)

    resid = x_next - (c_hat + b_hat * x_t)
    resid_std = float(np.sqrt(np.mean(resid**2)))

    denom = 1.0 - np.exp(-2.0 * theta_hat * dt)
    sigma_hat = 0.0 if denom <= 0 else resid_std * np.sqrt(2.0 * theta_hat / denom)

    params = OUParams(mu=float(mu_hat), theta=float(theta_hat), sigma=float(sigma_hat), dt=float(dt))
    return OUFit(params=params, residual_std=resid_std)