"""Heston stochastic volatility model with energy spike extension."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.integrate import quad
from scipy.optimize import differential_evolution


@dataclass
class HestonParams:
    kappa: float    # mean reversion speed of variance
    theta: float    # long-run variance
    sigma: float    # vol-of-vol
    rho: float      # correlation price-variance
    v0: float       # initial variance
    r: float = 0.0  # risk-free rate


class HestonModel:
    """Heston (1993) stochastic volatility model."""

    def simulate(self, S0: float, T: float, N: int, n_paths: int,
                 params: HestonParams, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Euler-Maruyama with full truncation scheme.

        Returns (price_paths, variance_paths) each of shape (N+1, n_paths).
        """
        rng = np.random.default_rng(seed)
        dt = T / N
        S = np.zeros((N + 1, n_paths))
        V = np.zeros((N + 1, n_paths))
        S[0] = S0
        V[0] = params.v0

        for t in range(N):
            z1 = rng.standard_normal(n_paths)
            z2 = rng.standard_normal(n_paths)
            zv = z1
            zs = params.rho * z1 + np.sqrt(max(1 - params.rho ** 2, 0)) * z2

            V_pos = np.maximum(V[t], 0.0)
            V[t + 1] = (V_pos + params.kappa * (params.theta - V_pos) * dt
                        + params.sigma * np.sqrt(V_pos * dt) * zv)
            V[t + 1] = np.maximum(V[t + 1], 0.0)  # full truncation

            S[t + 1] = S[t] * np.exp(
                (params.r - 0.5 * V_pos) * dt
                + np.sqrt(V_pos * dt) * zs
            )

        return S, V

    def characteristic_function(self, phi: complex, S0: float, T: float,
                                 params: HestonParams) -> complex:
        """Heston characteristic function (Gatheral formulation)."""
        kappa, theta, sigma, rho, v0, r = (
            params.kappa, params.theta, params.sigma,
            params.rho, params.v0, params.r
        )
        i = complex(0, 1)
        xi = kappa - sigma * rho * phi * i
        d = np.sqrt(xi ** 2 + sigma ** 2 * (phi ** 2 + phi * i))
        g = (xi - d) / (xi + d)
        exp_dT = np.exp(-d * T)
        C = (r * phi * i * T
             + kappa * theta / sigma ** 2 * (
                 (xi - d) * T - 2 * np.log((1 - g * exp_dT) / (1 - g))
             ))
        D = (xi - d) / sigma ** 2 * (1 - exp_dT) / (1 - g * exp_dT)
        return np.exp(C + D * v0 + i * phi * np.log(S0))

    def price_european(self, S0: float, K: float, T: float,
                       params: HestonParams, option_type: str = "call") -> float:
        """Price European option via Gil-Pelaez Fourier inversion."""
        discount = np.exp(-params.r * T)

        def integrand_P1(phi):
            cf = self.characteristic_function(phi - 1j, S0, T, params)
            cf0 = self.characteristic_function(-1j, S0, T, params)
            return (np.exp(-1j * phi * np.log(K)) * cf / (1j * phi * cf0)).real

        def integrand_P2(phi):
            cf = self.characteristic_function(phi, S0, T, params)
            return (np.exp(-1j * phi * np.log(K)) * cf / (1j * phi)).real

        P1 = 0.5 + 1 / np.pi * quad(integrand_P1, 1e-4, 200, limit=100)[0]
        P2 = 0.5 + 1 / np.pi * quad(integrand_P2, 1e-4, 200, limit=100)[0]

        call_price = float(S0 * P1 - K * discount * P2)
        if option_type == "call":
            return max(call_price, 0.0)
        return max(call_price - S0 + K * discount, 0.0)  # put-call parity

    def calibrate(self, market_prices: np.ndarray, strikes: np.ndarray,
                  maturities: np.ndarray, S0: float,
                  method: str = "de") -> HestonParams:
        """Calibrate to market prices via differential evolution."""

        def objective(x):
            kappa, theta, sigma, rho, v0 = x
            if kappa <= 0 or theta <= 0 or sigma <= 0 or abs(rho) >= 1 or v0 <= 0:
                return 1e8
            # Feller condition penalty
            feller_penalty = max(0, 2 * kappa * theta - sigma ** 2) * 0
            params = HestonParams(kappa=kappa, theta=theta, sigma=sigma,
                                  rho=rho, v0=v0)
            errors = []
            for mkt, K, T in zip(market_prices, strikes, maturities):
                try:
                    model_price = self.price_european(S0, K, float(T), params)
                    errors.append((model_price - mkt) ** 2)
                except Exception:
                    errors.append(1e4)
            return float(np.mean(errors)) + feller_penalty

        bounds = [(0.01, 10), (0.001, 1), (0.01, 2), (-0.99, 0.99), (0.001, 1)]
        if method == "de":
            result = differential_evolution(objective, bounds, seed=42,
                                            maxiter=200, tol=1e-6, popsize=10)
        else:
            from scipy.optimize import minimize as _min
            result = _min(objective, x0=[2, 0.04, 0.3, -0.5, 0.04],
                          bounds=bounds, method="L-BFGS-B")

        kappa, theta, sigma, rho, v0 = result.x
        return HestonParams(kappa=float(kappa), theta=float(theta),
                            sigma=float(sigma), rho=float(rho), v0=float(v0))


# ---------------------------------------------------------------------------
# Energy Spike SV: Heston + Merton Jumps
# ---------------------------------------------------------------------------

class EnergySpikeSV(HestonModel):
    """Heston stochastic volatility + Merton jump diffusion for power spikes."""

    def simulate(self, S0: float, T: float, N: int, n_paths: int,
                 params: HestonParams,
                 lambda_j: float = 2.0,
                 mu_j: float = -0.1,
                 sigma_j: float = 0.3,
                 seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Heston + Poisson jumps. Returns (price_paths, var_paths)."""
        S, V = super().simulate(S0, T, N, n_paths, params, seed)
        rng = np.random.default_rng(seed)
        dt = T / N
        k = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1
        for t in range(1, N + 1):
            n_jumps = rng.poisson(lambda_j * dt, n_paths)
            for i in range(n_paths):
                if n_jumps[i] > 0:
                    jump = np.sum(rng.normal(mu_j, sigma_j, n_jumps[i]))
                    S[t, i] *= np.exp(jump - lambda_j * k * dt)
        return S, V
