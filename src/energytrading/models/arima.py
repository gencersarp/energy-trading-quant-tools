"""ARIMA, SARIMA, GARCH, and EGARCH models for energy price volatility."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


# ---------------------------------------------------------------------------
# ARIMA (manual AR + differencing; statsmodels used if available)
# ---------------------------------------------------------------------------

@dataclass
class ARIMAParams:
    p: int
    d: int
    q: int


class ARIMAModel:
    """ARIMA(p,d,q) model. Uses statsmodels if installed, else manual AR(p) via OLS."""

    def __init__(self, p: int = 1, d: int = 1, q: int = 1):
        self.p = p
        self.d = d
        self.q = q
        self._model = None
        self._result = None
        self._manual_ar_coefs: np.ndarray | None = None
        self._manual_mean: float = 0.0
        self._orig_series: np.ndarray | None = None

    def fit(self, series: np.ndarray | pd.Series) -> "ARIMAModel":
        y = np.asarray(series, dtype=float)
        self._orig_series = y.copy()
        try:
            from statsmodels.tsa.arima.model import ARIMA as _ARIMA
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model = _ARIMA(y, order=(self.p, self.d, self.q))
                self._result = self._model.fit()
        except ImportError:
            # Manual AR(p) on differenced series
            yd = np.diff(y, n=self.d)
            self._manual_mean = yd.mean()
            yd_c = yd - self._manual_mean
            X = np.column_stack([yd_c[i: len(yd_c) - self.p + i] for i in range(self.p)])
            yy = yd_c[self.p:]
            coefs, *_ = np.linalg.lstsq(X, yy, rcond=None)
            self._manual_ar_coefs = coefs
        return self

    def forecast(self, steps: int) -> np.ndarray:
        if self._result is not None:
            fc = self._result.forecast(steps=steps)
            return np.asarray(fc)
        # Manual reconstruction
        y = self._orig_series.copy()
        yd = np.diff(y, n=self.d)
        yd_c = yd - self._manual_mean
        preds = []
        history = list(yd_c)
        for _ in range(steps):
            lag_vals = np.array(history[-self.p:])
            pred = self._manual_mean + float(self._manual_ar_coefs @ lag_vals)
            preds.append(pred)
            history.append(pred)
        # Invert differencing (approximate: add to last observed level)
        out = np.array(preds)
        last_val = y[-1]
        return np.cumsum(np.insert(out, 0, last_val))[1:]

    @property
    def residuals(self) -> np.ndarray:
        if self._result is not None:
            return np.asarray(self._result.resid)
        raise ValueError("Fit with statsmodels to access residuals")

    @property
    def aic(self) -> float:
        if self._result is not None:
            return float(self._result.aic)
        return np.inf


# ---------------------------------------------------------------------------
# SARIMA
# ---------------------------------------------------------------------------

class SARIMAModel:
    """SARIMA(p,d,q)(P,D,Q,s) — wraps statsmodels or falls back to ARIMA."""

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24)):
        self.order = order
        self.seasonal_order = seasonal_order
        self._result = None
        self._fallback = ARIMAModel(*order)

    def fit(self, series: np.ndarray | pd.Series) -> "SARIMAModel":
        y = np.asarray(series, dtype=float)
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False)
                self._result = m.fit(disp=False)
        except (ImportError, Exception):
            self._fallback.fit(y)
        return self

    def forecast(self, steps: int) -> np.ndarray:
        if self._result is not None:
            return np.asarray(self._result.forecast(steps=steps))
        return self._fallback.forecast(steps)

    @property
    def aic(self) -> float:
        if self._result is not None:
            return float(self._result.aic)
        return self._fallback.aic


# ---------------------------------------------------------------------------
# GARCH(1,1)
# ---------------------------------------------------------------------------

@dataclass
class GARCHResult:
    omega: float
    alpha: float
    beta: float
    log_likelihood: float
    aic: float


class GARCHModel:
    """GARCH(1,1) via MLE with scipy."""

    def __init__(self):
        self.result: GARCHResult | None = None
        self._returns: np.ndarray | None = None

    def _neg_log_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        n = len(returns)
        h = np.empty(n)
        h[0] = np.var(returns)
        for t in range(1, n):
            h[t] = omega + alpha * returns[t - 1] ** 2 + beta * h[t - 1]
            if h[t] <= 0:
                return 1e10
        nll = 0.5 * np.sum(np.log(h) + returns ** 2 / h)
        return float(nll)

    def fit(self, returns: np.ndarray) -> "GARCHModel":
        r = np.asarray(returns, dtype=float)
        self._returns = r
        var0 = float(np.var(r))
        x0 = np.array([var0 * 0.05, 0.05, 0.90])
        bounds = [(1e-8, None), (0, 1), (0, 1)]
        res = minimize(self._neg_log_likelihood, x0, args=(r,), method="L-BFGS-B", bounds=bounds)
        omega, alpha, beta = res.x
        ll = -res.fun
        k = 3
        aic = 2 * k - 2 * ll
        self.result = GARCHResult(omega=float(omega), alpha=float(alpha),
                                  beta=float(beta), log_likelihood=float(ll), aic=float(aic))
        return self

    def conditional_variance(self, returns: np.ndarray) -> np.ndarray:
        if self.result is None:
            raise ValueError("Fit model first")
        r = np.asarray(returns, dtype=float)
        h = np.empty(len(r))
        h[0] = np.var(r)
        for t in range(1, len(r)):
            h[t] = (self.result.omega + self.result.alpha * r[t - 1] ** 2
                    + self.result.beta * h[t - 1])
        return h

    def forecast_variance(self, h: int = 10) -> np.ndarray:
        if self.result is None:
            raise ValueError("Fit model first")
        g = self.result
        long_run = g.omega / (1 - g.alpha - g.beta)
        h_last = float(np.var(self._returns))
        forecasts = np.empty(h)
        persistence = g.alpha + g.beta
        for i in range(h):
            h_last = g.omega + persistence * h_last
            forecasts[i] = h_last
        return forecasts


# ---------------------------------------------------------------------------
# EGARCH(1,1)
# ---------------------------------------------------------------------------

class EGARCHModel:
    """EGARCH(1,1) — captures leverage effects in energy markets."""

    def __init__(self):
        self._params: np.ndarray | None = None
        self._returns: np.ndarray | None = None

    def _log_variance(self, params: np.ndarray, returns: np.ndarray) -> np.ndarray:
        omega, alpha, gamma, beta = params
        n = len(returns)
        log_h = np.empty(n)
        log_h[0] = np.log(np.var(returns) + 1e-8)
        for t in range(1, n):
            h_prev = np.exp(log_h[t - 1])
            std_resid = returns[t - 1] / (np.sqrt(h_prev) + 1e-8)
            log_h[t] = (omega + beta * log_h[t - 1]
                        + alpha * (np.abs(std_resid) - np.sqrt(2 / np.pi))
                        + gamma * std_resid)
        return log_h

    def _nll(self, params: np.ndarray, returns: np.ndarray) -> float:
        if abs(params[3]) >= 1:
            return 1e10
        log_h = self._log_variance(params, returns)
        nll = 0.5 * np.sum(log_h + returns ** 2 / (np.exp(log_h) + 1e-8))
        return float(nll)

    def fit(self, returns: np.ndarray) -> "EGARCHModel":
        r = np.asarray(returns, dtype=float)
        self._returns = r
        x0 = np.array([-0.1, 0.1, -0.05, 0.9])
        res = minimize(self._nll, x0, args=(r,), method="Nelder-Mead",
                       options={"maxiter": 5000, "xatol": 1e-6})
        self._params = res.x
        return self

    def conditional_std(self, returns: np.ndarray) -> np.ndarray:
        if self._params is None:
            raise ValueError("Fit model first")
        return np.sqrt(np.exp(self._log_variance(self._params, np.asarray(returns))))


# ---------------------------------------------------------------------------
# Auto ARIMA
# ---------------------------------------------------------------------------

def fit_arima_auto(series: np.ndarray | pd.Series, max_p: int = 3,
                   max_d: int = 2, max_q: int = 3) -> ARIMAModel:
    """Selects best ARIMA(p,d,q) by AIC over grid search."""
    best_aic = np.inf
    best_model: ARIMAModel | None = None
    for d in range(max_d + 1):
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    m = ARIMAModel(p=p, d=d, q=q)
                    m.fit(series)
                    if m.aic < best_aic:
                        best_aic = m.aic
                        best_model = m
                except Exception:
                    continue
    if best_model is None:
        best_model = ARIMAModel(1, 1, 1)
        best_model.fit(series)
    return best_model
