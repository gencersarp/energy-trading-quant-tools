"""Cointegration, VAR, ECM, and pairs trading models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Cointegration Analysis
# ---------------------------------------------------------------------------

@dataclass
class EGTestResult:
    test_statistic: float
    p_value: float
    is_cointegrated: bool
    cointegrating_vector: np.ndarray
    spread: np.ndarray


class CointegrationAnalyzer:
    """Engle-Granger and Johansen cointegration tests."""

    @staticmethod
    def _adf_test(x: np.ndarray) -> Tuple[float, float]:
        """Augmented Dickey-Fuller via OLS. Returns (stat, approx_pvalue)."""
        x = np.asarray(x, dtype=float)
        dx = np.diff(x)
        x_lag = x[:-1]
        X = np.column_stack([x_lag, np.ones(len(x_lag))])
        beta, *_ = np.linalg.lstsq(X, dx, rcond=None)
        resid = dx - X @ beta
        se = np.sqrt(np.sum(resid ** 2) / (len(dx) - 2) * np.linalg.inv(X.T @ X)[0, 0])
        t_stat = float(beta[0] / (se + 1e-10))
        # MacKinnon approx critical values: -3.96(1%), -3.41(5%), -3.13(10%)
        p_approx = float(np.clip(0.5 + 0.5 * np.tanh((t_stat + 3.41) / 0.5), 0.0, 1.0))
        return t_stat, p_approx

    def test_eg(self, y1: np.ndarray, y2: np.ndarray,
                significance: float = 0.05) -> EGTestResult:
        """Engle-Granger two-step cointegration test."""
        y1 = np.asarray(y1, dtype=float)
        y2 = np.asarray(y2, dtype=float)
        # Step 1: OLS regression y1 on y2
        X = np.column_stack([y2, np.ones(len(y2))])
        beta, *_ = np.linalg.lstsq(X, y1, rcond=None)
        spread = y1 - X @ beta
        # Step 2: ADF on residuals
        stat, pval = self._adf_test(spread)
        return EGTestResult(
            test_statistic=stat,
            p_value=pval,
            is_cointegrated=pval < significance,
            cointegrating_vector=beta,
            spread=spread,
        )

    def cointegrating_vector(self, y1: np.ndarray,
                             y2: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """Returns (beta, alpha, spread) from OLS regression."""
        result = self.test_eg(y1, y2)
        beta, alpha = result.cointegrating_vector
        return float(beta), float(alpha), result.spread

    def test_johansen(self, data: np.ndarray, r: int = 1,
                      lags: int = 1) -> Dict:
        """Johansen trace test (eigenvalue decomposition approach)."""
        data = np.asarray(data, dtype=float)
        T, n = data.shape
        dx = np.diff(data, axis=0)
        x_lag = data[:-1]
        # Residuals from regressing dx and x_lag on lags of dx
        def demean_resid(Y, X):
            beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
            return Y - X @ beta

        X_lags = np.column_stack(
            [dx[i: T - 1 - lags + i] for i in range(lags - 1)] + [np.ones(T - lags)]
        ) if lags > 1 else np.ones((T - 1, 1))
        R0 = demean_resid(dx[lags - 1:], X_lags)
        R1 = demean_resid(x_lag[lags - 1:], X_lags)
        # Moment matrices
        S00 = R0.T @ R0 / T
        S11 = R1.T @ R1 / T
        S01 = R0.T @ R1 / T
        # Eigenvalue problem
        S11_inv = np.linalg.inv(S11 + np.eye(n) * 1e-8)
        M = S11_inv @ S01.T @ np.linalg.inv(S00 + np.eye(n) * 1e-8) @ S01
        eigenvalues, eigenvectors = np.linalg.eig(M)
        idx = np.argsort(eigenvalues.real)[::-1]
        eigenvalues = eigenvalues.real[idx]
        eigenvectors = eigenvectors.real[:, idx]
        # Trace statistic
        trace_stat = -T * np.sum(np.log(1 - np.clip(eigenvalues[:n - r], 0, 0.9999)))
        return {
            "trace_statistic": float(trace_stat),
            "eigenvalues": eigenvalues.tolist(),
            "eigenvectors": eigenvectors,
            "n_cointegrating_relations": r,
        }


# ---------------------------------------------------------------------------
# VAR Model
# ---------------------------------------------------------------------------

class VARModel:
    """Vector Autoregression model via OLS."""

    def __init__(self, lags: int = 1):
        self.lags = lags
        self._coefs: np.ndarray | None = None
        self._intercept: np.ndarray | None = None
        self._resid: np.ndarray | None = None
        self._data: np.ndarray | None = None

    def fit(self, data: np.ndarray | pd.DataFrame) -> "VARModel":
        if isinstance(data, pd.DataFrame):
            data = data.values
        data = np.asarray(data, dtype=float)
        self._data = data
        T, n = data.shape
        p = self.lags
        Y = data[p:]
        X_list = [np.ones((T - p, 1))]
        for lag in range(1, p + 1):
            X_list.append(data[p - lag: T - lag])
        X = np.hstack(X_list)
        B, *_ = np.linalg.lstsq(X, Y, rcond=None)
        self._intercept = B[0]
        self._coefs = B[1:].reshape(p, n, n)
        self._resid = Y - X @ B
        return self

    def forecast(self, steps: int) -> np.ndarray:
        if self._coefs is None:
            raise ValueError("Fit model first")
        history = self._data[-self.lags:].copy()
        preds = []
        for _ in range(steps):
            fc = self._intercept.copy()
            for lag in range(self.lags):
                fc += self._coefs[lag] @ history[-(lag + 1)]
            preds.append(fc)
            history = np.vstack([history, fc])
        return np.array(preds)

    def granger_causality(self, cause_idx: int, effect_idx: int,
                          maxlag: int = 5) -> Dict[str, float]:
        """Tests if cause Granger-causes effect."""
        if self._data is None:
            raise ValueError("Fit model first")
        y = self._data[:, effect_idx]
        x = self._data[:, cause_idx]
        T = len(y)
        # Restricted: y regressed on own lags only
        Xr = np.column_stack([np.ones(T - maxlag)] +
                              [y[maxlag - i - 1: T - i - 1] for i in range(maxlag)])
        Xu = np.column_stack([Xr] +
                             [x[maxlag - i - 1: T - i - 1] for i in range(maxlag)])
        yr = y[maxlag:]
        _, rss_r, _, _ = np.linalg.lstsq(Xr, yr, rcond=None)
        _, rss_u, _, _ = np.linalg.lstsq(Xu, yr, rcond=None)
        rss_r = float(np.sum((yr - Xr @ np.linalg.lstsq(Xr, yr, rcond=None)[0]) ** 2))
        rss_u = float(np.sum((yr - Xu @ np.linalg.lstsq(Xu, yr, rcond=None)[0]) ** 2))
        df1, df2 = maxlag, T - 2 * maxlag - 1
        F = ((rss_r - rss_u) / df1) / (rss_u / df2) if df2 > 0 else 0.0
        p_val = float(1 - stats.f.cdf(F, df1, df2)) if df2 > 0 else 1.0
        return {"F_statistic": float(F), "p_value": p_val, "lags": maxlag}


# ---------------------------------------------------------------------------
# Error Correction Model
# ---------------------------------------------------------------------------

class ECMModel:
    """Error Correction Model for cointegrated pairs."""

    def __init__(self, lags: int = 1):
        self.lags = lags
        self._ec_coef: float = 0.0
        self._coefs: np.ndarray | None = None
        self._beta: float = 1.0
        self._alpha: float = 0.0

    def fit(self, y1: np.ndarray, y2: np.ndarray) -> "ECMModel":
        y1, y2 = np.asarray(y1, float), np.asarray(y2, float)
        # Step 1: cointegrating relationship
        analyzer = CointegrationAnalyzer()
        self._beta, self._alpha, spread = analyzer.cointegrating_vector(y1, y2)
        # Step 2: ECM regression
        dy1 = np.diff(y1)
        ec_lag = spread[:-1]
        T = len(dy1)
        p = self.lags
        X_list = [np.ones(T - p), ec_lag[p:]]
        for lag in range(1, p + 1):
            X_list.append(dy1[p - lag: T - lag])
        X = np.column_stack(X_list)
        y = dy1[p:]
        coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        self._ec_coef = float(coefs[1])
        self._coefs = coefs
        self._spread = spread
        return self

    def forecast(self, y1_last: float, y2_last: float,
                 steps: int = 5) -> np.ndarray:
        """Simple h-step ahead forecast of y1."""
        spread = y1_last - self._beta * y2_last - self._alpha
        preds = []
        y1 = y1_last
        for _ in range(steps):
            dy1 = self._coefs[0] + self._ec_coef * spread
            y1 = y1 + dy1
            spread = y1 - self._beta * y2_last - self._alpha
            preds.append(y1)
        return np.array(preds)


# ---------------------------------------------------------------------------
# Spread Trading Model
# ---------------------------------------------------------------------------

class SpreadTradingModel:
    """Wraps cointegration for mean-reverting pairs trading."""

    def __init__(self, window: int = 60):
        self.window = window
        self._beta: float = 1.0
        self._alpha: float = 0.0
        self._spread: np.ndarray | None = None

    def fit(self, y1: np.ndarray, y2: np.ndarray) -> "SpreadTradingModel":
        y1, y2 = np.asarray(y1, float), np.asarray(y2, float)
        X = np.column_stack([y2, np.ones(len(y2))])
        beta, *_ = np.linalg.lstsq(X, y1, rcond=None)
        self._beta, self._alpha = float(beta[0]), float(beta[1])
        self._spread = y1 - self._beta * y2 - self._alpha
        return self

    def z_score(self) -> float:
        if self._spread is None:
            raise ValueError("Fit model first")
        mu = np.mean(self._spread[-self.window:])
        sigma = np.std(self._spread[-self.window:]) + 1e-8
        return float((self._spread[-1] - mu) / sigma)

    def half_life(self) -> float:
        """OU half-life in observations."""
        if self._spread is None:
            raise ValueError("Fit model first")
        s = self._spread
        s_lag = s[:-1]
        ds = np.diff(s)
        X = np.column_stack([s_lag, np.ones(len(s_lag))])
        beta, *_ = np.linalg.lstsq(X, ds, rcond=None)
        lam = float(beta[0])
        if lam >= 0:
            return np.inf
        return float(-np.log(2) / lam)

    def entry_signal(self, z_entry: float = 2.0,
                     z_exit: float = 0.5) -> int:
        """Returns -1 (short spread), 0 (flat), +1 (long spread)."""
        z = self.z_score()
        if z < -z_entry:
            return 1
        if z > z_entry:
            return -1
        return 0
