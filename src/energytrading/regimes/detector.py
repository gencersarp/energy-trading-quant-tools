"""Enhanced regime detection: HMM, threshold, change-point, Markov chain."""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class RegimeType(str, Enum):
    NORMAL = "normal"
    SPIKE = "spike"
    CRASH = "crash"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"


class RegimeDetector:
    """Multiple regime detection methods for energy price series."""

    def hmm_regimes(self, returns: pd.Series | np.ndarray,
                     n_states: int = 3) -> pd.Series:
        """Gaussian HMM regime detection."""
        r = np.asarray(returns).reshape(-1, 1)
        index = getattr(returns, "index", range(len(r)))
        try:
            from hmmlearn.hmm import GaussianHMM
            model = GaussianHMM(n_components=n_states, covariance_type="diag",
                                 n_iter=200, random_state=42)
            model.fit(r)
            states = model.predict(r)
            # Sort states by mean (low=0, mid=1, high=2)
            means = [float(model.means_[i][0]) for i in range(n_states)]
            sort_order = np.argsort(means)
            state_map = {old: new for new, old in enumerate(sort_order)}
            states = np.array([state_map[s] for s in states])
        except ImportError:
            states = self.threshold_regimes(pd.Series(r.flatten()), n_states=n_states).values
        return pd.Series(states, index=index, name="regime")

    def threshold_regimes(self, prices: pd.Series, vol_window: int = 20,
                           n_states: int = 2, vol_threshold_high: float = 2.0,
                           vol_threshold_low: float = 0.5) -> pd.Series:
        """Volatility-threshold regime classification."""
        returns = prices.pct_change().fillna(0)
        roll_vol = returns.rolling(vol_window).std().fillna(method="bfill")
        mu_vol = roll_vol.mean()
        regimes = pd.Series(0, index=prices.index, dtype=int)
        regimes[roll_vol > mu_vol * vol_threshold_high] = 2  # spike
        regimes[roll_vol < mu_vol * vol_threshold_low] = 0    # quiet
        regimes[(roll_vol >= mu_vol * vol_threshold_low) & (roll_vol <= mu_vol * vol_threshold_high)] = 1  # normal
        return regimes

    def change_point_detection(self, series: pd.Series | np.ndarray,
                                min_size: int = 20) -> List[int]:
        """Binary segmentation change-point detection (PELT approximation)."""
        x = np.asarray(series)
        n = len(x)
        breakpoints = []

        def cost(lo: int, hi: int) -> float:
            segment = x[lo:hi]
            if len(segment) < 2:
                return 0.0
            return float((len(segment) - 1) * np.log(np.var(segment) + 1e-10))

        def best_split(lo: int, hi: int) -> Optional[int]:
            if hi - lo < 2 * min_size:
                return None
            best_cost = cost(lo, hi)
            best_k = None
            for k in range(lo + min_size, hi - min_size):
                c = cost(lo, k) + cost(k, hi)
                if c < best_cost - 5.0:  # penalty
                    best_cost = c
                    best_k = k
            return best_k

        def segment(lo: int, hi: int) -> None:
            k = best_split(lo, hi)
            if k is not None:
                breakpoints.append(k)
                segment(lo, k)
                segment(k, hi)

        segment(0, n)
        return sorted(breakpoints)

    def markov_chain_regimes(self, returns: pd.Series,
                              n_states: int = 2) -> Tuple[pd.Series, np.ndarray]:
        """Classify returns into regimes and estimate transition matrix."""
        quantiles = [i / n_states for i in range(1, n_states)]
        thresholds = [float(returns.quantile(q)) for q in quantiles]
        states = pd.Series(0, index=returns.index, dtype=int)
        for i, thresh in enumerate(thresholds):
            states[returns > thresh] = i + 1
        # Estimate transition matrix
        P = np.zeros((n_states, n_states))
        state_arr = states.values
        for t in range(len(state_arr) - 1):
            i, j = int(state_arr[t]), int(state_arr[t + 1])
            if 0 <= i < n_states and 0 <= j < n_states:
                P[i, j] += 1
        row_sums = P.sum(axis=1, keepdims=True)
        P = P / (row_sums + 1e-8)
        return states, P

    def current_regime_probability(self, returns: pd.Series,
                                    model: Optional[object] = None) -> Dict[int, float]:
        """Compute current regime probability using recent volatility."""
        recent = returns.iloc[-20:] if len(returns) >= 20 else returns
        vol = float(recent.std())
        all_vol = float(returns.std())
        vol_ratio = vol / (all_vol + 1e-8)
        if vol_ratio > 1.5:
            return {0: 0.1, 1: 0.3, 2: 0.6}  # spike regime
        elif vol_ratio < 0.5:
            return {0: 0.7, 1: 0.2, 2: 0.1}  # quiet regime
        return {0: 0.2, 1: 0.6, 2: 0.2}  # normal


class RegimeTransitionModel:
    """Markov chain model for regime dynamics."""

    def __init__(self):
        self._P: Optional[np.ndarray] = None
        self._pi: Optional[np.ndarray] = None
        self._regime_stats: Dict[int, Dict] = {}

    def fit(self, regimes: pd.Series) -> "RegimeTransitionModel":
        states = regimes.values.astype(int)
        n_states = int(states.max()) + 1
        P = np.zeros((n_states, n_states))
        for t in range(len(states) - 1):
            P[states[t], states[t + 1]] += 1
        row_sums = P.sum(axis=1, keepdims=True)
        self._P = P / (row_sums + 1e-8)
        # Steady state via eigendecomposition
        eigvals, eigvecs = np.linalg.eig(self._P.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        pi = eigvecs[:, idx].real
        self._pi = np.abs(pi) / np.abs(pi).sum()
        return self

    def expected_duration(self, regime: int) -> float:
        """Expected periods in regime = 1 / (1 - P_{ii})."""
        if self._P is None:
            raise ValueError("Fit first")
        pii = self._P[regime, regime]
        return float(1.0 / (1.0 - pii + 1e-8))

    def forecast_regime_probability(self, current_regime: int,
                                     horizon: int = 5) -> np.ndarray:
        """Forecast regime probability vector h steps ahead."""
        if self._P is None:
            raise ValueError("Fit first")
        n = self._P.shape[0]
        e = np.zeros(n)
        e[current_regime] = 1.0
        probs = np.zeros((horizon, n))
        for t in range(horizon):
            e = e @ self._P
            probs[t] = e
        return probs

    def regime_conditional_moments(self, returns: pd.Series,
                                    regimes: pd.Series) -> Dict[int, Dict]:
        result = {}
        for regime in regimes.unique():
            mask = regimes == regime
            r = returns[mask]
            result[int(regime)] = {
                "mu": float(r.mean()),
                "sigma": float(r.std()),
                "skew": float(stats.skew(r.dropna())),
                "kurt": float(stats.kurtosis(r.dropna())),
                "n_obs": int(mask.sum()),
            }
        return result


class CrisisIndicator:
    """Systemic risk and crisis detection indicators."""

    def mahalanobis_distance(self, returns: pd.DataFrame,
                              lookback: int = 252) -> pd.Series:
        """Turbulence index via Mahalanobis distance."""
        result = pd.Series(index=returns.index, dtype=float)
        for i in range(lookback, len(returns)):
            hist = returns.iloc[max(0, i - lookback): i].dropna()
            mu = hist.mean().values
            cov = hist.cov().values + np.eye(len(mu)) * 1e-6
            r = returns.iloc[i].fillna(0).values
            try:
                cov_inv = np.linalg.inv(cov)
                d = float((r - mu) @ cov_inv @ (r - mu))
                result.iloc[i] = d
            except Exception:
                result.iloc[i] = 0.0
        return result.fillna(0)

    def absorption_ratio(self, returns: pd.DataFrame,
                          n_factors: int = 1) -> pd.Series:
        """Absorption ratio: variance of top PCA components / total."""
        result = pd.Series(index=returns.index, dtype=float)
        window = 60
        for i in range(window, len(returns)):
            r = returns.iloc[max(0, i - window): i].dropna()
            if len(r) < 10 or r.shape[1] < 2:
                continue
            cov = r.cov().values
            eigvals = np.linalg.eigvalsh(cov)[::-1]
            ar = eigvals[:n_factors].sum() / (eigvals.sum() + 1e-8)
            result.iloc[i] = ar
        return result.fillna(0)

    def correlation_surprise(self, current_corr: np.ndarray,
                               historical_corr: np.ndarray) -> float:
        """Frob norm distance between current and historical correlation."""
        return float(np.linalg.norm(current_corr - historical_corr, "fro"))
