"""Signal ensemble, meta-labeling, and Bayesian signal aggregation."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class SignalStack:
    """
    Combine multiple raw signals into a composite signal using various
    aggregation methods: mean, median, rank, z-score normalization.
    """

    def __init__(self, method: str = "mean", normalize: bool = True):
        self.method = method
        self.normalize = normalize
        self._weights: Optional[np.ndarray] = None

    def fit_weights(self, signals: pd.DataFrame,
                    returns: pd.Series,
                    lookback: int = 63) -> "SignalStack":
        """Fit signal weights via IC (information coefficient) weighting."""
        n_signals = len(signals.columns)
        ics = np.zeros(n_signals)
        future_returns = returns.shift(-1)
        for i, col in enumerate(signals.columns):
            combined = pd.concat([signals[col], future_returns], axis=1).dropna()
            if len(combined) > 5:
                ics[i] = float(np.corrcoef(
                    combined.iloc[:, 0].values,
                    combined.iloc[:, 1].values)[0, 1])
        ics = np.nan_to_num(ics)
        self._weights = np.maximum(ics, 0)
        s = self._weights.sum()
        if s > 0:
            self._weights /= s
        else:
            self._weights = np.ones(n_signals) / n_signals
        return self

    def transform(self, signals: pd.DataFrame) -> pd.Series:
        if self.normalize:
            # Z-score normalize each signal
            normalized = (signals - signals.rolling(63, min_periods=5).mean()) / \
                         (signals.rolling(63, min_periods=5).std() + 1e-8)
        else:
            normalized = signals

        if self.method == "mean":
            composite = normalized.mean(axis=1)
        elif self.method == "median":
            composite = normalized.median(axis=1)
        elif self.method == "weighted":
            if self._weights is not None:
                w = self._weights[:len(normalized.columns)]
                composite = normalized.values @ w
                composite = pd.Series(composite, index=normalized.index)
            else:
                composite = normalized.mean(axis=1)
        elif self.method == "rank":
            ranked = normalized.rank(axis=1, pct=True) - 0.5
            composite = ranked.mean(axis=1)
        elif self.method == "vote":
            composite = np.sign(normalized).sum(axis=1)
        else:
            composite = normalized.mean(axis=1)

        return composite.rename("composite_signal")

    def fit_transform(self, signals: pd.DataFrame,
                      returns: pd.Series) -> pd.Series:
        return self.fit_weights(signals, returns).transform(signals)


class MetaLabeler:
    """
    Meta-labeling (Lopez de Prado, 2018):
    A secondary model that filters a primary signal by predicting
    whether the primary signal will be profitable.
    """

    def __init__(self, threshold: float = 0.55):
        self.threshold = threshold
        self._model = None

    def _make_features(self, primary_signal: pd.Series,
                       price_returns: pd.Series,
                       vol: pd.Series) -> pd.DataFrame:
        """Features for the meta-labeling classifier."""
        features = pd.DataFrame(index=primary_signal.index)
        features["signal_strength"] = primary_signal.abs()
        features["signal_direction"] = np.sign(primary_signal)
        features["vol"] = vol.reindex(primary_signal.index)
        features["recent_return"] = price_returns.rolling(5).mean().reindex(
            primary_signal.index)
        features["return_vol_ratio"] = (
            features["recent_return"] / (features["vol"] + 1e-8))
        return features.fillna(0)

    def fit(self, primary_signal: pd.Series, price_returns: pd.Series,
            vol: pd.Series) -> "MetaLabeler":
        """
        Train on whether primary signal was profitable in hindsight.
        Uses a simple logistic model (sklearn if available, else threshold).
        """
        # Create labels: 1 if primary signal direction matches next-period return
        future_returns = price_returns.shift(-1)
        labels = (np.sign(primary_signal) == np.sign(future_returns)).astype(int)
        idx = labels.dropna().index.intersection(primary_signal.dropna().index)

        X = self._make_features(primary_signal.reindex(idx),
                                 price_returns.reindex(idx),
                                 vol.reindex(idx)).values
        y = labels.reindex(idx).values

        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            clf = LogisticRegression(C=1.0, max_iter=200)
            clf.fit(X_scaled, y)
            self._model = (scaler, clf)
        except ImportError:
            # Fallback: use signal strength as meta-filter
            self._model = None

        return self

    def predict_confidence(self, primary_signal: pd.Series,
                            price_returns: pd.Series,
                            vol: pd.Series) -> pd.Series:
        """Return predicted probability that primary signal is correct."""
        X = self._make_features(primary_signal, price_returns, vol)

        if self._model is not None:
            scaler, clf = self._model
            X_scaled = scaler.transform(X.fillna(0).values)
            proba = clf.predict_proba(X_scaled)[:, 1]
            return pd.Series(proba, index=primary_signal.index)
        else:
            # Fallback: confidence proportional to signal strength
            raw = primary_signal.abs() / (primary_signal.abs().rolling(21).mean() + 1e-8)
            return raw.clip(0, 1).rename("meta_confidence")

    def filter_signal(self, primary_signal: pd.Series,
                       price_returns: pd.Series,
                       vol: pd.Series) -> pd.Series:
        """Apply meta-label filter: zero out low-confidence signals."""
        confidence = self.predict_confidence(primary_signal, price_returns, vol)
        filtered = primary_signal.copy()
        filtered[confidence < self.threshold] = 0.0
        return filtered.rename("meta_filtered_signal")


class BayesianSignalAggregator:
    """
    Bayesian combination of multiple signals using prior beliefs
    and likelihood updates from recent performance.
    """

    def __init__(self, prior_weights: Optional[np.ndarray] = None,
                 update_window: int = 63,
                 alpha: float = 0.1):
        self.prior_weights = prior_weights
        self.update_window = update_window
        self.alpha = alpha  # learning rate for weight updates
        self._posterior_weights: Optional[np.ndarray] = None

    def update_weights(self, signals: pd.DataFrame,
                       returns: pd.Series) -> np.ndarray:
        """Update posterior weights using recent signal ICs."""
        n = len(signals.columns)
        likelihoods = np.zeros(n)

        for i, col in enumerate(signals.columns):
            s = signals[col].iloc[-self.update_window:]
            r = returns.reindex(s.index).shift(-1).dropna()
            s = s.reindex(r.index)
            if len(s) > 5:
                ic = float(np.corrcoef(s.values, r.values)[0, 1])
                likelihoods[i] = max(ic, 0)

        prior = self.prior_weights if self.prior_weights is not None else \
                np.ones(n) / n

        # Bayesian update: posterior ∝ prior × likelihood
        posterior = prior * (1 - self.alpha) + likelihoods * self.alpha
        total = posterior.sum()
        if total > 0:
            posterior /= total
        else:
            posterior = np.ones(n) / n

        self._posterior_weights = posterior
        return posterior

    def aggregate(self, signals: pd.DataFrame,
                  returns: Optional[pd.Series] = None) -> pd.Series:
        n = len(signals.columns)

        if returns is not None and len(returns) > self.update_window:
            weights = self.update_weights(signals, returns)
        elif self._posterior_weights is not None:
            weights = self._posterior_weights
        elif self.prior_weights is not None:
            weights = self.prior_weights
        else:
            weights = np.ones(n) / n

        # Normalize signals
        normalized = (signals - signals.rolling(self.update_window,
                                                 min_periods=5).mean()) / \
                     (signals.rolling(self.update_window,
                                      min_periods=5).std() + 1e-8)
        composite = normalized.values @ weights[:len(normalized.columns)]
        return pd.Series(composite, index=signals.index,
                         name="bayesian_composite")

    def get_weights(self) -> Optional[np.ndarray]:
        return self._posterior_weights
