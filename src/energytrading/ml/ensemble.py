"""Signal/model ensemble infrastructure with meta-labeling and Bayesian weighting."""
from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


class EnsembleMethod(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    WEIGHTED = "weighted"
    STACKING = "stacking"
    BAYESIAN = "bayesian"


class SignalEnsemble:
    """Combines multiple trading signals / forecasts into one composite signal."""

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._weights: Dict[str, float] = {}
        self._stacker: Optional[Any] = None
        self._scaler = StandardScaler()

    def add_model(self, name: str, model: Any, weight: float = 1.0) -> None:
        self._models[name] = model
        self._weights[name] = weight

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SignalEnsemble":
        """Fit individual models and learn stacking weights."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        # Collect predictions from each model
        preds = {}
        for name, model in self._models.items():
            try:
                model.fit(X, y)
                preds[name] = model.predict(X)
            except Exception:
                pass
        if preds:
            P = np.column_stack(list(preds.values()))
            self._stacker = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
            self._stacker.fit(P, y)
            coefs = np.maximum(self._stacker.coef_, 0)
            total = coefs.sum()
            if total > 0:
                for name, coef in zip(preds.keys(), coefs):
                    self._weights[name] = float(coef / total)
        return self

    def predict(self, X: np.ndarray,
                method: EnsembleMethod = EnsembleMethod.WEIGHTED) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        preds = {}
        for name, model in self._models.items():
            try:
                preds[name] = np.asarray(model.predict(X), dtype=float)
            except Exception:
                preds[name] = np.zeros(len(X))

        if not preds:
            return np.zeros(len(X))

        P = np.column_stack(list(preds.values()))
        names = list(preds.keys())

        if method == EnsembleMethod.MEAN:
            return P.mean(axis=1)
        if method == EnsembleMethod.MEDIAN:
            return np.median(P, axis=1)
        if method == EnsembleMethod.STACKING and self._stacker is not None:
            return self._stacker.predict(P)
        # Weighted
        weights = np.array([self._weights.get(n, 1.0) for n in names])
        weights = weights / (weights.sum() + 1e-8)
        return P @ weights

    def update_weights_online(self, y_true: float,
                               predictions_dict: Dict[str, float]) -> None:
        """Bayesian weight update based on recent prediction error."""
        for name, pred in predictions_dict.items():
            if name in self._weights:
                error = abs(pred - y_true)
                likelihood = np.exp(-error)
                self._weights[name] *= likelihood
        # Normalize
        total = sum(self._weights.values()) + 1e-8
        for name in self._weights:
            self._weights[name] /= total

    def get_weights(self) -> Dict[str, float]:
        return dict(self._weights)


class MetaLabelingFilter:
    """Uses a secondary classifier to determine size/confidence of primary signals."""

    def __init__(self, classifier: Any = None):
        self._clf = classifier or RandomForestClassifier(
            n_estimators=100, max_depth=4, random_state=42)
        self._fitted = False

    def fit(self, primary_signals: pd.Series,
            features: pd.DataFrame,
            y_true: pd.Series) -> "MetaLabelingFilter":
        """
        primary_signals: {-1, 0, 1} series
        y_true: actual returns aligned with signals
        """
        # Target: 1 if primary signal was correct, 0 otherwise
        signal_pnl = primary_signals * y_true
        meta_target = (signal_pnl > 0).astype(int)
        # Only fit on non-zero signals
        mask = primary_signals != 0
        if mask.sum() < 10:
            self._fitted = False
            return self
        X = features[mask].values
        y = meta_target[mask].values
        self._clf.fit(X, y)
        self._fitted = True
        return self

    def filter(self, primary_signals: pd.Series,
               features: pd.DataFrame) -> pd.DataFrame:
        """Returns filtered signal with confidence score."""
        mask = primary_signals != 0
        if not self._fitted or mask.sum() == 0:
            return pd.DataFrame({
                "signal": primary_signals,
                "confidence": np.where(mask, 0.5, 0.0),
                "sized_signal": primary_signals * 0.5,
            })
        proba = np.zeros(len(primary_signals))
        proba[mask] = self._clf.predict_proba(features[mask].values)[:, 1]
        sized = primary_signals * proba
        return pd.DataFrame({
            "signal": primary_signals,
            "confidence": proba,
            "sized_signal": sized,
        }, index=primary_signals.index)


class BayesianModelAveraging:
    """Bayesian Model Averaging with log marginal likelihood weighting."""

    def __init__(self, prior_weight: float = 1.0):
        self._prior = prior_weight
        self._log_likelihoods: Dict[str, float] = {}
        self._models: Dict[str, Any] = {}

    def add_model(self, name: str, model: Any) -> None:
        self._models[name] = model
        self._log_likelihoods[name] = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianModelAveraging":
        y = np.asarray(y, dtype=float)
        for name, model in self._models.items():
            try:
                model.fit(X, y)
                preds = np.asarray(model.predict(X), dtype=float)
                resid = y - preds
                sigma2 = np.var(resid) + 1e-8
                n = len(y)
                log_lik = -0.5 * n * np.log(2 * np.pi * sigma2) - 0.5 * np.sum(resid ** 2) / sigma2
                self._log_likelihoods[name] = float(log_lik)
            except Exception:
                self._log_likelihoods[name] = -1e8
        return self

    def _bma_weights(self) -> Dict[str, float]:
        log_liks = np.array(list(self._log_likelihoods.values()))
        log_liks -= log_liks.max()
        weights = np.exp(log_liks) * self._prior
        weights /= weights.sum() + 1e-8
        return dict(zip(self._log_likelihoods.keys(), weights))

    def predict(self, X: np.ndarray) -> np.ndarray:
        weights = self._bma_weights()
        result = np.zeros(len(X))
        for name, model in self._models.items():
            try:
                result += weights[name] * np.asarray(model.predict(X), dtype=float)
            except Exception:
                pass
        return result

    def model_probabilities(self) -> Dict[str, float]:
        return self._bma_weights()
