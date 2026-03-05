"""Turbulence index, absorption ratio, and composite stress indicators."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class TurbulenceIndex:
    """Mahalanobis-distance-based turbulence index for energy markets."""

    def compute(self, returns_df: pd.DataFrame, lookback: int = 252) -> pd.Series:
        result = pd.Series(index=returns_df.index, dtype=float, name="turbulence")
        for i in range(lookback, len(returns_df)):
            hist = returns_df.iloc[max(0, i - lookback): i].dropna()
            mu = hist.mean().values
            cov = hist.cov().values + np.eye(len(mu)) * 1e-6
            r = returns_df.iloc[i].fillna(0).values
            try:
                cov_inv = np.linalg.inv(cov)
                turb = float((r - mu) @ cov_inv @ (r - mu))
                result.iloc[i] = turb
            except np.linalg.LinAlgError:
                result.iloc[i] = 0.0
        return result.fillna(0)

    def threshold(self, turbulence_series: pd.Series,
                   percentile: float = 95) -> pd.Series:
        """Return boolean mask of crisis periods."""
        thresh = turbulence_series.quantile(percentile / 100)
        return (turbulence_series > thresh).rename("is_crisis")


class AbsorptionRatio:
    """Measures systemic risk via PCA variance concentration."""

    def compute(self, returns_df: pd.DataFrame, n_eigenvectors: int | None = None,
                 fraction: float = 0.20, window: int = 60) -> pd.Series:
        """
        AR_t = variance explained by top n_eigenvectors / total variance.
        Increasing AR → more systemic risk.
        """
        n_assets = returns_df.shape[1]
        n_eig = n_eigenvectors or max(1, int(n_assets * fraction))
        result = pd.Series(index=returns_df.index, dtype=float, name="absorption_ratio")
        for i in range(window, len(returns_df)):
            r = returns_df.iloc[max(0, i - window): i].dropna()
            if r.shape[0] < 10 or r.shape[1] < 2:
                continue
            try:
                pca = PCA(n_components=min(n_eig, r.shape[1]))
                pca.fit(r)
                ar = float(pca.explained_variance_ratio_[:n_eig].sum())
                result.iloc[i] = ar
            except Exception:
                result.iloc[i] = 0.0
        return result.fillna(0)


class CorrelationRegimeDetector:
    """Detects correlation regime breaks in multi-asset energy returns."""

    def rolling_correlation_matrix(self, returns_df: pd.DataFrame,
                                    window: int = 60) -> list[np.ndarray]:
        matrices = []
        for i in range(window, len(returns_df)):
            r = returns_df.iloc[max(0, i - window): i].dropna()
            corr = r.corr().values if r.shape[0] > 2 else np.eye(r.shape[1])
            matrices.append(corr)
        return matrices

    def average_correlation(self, returns_df: pd.DataFrame,
                             window: int = 60) -> pd.Series:
        result = pd.Series(index=returns_df.index[window:], dtype=float,
                           name="avg_correlation")
        matrices = self.rolling_correlation_matrix(returns_df, window)
        for i, mat in enumerate(matrices):
            n = mat.shape[0]
            if n > 1:
                off_diag = mat[np.triu_indices(n, k=1)]
                result.iloc[i] = float(off_diag.mean())
        return result

    def detect_correlation_breaks(self, corr_matrices: list[np.ndarray],
                                   threshold: float = 0.3) -> list[int]:
        """Detect large jumps in correlation structure."""
        breaks = []
        for i in range(1, len(corr_matrices)):
            diff = np.linalg.norm(corr_matrices[i] - corr_matrices[i - 1], "fro")
            if diff > threshold:
                breaks.append(i)
        return breaks


class MarketStressComposite:
    """Composite stress score combining turbulence + absorption ratio + correlation."""

    def __init__(self):
        self._ti = TurbulenceIndex()
        self._ar = AbsorptionRatio()
        self._cr = CorrelationRegimeDetector()

    def compute(self, returns_df: pd.DataFrame,
                 weights: tuple = (0.4, 0.3, 0.3)) -> pd.Series:
        """Returns composite stress in [0,1]."""
        turb = self._ti.compute(returns_df)
        absorb = self._ar.compute(returns_df)
        avg_corr = self._cr.average_correlation(returns_df)

        # Normalize each to [0,1]
        def _normalize(s: pd.Series) -> pd.Series:
            s_clean = s.dropna()
            if s_clean.empty or s_clean.std() == 0:
                return pd.Series(0.0, index=s.index)
            return (s - s_clean.min()) / (s_clean.max() - s_clean.min() + 1e-8)

        t_n = _normalize(turb)
        a_n = _normalize(absorb)
        c_n = _normalize(avg_corr.reindex(returns_df.index).fillna(0))

        composite = (weights[0] * t_n + weights[1] * a_n + weights[2] * c_n)
        return composite.clip(0, 1).rename("stress_composite")

    def alert(self, returns_df: pd.DataFrame, threshold: float = 0.8) -> bool:
        stress = self.compute(returns_df)
        return bool(stress.dropna().iloc[-1] >= threshold if len(stress.dropna()) > 0 else False)
