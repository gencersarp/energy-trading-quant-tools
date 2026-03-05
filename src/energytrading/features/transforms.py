"""Signal transforms: Fourier, wavelet, rolling features, spike detection."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class FourierFeatures:
    """Discrete Fourier transform feature extraction."""

    def __init__(self, n_harmonics: int = 10):
        self.n_harmonics = n_harmonics
        self._freqs: Optional[np.ndarray] = None
        self._amplitudes: Optional[np.ndarray] = None

    def fit(self, series: np.ndarray) -> "FourierFeatures":
        s = np.asarray(series, dtype=float)
        fft = np.fft.rfft(s)
        n = len(s)
        freqs = np.fft.rfftfreq(n)
        amplitudes = np.abs(fft) / n
        # Keep top n_harmonics by amplitude (excluding DC)
        idx = np.argsort(amplitudes[1:])[::-1][: self.n_harmonics] + 1
        self._freqs = freqs[idx]
        self._amplitudes = amplitudes[idx]
        return self

    def transform(self, length: int) -> np.ndarray:
        """Reconstruct approximation from top harmonics."""
        if self._freqs is None:
            raise RuntimeError("Call fit() first")
        t = np.arange(length)
        approx = np.zeros(length)
        for freq, amp in zip(self._freqs, self._amplitudes):
            approx += amp * np.cos(2 * np.pi * freq * t)
        return approx

    def dominant_frequencies(self) -> np.ndarray:
        return self._freqs if self._freqs is not None else np.array([])

    @staticmethod
    def power_spectrum(series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s = np.asarray(series, dtype=float)
        fft = np.fft.rfft(s - s.mean())
        freqs = np.fft.rfftfreq(len(s))
        power = (np.abs(fft) ** 2) / len(s)
        return freqs, power

    @staticmethod
    def bandpass_filter(series: np.ndarray,
                        low_freq: float, high_freq: float) -> np.ndarray:
        """Simple FFT-based bandpass filter."""
        s = np.asarray(series, dtype=float)
        fft = np.fft.rfft(s)
        freqs = np.fft.rfftfreq(len(s))
        mask = (freqs >= low_freq) & (freqs <= high_freq)
        fft_filtered = fft * mask
        return np.fft.irfft(fft_filtered, n=len(s))


class WaveletFeatures:
    """Haar wavelet multi-resolution analysis."""

    @staticmethod
    def haar_decompose(series: np.ndarray,
                       levels: int = 3) -> List[np.ndarray]:
        """
        Returns [approximation, detail_1, detail_2, ...] coefficients.
        Pads to next power of 2 if necessary.
        """
        s = np.asarray(series, dtype=float)
        # Pad to power of 2
        n = len(s)
        n2 = 1
        while n2 < n:
            n2 <<= 1
        padded = np.zeros(n2)
        padded[:n] = s

        components = []
        current = padded.copy()
        for _ in range(levels):
            n_cur = len(current)
            half = n_cur // 2
            approx = (current[:n_cur - n_cur % 2:2] +
                      current[1:n_cur - n_cur % 2:2]) / np.sqrt(2)
            detail = (current[:n_cur - n_cur % 2:2] -
                      current[1:n_cur - n_cur % 2:2]) / np.sqrt(2)
            components.append(detail[:half])
            current = approx[:half]
        components.append(current)  # final approximation
        return components[::-1]  # [approx, detail_coarse, ..., detail_fine]

    @staticmethod
    def energy_by_level(series: np.ndarray, levels: int = 3) -> np.ndarray:
        components = WaveletFeatures.haar_decompose(series, levels)
        return np.array([np.sum(c ** 2) for c in components])

    @staticmethod
    def denoise(series: np.ndarray, levels: int = 3,
                threshold_factor: float = 0.5) -> np.ndarray:
        """Soft-threshold wavelet denoising."""
        components = WaveletFeatures.haar_decompose(series, levels)
        # Soft threshold on detail coefficients
        denoised_components = [components[0]]  # keep approximation
        for detail in components[1:]:
            threshold = threshold_factor * np.median(np.abs(detail)) / 0.6745
            soft = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0)
            denoised_components.append(soft)
        # Reconstruct (simplified: just use approximation for denoised signal)
        # Return the approximation upsampled to original length
        approx = components[0]
        scale = len(series) / len(approx)
        indices = np.minimum(
            (np.arange(len(series)) / scale).astype(int), len(approx) - 1)
        return approx[indices]


class RollingFeatureExtractor:
    """Standard rolling statistical features for time series."""

    def __init__(self, windows: List[int] = None):
        self.windows = windows or [5, 10, 21, 63]

    def transform(self, series: pd.Series,
                  features: Optional[List[str]] = None) -> pd.DataFrame:
        feat = features or ["mean", "std", "min", "max", "zscore",
                            "skew", "momentum", "roc"]
        cols: Dict[str, pd.Series] = {}
        for w in self.windows:
            roll = series.rolling(w, min_periods=1)
            if "mean" in feat:
                cols[f"mean_{w}"] = roll.mean()
            if "std" in feat:
                cols[f"std_{w}"] = roll.std()
            if "min" in feat:
                cols[f"min_{w}"] = roll.min()
            if "max" in feat:
                cols[f"max_{w}"] = roll.max()
            if "zscore" in feat:
                mean = roll.mean()
                std = roll.std()
                cols[f"zscore_{w}"] = (series - mean) / (std + 1e-8)
            if "skew" in feat:
                cols[f"skew_{w}"] = roll.skew()
            if "momentum" in feat:
                cols[f"momentum_{w}"] = series - series.shift(w)
            if "roc" in feat:
                cols[f"roc_{w}"] = series.pct_change(w)
        return pd.DataFrame(cols, index=series.index)

    def transform_df(self, df: pd.DataFrame,
                     columns: Optional[List[str]] = None) -> pd.DataFrame:
        cols_to_use = columns or df.select_dtypes(include=np.number).columns.tolist()
        frames = []
        for col in cols_to_use:
            sub = self.transform(df[col])
            sub.columns = [f"{col}_{c}" for c in sub.columns]
            frames.append(sub)
        return pd.concat([df] + frames, axis=1)


class SpikeDetector:
    """Detect and flag price spikes using multiple methods."""

    def __init__(self, method: str = "zscore", threshold: float = 3.0):
        self.method = method
        self.threshold = threshold

    def detect(self, series: pd.Series,
               window: int = 21) -> pd.Series:
        """Returns boolean Series: True = spike."""
        s = np.asarray(series)
        n = len(s)
        flags = np.zeros(n, dtype=bool)

        if self.method == "zscore":
            roll = series.rolling(window, min_periods=3)
            mean = roll.mean()
            std = roll.std()
            z = (series - mean) / (std + 1e-8)
            flags = np.abs(z.values) > self.threshold

        elif self.method == "iqr":
            roll = series.rolling(window, min_periods=3)
            q25 = roll.quantile(0.25)
            q75 = roll.quantile(0.75)
            iqr = q75 - q25
            lower = q25 - self.threshold * iqr
            upper = q75 + self.threshold * iqr
            flags = ((series < lower) | (series > upper)).values

        elif self.method == "mad":
            roll_median = series.rolling(window, min_periods=3).median()
            mad = (series - roll_median).abs().rolling(window, min_periods=3).median()
            modified_z = 0.6745 * (series - roll_median) / (mad + 1e-8)
            flags = np.abs(modified_z.values) > self.threshold

        elif self.method == "gesd":
            # Generalized ESD test (simplified iterative)
            vals = s.copy()
            for _ in range(int(self.threshold)):
                m, s_std = vals.mean(), vals.std()
                if s_std < 1e-10:
                    break
                z_scores = np.abs(vals - m) / s_std
                idx_max = np.argmax(z_scores)
                if z_scores[idx_max] > 2.5:
                    flags[idx_max] = True
                    vals[idx_max] = m  # mask out
                else:
                    break

        return pd.Series(flags, index=series.index, name="is_spike")

    def clip_spikes(self, series: pd.Series, window: int = 21) -> pd.Series:
        """Replace spikes with rolling median."""
        spikes = self.detect(series, window)
        roll_median = series.rolling(window, min_periods=1).median()
        return series.where(~spikes, roll_median)

    def spike_statistics(self, series: pd.Series,
                         window: int = 21) -> Dict[str, float]:
        spikes = self.detect(series, window)
        spike_vals = series[spikes]
        return {
            "n_spikes": int(spikes.sum()),
            "spike_rate": float(spikes.mean()),
            "mean_spike_magnitude": float(spike_vals.abs().mean()) if len(spike_vals) > 0 else 0.0,
            "max_spike": float(spike_vals.max()) if len(spike_vals) > 0 else 0.0,
            "min_spike": float(spike_vals.min()) if len(spike_vals) > 0 else 0.0,
        }


class FeatureSelector:
    """Select informative features using correlation and importance metrics."""

    @staticmethod
    def remove_correlated(df: pd.DataFrame,
                          threshold: float = 0.95) -> pd.DataFrame:
        """Drop features with pairwise correlation above threshold."""
        corr = df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
        return df.drop(columns=to_drop)

    @staticmethod
    def variance_threshold(df: pd.DataFrame,
                           min_var: float = 1e-4) -> pd.DataFrame:
        """Drop near-constant features."""
        variances = df.var()
        keep = variances[variances > min_var].index
        return df[keep]

    @staticmethod
    def mutual_information_scores(X: np.ndarray,
                                  y: np.ndarray) -> np.ndarray:
        """Approximate mutual information via correlation ratio."""
        n_features = X.shape[1] if X.ndim > 1 else 1
        X2d = X.reshape(-1, n_features)
        scores = np.zeros(n_features)
        for i in range(n_features):
            xi = X2d[:, i]
            corr = np.corrcoef(xi, y)[0, 1]
            scores[i] = corr ** 2
        return scores
