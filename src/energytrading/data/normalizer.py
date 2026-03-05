"""Data normalization: timezone alignment, contract rolling, forward curve stitching."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class TimezoneAligner:
    """
    Align time series from different timezones into a common UTC index.
    Handles DST gaps and overlaps gracefully.
    """

    def __init__(self, target_tz: str = "UTC"):
        self.target_tz = target_tz

    def localize(self, series: pd.Series, source_tz: str,
                 ambiguous: str = "NaT",
                 nonexistent: str = "NaT") -> pd.Series:
        idx = pd.DatetimeIndex(series.index)
        if idx.tz is None:
            try:
                idx = idx.tz_localize(source_tz, ambiguous=ambiguous,
                                      nonexistent=nonexistent)
            except Exception:
                idx = idx.tz_localize(source_tz, ambiguous="NaT")
        idx_utc = idx.tz_convert(self.target_tz)
        return series.set_axis(idx_utc).sort_index()

    def align_multiple(self, series_dict: Dict[str, pd.Series],
                       freq: str = "H",
                       fill_method: Optional[str] = "ffill") -> pd.DataFrame:
        """
        Align multiple series with potentially different timezones
        onto a common UTC DatetimeIndex at the given frequency.
        """
        aligned = {}
        for name, s in series_dict.items():
            if hasattr(s.index, "tz") and s.index.tz is not None:
                s = s.copy()
                s.index = s.index.tz_convert(self.target_tz)
            aligned[name] = s

        # Create common index
        start = min(s.index.min() for s in aligned.values())
        end = max(s.index.max() for s in aligned.values())
        common_idx = pd.date_range(start, end, freq=freq, tz=self.target_tz)

        df = pd.DataFrame(index=common_idx)
        for name, s in aligned.items():
            df[name] = s.reindex(common_idx)

        if fill_method:
            df = getattr(df, fill_method)()
        return df


class DataNormalizer:
    """
    Normalize/standardize price series and handle outliers.
    Supports z-score, min-max, robust (IQR), and log transforms.
    """

    def __init__(self, method: str = "zscore"):
        self.method = method
        self._params: Dict = {}

    def fit(self, df: pd.DataFrame) -> "DataNormalizer":
        for col in df.select_dtypes(include=np.number).columns:
            s = df[col].dropna()
            if self.method == "zscore":
                self._params[col] = {"mean": s.mean(), "std": s.std()}
            elif self.method == "minmax":
                self._params[col] = {"min": s.min(), "max": s.max()}
            elif self.method == "robust":
                q25, q75 = s.quantile(0.25), s.quantile(0.75)
                self._params[col] = {"median": s.median(), "iqr": q75 - q25}
            elif self.method == "log":
                self._params[col] = {"shift": max(0, -s.min() + 1e-6)}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for col in df.select_dtypes(include=np.number).columns:
            if col not in self._params:
                continue
            p = self._params[col]
            if self.method == "zscore":
                result[col] = (df[col] - p["mean"]) / (p["std"] + 1e-8)
            elif self.method == "minmax":
                result[col] = (df[col] - p["min"]) / (p["max"] - p["min"] + 1e-8)
            elif self.method == "robust":
                result[col] = (df[col] - p["median"]) / (p["iqr"] + 1e-8)
            elif self.method == "log":
                result[col] = np.log(df[col] + p["shift"] + 1e-8)
        return result

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for col in df.select_dtypes(include=np.number).columns:
            if col not in self._params:
                continue
            p = self._params[col]
            if self.method == "zscore":
                result[col] = df[col] * p["std"] + p["mean"]
            elif self.method == "minmax":
                result[col] = df[col] * (p["max"] - p["min"]) + p["min"]
            elif self.method == "robust":
                result[col] = df[col] * p["iqr"] + p["median"]
            elif self.method == "log":
                result[col] = np.exp(df[col]) - p["shift"]
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def clip_outliers(self, df: pd.DataFrame,
                      n_sigma: float = 4.0) -> pd.DataFrame:
        result = df.copy()
        for col in df.select_dtypes(include=np.number).columns:
            mean = df[col].mean()
            std = df[col].std()
            result[col] = df[col].clip(mean - n_sigma * std,
                                       mean + n_sigma * std)
        return result


class ContractRoller:
    """
    Roll futures contract series: stitch front-month contracts into
    a continuous price series with back-adjust or panama canal methods.
    """

    def __init__(self, method: str = "back_adjust"):
        self.method = method

    def roll(self, contracts: List[Tuple[pd.Timestamp, pd.Series]],
             roll_day: int = -5) -> pd.Series:
        """
        contracts: list of (expiry_date, price_series) tuples, sorted by expiry.
        roll_day: number of business days before expiry to roll.
        Returns continuous price series.
        """
        if not contracts:
            return pd.Series(dtype=float)

        pieces: List[pd.Series] = []
        cumulative_adjustment = 0.0

        for i, (expiry, series) in enumerate(contracts):
            if i < len(contracts) - 1:
                next_expiry, next_series = contracts[i + 1]
                roll_date = expiry - pd.offsets.BusinessDay(abs(roll_day))
                current_slice = series[series.index <= roll_date]
                next_slice_at_roll = next_series.reindex([roll_date], method="nearest")
                current_at_roll = series.reindex([roll_date], method="nearest")

                if len(current_slice) == 0:
                    continue

                if self.method == "back_adjust":
                    if (len(next_slice_at_roll) > 0 and
                            len(current_at_roll) > 0 and
                            not next_slice_at_roll.empty and
                            not current_at_roll.empty):
                        gap = (float(next_slice_at_roll.iloc[0]) -
                               float(current_at_roll.iloc[0]))
                        cumulative_adjustment += gap
                    adjusted = current_slice - cumulative_adjustment
                    pieces.append(adjusted)
                else:  # panama/no adjustment
                    pieces.append(current_slice)
            else:
                # Last contract: take everything after last roll
                if pieces:
                    last_date = pieces[-1].index[-1]
                    remaining = series[series.index > last_date]
                    pieces.append(remaining - cumulative_adjustment
                                  if self.method == "back_adjust" else remaining)
                else:
                    pieces.append(series)

        if not pieces:
            return pd.Series(dtype=float)
        return pd.concat(pieces).sort_index().groupby(level=0).first()


class ForwardCurveStitcher:
    """
    Build a continuous forward curve from overlapping contract quotes.
    Handles seasonal products (M1, Q1, Cal, etc.).
    """

    def __init__(self, interpolation: str = "linear"):
        self.interpolation = interpolation

    def stitch(self, quotes: Dict[str, Tuple[pd.Timestamp, pd.Timestamp, float]],
               start_date: Optional[pd.Timestamp] = None,
               end_date: Optional[pd.Timestamp] = None) -> pd.Series:
        """
        quotes: {contract_name: (start, end, price)}.
        Returns daily forward price curve.
        """
        if not quotes:
            return pd.Series(dtype=float)

        all_dates = []
        all_prices = []

        for name, (start, end, price) in quotes.items():
            dates = pd.date_range(start, end, freq="D")
            all_dates.extend(dates)
            all_prices.extend([price] * len(dates))

        curve = pd.Series(all_prices, index=pd.DatetimeIndex(all_dates))
        curve = curve.groupby(level=0).mean()  # average where overlapping

        if start_date and end_date:
            full_idx = pd.date_range(start_date, end_date, freq="D")
            curve = curve.reindex(full_idx)
            if self.interpolation == "linear":
                curve = curve.interpolate(method="linear")
            elif self.interpolation == "cubic":
                curve = curve.interpolate(method="cubic")
            curve = curve.fillna(method="ffill").fillna(method="bfill")

        return curve

    def flatten_seasonality(self, curve: pd.Series,
                             reference_year: int = None) -> pd.Series:
        """Remove seasonal component from forward curve using ratio method."""
        if reference_year is None:
            reference_year = curve.index[0].year
        monthly_avg = curve.groupby(curve.index.month).mean()
        annual_avg = monthly_avg.mean()
        seasonal_index = monthly_avg / (annual_avg + 1e-8)
        deseasonalized = curve.copy()
        for month, factor in seasonal_index.items():
            mask = curve.index.month == month
            deseasonalized[mask] = curve[mask] / (factor + 1e-8)
        return deseasonalized
