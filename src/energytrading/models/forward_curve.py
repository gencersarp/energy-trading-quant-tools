import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

class ForwardCurveBuilder:
    """Builds smooth continuous forward curves from discrete market quotes."""
    def __init__(self):
        self.curve = None
        self.daily_curve = None

    def bootstrap_from_blocks(self, blocks: pd.DataFrame) -> pd.Series:
        """
        Bootstraps a daily curve from coarse block contracts (Monthly/Quarterly).
        blocks: DataFrame with columns ['start', 'end', 'price']
        """
        dates = []
        prices = []
        for _, row in blocks.iterrows():
            dr = pd.date_range(row['start'], row['end'], freq='D')
            dates.extend(dr)
            prices.extend([row['price']] * len(dr))
        
        self.daily_curve = pd.Series(prices, index=dates)
        return self.daily_curve

    def smooth_spline(self, anchor_dates: pd.DatetimeIndex, prices: np.ndarray) -> pd.Series:
        """
        Applies a cubic spline over specific anchor dates (e.g. mid-month)
        to generate a perfectly smooth daily forward curve.
        """
        numeric_dates = (anchor_dates - anchor_dates[0]).days.values
        cs = CubicSpline(numeric_dates, prices, bc_type='natural')
        
        full_dates = pd.date_range(anchor_dates[0], anchor_dates[-1], freq='D')
        full_numeric = (full_dates - anchor_dates[0]).days.values
        
        smoothed_prices = cs(full_numeric)
        self.daily_curve = pd.Series(smoothed_prices, index=full_dates)
        return self.daily_curve