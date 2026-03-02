import numpy as np
from scipy.interpolate import CubicSpline

class ForwardCurveBuilder:
    """Builds smooth continuous forward curves from discrete market quotes."""
    def __init__(self):
        self.curve = None

    def build(self, tenors: np.ndarray, prices: np.ndarray, method: str = 'cubic') -> None:
        """
        Fits a curve to market forward prices.
        tenors: Time to maturity in years
        prices: Market quoted prices
        """
        if method == 'cubic':
            self.curve = CubicSpline(tenors, prices, bc_type='natural')
        else:
            raise ValueError(f"Unsupported interpolation method: {method}")

    def get_price(self, t: float) -> float:
        if self.curve is None:
            raise ValueError("Forward curve has not been built yet. Call build() first.")
        return float(self.curve(t))