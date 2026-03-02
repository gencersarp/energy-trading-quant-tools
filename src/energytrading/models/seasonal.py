import numpy as np
from sklearn.linear_model import LinearRegression

class SeasonalTrendModel:
    """Fourier-based seasonal and trend decomposition for power prices."""
    def __init__(self, fourier_order: int = 3):
        self.fourier_order = fourier_order
        self.model = LinearRegression()

    def _create_features(self, t: np.ndarray) -> np.ndarray:
        features = [t] # Linear trend
        # Annual seasonality assuming daily time steps (365.25)
        for i in range(1, self.fourier_order + 1):
            features.append(np.sin(2 * np.pi * i * t / 365.25))
            features.append(np.cos(2 * np.pi * i * t / 365.25))
        return np.column_stack(features)

    def fit(self, t: np.ndarray, y: np.ndarray) -> None:
        X = self._create_features(t)
        self.model.fit(X, y)

    def predict(self, t: np.ndarray) -> np.ndarray:
        X = self._create_features(t)
        return self.model.predict(X)