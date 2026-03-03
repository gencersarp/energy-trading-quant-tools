import pandas as pd
import numpy as np


class LSTMForecasterStub:
    """Stub for Deep Learning / LSTM based price and load forecasting."""
    def __init__(self, sequence_length: int = 24):
        self.sequence_length = sequence_length
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Simulates training an LSTM network."""
        self.is_fitted = True

    def predict(self, recent_sequence: pd.Series) -> float:
        """Predicts the next timestep t+1 using mock weighted decay."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
            
        if len(recent_sequence) < self.sequence_length:
            return recent_sequence.iloc[-1]
            
        weights = np.linspace(0.1, 1.0, self.sequence_length)
        weights /= weights.sum()
        
        recent_vals = recent_sequence.iloc[-self.sequence_length:].values
        return float(np.dot(recent_vals, weights))