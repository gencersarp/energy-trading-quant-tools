import numpy as np
from hmmlearn import hmm


class RegimeSwitchingModel:
    """2-Regime Markov Switching Model for Base vs Spike regimes."""
    def __init__(self):
        self.model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)

    def fit(self, prices: np.ndarray) -> None:
        """Fits the Hidden Markov Model to price returns."""
        returns = np.diff(np.log(prices))
        self.model.fit(returns.reshape(-1, 1))

    def predict_regime(self, prices: np.ndarray) -> np.ndarray:
        """Predicts the current regime (0 or 1)."""
        returns = np.diff(np.log(prices))
        states = self.model.predict(returns.reshape(-1, 1))
        return np.insert(states, 0, states[0])