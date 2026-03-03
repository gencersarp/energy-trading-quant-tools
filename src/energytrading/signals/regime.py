import numpy as np
import pandas as pd


class RegimeSwitchingHMM:
    """
    Hidden Markov Model (HMM) for detecting 'Normal' vs 'Spike' regimes in power prices.
    Uses GaussianHMM if hmmlearn is installed, otherwise falls back to volatility thresholds.
    """
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.model = None

    def fit_predict(self, returns: pd.Series) -> pd.Series:
        """Fits a Gaussian HMM and decodes the hidden states (regimes)."""
        try:
            from hmmlearn.hmm import GaussianHMM
            X = returns.dropna().values.reshape(-1, 1)
            
            self.model = GaussianHMM(n_components=self.n_components, covariance_type="diag", n_iter=100)
            self.model.fit(X)
            states = self.model.predict(X)
            
            # Align state 1 with higher volatility (spike regime)
            vol_0 = np.var(X[states == 0])
            vol_1 = np.var(X[states == 1])
            if vol_0 > vol_1:
                states = 1 - states
                
            return pd.Series(states, index=returns.dropna().index)
            
        except ImportError:
            # Fallback simple rolling volatility threshold
            roll_vol = returns.rolling(window=24).std()
            thresh = roll_vol.quantile(0.90)
            states = (roll_vol > thresh).astype(int)
            return states.fillna(0)