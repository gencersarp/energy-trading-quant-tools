import lightgbm as lgb
import polars as pl
import joblib
import os

class IntradayPriceForecaster:
    """
    Senior Quant Upgrade: Production-ready LGBM forecaster.
    Adds serialization (save/load) for deployment to Kubernetes and strict typing.
    """
    def __init__(self, model_path: str = "models/saved/lgbm_intraday.pkl"):
        self.model_path = model_path
        self.model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            objective="regression",
            metric="rmse",
            n_jobs=-1
        )

    def train(self, X_train: pl.DataFrame, y_train: pl.Series):
        """Fits the model using Polars dataframes."""
        self.model.fit(X_train.to_pandas(), y_train.to_pandas())

    def predict(self, X_test: pl.DataFrame) -> pl.Series:
        """Scores live data for intraday trading signals."""
        preds = self.model.predict(X_test.to_pandas())
        return pl.Series("price_forecast", preds)

    def save(self):
        """Serializes the model for production serving."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def load(self):
        """Loads a pre-trained model from disk."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(f"Model file {self.model_path} not found. Train first.")
