import lightgbm as lgb
import polars as pl

class IntradayPriceForecaster:
    """
    Gradient Boosting forecaster for short-term Nord Pool / EPEX intraday power prices.
    Uses LightGBM for non-linear feature interactions (e.g., wind dropping + tight spread).
    """
    def __init__(self):
        self.model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            objective="regression",
            metric="rmse"
        )

    def train(self, X_train: pl.DataFrame, y_train: pl.Series):
        """Fits the model using Polars dataframes converted to Pandas temporarily."""
        self.model.fit(X_train.to_pandas(), y_train.to_pandas())

    def predict(self, X_test: pl.DataFrame) -> pl.Series:
        """Scores live data for intraday trading signals."""
        preds = self.model.predict(X_test.to_pandas())
        return pl.Series("price_forecast", preds)
