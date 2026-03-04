import polars as pl
import logging

logging.basicConfig(level=logging.INFO)

class WindShockDetector:
    """
    Intern Project: Intraday Wind Shock Detection System
    Ingests weather forecasts, detects rapid revisions, estimates impact, and alerts traders.
    """
    def __init__(self, shock_threshold_mw: float = 500.0):
        self.threshold = shock_threshold_mw

    def detect_shocks(self, forecast_history: pl.DataFrame) -> pl.DataFrame:
        """
        Expects a bitemporal dataframe of wind forecasts.
        Calculates the delta between the current forecast and the prior forecast for the same event_time.
        """
        # Sort by delivery time (event), then by when we learned the forecast (knowledge)
        df = forecast_history.sort(["event_time", "knowledge_time"])
        
        # Calculate the MW change from the previous forecast for the exact same delivery hour
        df = df.with_columns(
            pl.col("wind_forecast_mw").diff().over("event_time").alias("forecast_revision_mw")
        )
        
        # Filter for shocks exceeding the threshold
        shocks = df.filter(pl.col("forecast_revision_mw").abs() >= self.threshold)
        
        for row in shocks.iter_rows(named=True):
            logging.warning(
                f"🚨 WIND SHOCK DETECTED! Delivery: {row['event_time']} | "
                f"Revision: {row['forecast_revision_mw']} MW | "
                f"Knowledge Time: {row['knowledge_time']}"
            )
            
        return shocks
