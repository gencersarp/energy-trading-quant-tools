import polars as pl

def generate_orderbook_features(df_ob: pl.DataFrame) -> pl.DataFrame:
    """
    Generates real-time market microstructure features.
    Designed to run on Kafka streams or ClickHouse ticks.
    """
    return df_ob.with_columns([
        # Microstructure Imbalance
        ((pl.col("bid_vol") - pl.col("ask_vol")) / (pl.col("bid_vol") + pl.col("ask_vol"))).alias("ob_imbalance"),
        
        # Volatility & Liquidity
        pl.col("mid_price").rolling_std(window_size="5m").alias("rolling_vol_5m"),
        (pl.col("ask_price") - pl.col("bid_price")).alias("spread")
    ])

def align_fundamentals(df_market: pl.DataFrame, df_weather: pl.DataFrame) -> pl.DataFrame:
    """
    Aligns lower-frequency weather data (e.g. hourly ECMWF updates) 
    to high-frequency market tick data using a strict backward as-of join 
    on the bitemporal knowledge_time.
    """
    return df_market.join_asof(
        df_weather,
        left_on="timestamp",
        right_on="knowledge_time",
        strategy="backward"
    )
