import polars as pl
from dataclasses import dataclass
from datetime import datetime

@dataclass
class BitemporalRecord:
    event_time: datetime      # When the physical event (e.g., power delivery) happens
    knowledge_time: datetime  # When the data actually arrived at our systems (API timestamp)
    value: float

def filter_as_of(df: pl.DataFrame, as_of_time: datetime) -> pl.DataFrame:
    """
    Filters a bitemporal dataframe to prevent lookahead bias in backtesting.
    Ensures that for any given historical moment, the model only sees the 
    weather/market forecast exactly as it was known at that exact microsecond.
    """
    return (
        df.filter(pl.col("knowledge_time") <= as_of_time)
        .sort(["event_time", "knowledge_time"])
        .group_by("event_time")
        .last()
    )

def query_clickhouse_bitemporal(query: str) -> pl.DataFrame:
    """
    Stub for querying ClickHouse columnar storage for tick/orderbook data.
    In a real setup, this would utilize clickhouse-connect for ultra-fast reads.
    """
    pass
