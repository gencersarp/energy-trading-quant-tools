from dataclasses import dataclass
import os

@dataclass
class TradingConfig:
    """
    Centralized configuration for the trading infrastructure.
    In a real hedge fund, these would be loaded from HashiCorp Vault or Kubernetes ConfigMaps.
    """
    gate_closure_mins: int = int(os.getenv("TRADING_GATE_CLOSURE_MINS", "60"))
    wind_shock_threshold_mw: float = float(os.getenv("TRADING_WIND_SHOCK_THRESHOLD_MW", "500.0"))
    max_position_mw: float = float(os.getenv("TRADING_MAX_POSITION_MW", "100.0"))
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    kafka_broker: str = os.getenv("KAFKA_BROKER", "localhost:9092")
