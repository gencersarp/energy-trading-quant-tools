import logging
from typing import Dict, Optional
from ..core.events import ForecastUpdate, SignalEvent
from ..core.config import TradingConfig

logger = logging.getLogger(__name__)

class StreamingWindShockDetector:
    """
    Senior Quant Upgrade: Transformed from a static batch script to a stateful, event-driven streaming service.
    Maintains an internal cache (which would be Redis in production) to instantly detect delta changes 
    the millisecond a new forecast arrives via Kafka.
    """
    def __init__(self, config: TradingConfig):
        self.threshold = config.wind_shock_threshold_mw
        # Simulates a fast-path Redis cache: { (zone, delivery_time): last_forecast_mw }
        self._state_cache: Dict[str, float] = {}

    def process_forecast(self, event: ForecastUpdate) -> Optional[SignalEvent]:
        state_key = f"{event.zone}_{event.delivery_time.isoformat()}"
        last_forecast = self._state_cache.get(state_key)
        
        # Update state with new forecast
        self._state_cache[state_key] = event.forecast_mw
        
        if last_forecast is not None:
            delta = event.forecast_mw - last_forecast
            if abs(delta) >= self.threshold:
                logger.warning(f"🚨 SHOCK: {event.zone} | Delta: {delta} MW | Delivery: {event.delivery_time}")
                
                # Normalize strength for the execution algorithm (-1.0 to 1.0)
                # If wind drops, power prices spike (Buy signal = Positive strength)
                strength = max(-1.0, min(1.0, -delta / 2000.0)) 
                
                return SignalEvent(
                    timestamp=event.timestamp,
                    zone=event.zone,
                    delivery_time=event.delivery_time,
                    signal_type="WIND_SHOCK",
                    strength=strength
                )
        return None
