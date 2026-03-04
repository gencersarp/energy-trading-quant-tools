import pytest
from datetime import datetime, timedelta
from energytrading.core.config import TradingConfig
from energytrading.core.events import ForecastUpdate, OrderEvent
from energytrading.signals.wind_shock import StreamingWindShockDetector
from energytrading.execution.engine import EventDrivenExecution

def test_wind_shock_streaming_detector():
    """Validates that the stateful streaming detector correctly catches forecast deltas."""
    config = TradingConfig(wind_shock_threshold_mw=400.0)
    detector = StreamingWindShockDetector(config)
    
    now = datetime.now()
    deliv = now + timedelta(hours=2)
    
    # Event 1: Initial forecast (No shock)
    ev1 = ForecastUpdate(timestamp=now, zone="DK1", delivery_time=deliv, forecast_mw=1200.0, provider="ECMWF")
    assert detector.process_forecast(ev1) is None
    
    # Event 2: Minor revision (No shock)
    ev2 = ForecastUpdate(timestamp=now+timedelta(minutes=15), zone="DK1", delivery_time=deliv, forecast_mw=1100.0, provider="ECMWF")
    assert detector.process_forecast(ev2) is None
    
    # Event 3: Massive drop in wind (SHOCK!)
    ev3 = ForecastUpdate(timestamp=now+timedelta(minutes=30), zone="DK1", delivery_time=deliv, forecast_mw=600.0, provider="ECMWF")
    signal = detector.process_forecast(ev3)
    
    assert signal is not None
    assert signal.signal_type == "WIND_SHOCK"
    assert signal.strength > 0  # Wind dropped, prices go up -> Positive (BUY) strength

def test_execution_engine_risk_limits():
    """Validates that the Execution Engine enforces hedge fund laws and physical grid laws."""
    config = TradingConfig(gate_closure_mins=60, max_position_mw=50.0)
    engine = EventDrivenExecution(config)
    now = datetime.now()
    
    # Test 1: Successful Trade (Compliant)
    order1 = OrderEvent(
        timestamp=now, order_id="O1", zone="DK1", 
        delivery_time=now + timedelta(minutes=120), 
        side="BUY", qty_mw=40.0, price_limit=50.0
    )
    assert engine.on_order(order1) == True
    assert engine.positions["DK1"] == 40.0
    
    # Test 2: Position Limit Breach (Risk Block)
    order2 = OrderEvent(
        timestamp=now, order_id="O2", zone="DK1", 
        delivery_time=now + timedelta(minutes=120), 
        side="BUY", qty_mw=20.0, price_limit=50.0
    )
    assert engine.on_order(order2) == False  # 40 + 20 = 60 > 50 (Blocked)
    
    # Test 3: Gate Closure Breach (TSO Block)
    order3 = OrderEvent(
        timestamp=now, order_id="O3", zone="DK1", 
        delivery_time=now + timedelta(minutes=30), # Inside 60m gate
        side="SELL", qty_mw=10.0, price_limit=50.0
    )
    assert engine.on_order(order3) == False # Blocked due to grid delivery constraint
