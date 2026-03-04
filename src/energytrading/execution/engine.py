import logging
from datetime import datetime
from ..core.events import OrderEvent
from ..core.config import TradingConfig

logger = logging.getLogger(__name__)

class EventDrivenExecution:
    """
    Senior Quant Upgrade: A strict, risk-managed Execution Engine.
    Processes OrderEvents asynchronously. It strictly enforces physical trading laws
    (TSO gate closures) and quantitative hedge fund laws (Hard Position Limits).
    """
    def __init__(self, config: TradingConfig):
        self.config = config
        # Tracks the Net Open Position (NOP) per bidding zone
        self.positions = {"DK1": 0.0, "DK2": 0.0, "DE": 0.0, "SE3": 0.0, "NO2": 0.0}

    def on_order(self, order: OrderEvent) -> bool:
        time_to_delivery = (order.delivery_time - order.timestamp).total_seconds() / 60.0
        
        # 1. Physical Grid Constraint: TSO Gate Closure
        if time_to_delivery < self.config.gate_closure_mins:
            logger.error(f"[RISK REJECT] Order {order.order_id}: Past {self.config.gate_closure_mins}m gate closure.")
            return False
            
        # 2. Risk Management: Position Limit Breach Check
        qty_signed = order.qty_mw if order.side == "BUY" else -order.qty_mw
        new_position = self.positions.get(order.zone, 0.0) + qty_signed
        
        if abs(new_position) > self.config.max_position_mw:
            logger.error(f"[RISK REJECT] Order {order.order_id}: NOP {new_position} exceeds limit {self.config.max_position_mw}MW.")
            return False

        # 3. Market Microstructure: Simulated Slippage
        slippage = 0.1 * (order.qty_mw / 10.0)
        fill_price = order.price_limit + slippage if order.side == "BUY" else order.price_limit - slippage
        
        # 4. State Update
        self.positions[order.zone] = new_position
        logger.info(f"[FILLED] {order.order_id} | {order.side} {order.qty_mw}MW @ {fill_price:.2f} EUR/MWh | NOP {order.zone}: {self.positions[order.zone]}MW")
        
        return True
