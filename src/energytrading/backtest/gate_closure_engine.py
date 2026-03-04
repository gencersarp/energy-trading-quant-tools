from dataclasses import dataclass
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)

@dataclass
class Order:
    id: str
    zone: str               # e.g., "DK1", "DK2"
    delivery_start: datetime # The physical delivery hour
    submit_time: datetime    # When the algo fired the order
    side: str               # "BUY" or "SELL"
    price: float
    qty_mw: float

class ExecutionEngine:
    """
    Realistic execution simulator for European Intraday Power Markets.
    Enforces Transmission System Operator (TSO) gate closures and models slippage.
    """
    def __init__(self, gate_closure_mins: int = 60):
        self.gate_closure_mins = gate_closure_mins
        self.positions = {"DK1": 0.0, "DK2": 0.0, "DE": 0.0}

    def submit_order(self, order: Order) -> bool:
        time_to_delivery = (order.delivery_start - order.submit_time).total_seconds() / 60.0
        
        # Realism Constraint: TSO Gate Closure
        # You cannot trade physical power intraday once the grid operator locks the hour.
        if time_to_delivery < self.gate_closure_mins:
            logging.error(f"[REJECTED] Order {order.id}: Passed gate closure for {order.delivery_start}")
            return False
        
        # Realism Constraint: Slippage Modeling
        # Assuming clearing the book moves the price against us by 0.1 EUR/MWh per 10 MW
        slippage = 0.1 * (order.qty_mw / 10.0) 
        fill_price = order.price + slippage if order.side == "BUY" else order.price - slippage
        
        # Position Management
        qty_signed = order.qty_mw if order.side == "BUY" else -order.qty_mw
        self.positions[order.zone] += qty_signed
        
        logging.info(f"[FILLED] Order {order.id} | {order.side} {order.qty_mw}MW @ {fill_price:.2f} EUR/MWh | NOP {order.zone}: {self.positions[order.zone]} MW")
        return True
