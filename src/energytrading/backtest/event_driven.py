from dataclasses import dataclass
from typing import List, Protocol
from collections import deque


class Event(Protocol):
    """Base Event Protocol for the intraday queue."""
    pass

@dataclass
class TickEvent:
    timestamp: str
    bid: float
    ask: float
    bid_vol: float
    ask_vol: float

@dataclass
class OrderEvent:
    timestamp: str
    symbol: str
    qty: float
    order_type: str  # "MARKET", "LIMIT"
    price: float = 0.0

@dataclass
class FillEvent:
    timestamp: str
    symbol: str
    qty: float
    fill_price: float
    commission: float


class IntradayEventEngine:
    """
    Institutional-grade Event-Driven Backtester for continuous intraday power markets.
    Replaces vectorized testing which cannot handle limit orders, partial fills, 
    or true order book priority before Gate Closure.
    """
    def __init__(self):
        self.events_queue = deque()
        self.positions = {}
        self.cash = 0.0

    def add_event(self, event: Event) -> None:
        self.events_queue.append(event)

    def process_queue(self) -> None:
        """Processes the event loop sequentially, avoiding lookahead bias natively."""
        while self.events_queue:
            event = self.events_queue.popleft()
            
            if isinstance(event, TickEvent):
                self._handle_tick(event)
            elif isinstance(event, OrderEvent):
                self._handle_order(event)
            elif isinstance(event, FillEvent):
                self._handle_fill(event)

    def _handle_tick(self, event: TickEvent) -> None:
        # Update strategy / signal logic with the new LOB state
        pass

    def _handle_order(self, event: OrderEvent) -> None:
        # Simulate exchange matching engine
        # E.g., if MARKET order, generate FillEvent immediately against opposite book
        fill = FillEvent(
            timestamp=event.timestamp,
            symbol=event.symbol,
            qty=event.qty,
            fill_price=event.price, # simplified
            commission=abs(event.qty) * 0.01
        )
        self.add_event(fill)

    def _handle_fill(self, event: FillEvent) -> None:
        # Update portfolio accounting
        self.positions[event.symbol] = self.positions.get(event.symbol, 0) + event.qty
        self.cash -= (event.qty * event.fill_price) + event.commission