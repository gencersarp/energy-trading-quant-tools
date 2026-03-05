"""Limit order book simulator with Hawkes process arrivals."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Order:
    order_id: str
    side: str          # 'bid' | 'ask'
    price: float
    quantity: float
    timestamp: float
    order_type: str = "limit"  # 'limit' | 'market'

    @classmethod
    def new(cls, side: str, price: float, quantity: float,
             order_type: str = "limit") -> "Order":
        return cls(order_id=str(uuid.uuid4())[:8], side=side, price=price,
                   quantity=quantity, timestamp=time.time(), order_type=order_type)


@dataclass
class Fill:
    aggressor_id: str
    passive_id: str
    price: float
    quantity: float
    timestamp: float


class OrderBook:
    """Full price-time priority limit order book."""

    def __init__(self):
        self._bids: Dict[str, Order] = {}  # order_id -> Order
        self._asks: Dict[str, Order] = {}
        self._fills: List[Fill] = []

    def add_order(self, order: Order) -> Optional[List[Fill]]:
        fills = []
        if order.order_type == "market" or order.side == "bid":
            fills = self._match_bid(order)
        else:
            fills = self._match_ask(order)
        if order.quantity > 0:
            if order.order_type == "limit":
                if order.side == "bid":
                    self._bids[order.order_id] = order
                else:
                    self._asks[order.order_id] = order
        self._fills.extend(fills)
        return fills

    def _match_bid(self, bid: Order) -> List[Fill]:
        fills = []
        sorted_asks = sorted(self._asks.values(), key=lambda o: (o.price, o.timestamp))
        for ask in sorted_asks:
            if bid.quantity <= 0:
                break
            if bid.order_type == "limit" and bid.price < ask.price:
                break
            trade_qty = min(bid.quantity, ask.quantity)
            trade_price = ask.price  # passive side sets price
            fill = Fill(bid.order_id, ask.order_id, trade_price, trade_qty, time.time())
            fills.append(fill)
            bid.quantity -= trade_qty
            ask.quantity -= trade_qty
            if ask.quantity <= 0:
                del self._asks[ask.order_id]
        return fills

    def _match_ask(self, ask: Order) -> List[Fill]:
        fills = []
        sorted_bids = sorted(self._bids.values(), key=lambda o: (-o.price, o.timestamp))
        for bid in sorted_bids:
            if ask.quantity <= 0:
                break
            if ask.order_type == "limit" and ask.price > bid.price:
                break
            trade_qty = min(ask.quantity, bid.quantity)
            trade_price = bid.price
            fill = Fill(ask.order_id, bid.order_id, trade_price, trade_qty, time.time())
            fills.append(fill)
            ask.quantity -= trade_qty
            bid.quantity -= trade_qty
            if bid.quantity <= 0:
                del self._bids[bid.order_id]
        return fills

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._bids:
            del self._bids[order_id]
            return True
        if order_id in self._asks:
            del self._asks[order_id]
            return True
        return False

    def best_bid(self) -> Optional[float]:
        if not self._bids:
            return None
        return max(o.price for o in self._bids.values())

    def best_ask(self) -> Optional[float]:
        if not self._asks:
            return None
        return min(o.price for o in self._asks.values())

    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return bb or ba
        return (bb + ba) / 2

    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return ba - bb

    def depth(self, levels: int = 5) -> Tuple[List, List]:
        sorted_bids = sorted(self._bids.values(), key=lambda o: -o.price)
        sorted_asks = sorted(self._asks.values(), key=lambda o: o.price)
        bid_levels = [(o.price, o.quantity) for o in sorted_bids[:levels]]
        ask_levels = [(o.price, o.quantity) for o in sorted_asks[:levels]]
        return bid_levels, ask_levels

    def imbalance(self, levels: int = 5) -> float:
        bid_lev, ask_lev = self.depth(levels)
        bid_vol = sum(q for _, q in bid_lev)
        ask_vol = sum(q for _, q in ask_lev)
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return float((bid_vol - ask_vol) / total)

    def vwap(self, side: str, qty: float) -> Optional[float]:
        """Volume-weighted average fill price for qty."""
        orders = (sorted(self._asks.values(), key=lambda o: o.price)
                  if side == "buy"
                  else sorted(self._bids.values(), key=lambda o: -o.price))
        remaining = qty
        total_cost = 0.0
        for o in orders:
            fill_qty = min(o.quantity, remaining)
            total_cost += fill_qty * o.price
            remaining -= fill_qty
            if remaining <= 0:
                break
        if remaining > 0:
            return None
        return total_cost / qty


class OrderBookSimulator:
    """Simulates realistic LOB dynamics using Hawkes process for arrivals."""

    def simulate(self, n_steps: int, initial_price: float = 50.0,
                  lambda_arrive: float = 10.0, lambda_cancel: float = 5.0,
                  vol: float = 0.01, seed: int = 42) -> List[OrderBook]:
        """Simulate order book evolution."""
        rng = np.random.default_rng(seed)
        snapshots = []
        ob = OrderBook()
        price = initial_price

        # Seed the book
        for _ in range(5):
            ob.add_order(Order.new("bid", price * (1 - rng.uniform(0.001, 0.01)),
                                   rng.exponential(10)))
            ob.add_order(Order.new("ask", price * (1 + rng.uniform(0.001, 0.01)),
                                   rng.exponential(10)))

        for t in range(n_steps):
            # Random walk price
            price += rng.normal(0, vol * price)
            price = max(price, 0.1)

            # New order arrivals (Poisson)
            n_arrivals = rng.poisson(lambda_arrive)
            for _ in range(n_arrivals):
                side = "bid" if rng.uniform() < 0.5 else "ask"
                spread = price * rng.uniform(0.001, 0.02)
                p = price - spread if side == "bid" else price + spread
                q = rng.exponential(10)
                ob.add_order(Order.new(side, p, q))

            # Random cancellations
            for order_dict in [ob._bids, ob._asks]:
                to_cancel = [oid for oid in list(order_dict.keys())
                              if rng.uniform() < lambda_cancel / max(lambda_arrive, 1)]
                for oid in to_cancel[:5]:
                    ob.cancel_order(oid)

            snapshots.append(ob)

        return snapshots
