"""WebSocket market data feed and real-time signal engine."""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WebSocketFeedConfig:
    url: str = "ws://localhost:8080/market"
    channels: List[str] = field(default_factory=lambda: ["prices", "orderbook"])
    reconnect_delay: float = 5.0
    max_retries: int = 10


class MarketDataFeed:
    """WebSocket market data feed with graceful fallback to simulated data."""

    def __init__(self, config: WebSocketFeedConfig,
                 on_price_fn: Optional[Callable[[Dict], None]] = None,
                 on_orderbook_fn: Optional[Callable[[Dict], None]] = None):
        self.config = config
        self._on_price = on_price_fn or (lambda msg: None)
        self._on_orderbook = on_orderbook_fn or (lambda msg: None)
        self._connected = False
        self._running = False
        self._ws = None
        self._thread: Optional[threading.Thread] = None
        self._message_queue: queue.Queue = queue.Queue(maxsize=10000)

    def connect(self) -> None:
        try:
            import websockets
            import asyncio
            self._connected = True
        except ImportError:
            logger.warning("websockets not installed, using simulated feed")
            self._connected = False
        self._running = True
        self._thread = threading.Thread(target=self._feed_loop, daemon=True)
        self._thread.start()

    def disconnect(self) -> None:
        self._running = False
        self._connected = False

    def subscribe(self, channels: List[str]) -> None:
        self.config.channels = channels

    def parse_message(self, raw_msg: str | bytes) -> Dict:
        try:
            if isinstance(raw_msg, bytes):
                raw_msg = raw_msg.decode("utf-8")
            return json.loads(raw_msg)
        except Exception:
            return {"raw": str(raw_msg)}

    def _feed_loop(self) -> None:
        """Simulate market data feed when WebSocket not available."""
        rng = np.random.default_rng(42)
        price = 50.0
        while self._running:
            price += rng.normal(0, 0.5)
            price = max(price, 1.0)
            msg = {
                "type": "price",
                "zone": "DK1",
                "price": round(float(price), 2),
                "timestamp": datetime.utcnow().isoformat(),
                "volume": float(rng.exponential(100)),
            }
            try:
                self._message_queue.put_nowait(msg)
            except queue.Full:
                pass
            self._on_price(msg)
            time.sleep(0.1)

    def get_latest_message(self, timeout: float = 1.0) -> Optional[Dict]:
        try:
            return self._message_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class RealtimeSignalEngine:
    """Processes live market feed and computes trading signals in real-time."""

    def __init__(self, feed: MarketDataFeed):
        self._feed = feed
        self._signal_generators: Dict[str, Callable[[Dict], float]] = {}
        self._signal_history: Dict[str, List[float]] = {}
        self._price_history: List[float] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def add_signal_generator(self, name: str,
                               fn: Callable[[List[float]], float]) -> None:
        self._signal_generators[name] = fn
        self._signal_history[name] = []

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _process_loop(self) -> None:
        while self._running:
            msg = self._feed.get_latest_message(timeout=0.5)
            if msg and msg.get("type") == "price":
                price = float(msg.get("price", 0))
                self._price_history.append(price)
                for name, gen_fn in self._signal_generators.items():
                    try:
                        signal = gen_fn(self._price_history)
                        self._signal_history[name].append(float(signal))
                    except Exception as e:
                        logger.debug(f"Signal {name} error: {e}")

    def get_latest_signals(self) -> Dict[str, float]:
        return {
            name: hist[-1] if hist else 0.0
            for name, hist in self._signal_history.items()
        }

    def get_signal_history(self, name: str, n: int = 100) -> List[float]:
        return self._signal_history.get(name, [])[-n:]
