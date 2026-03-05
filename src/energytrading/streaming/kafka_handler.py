"""Kafka streaming handler for real-time energy market data."""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class KafkaConfig:
    bootstrap_servers: str = "localhost:9092"
    topic: str = "energy-prices"
    group_id: str = "energy-quant"
    auto_offset_reset: str = "latest"
    batch_size: int = 100


class EnergyPriceProducer:
    """Wraps KafkaProducer for energy price publishing."""

    def __init__(self, config: KafkaConfig):
        self.config = config
        self._producer = None
        self._mock_queue: List[Dict] = []
        self._use_mock = False
        try:
            from kafka import KafkaProducer as _KP
            self._producer = _KP(
                bootstrap_servers=config.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
        except (ImportError, Exception) as e:
            logger.warning(f"Kafka unavailable ({e}), using in-memory mock")
            self._use_mock = True

    def publish_price(self, zone: str, price: float, timestamp: Optional[str] = None,
                      metadata: Optional[Dict] = None) -> None:
        record = {
            "zone": zone,
            "price": price,
            "timestamp": timestamp or datetime.utcnow().isoformat(),
            **(metadata or {}),
        }
        if self._use_mock:
            self._mock_queue.append(record)
        else:
            self._producer.send(self.config.topic, record)

    def publish_batch(self, records: List[Dict]) -> None:
        for rec in records:
            if self._use_mock:
                self._mock_queue.append(rec)
            else:
                self._producer.send(self.config.topic, rec)

    def flush(self) -> None:
        if self._producer:
            self._producer.flush()

    def get_mock_messages(self) -> List[Dict]:
        return list(self._mock_queue)


class EnergyPriceConsumer:
    """Wraps KafkaConsumer for energy price subscription."""

    def __init__(self, config: KafkaConfig, handler_fn: Callable[[Dict], None]):
        self.config = config
        self._handler = handler_fn
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._consumer = None
        self._use_mock = False
        try:
            from kafka import KafkaConsumer as _KC
            self._consumer = _KC(
                config.topic,
                bootstrap_servers=config.bootstrap_servers,
                group_id=config.group_id,
                auto_offset_reset=config.auto_offset_reset,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            )
        except (ImportError, Exception):
            self._use_mock = True

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._consume_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._consumer:
            self._consumer.close()

    def _consume_loop(self) -> None:
        if self._use_mock:
            # Simulate consuming from mock queue
            while self._running:
                time.sleep(0.1)
        else:
            for msg in self._consumer:
                if not self._running:
                    break
                self.process_message(msg.value)

    def process_message(self, msg: Dict) -> None:
        try:
            self._handler(msg)
        except Exception as e:
            logger.error(f"Error processing message: {e}")


class StreamProcessor:
    """Stateful stream processor with pipeline of transformers."""

    def __init__(self):
        self._transformers: List[Callable] = []
        self._aggregators: List[Dict] = []
        self._window_buffers: Dict[str, List] = {}

    def add_transformer(self, fn: Callable[[Dict], Dict]) -> "StreamProcessor":
        self._transformers.append(fn)
        return self

    def add_aggregator(self, window_seconds: float,
                        agg_fn: Callable[[List], Any],
                        name: str = "agg") -> "StreamProcessor":
        self._aggregators.append({
            "window": window_seconds,
            "fn": agg_fn,
            "name": name,
            "buffer": [],
            "last_flush": time.time(),
        })
        return self

    def process(self, msg: Dict) -> Dict:
        result = dict(msg)
        for tf in self._transformers:
            result = tf(result)
        for agg in self._aggregators:
            agg["buffer"].append(result)
            now = time.time()
            if now - agg["last_flush"] >= agg["window"]:
                agg_result = agg["fn"](agg["buffer"])
                result[agg["name"]] = agg_result
                agg["buffer"] = []
                agg["last_flush"] = now
        return result


class MockKafkaStream:
    """In-memory mock Kafka stream for testing without Kafka infrastructure."""

    def __init__(self, topic: str = "test"):
        self.topic = topic
        self._messages: queue.Queue = queue.Queue()
        self._subscribers: List[Callable] = []

    def publish(self, message: Dict) -> None:
        self._messages.put(message)
        for sub in self._subscribers:
            try:
                sub(message)
            except Exception:
                pass

    def subscribe(self, fn: Callable[[Dict], None]) -> None:
        self._subscribers.append(fn)

    def consume(self, block: bool = False) -> Optional[Dict]:
        try:
            return self._messages.get(block=block, timeout=0.1)
        except queue.Empty:
            return None

    def consume_all(self) -> List[Dict]:
        messages = []
        while not self._messages.empty():
            try:
                messages.append(self._messages.get_nowait())
            except queue.Empty:
                break
        return messages
