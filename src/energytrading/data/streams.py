from kafka import KafkaConsumer, KafkaProducer
import json
import logging

logger = logging.getLogger(__name__)

class MarketDataStreamer:
    """Kafka stream handlers for EEX/ICE real-time data."""
    def __init__(self, brokers: list[str]):
        self.brokers = brokers
        
    def get_consumer(self, topic: str):
        return KafkaConsumer(
            topic,
            bootstrap_servers=self.brokers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
    def get_producer(self):
        return KafkaProducer(
            bootstrap_servers=self.brokers,
            value_serializer=lambda m: json.dumps(m).encode('utf-8')
        )