"""
Kafka producer for streaming transactions.
"""
import json
import logging
import time
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURES_FILE = Path("data/features.parquet")
KAFKA_TOPIC = "transactions"


def produce_transactions(rate_per_second: int = 100):
    """Produce transactions to Kafka at given rate."""
    logger.info(f"Starting producer at {rate_per_second} tx/sec")

    df = pd.read_parquet(FEATURES_FILE)
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()

    producer = KafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    interval = 1.0 / rate_per_second

    for _, row in df.iterrows():
        tx = {
            "step": int(row["step"]),
            "type": row["type"],
            "amount": float(row["amount"]),
            "nameOrig": row["nameOrig"],
            "oldbalanceOrg": float(row["oldbalanceOrg"]),
            "newbalanceOrg": float(row["newbalanceOrg"]),
            "nameDest": row["nameDest"],
            "oldbalanceDest": float(row["oldbalanceDest"]),
            "newbalanceDest": float(row["newbalanceDest"]),
            "isFraud": int(row["isFraud"]),
        }

        producer.send(KAFKA_TOPIC, value=tx)
        time.sleep(interval)

    producer.flush()
    logger.info("Producer complete")


if __name__ == "__main__":
    produce_transactions(rate_per_second=50)
