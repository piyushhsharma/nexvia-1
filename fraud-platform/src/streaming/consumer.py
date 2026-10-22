"""
FastAPI consumer for scoring transactions from Kafka.
"""
import json
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from kafka import KafkaConsumer
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path("data/alerts.db")
app = FastAPI()


class Transaction(BaseModel):
    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrg: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float
    isFraud: int


class ScoreResponse(BaseModel):
    fraud_score: float
    is_fraud: bool
    latency_ms: float


def setup_db():
    """Setup SQLite database for alerts."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT,
            account_id TEXT,
            fraud_score REAL,
            is_fraud INTEGER,
            timestamp TIMESTAMP,
            latency_ms REAL
        )
    """)

    conn.commit()
    conn.close()


def get_features(tx: Dict) -> Dict:
    """Get features for a transaction (simplified)."""
    # In production, would fetch from Feast online store
    return {
        "tx_count_1h": 1,
        "tx_count_24h": 5,
        "amount_mean_24h": tx["amount"],
        "amount_vs_mean_ratio": 1.0,
        "amount_zscore": 0.0,
        "new_destination": 1,
        "is_transfer": 1 if tx["type"] == "TRANSFER" else 0,
        "is_cash_out": 1 if tx["type"] == "CASH_OUT" else 0,
        "org_balance_mismatch": 0,
        "amount_log": np.log1p(tx["amount"]),
    }


@app.post("/score", response_model=ScoreResponse)
async def score_transaction(tx: Transaction):
    """Score a single transaction."""
    start_time = time.time()

    # Simplified scoring (would use loaded model in production)
    features = get_features(tx.dict())
    fraud_score = 0.01  # Placeholder

    is_fraud = fraud_score > 0.5
    latency_ms = (time.time() - start_time) * 1000

    return ScoreResponse(
        fraud_score=fraud_score,
        is_fraud=is_fraud,
        latency_ms=latency_ms,
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/metrics")
async def metrics():
    """Prometheus-style metrics."""
    return {
        "requests_total": 0,
        "latency_p50_ms": 0.0,
        "latency_p95_ms": 0.0,
        "latency_p99_ms": 0.0,
    }


@app.get("/cases")
async def get_cases():
    """Get fraud cases."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 100")
    rows = cursor.fetchall()

    conn.close()

    return {"cases": rows}


def consume_transactions():
    """Consume transactions from Kafka and score them."""
    logger.info("Starting consumer...")

    consumer = KafkaConsumer(
        "transactions",
        bootstrap_servers="localhost:9092",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )

    for message in consumer:
        tx = message.value
        start_time = time.time()

        # Score transaction
        features = get_features(tx)
        fraud_score = 0.01  # Placeholder

        is_fraud = fraud_score > 0.5
        latency_ms = (time.time() - start_time) * 1000

        # Save to DB if fraud
        if is_fraud:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO alerts (transaction_id, account_id, fraud_score, is_fraud, timestamp, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (str(tx["step"]), tx["nameOrig"], fraud_score, int(is_fraud), datetime.now(), latency_ms))

            conn.commit()
            conn.close()


if __name__ == "__main__":
    import uvicorn

    setup_db()
    uvicorn.run(app, host="0.0.0.0", port=8000)
