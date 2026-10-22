"""
Simple feature store using parquet for offline and SQLite for online serving.
"""
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import sqlite3

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURES_FILE = Path("data/features.parquet")
ONLINE_DB = Path("data/online_store.db")


def setup_online_store():
    """Setup SQLite online store."""
    conn = sqlite3.connect(ONLINE_DB)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS account_features (
            account_id TEXT PRIMARY KEY,
            tx_count_1h INTEGER,
            tx_count_24h INTEGER,
            amount_mean_24h REAL,
            amount_vs_mean_ratio REAL,
            amount_zscore REAL,
            new_destination INTEGER,
            is_transfer INTEGER,
            is_cash_out INTEGER,
            org_balance_mismatch INTEGER,
            amount_log REAL,
            updated_at TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    logger.info(f"Online store setup at {ONLINE_DB}")


def get_online_features(account_id: str) -> dict:
    """Get features for a single account from online store."""
    conn = sqlite3.connect(ONLINE_DB)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT tx_count_1h, tx_count_24h, amount_mean_24h, amount_vs_mean_ratio,
               amount_zscore, new_destination, is_transfer, is_cash_out,
               org_balance_mismatch, amount_log
        FROM account_features
        WHERE account_id = ?
    """, (account_id,))

    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            "tx_count_1h": row[0],
            "tx_count_24h": row[1],
            "amount_mean_24h": row[2],
            "amount_vs_mean_ratio": row[3],
            "amount_zscore": row[4],
            "new_destination": row[5],
            "is_transfer": row[6],
            "is_cash_out": row[7],
            "org_balance_mismatch": row[8],
            "amount_log": row[9],
        }
    return None


def update_online_features(account_id: str, features: dict):
    """Update features for an account in online store."""
    conn = sqlite3.connect(ONLINE_DB)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO account_features
        (account_id, tx_count_1h, tx_count_24h, amount_mean_24h, amount_vs_mean_ratio,
         amount_zscore, new_destination, is_transfer, is_cash_out, org_balance_mismatch,
         amount_log, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        account_id,
        features["tx_count_1h"],
        features["tx_count_24h"],
        features["amount_mean_24h"],
        features["amount_vs_mean_ratio"],
        features["amount_zscore"],
        features["new_destination"],
        features["is_transfer"],
        features["is_cash_out"],
        features["org_balance_mismatch"],
        features["amount_log"],
        datetime.now(),
    ))

    conn.commit()
    conn.close()


def materialize_to_online(limit: int = 10000):
    """Materialize latest features to online store."""
    logger.info("Materializing features to online store...")

    df = pd.read_parquet(FEATURES_FILE)

    # Get latest features per account
    feature_cols = [
        "nameOrig", "tx_count_1h", "tx_count_24h", "amount_mean_24h",
        "amount_vs_mean_ratio", "amount_zscore", "new_destination",
        "is_transfer", "is_cash_out", "org_balance_mismatch", "amount_log"
    ]

    latest_df = df.sort_values("step").groupby("nameOrig").last().reset_index()
    latest_df = latest_df[feature_cols]

    for _, row in latest_df.head(limit).iterrows():
        features = {
            "tx_count_1h": int(row["tx_count_1h"]),
            "tx_count_24h": int(row["tx_count_24h"]),
            "amount_mean_24h": float(row["amount_mean_24h"]),
            "amount_vs_mean_ratio": float(row["amount_vs_mean_ratio"]),
            "amount_zscore": float(row["amount_zscore"]),
            "new_destination": int(row["new_destination"]),
            "is_transfer": int(row["is_transfer"]),
            "is_cash_out": int(row["is_cash_out"]),
            "org_balance_mismatch": int(row["org_balance_mismatch"]),
            "amount_log": float(row["amount_log"]),
        }
        update_online_features(row["nameOrig"], features)

    logger.info(f"Materialized {min(limit, len(latest_df))} accounts to online store")


def main():
    setup_online_store()
    materialize_to_online()


if __name__ == "__main__":
    main()
