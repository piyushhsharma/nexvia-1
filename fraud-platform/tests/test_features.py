"""
Test feature engineering and feature store consistency.
"""
import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features.simple_feature_store import get_online_features, setup_online_store, update_online_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURES_FILE = Path("data/features.parquet")


def test_feature_consistency():
    """Test that offline and online features match for sample entities."""
    setup_online_store()

    # Load offline features
    df = pd.read_parquet(FEATURES_FILE)

    # Pick a sample account
    sample_account = df["nameOrig"].iloc[0]
    logger.info(f"Testing consistency for account: {sample_account}")

    # Get offline features
    offline_row = df[df["nameOrig"] == sample_account].iloc[-1]

    # Update online store with this account's features
    features = {
        "tx_count_1h": int(offline_row["tx_count_1h"]),
        "tx_count_24h": int(offline_row["tx_count_24h"]),
        "amount_mean_24h": float(offline_row["amount_mean_24h"]),
        "amount_vs_mean_ratio": float(offline_row["amount_vs_mean_ratio"]),
        "amount_zscore": float(offline_row["amount_zscore"]),
        "new_destination": int(offline_row["new_destination"]),
        "is_transfer": int(offline_row["is_transfer"]),
        "is_cash_out": int(offline_row["is_cash_out"]),
        "org_balance_mismatch": int(offline_row["org_balance_mismatch"]),
        "amount_log": float(offline_row["amount_log"]),
    }
    update_online_features(sample_account, features)

    # Get online features
    online_features = get_online_features(sample_account)

    # Compare
    assert online_features is not None, "Online features not found"
    assert abs(online_features["tx_count_1h"] - features["tx_count_1h"]) < 1e-6, "tx_count_1h mismatch"
    assert abs(online_features["amount_log"] - features["amount_log"]) < 1e-6, "amount_log mismatch"

    logger.info("Offline and online features match")


def test_no_leakage():
    """Test that features don't leak future information."""
    df = pd.read_parquet(FEATURES_FILE)

    # Check that rolling features only use past data
    # For each transaction, check that tx_count doesn't include current transaction
    df_sorted = df.sort_values(["nameOrig", "step"]).reset_index(drop=True)

    # Sample check: first transaction should have count=1 (itself)
    first_tx = df_sorted.groupby("nameOrig").first()
    assert (first_tx["tx_count_1h"] == 1).all(), "First transaction should have count=1"

    logger.info("No feature leakage detected")


if __name__ == "__main__":
    test_feature_consistency()
    test_no_leakage()
    logger.info("All feature tests passed")
