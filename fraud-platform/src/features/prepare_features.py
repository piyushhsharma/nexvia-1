"""
Prepare and compute features from raw transaction data.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_FILE = Path("data/paysim_sampled.csv")
FEATURES_FILE = Path("data/features.parquet")


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features for fraud detection."""
    logger.info("Computing features...")

    df = df.copy()
    df = df.sort_values(["nameOrig", "step"]).reset_index(drop=True)

    # Transaction type encoding
    df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
    df["is_cash_out"] = (df["type"] == "CASH_OUT").astype(int)

    # Balance delta features
    df["balance_delta_org"] = df["oldbalanceOrg"] - df["newbalanceOrg"] - df["amount"]
    df["balance_delta_dest"] = df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]
    df["org_balance_mismatch"] = (df["balance_delta_org"].abs() > 0.01).astype(int)

    # Amount features
    df["amount_log"] = np.log1p(df["amount"])

    # Per-account transaction counts using shift and cumsum
    df["prev_step"] = df.groupby("nameOrig")["step"].shift(1)
    df["time_since_prev"] = df["step"] - df["prev_step"]
    df["tx_count_1h"] = df.groupby("nameOrig").apply(lambda x: x["step"].expanding().count(), include_groups=False).reset_index(level=0, drop=True)
    df["tx_count_24h"] = df.groupby("nameOrig").apply(lambda x: x["step"].expanding().count(), include_groups=False).reset_index(level=0, drop=True)

    # Rolling average amount using expanding mean
    df["amount_mean_24h"] = df.groupby("nameOrig")["amount"].apply(lambda x: x.expanding().mean()).reset_index(level=0, drop=True)
    amount_std_24h = df.groupby("nameOrig")["amount"].apply(lambda x: x.expanding().std()).reset_index(level=0, drop=True)

    # Amount vs rolling average
    df["amount_vs_mean_ratio"] = df["amount"] / (df["amount_mean_24h"] + 1e-6)
    df["amount_zscore"] = (df["amount"] - df["amount_mean_24h"]) / (amount_std_24h + 1e-6)

    # New destination flag
    df["dest_seen_before"] = df.groupby("nameDest").cumcount() > 0
    df["new_destination"] = (~df["dest_seen_before"]).astype(int)

    # Fill NaN values from rolling windows
    df["tx_count_1h"] = df["tx_count_1h"].fillna(0)
    df["tx_count_24h"] = df["tx_count_24h"].fillna(0)
    df["amount_mean_24h"] = df["amount_mean_24h"].fillna(df["amount"])
    df["amount_vs_mean_ratio"] = df["amount_vs_mean_ratio"].fillna(1.0)
    df["amount_zscore"] = df["amount_zscore"].fillna(0)

    # Drop temporary columns
    df = df.drop(columns=["prev_step", "time_since_prev"], errors="ignore")

    logger.info(f"Computed {len(df)} feature rows")
    return df


def main():
    logger.info(f"Loading data from {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)

    features_df = compute_features(df)

    FEATURES_FILE.parent.mkdir(exist_ok=True)
    features_df.to_parquet(FEATURES_FILE, index=False)
    logger.info(f"Saved features to {FEATURES_FILE}")


if __name__ == "__main__":
    main()
