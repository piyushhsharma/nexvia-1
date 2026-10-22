"""
Download PaySim dataset from Kaggle or generate synthetic fallback.
"""
import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
RAW_FILE = DATA_DIR / "paysim_raw.csv"
SAMPLED_FILE = DATA_DIR / "paysim_sampled.csv"


def generate_synthetic_paysim(n_rows: int = 100000, fraud_rate: float = 0.001) -> pd.DataFrame:
    """Generate synthetic PaySim-like data for testing."""
    logger.info(f"Generating {n_rows} synthetic transactions...")

    np.random.seed(42)

    transaction_types = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    type_dist = [0.2, 0.25, 0.1, 0.35, 0.1]

    n_fraud = int(n_rows * fraud_rate)
    n_legit = n_rows - n_fraud

    types = np.random.choice(transaction_types, n_rows, p=type_dist)

    # PaySim pattern: only TRANSFER and CASH_OUT have fraud
    fraud_mask = np.zeros(n_rows, dtype=bool)
    fraud_eligible = np.where(np.isin(types, ["TRANSFER", "CASH_OUT"]))[0]
    fraud_indices = np.random.choice(fraud_eligible, min(n_fraud, len(fraud_eligible)), replace=False)
    fraud_mask[fraud_indices] = True

    steps = np.random.randint(0, 743, n_rows)  # 743 steps = 30 days
    amounts = np.random.lognormal(mean=8, sigma=1.5, size=n_rows)
    amounts = np.clip(amounts, 1, 1e7)

    # Account balances
    old_balance_org = np.random.lognormal(mean=10, sigma=2, size=n_rows)
    old_balance_org = np.clip(old_balance_org, 0, 1e8)
    new_balance_org = old_balance_org - amounts
    new_balance_org = np.maximum(new_balance_org, 0)

    old_balance_dest = np.random.lognormal(mean=9, sigma=2, size=n_rows)
    old_balance_dest = np.clip(old_balance_dest, 0, 1e7)
    new_balance_dest = old_balance_dest + amounts
    new_balance_dest = np.clip(new_balance_dest, 0, 1e8)

    df = pd.DataFrame({
        "step": steps.astype(int),
        "type": types,
        "amount": amounts,
        "nameOrig": [f"C{np.random.randint(1000000, 9999999)}" for _ in range(n_rows)],
        "oldbalanceOrg": old_balance_org,
        "newbalanceOrg": new_balance_org,
        "nameDest": [f"M{np.random.randint(1000000, 9999999)}" for _ in range(n_rows)],
        "oldbalanceDest": old_balance_dest,
        "newbalanceDest": new_balance_dest,
        "isFraud": fraud_mask.astype(int),
        "isFlaggedFraud": np.zeros(n_rows, dtype=int),
    })

    logger.info(f"Generated {n_rows} rows with {df['isFraud'].sum()} fraud cases ({df['isFraud'].mean():.4%})")
    return df


def download_from_kaggle() -> bool:
    """Download PaySim dataset from Kaggle."""
    try:
        import kaggle
    except ImportError:
        logger.error("kaggle package not installed. Install with: pip install kaggle")
        return False

    logger.info("Attempting to download from Kaggle...")
    try:
        kaggle.api.dataset_download_files(
            "ealaxiani/paysim1",
            path=str(DATA_DIR),
            unzip=True,
        )
        logger.info("Download complete")
        return True
    except Exception as e:
        logger.warning(f"Kaggle download failed: {e}")
        return False


def sample_data(df: pd.DataFrame, target_rows: int = 500000) -> pd.DataFrame:
    """Sample data while keeping all fraud rows and stratifying by time."""
    logger.info(f"Sampling to ~{target_rows} rows while preserving all fraud...")

    fraud_df = df[df["isFraud"] == 1].copy()
    legit_df = df[df["isFraud"] == 0].copy()

    logger.info(f"Original: {len(df)} rows, {len(fraud_df)} fraud ({len(fraud_df)/len(df):.4%})")

    # Stratify legit sample by step to preserve time distribution
    legit_sampled_list = []
    target_legit = target_rows - len(fraud_df)
    for step, group in legit_df.groupby("step"):
        sample_size = min(len(group), int(target_legit * len(group) / len(legit_df)))
        if sample_size > 0:
            legit_sampled_list.append(group.sample(n=sample_size, random_state=42))

    legit_sampled = pd.concat(legit_sampled_list, ignore_index=True)
    result = pd.concat([fraud_df, legit_sampled], ignore_index=True).sample(frac=1, random_state=42)

    logger.info(f"Sampled: {len(result)} rows, {result['isFraud'].sum()} fraud ({result['isFraud'].mean():.4%})")
    return result


def main():
    parser = argparse.ArgumentParser(description="Download or generate PaySim data")
    parser.add_argument("--force-kaggle", action="store_true", help="Force Kaggle download")
    parser.add_argument("--rows", type=int, default=500000, help="Target rows for sampling")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    # Try Kaggle first if file doesn't exist or forced
    raw_file = RAW_FILE
    if args.force_kaggle or not raw_file.exists():
        if download_from_kaggle():
            # Find the downloaded file (might have different name)
            csv_files = list(DATA_DIR.glob("*.csv"))
            if csv_files:
                raw_file = csv_files[0]
                logger.info(f"Using downloaded file: {raw_file}")
            else:
                logger.error("No CSV file found after download")
                return

    # Load or generate
    if raw_file.exists():
        logger.info(f"Loading data from {raw_file}")
        df = pd.read_csv(raw_file)
    else:
        logger.info("Generating synthetic data as fallback")
        df = generate_synthetic_paysim(n_rows=args.rows)

    # Sample
    sampled_df = sample_data(df, target_rows=args.rows)

    # Save
    sampled_df = sampled_df.reset_index(drop=True)
    sampled_df.to_csv(SAMPLED_FILE, index=False)
    logger.info(f"Saved sampled data to {SAMPLED_FILE}")


if __name__ == "__main__":
    main()
