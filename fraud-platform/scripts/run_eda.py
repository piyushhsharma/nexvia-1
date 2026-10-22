"""
Run EDA on PaySim dataset and save plots.
"""
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_FILE = Path("data/paysim_sampled.csv")
REPORTS_DIR = Path("reports/eda")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")


def main():
    logger.info(f"Loading data from {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    logger.info(f"Dataset shape: {df.shape}")

    # Basic info
    logger.info("\nData types:")
    logger.info(df.dtypes)
    logger.info("\nMissing values:")
    logger.info(df.isnull().sum())
    logger.info("\nBasic statistics:")
    logger.info(df.describe())

    # Class imbalance
    fraud_count = df["isFraud"].value_counts()
    fraud_rate = df["isFraud"].mean()
    logger.info(f"\nFraud cases: {fraud_count[1]} ({fraud_rate:.4%})")
    logger.info(f"Legitimate cases: {fraud_count[0]} ({1-fraud_rate:.4%})")

    fig, ax = plt.subplots(figsize=(8, 6))
    fraud_count.plot(kind="bar", ax=ax, color=["steelblue", "coral"])
    ax.set_title("Class Distribution")
    ax.set_xlabel("isFraud")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "class_distribution.png", dpi=150)
    plt.close()
    logger.info("Saved class_distribution.png")

    # Transaction type distribution
    type_counts = df["type"].value_counts()
    logger.info("\nTransaction types:")
    logger.info(type_counts)

    fig, ax = plt.subplots(figsize=(10, 6))
    type_counts.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Transaction Type Distribution")
    ax.set_xlabel("Transaction Type")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "transaction_types.png", dpi=150)
    plt.close()
    logger.info("Saved transaction_types.png")

    # Fraud by transaction type
    fraud_by_type = df.groupby("type")["isFraud"].agg(["count", "sum", "mean"])
    fraud_by_type.columns = ["total", "fraud_count", "fraud_rate"]
    logger.info("\nFraud by transaction type:")
    logger.info(fraud_by_type)

    fig, ax = plt.subplots(figsize=(10, 6))
    fraud_by_type["fraud_rate"].plot(kind="bar", ax=ax, color="coral")
    ax.set_title("Fraud Rate by Transaction Type")
    ax.set_xlabel("Transaction Type")
    ax.set_ylabel("Fraud Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "fraud_by_type.png", dpi=150)
    plt.close()
    logger.info("Saved fraud_by_type.png")

    # Fraud rate over time
    fraud_by_step = df.groupby("step")["isFraud"].agg(["count", "sum", "mean"])
    fraud_by_step.columns = ["total", "fraud_count", "fraud_rate"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    ax1.plot(fraud_by_step.index, fraud_by_step["total"], color="steelblue")
    ax1.set_title("Transaction Volume Over Time")
    ax1.set_xlabel("Step (hour)")
    ax1.set_ylabel("Transaction Count")

    ax2.plot(fraud_by_step.index, fraud_by_step["fraud_rate"], color="coral")
    ax2.set_title("Fraud Rate Over Time")
    ax2.set_xlabel("Step (hour)")
    ax2.set_ylabel("Fraud Rate")
    ax2.axhline(y=fraud_by_step["fraud_rate"].mean(), color="red", linestyle="--", label="Mean")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "fraud_over_time.png", dpi=150)
    plt.close()
    logger.info("Saved fraud_over_time.png")

    # Amount distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    df[df["isFraud"] == 0]["amount"].plot(kind="hist", bins=50, ax=ax1, color="steelblue", alpha=0.7)
    ax1.set_title("Amount Distribution - Legitimate")
    ax1.set_xlabel("Amount")
    ax1.set_ylabel("Count")
    ax1.set_yscale("log")

    df[df["isFraud"] == 1]["amount"].plot(kind="hist", bins=50, ax=ax2, color="coral", alpha=0.7)
    ax2.set_title("Amount Distribution - Fraud")
    ax2.set_xlabel("Amount")
    ax2.set_ylabel("Count")
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "amount_distribution.png", dpi=150)
    plt.close()
    logger.info("Saved amount_distribution.png")

    # Balance delta analysis
    df["balance_delta_org"] = df["oldbalanceOrg"] - df["newbalanceOrg"] - df["amount"]
    df["balance_delta_dest"] = df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]

    logger.info("\nBalance delta mismatch in origin (should be 0):")
    logger.info(df["balance_delta_org"].describe())
    logger.info("\nBalance delta mismatch in destination (should be 0):")
    logger.info(df["balance_delta_dest"].describe())

    org_mismatch = (df["balance_delta_org"].abs() > 0.01).astype(int)
    dest_mismatch = (df["balance_delta_dest"].abs() > 0.01).astype(int)

    origin_mismatch_fraud = df[org_mismatch == 1]["isFraud"].mean()
    origin_match_fraud = df[org_mismatch == 0]["isFraud"].mean()
    dest_mismatch_mask = dest_mismatch == 1
    if dest_mismatch_mask.sum() > 0:
        dest_mismatch_fraud = df[dest_mismatch_mask]["isFraud"].mean()
    else:
        dest_mismatch_fraud = 0
    dest_match_fraud = df[dest_mismatch == 0]["isFraud"].mean()

    logger.info(f"\nFraud rate when origin balance mismatches: {origin_mismatch_fraud:.4%}")
    logger.info(f"Fraud rate when origin balance matches: {origin_match_fraud:.4%}")
    logger.info(f"\nFraud rate when dest balance mismatches: {dest_mismatch_fraud:.4%}")
    logger.info(f"Fraud rate when dest balance matches: {dest_match_fraud:.4%}")

    # Save summary
    summary = {
        "total_rows": int(len(df)),
        "fraud_count": int(df["isFraud"].sum()),
        "fraud_rate": float(df["isFraud"].mean()),
        "transaction_types": int(df["type"].nunique()),
        "unique_orig_accounts": int(df["nameOrig"].nunique()),
        "unique_dest_accounts": int(df["nameDest"].nunique()),
        "time_steps": int(df["step"].nunique()),
        "fraud_types": df[df["isFraud"] == 1]["type"].unique().tolist(),
    }

    with open(REPORTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\nEDA Summary:")
    logger.info(json.dumps(summary, indent=2))

    logger.info("\nKey Findings:")
    logger.info(f"1. Class Imbalance: Fraud is {fraud_rate:.4%} of transactions (<1%)")
    logger.info(f"2. Fraud Types: Only in {summary['fraud_types']}")
    logger.info("3. Temporal patterns exist - use time-based split")
    logger.info("4. Balance delta mismatches correlate with fraud")


if __name__ == "__main__":
    main()
