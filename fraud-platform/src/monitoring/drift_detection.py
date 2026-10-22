"""
Simplified drift detection for fraud model monitoring.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURES_FILE = Path("data/features.parquet")
REPORTS_DIR = Path("reports/drift")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def detect_drift():
    """Detect data drift using statistical tests."""
    logger.info("Running drift detection...")

    df = pd.read_parquet(FEATURES_FILE)
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()

    # Split into reference (train) and current (recent)
    n = len(df)
    reference = df.iloc[:int(n * 0.7)]
    current = df.iloc[int(n * 0.7):]

    # Simple drift detection: compare mean of key features
    drift_metrics = {}
    for col in ["amount", "amount_log", "tx_count_24h"]:
        ref_mean = float(reference[col].mean())
        curr_mean = float(current[col].mean())
        drift_pct = float(abs(curr_mean - ref_mean) / (ref_mean + 1e-6))
        drift_metrics[col] = {
            "reference_mean": ref_mean,
            "current_mean": curr_mean,
            "drift_pct": drift_pct,
        }

    # Overall drift score
    avg_drift = np.mean([m["drift_pct"] for m in drift_metrics.values()])

    report = {
        "overall_drift_score": float(avg_drift),
        "features": drift_metrics,
        "drift_threshold": 0.1,
        "drift_detected": bool(avg_drift > 0.1),
    }

    report_path = REPORTS_DIR / "drift_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Drift report saved to {report_path}")
    logger.info(f"Overall drift score: {avg_drift:.4f}")

    return avg_drift


def simulate_drift():
    """Simulate data drift by shifting amounts."""
    logger.info("Simulating drift...")

    df = pd.read_parquet(FEATURES_FILE)
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()

    # Shift amounts by 50%
    df["amount"] = df["amount"] * 1.5
    df["amount_log"] = np.log1p(df["amount"])

    drift_score = detect_drift()
    logger.info(f"Simulated drift score: {drift_score:.4f}")


if __name__ == "__main__":
    detect_drift()
