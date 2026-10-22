"""
Train XGBoost model for fraud detection with MLflow tracking.
"""
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURES_FILE = Path("data/features.parquet")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "tx_count_1h",
    "tx_count_24h",
    "amount_mean_24h",
    "amount_vs_mean_ratio",
    "amount_zscore",
    "new_destination",
    "is_transfer",
    "is_cash_out",
    "org_balance_mismatch",
    "amount_log",
]


def time_based_split(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Split data by time (step) to avoid leakage."""
    df_sorted = df.sort_values("step").reset_index(drop=True)

    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:train_end]
    val_df = df_sorted.iloc[train_end:val_end]
    test_df = df_sorted.iloc[val_end:]

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def train_xgboost_baseline(X_train, y_train, X_val, y_val):
    """Train XGBoost with scale_pos_weight for imbalance."""
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    logger.info(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="aucpr",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    return model


def train_xgboost_smote(X_train, y_train, X_val, y_val):
    """Train XGBoost with SMOTE on training data."""
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    logger.info(f"After SMOTE: {len(X_train_smote)} samples, fraud rate: {y_train_smote.mean():.4%}")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="aucpr",
    )

    model.fit(
        X_train_smote,
        y_train_smote,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    return model


def evaluate_model(model, X, y, prefix: str = ""):
    """Evaluate model and return metrics."""
    y_pred_proba = model.predict_proba(X)[:, 1]

    pr_auc = average_precision_score(y, y_pred_proba)

    # Find threshold for 90% precision
    precisions, recalls, thresholds = precision_recall_curve(y, y_pred_proba)
    # precision_recall_curve returns len(thresholds) + 1 precision/recall values
    # We need to map precision indices to threshold indices correctly
    idx = np.where(precisions[:-1] >= 0.9)[0]  # Exclude last element which has no threshold
    if len(idx) > 0:
        threshold_90_prec = thresholds[idx[0]]
        recall_at_90_prec = recalls[idx[0]]
    else:
        threshold_90_prec = 0.5
        recall_at_90_prec = 0.0

    # Confusion matrix at default threshold (0.5)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    cm = confusion_matrix(y, y_pred)

    metrics = {
        f"{prefix}pr_auc": float(pr_auc),
        f"{prefix}threshold_90_precision": float(threshold_90_prec),
        f"{prefix}recall_at_90_precision": float(recall_at_90_prec),
        f"{prefix}tn": int(cm[0, 0]),
        f"{prefix}fp": int(cm[0, 1]),
        f"{prefix}fn": int(cm[1, 0]),
        f"{prefix}tp": int(cm[1, 1]),
    }

    return metrics, y_pred_proba


def plot_pr_curve(y_true, y_pred_proba, save_path: Path):
    """Plot precision-recall curve."""
    from sklearn.metrics import precision_recall_curve

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    mlflow.set_experiment("fraud_detection_xgboost")

    with mlflow.start_run():
        logger.info("Loading features...")
        df = pd.read_parquet(FEATURES_FILE)

        # Filter to only fraud-relevant transaction types
        df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
        logger.info(f"Filtered to {len(df)} transactions (TRANSFER and CASH_OUT only)")

        # Time-based split
        train_df, val_df, test_df = time_based_split(df)

        # Prepare features
        X_train = train_df[FEATURE_COLS]
        y_train = train_df["isFraud"]
        X_val = val_df[FEATURE_COLS]
        y_val = val_df["isFraud"]
        X_test = test_df[FEATURE_COLS]
        y_test = test_df["isFraud"]

        logger.info(f"Train fraud rate: {y_train.mean():.4%}")
        logger.info(f"Val fraud rate: {y_val.mean():.4%}")
        logger.info(f"Test fraud rate: {y_test.mean():.4%}")

        # Train baseline (scale_pos_weight)
        logger.info("Training baseline XGBoost...")
        model_baseline = train_xgboost_baseline(X_train, y_train, X_val, y_val)

        # Train with SMOTE
        logger.info("Training XGBoost with SMOTE...")
        model_smote = train_xgboost_smote(X_train, y_train, X_val, y_val)

        # Evaluate both
        logger.info("Evaluating models...")
        baseline_metrics, _ = evaluate_model(model_baseline, X_test, y_test, "baseline_")
        smote_metrics, _ = evaluate_model(model_smote, X_test, y_test, "smote_")

        # Log all metrics
        all_metrics = {**baseline_metrics, **smote_metrics}
        mlflow.log_metrics(all_metrics)

        logger.info("Test Metrics:")
        for key, value in all_metrics.items():
            logger.info(f"  {key}: {value}")

        # Log params
        mlflow.log_params({
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
        })

        # Plot PR curve for baseline
        y_pred_proba = model_baseline.predict_proba(X_test)[:, 1]
        plot_pr_curve(y_test, y_pred_proba, REPORTS_DIR / "pr_curve.png")
        mlflow.log_artifact(str(REPORTS_DIR / "pr_curve.png"))

        # Save confusion matrix
        cm_path = REPORTS_DIR / "confusion_matrix.json"
        with open(cm_path, "w") as f:
            json.dump({
                "baseline": {
                    "tn": int(baseline_metrics["baseline_tn"]),
                    "fp": int(baseline_metrics["baseline_fp"]),
                    "fn": int(baseline_metrics["baseline_fn"]),
                    "tp": int(baseline_metrics["baseline_tp"]),
                },
                "smote": {
                    "tn": int(smote_metrics["smote_tn"]),
                    "fp": int(smote_metrics["smote_fp"]),
                    "fn": int(smote_metrics["smote_fn"]),
                    "tp": int(smote_metrics["smote_tp"]),
                },
            }, f, indent=2)
        mlflow.log_artifact(str(cm_path))

        # Register the better model
        if baseline_metrics["baseline_pr_auc"] >= smote_metrics["smote_pr_auc"]:
            best_model = model_baseline
            best_model_name = "xgboost_baseline"
            logger.info("Baseline model is better")
        else:
            best_model = model_smote
            best_model_name = "xgboost_smote"
            logger.info("SMOTE model is better")

        mlflow.xgboost.log_model(best_model, "model")
        mlflow.log_param("best_model", best_model_name)

        # Business rule: threshold for 90% precision
        threshold = baseline_metrics["baseline_threshold_90_precision"]
        logger.info(f"Selected threshold for 90% precision: {threshold:.4f}")

        # Save threshold
        threshold_path = REPORTS_DIR / "threshold.json"
        with open(threshold_path, "w") as f:
            json.dump({"threshold": float(threshold)}, f, indent=2)

        logger.info("Training complete")


if __name__ == "__main__":
    main()
