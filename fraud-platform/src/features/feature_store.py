"""
Feast feature store setup and materialization.
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from feast import FeatureStore, RepoConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURES_FILE = Path("data/features.parquet")
REPO_PATH = Path("feature_repo")


def setup_feature_store() -> FeatureStore:
    """Initialize Feast feature store."""
    REPO_PATH.mkdir(exist_ok=True)

    config = RepoConfig(
        project="fraud_detection",
        provider="local",
        offline_store={
            "type": "file",
        },
        online_store={
            "type": "sqlite",
            "path": str(REPO_PATH / "online_store.db"),
        },
    )

    store = FeatureStore(config=config, repo_path=str(REPO_PATH))
    logger.info("Feature store initialized")
    return store


def materialize_features(store: FeatureStore):
    """Materialize features from parquet source."""
    logger.info("Materializing features...")

    # Load features
    df = pd.read_parquet(FEATURES_FILE)

    # Create entity DataFrame for Feast
    # Map account_id to both origin and destination for feature serving
    entity_df_orig = df[["step", "nameOrig"]].copy()
    entity_df_orig = entity_df_orig.rename(columns={"nameOrig": "account_id"})
    entity_df_orig["event_timestamp"] = pd.to_datetime(df["step"], unit="h", origin="1970-01-01")

    entity_df_dest = df[["step", "nameDest"]].copy()
    entity_df_dest = entity_df_dest.rename(columns={"nameDest": "account_id"})
    entity_df_dest["event_timestamp"] = pd.to_datetime(df["step"], unit="h", origin="1970-01-01")

    # For now, we'll use a simpler approach - just save the parquet for offline retrieval
    # and setup online serving separately
    logger.info(f"Features ready at {FEATURES_FILE}")


def get_historical_features(store: FeatureStore, entity_df: pd.DataFrame) -> pd.DataFrame:
    """Get historical features for training."""
    feature_refs = [
        "account_features:tx_count_1h",
        "account_features:tx_count_24h",
        "account_features:amount_mean_24h",
        "account_features:amount_vs_mean_ratio",
        "account_features:amount_zscore",
        "account_features:new_destination",
        "account_features:is_transfer",
        "account_features:is_cash_out",
        "account_features:org_balance_mismatch",
        "account_features:amount_log",
    ]

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=feature_refs,
    ).to_df()

    return training_df


def main():
    store = setup_feature_store()
    materialize_features(store)


if __name__ == "__main__":
    main()
