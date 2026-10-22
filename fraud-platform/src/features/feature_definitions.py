"""
Feast feature definitions for fraud detection.
"""
from datetime import timedelta

import numpy as np
import pandas as pd
from feast import Entity, Field, FeatureView, ValueType
from feast.infra.offline_stores import ParquetSource
from feast.types import Float32, Int64

account_entity = Entity(
    name="account_id",
    value_type=ValueType.STRING,
    description="Account identifier (origin or destination)",
)

transaction_source = ParquetSource(
    path="data/paysim_sampled.csv",
    timestamp_field="step",
    created_timestamp_column="step",
)


def compute_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute transaction-level features for Feast."""
    df = df.copy()

    # Transaction type encoding (only TRANSFER and CASH_OUT matter for fraud)
    df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
    df["is_cash_out"] = (df["type"] == "CASH_OUT").astype(int)

    # Balance delta features (PaySim-specific fraud signal)
    df["balance_delta_org"] = df["oldbalanceOrg"] - df["newbalanceOrg"] - df["amount"]
    df["balance_delta_dest"] = df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]
    df["org_balance_mismatch"] = (df["balance_delta_org"].abs() > 0.01).astype(int)

    # Amount features
    df["amount_log"] = np.log1p(df["amount"])

    return df


def compute_account_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute account-level rolling features."""
    df = df.copy()
    df = df.sort_values("step")

    # Per-account transaction counts in time windows
    df["tx_count_1h"] = df.groupby("nameOrig").rolling(window=1, on="step")["step"].count().reset_index(0, drop=True)
    df["tx_count_24h"] = df.groupby("nameOrig").rolling(window=24, on="step")["step"].count().reset_index(0, drop=True)

    # Rolling average amount
    df["amount_mean_24h"] = df.groupby("nameOrig").rolling(window=24, on="step")["amount"].mean().reset_index(0, drop=True)

    # Amount vs rolling average (ratio and z-score)
    df["amount_vs_mean_ratio"] = df["amount"] / (df["amount_mean_24h"] + 1e-6)
    df["amount_zscore"] = (df["amount"] - df["amount_mean_24h"]) / (df.groupby("nameOrig").rolling(window=24, on="step")["amount"].std().reset_index(0, drop=True) + 1e-6)

    # New destination flag
    df["dest_seen_before"] = df.groupby("nameDest").cumcount() > 0
    df["new_destination"] = (~df["dest_seen_before"]).astype(int)

    return df


account_features_view = FeatureView(
    name="account_features",
    entities=["account_id"],
    ttl=timedelta(days=7),
    schema=[
        Field(name="tx_count_1h", dtype=Int64),
        Field(name="tx_count_24h", dtype=Int64),
        Field(name="amount_mean_24h", dtype=Float32),
        Field(name="amount_vs_mean_ratio", dtype=Float32),
        Field(name="amount_zscore", dtype=Float32),
        Field(name="new_destination", dtype=Int64),
        Field(name="is_transfer", dtype=Int64),
        Field(name="is_cash_out", dtype=Int64),
        Field(name="org_balance_mismatch", dtype=Int64),
        Field(name="amount_log", dtype=Float32),
    ],
    source=transaction_source,
)
