"""
Build transaction graph for GNN-based fraud detection.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURES_FILE = Path("data/features.parquet")
GRAPH_FILE = Path("data/graph.pt")


def build_transaction_graph(df: pd.DataFrame, max_nodes: int = 10000) -> Data:
    """Build a graph from transactions."""
    logger.info("Building transaction graph...")

    # Limit to recent transactions and fraud-relevant types
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
    df = df.sort_values("step").tail(max_nodes)

    # Create node mappings
    all_accounts = set(df["nameOrig"].unique()) | set(df["nameDest"].unique())
    account_to_idx = {acc: i for i, acc in enumerate(all_accounts)}

    # Build edge list
    edge_indices = []
    edge_features = []
    edge_labels = []

    for _, row in df.iterrows():
        src = account_to_idx[row["nameOrig"]]
        dst = account_to_idx[row["nameDest"]]

        edge_indices.append([src, dst])
        edge_features.append([
            row["amount_log"],
            row["is_transfer"],
            row["is_cash_out"],
            row["org_balance_mismatch"],
        ])
        edge_labels.append(row["isFraud"])

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    edge_label = torch.tensor(edge_labels, dtype=torch.float)

    # Node features (simple identity-based)
    num_nodes = len(all_accounts)
    node_features = torch.eye(num_nodes, dtype=torch.float)

    # Node labels (fraud if any connected transaction is fraud)
    node_labels = torch.zeros(num_nodes, dtype=torch.float)
    for i, row in df.iterrows():
        src = account_to_idx[row["nameOrig"]]
        if row["isFraud"] == 1:
            node_labels[src] = 1

    graph_data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=node_labels,
        edge_y=edge_label,
    )

    logger.info(f"Graph built: {num_nodes} nodes, {len(edge_indices)} edges")
    return graph_data


def compute_graph_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute graph-derived signals."""
    logger.info("Computing graph signals...")

    # Connected component size (simplified: number of unique counterparties)
    df["num_counterparties"] = df.groupby("nameOrig")["nameDest"].transform("nunique")

    # Shared counterparty count (accounts that transact with same destination)
    dest_counts = df["nameDest"].value_counts()
    df["dest_popularity"] = df["nameDest"].map(dest_counts)

    logger.info("Graph signals computed")
    return df


def main():
    df = pd.read_parquet(FEATURES_FILE)

    # Compute graph signals
    df_with_signals = compute_graph_signals(df)
    df_with_signals.to_parquet(FEATURES_FILE, index=False)

    # Build graph for GNN
    graph = build_transaction_graph(df)
    torch.save(graph, GRAPH_FILE)
    logger.info(f"Graph saved to {GRAPH_FILE}")


if __name__ == "__main__":
    main()
