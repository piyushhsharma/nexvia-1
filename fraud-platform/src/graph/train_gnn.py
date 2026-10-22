"""
Train GNN model for fraud detection (simplified for CPU).
"""
import logging
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

GRAPH_FILE = Path("data/graph.pt")


class FraudGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 32)
        self.fc = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return torch.sigmoid(x).squeeze()


def train_gnn(graph: Data):
    """Train GNN model."""
    mlflow.set_experiment("fraud_detection_gnn")

    with mlflow.start_run():
        logger.info("Training GNN...")

        model = FraudGNN(graph.x.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Simple train/val split by nodes
        num_nodes = graph.x.shape[0]
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_idx = int(0.8 * num_nodes)
        train_mask[:train_idx] = True
        val_mask[train_idx:] = True

        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            out = model(graph.x, graph.edge_index)
            loss = F.binary_cross_entropy(out[train_mask], graph.y[train_mask])
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        # Evaluate
        model.eval()
        with torch.no_grad():
            out = model(graph.x, graph.edge_index)
            val_pred = out[val_mask]
            val_true = graph.y[val_mask]

            # Simple metrics
            pred_labels = (val_pred > 0.5).float()
            accuracy = (pred_labels == val_true).float().mean()

            logger.info(f"Validation accuracy: {accuracy:.4f}")
            mlflow.log_metric("val_accuracy", float(accuracy))

        # Skip model logging due to serialization issues with PyTorch 3.14
        # In production, would use compatible PyTorch version or save manually
        logger.info("GNN training complete")

        return model


def main():
    if not GRAPH_FILE.exists():
        logger.error(f"Graph file not found: {GRAPH_FILE}")
        logger.info("Run build_graph.py first")
        return

    graph = torch.load(GRAPH_FILE, weights_only=False)
    model = train_gnn(graph)


if __name__ == "__main__":
    main()
