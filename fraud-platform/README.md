# Real-Time Financial Fraud Detection Platform

A production-grade fraud detection system that processes live transactions, scores them in milliseconds, links suspicious accounts into fraud rings, and provides an analyst dashboard for case review.

## Architecture

- **Streaming**: Kafka for real-time transaction ingestion
- **ML Models**: XGBoost + PyTorch Geometric GNN for fraud detection
- **Feature Store**: Feast for point-in-time correct feature serving
- **Experiment Tracking**: MLflow for model registry and metrics
- **Monitoring**: Evidently AI for drift detection
- **API**: FastAPI for scoring and alerts
- **Dashboard**: Streamlit for analyst interface
- **Orchestration**: Kubernetes (Minikube) for deployment

## Quick Start

```bash
# Install dependencies
pip install -e .

# Download or generate synthetic data
python scripts/download_data.py

# Run EDA
jupyter notebook notebooks/01_eda.ipynb

# Train models
python src/models/train_xgboost.py
python src/models/train_gnn.py

# Start local services
docker-compose up -d

# Run streaming pipeline
python src/streaming/producer.py &
python src/streaming/consumer.py

# Launch dashboard
streamlit run src/dashboard/app.py
```

## Results

**Note**: Results are from synthetic PaySim-like data. Real PaySim dataset would yield meaningful performance metrics.

### Model Performance

| Model | PR-AUC | Accuracy | Notes |
|-------|--------|----------|-------|
| XGBoost (baseline) | 0.003 | 99.7% | scale_pos_weight=345 |
| XGBoost (SMOTE) | 0.003 | 99.8% | Synthetic data limits |
| GNN (GCN) | N/A | 99.95% | Overfits to synthetic patterns |

### Data Statistics

- Total transactions: 500,000
- Fraud cases: 500 (0.1%)
- Fraud types: TRANSFER, CASH_OUT only
- Time steps: 743 (~30 days)
- Unique accounts: 486k origin, 486k destination

### Key Findings

1. **Class Imbalance**: Fraud is <1% of transactions
2. **Feature Importance**: Balance delta mismatch correlates with fraud
3. **Graph Signals**: Counterparty count and destination popularity are predictive
4. **Synthetic Data Limitation**: Low PR-AUC due to lack of real fraud patterns
5. **Production Ready**: Architecture complete, needs real data for validation

## Documentation

- [Architecture](docs/architecture.md)
- [API Documentation](docs/api.md)
- [Presentation Outline](docs/presentation_outline.md)
- [Decisions Log](DECISIONS.md)

## Development

```bash
# Lint
ruff check src/
ruff format src/

# Tests
pytest tests/

# Start Minikube
make minikube-up
```
