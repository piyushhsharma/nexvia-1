# 10-Minute Presentation Outline

## Problem (2 minutes)
- Financial fraud causes billions in losses annually
- Real-time detection is critical but challenging
- Need to score transactions in milliseconds while maintaining low false positive rates
- Fraud rings are complex - individual transaction analysis misses network patterns

## Design (3 minutes)
- **Architecture**: Kafka for streaming, Feast for features, XGBoost + GNN for detection
- **Features**: Transaction velocity, rolling averages, balance delta mismatches, graph signals
- **Models**: XGBoost for tabular patterns, GNN for relationship patterns, blender for combined score
- **Point-in-time correctness**: Time-based splits, expanding windows, no future data leakage
- **Monitoring**: Evidently for drift detection, automated retraining triggers

## Challenges (2 minutes)
- **Class imbalance**: Fraud is <1% of transactions - used scale_pos_weight and PR-AUC
- **Temporal leakage**: Time-based splits prevent using future information
- **Graph scale**: 500k transactions create large graphs - used neighbor sampling
- **Real-time latency**: Target <10ms per transaction - optimized feature serving
- **Synthetic data limitations**: Real PaySim data needed for production validation

## Results (2 minutes)
- **XGBoost**: PR-AUC 0.003 on synthetic data (expected - synthetic lacks real patterns)
- **GNN**: 99.95% accuracy on synthetic data (overfitting to synthetic patterns)
- **Feature engineering**: Balance delta mismatch strongly correlates with fraud
- **Graph signals**: Counterparty count and destination popularity are predictive
- **Drift detection**: Implemented with configurable thresholds
- **Note**: Real-world performance would be validated on actual PaySim dataset

## Next Steps (1 minute)
- Deploy on Minikube with Kafka, MLflow, Feast
- Integration testing with real transaction replay
- Production monitoring with Prometheus/Grafana
- A/B testing against existing fraud detection system
- Continuous retraining pipeline with human-in-the-loop feedback
