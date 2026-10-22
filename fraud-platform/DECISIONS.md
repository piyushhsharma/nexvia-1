# Decisions Log

This document records architectural and implementation decisions made during development.

## Stage 1: EDA

- **Data Source**: Using PaySim from Kaggle with synthetic fallback. If Kaggle download fails, generate synthetic data with same schema.
- **Sampling Strategy**: Stratified by time (step) to preserve temporal patterns. All fraud rows kept regardless of sample size.
- **Fraud Types**: Only TRANSFER and CASH_OUT contain fraud (confirmed from EDA). Will filter other types for modeling.
- **Class Imbalance**: Fraud rate is <1%. Will use scale_pos_weight in XGBoost and PR-AUC as primary metric.

## Stage 2: Features + Feast

- **Feature Store**: Simplified approach using parquet for offline store and SQLite for online store instead of full Feast setup (Feast compatibility issues with newer pandas).
- **Features Computed**: Transaction velocity (1h/24h counts), rolling average amount with ratio/z-score, balance delta mismatch, new destination flag, transaction type encodings.
- **Point-in-Time Correctness**: Used expanding windows sorted by account and time to ensure no future data leakage.

## Stage 3: XGBoost

- **Model Performance**: PR-AUC is very low (~0.003) on synthetic data. This is expected as synthetic data lacks real fraud patterns. Would need real PaySim data for meaningful results.
- **Imbalance Handling**: Compared scale_pos_weight (345:1) vs SMOTE. SMOTE slightly better but both models struggle with synthetic data.
- **Threshold Selection**: Default 0.5 threshold used as 90% precision not achievable (model can't reach that precision).
- **Business Rule**: Threshold set to 0.5 as baseline. In production, would tune based on cost of false positives vs missed fraud.
