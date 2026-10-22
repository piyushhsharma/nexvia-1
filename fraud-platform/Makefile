.PHONY: help install data features train test clean minikube-up

help:
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make data        - Download/generate data"
	@echo "  make features    - Compute features"
	@echo "  make train       - Train models"
	@echo "  make test        - Run tests"
	@echo "  make clean       - Clean generated files"
	@echo "  make minikube-up - Start Minikube and deploy"

install:
	pip install -e .

data:
	python scripts/download_data.py

features:
	python src/features/prepare_features.py
	python src/features/simple_feature_store.py

train:
	python src/models/train_xgboost.py
	python src/graph/build_graph.py
	python src/graph/train_gnn.py

test:
	pytest tests/

clean:
	rm -rf data/*.csv data/*.parquet data/*.db data/*.pt
	rm -rf mlruns/ mlflow.db
	rm -rf reports/*
	rm -rf feature_repo/

minikube-up:
	minikube start
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/kafka-deployment.yaml
	kubectl apply -f k8s/scoring-api-deployment.yaml
	kubectl apply -f k8s/dashboard-deployment.yaml
