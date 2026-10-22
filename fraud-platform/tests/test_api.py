"""
API contract tests for scoring service.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from src.streaming.consumer import app


def test_health_endpoint():
    """Test health check endpoint."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_score_endpoint():
    """Test scoring endpoint."""
    client = TestClient(app)

    tx = {
        "step": 100,
        "type": "TRANSFER",
        "amount": 5000.0,
        "nameOrig": "C1234567",
        "oldbalanceOrg": 10000.0,
        "newbalanceOrg": 5000.0,
        "nameDest": "M1234567",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 5000.0,
        "isFraud": 0,
    }

    response = client.post("/score", json=tx)
    assert response.status_code == 200

    data = response.json()
    assert "fraud_score" in data
    assert "is_fraud" in data
    assert "latency_ms" in data


if __name__ == "__main__":
    test_health_endpoint()
    test_score_endpoint()
    print("API tests passed")
