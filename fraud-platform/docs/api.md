# API Documentation

## Scoring API

### POST /score

Score a single transaction for fraud.

**Request Body:**
```json
{
  "step": 100,
  "type": "TRANSFER",
  "amount": 5000.0,
  "nameOrig": "C1234567",
  "oldbalanceOrg": 10000.0,
  "newbalanceOrg": 5000.0,
  "nameDest": "M1234567",
  "oldbalanceDest": 0.0,
  "newbalanceDest": 5000.0,
  "isFraud": 0
}
```

**Response:**
```json
{
  "fraud_score": 0.85,
  "is_fraud": true,
  "latency_ms": 5.2
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

### GET /metrics

Prometheus-style metrics.

**Response:**
```json
{
  "requests_total": 1000,
  "latency_p50_ms": 5.0,
  "latency_p95_ms": 15.0,
  "latency_p99_ms": 30.0
}
```

### GET /cases

Get recent fraud cases.

**Response:**
```json
{
  "cases": [
    {
      "id": 1,
      "transaction_id": "TX001",
      "account_id": "C1234567",
      "fraud_score": 0.85,
      "is_fraud": 1,
      "timestamp": "2026-10-21T10:30:00",
      "latency_ms": 5.2
    }
  ]
}
```
