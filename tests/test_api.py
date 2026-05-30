import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from src.api.main import app

client = TestClient(app)


def test_root_returns_200():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "endpoints" in data


def test_health_endpoint_exists():
    # Health may return 200 (ok) or 200 (degraded) — both are 200
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "qdrant" in data
    assert "redis" in data


@pytest.mark.asyncio
async def test_generate_signal_calls_run_analysis():
    mock_output = {
        "symbol": "AAPL",
        "timeframe": "1h",
        "timestamp": "2024-01-01T00:00:00Z",
        "direction": "LONG",
        "confidence": 0.8,
        "entry_price": 150.0,
        "stop_loss": 147.0,
        "take_profit": 156.0,
        "risk_reward_ratio": 2.0,
        "risk_level": "MEDIUM",
        "position_size_pct": 2.0,
        "technical_score": 0.7,
        "sentiment_score": 0.6,
        "reasoning": "Bullish EMA alignment",
        "warnings": [],
        "risk_approved": True,
    }

    with patch("src.api.routes.signals.run_analysis", new_callable=AsyncMock, return_value=mock_output):
        response = client.post("/api/v1/signals/generate", json={"symbol": "AAPL", "timeframe": "1h"})

    assert response.status_code == 200
    data = response.json()
    assert data["direction"] == "LONG"
    assert data["symbol"] == "AAPL"
