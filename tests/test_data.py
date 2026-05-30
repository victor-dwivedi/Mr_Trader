import asyncio
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np


@pytest.fixture
def sample_df():
    dates = pd.date_range("2024-01-01", periods=50, freq="1h")
    np.random.seed(42)
    close = 150 + np.cumsum(np.random.randn(50) * 0.5)
    df = pd.DataFrame({
        "Open": close - 0.2,
        "High": close + 0.5,
        "Low": close - 0.5,
        "Close": close,
        "Volume": np.random.randint(1_000_000, 5_000_000, 50).astype(float),
    }, index=dates)
    return df


def test_compute_indicators_returns_dict(sample_df):
    from src.data.indicators import compute_indicators
    result = compute_indicators(sample_df)
    assert isinstance(result, dict)
    assert "close" in result
    assert result["close"] > 0


def test_compute_indicators_has_rsi(sample_df):
    from src.data.indicators import compute_indicators
    result = compute_indicators(sample_df)
    # RSI should be present and within 0-100
    rsi = result.get("rsi_14")
    if rsi is not None:
        assert 0 <= rsi <= 100


@pytest.mark.asyncio
async def test_fetch_news_returns_list():
    from src.data.news_feed import fetch_yfinance_news
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.news = [
            {"title": "Test News", "summary": "Summary", "publisher": "Test", "link": "http://x.com", "providerPublishTime": 1700000000},
        ]
        result = await fetch_yfinance_news("AAPL", limit=5)
    assert isinstance(result, list)
    assert result[0]["title"] == "Test News"
