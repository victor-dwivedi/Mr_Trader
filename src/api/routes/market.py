from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

from src.config.settings import get_settings
from src.data.indicators import compute_indicators
from src.data.market_feed import fetch_market_data, fetch_ticker_info
from src.models.schemas import MarketDataResponse

router = APIRouter(prefix="/market", tags=["market"])


@router.get("/{symbol}", response_model=MarketDataResponse)
async def get_market_data(
    symbol: str,
    timeframe: str = Query(default="1h", description="yfinance interval (1m, 5m, 15m, 1h, 1d)"),
    period: str = Query(default="5d", description="yfinance period (1d, 5d, 1mo)"),
) -> MarketDataResponse:
    try:
        raw = await fetch_market_data(symbol.upper(), period=period, interval=timeframe)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch market data: {e}")

    df = raw.pop("_df", None)
    indicators = {}
    if df is not None:
        try:
            indicators = compute_indicators(df)
        except Exception:
            pass

    return MarketDataResponse(
        symbol=raw["symbol"],
        latest_price=raw["latest_price"],
        change_pct=raw["change_pct"],
        volume=raw["volume"],
        timestamp=datetime.fromisoformat(raw["timestamp"]),
        indicators=indicators,
    )


@router.get("/{symbol}/info")
async def get_ticker_info(symbol: str) -> dict:
    try:
        return await fetch_ticker_info(symbol.upper())
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/{symbol}/ohlcv")
async def get_ohlcv(
    symbol: str,
    timeframe: str = Query(default="1h"),
    period: str = Query(default="5d"),
) -> dict:
    try:
        raw = await fetch_market_data(symbol.upper(), period=period, interval=timeframe)
        raw.pop("_df", None)
        return raw
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
