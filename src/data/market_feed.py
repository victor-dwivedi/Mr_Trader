from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _fetch_ticker_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No market data returned for {symbol}")
    return df


async def fetch_market_data(
    symbol: str,
    period: str = "5d",
    interval: str = "1h",
) -> dict:
    loop = asyncio.get_event_loop()
    df = await loop.run_in_executor(None, _fetch_ticker_history, symbol, period, interval)

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    latest = df.iloc[-1]
    prev_close = df.iloc[-2]["Close"] if len(df) > 1 else latest["Close"]
    change_pct = ((latest["Close"] - prev_close) / prev_close) * 100

    ohlcv_records = []
    for ts, row in df.tail(100).iterrows():
        ohlcv_records.append({
            "timestamp": ts.isoformat(),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"]),
        })

    return {
        "symbol": symbol,
        "interval": interval,
        "latest_price": float(latest["Close"]),
        "open": float(latest["Open"]),
        "high": float(latest["High"]),
        "low": float(latest["Low"]),
        "volume": float(latest["Volume"]),
        "change_pct": round(float(change_pct), 2),
        "timestamp": datetime.now().isoformat(),
        "ohlcv": ohlcv_records,
        "_df": df,  # keep raw df for indicator computation
    }


async def fetch_ticker_info(symbol: str) -> dict:
    loop = asyncio.get_event_loop()

    def _get_info():
        t = yf.Ticker(symbol)
        info = t.info or {}
        return {
            "symbol": symbol,
            "name": info.get("longName", symbol),
            "sector": info.get("sector", "Unknown"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "avg_volume": info.get("averageVolume"),
        }

    return await loop.run_in_executor(None, _get_info)
