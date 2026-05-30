"""
Simple directional accuracy backtest.

Replays historical OHLCV data, calls the LangGraph pipeline per candle,
and compares the predicted direction to the actual next-candle direction.

Usage:
    python -m scripts.backtest --symbol AAPL --period 1mo --timeframe 1h
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.orchestrator import run_analysis


def direction_from_candle(open_price: float, close_price: float) -> str:
    if close_price > open_price * 1.001:
        return "LONG"
    elif close_price < open_price * 0.999:
        return "SHORT"
    return "HOLD"


async def backtest(symbol: str, period: str = "1mo", timeframe: str = "1h", sample: int = 20) -> dict:
    print(f"Fetching {symbol} history ({period}, {timeframe})...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=timeframe)

    if len(df) < 5:
        print("Not enough data.")
        return {}

    # Sample evenly to avoid hammering the LLM API in a tight loop
    indices = list(range(5, min(len(df) - 1, 5 + sample)))
    print(f"Running {len(indices)} signal evaluations...\n")

    correct = 0
    total = 0
    results = []

    for i in indices:
        # The "actual" direction is determined by the next candle
        next_candle = df.iloc[i + 1] if i + 1 < len(df) else None
        if next_candle is None:
            continue

        actual = direction_from_candle(float(next_candle["Open"]), float(next_candle["Close"]))

        try:
            signal = await run_analysis(symbol, timeframe)
            predicted = signal.get("direction", "HOLD")
        except Exception as e:
            print(f"  [candle {i}] Error: {e}")
            continue

        is_correct = predicted == actual or actual == "HOLD"
        if actual != "HOLD":
            total += 1
            if predicted == actual:
                correct += 1

        results.append({
            "candle": i,
            "timestamp": df.index[i].isoformat(),
            "predicted": predicted,
            "actual": actual,
            "correct": is_correct,
            "confidence": signal.get("confidence", 0),
        })

        print(
            f"  [{df.index[i].strftime('%Y-%m-%d %H:%M')}] "
            f"predicted={predicted:5s} actual={actual:5s} {'✓' if is_correct else '✗'} "
            f"conf={signal.get('confidence', 0):.2f}"
        )

    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\n--- Results for {symbol} ---")
    print(f"Directional accuracy: {accuracy:.1f}% ({correct}/{total})")

    return {
        "symbol": symbol,
        "period": period,
        "timeframe": timeframe,
        "accuracy_pct": round(accuracy, 2),
        "correct": correct,
        "total": total,
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest directional accuracy")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--period", default="1mo")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--sample", type=int, default=20, help="Number of candles to evaluate")
    args = parser.parse_args()
    asyncio.run(backtest(args.symbol, args.period, args.timeframe, args.sample))
