from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from src.agents.orchestrator import run_analysis
from src.config.settings import get_settings
from src.models.schemas import SignalRequest, SignalResponse

router = APIRouter(prefix="/signals", tags=["signals"])

# In-memory signal history (production: replace with Redis)
_signal_history: dict[str, list[dict]] = defaultdict(list)
_running_tasks: dict[str, bool] = {}


def _store_signal(symbol: str, signal: dict) -> None:
    settings = get_settings()
    history = _signal_history[symbol]
    history.append(signal)
    if len(history) > settings.max_signal_history:
        _signal_history[symbol] = history[-settings.max_signal_history :]


@router.post("/generate", response_model=SignalResponse)
async def generate_signal(request: SignalRequest) -> SignalResponse:
    """Run the full multi-agent analysis pipeline and return a trade signal."""
    symbol = request.symbol.upper()
    timeframe = request.timeframe.value

    try:
        output = await run_analysis(symbol, timeframe)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signal generation failed: {e}")

    if not output:
        raise HTTPException(status_code=500, detail="Agent returned no output")

    _store_signal(symbol, output)

    return SignalResponse(
        symbol=output["symbol"],
        timeframe=output["timeframe"],
        timestamp=datetime.fromisoformat(output["timestamp"].rstrip("Z")),
        direction=output["direction"],
        confidence=output["confidence"],
        entry_price=output["entry_price"] or 0,
        stop_loss=output.get("stop_loss") or 0,
        take_profit=output.get("take_profit") or 0,
        risk_reward_ratio=output.get("risk_reward_ratio") or 0,
        risk_level=output.get("risk_level", "UNKNOWN"),
        position_size_pct=output.get("position_size_pct") or 0,
        technical_score=output.get("technical_score") or 0,
        sentiment_score=output.get("sentiment_score") or 0,
        reasoning=output.get("reasoning") or "",
        warnings=output.get("warnings") or [],
    )


@router.get("/history/{symbol}")
async def get_signal_history(
    symbol: str,
    limit: int = Query(default=20, ge=1, le=100),
) -> list[dict]:
    """Return recent signal history for a symbol."""
    history = _signal_history.get(symbol.upper(), [])
    return history[-limit:]


@router.get("/batch")
async def batch_signals(
    symbols: str = Query(description="Comma-separated symbols e.g. AAPL,TSLA,SPY"),
    timeframe: str = Query(default="1h"),
) -> list[dict]:
    """Generate signals for multiple symbols concurrently."""
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:10]

    tasks = [run_analysis(sym, timeframe) for sym in symbol_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    output = []
    for sym, result in zip(symbol_list, results):
        if isinstance(result, Exception):
            output.append({"symbol": sym, "error": str(result)})
        else:
            _store_signal(sym, result)
            output.append(result)

    return output
