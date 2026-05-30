"""WebSocket endpoint for real-time signal streaming."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from src.agents.orchestrator import run_analysis
from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self) -> None:
        self._active: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, symbol: str) -> None:
        await websocket.accept()
        self._active.setdefault(symbol, []).append(websocket)
        logger.info("WS connected: %s (total=%d)", symbol, len(self._active[symbol]))

    def disconnect(self, websocket: WebSocket, symbol: str) -> None:
        sockets = self._active.get(symbol, [])
        if websocket in sockets:
            sockets.remove(websocket)
        logger.info("WS disconnected: %s (remaining=%d)", symbol, len(sockets))

    async def broadcast(self, symbol: str, message: Any) -> None:
        sockets = self._active.get(symbol, [])
        dead = []
        for ws in sockets:
            try:
                await ws.send_text(json.dumps(message, default=str))
            except Exception:
                dead.append(ws)
        for ws in dead:
            sockets.remove(ws)


manager = ConnectionManager()


async def signal_stream_handler(websocket: WebSocket, symbol: str) -> None:
    """
    Continuously generate trade signals at a configured interval and push
    them to connected WebSocket clients.
    """
    settings = get_settings()
    symbol = symbol.upper()
    await manager.connect(websocket, symbol)

    try:
        await websocket.send_text(json.dumps({
            "type": "connected",
            "symbol": symbol,
            "message": f"Streaming signals for {symbol}",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }))

        while True:
            try:
                signal = await run_analysis(symbol)
                await websocket.send_text(json.dumps({
                    "type": "signal",
                    "data": signal,
                }, default=str))
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }))

            # Wait for the next analysis cycle (default: 5 minutes)
            await asyncio.sleep(settings.cache_ttl_seconds)

    except WebSocketDisconnect:
        manager.disconnect(websocket, symbol)
    except Exception as e:
        logger.error("WebSocket error for %s: %s", symbol, e)
        manager.disconnect(websocket, symbol)
