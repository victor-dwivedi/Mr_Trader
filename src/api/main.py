from __future__ import annotations

import logging

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.health import router as health_router
from src.api.routes.market import router as market_router
from src.api.routes.signals import router as signals_router
from src.api.websocket import signal_stream_handler
from src.config.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

settings = get_settings()

app = FastAPI(
    title="Mr. Trader — HFT & Market Intelligence",
    description=(
        "Multi-agent AI trading intelligence system using LangGraph, Qdrant RAG, "
        "and Claude LLM for real-time market analysis and trade signal generation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST routes
app.include_router(health_router, prefix="/api/v1")
app.include_router(market_router, prefix="/api/v1")
app.include_router(signals_router, prefix="/api/v1")


# WebSocket route for real-time signal streaming
@app.websocket("/ws/signals/{symbol}")
async def websocket_signal_stream(websocket: WebSocket, symbol: str) -> None:
    await signal_stream_handler(websocket, symbol)


@app.get("/", tags=["root"])
async def root() -> dict:
    return {
        "name": "Mr. Trader — HFT Intelligence System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "endpoints": {
            "generate_signal": "POST /api/v1/signals/generate",
            "batch_signals": "GET /api/v1/signals/batch?symbols=AAPL,TSLA",
            "market_data": "GET /api/v1/market/{symbol}",
            "ws_stream": "WS /ws/signals/{symbol}",
        },
    }
