from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"


class Timeframe(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


# ── LLM structured outputs ──────────────────────────────────────────────────

class TechnicalAnalysis(BaseModel):
    trend: str = Field(description="BULLISH, BEARISH, or NEUTRAL")
    strength: float = Field(ge=0, le=1, description="Trend strength 0-1")
    rsi_signal: str = Field(description="OVERBOUGHT, OVERSOLD, or NEUTRAL")
    macd_signal: str = Field(description="BULLISH_CROSS, BEARISH_CROSS, or FLAT")
    support_level: float = Field(description="Nearest support price level")
    resistance_level: float = Field(description="Nearest resistance price level")
    key_signals: list[str] = Field(description="List of observed technical signals")
    score: float = Field(ge=-1, le=1, description="Composite technical score: -1 bearish to +1 bullish")
    summary: str = Field(description="Concise technical analysis summary")


class SentimentAnalysis(BaseModel):
    overall_sentiment: str = Field(description="POSITIVE, NEGATIVE, or NEUTRAL")
    sentiment_score: float = Field(ge=-1, le=1, description="-1 very negative to +1 very positive")
    key_themes: list[str] = Field(description="Main themes from news")
    catalyst_detected: bool = Field(description="Whether a major catalyst event was detected")
    catalyst_description: Optional[str] = Field(default=None, description="Description of catalyst if detected")
    confidence: float = Field(ge=0, le=1, description="Confidence in sentiment assessment")
    summary: str = Field(description="Concise sentiment analysis summary")


class TradeSignal(BaseModel):
    direction: Direction
    confidence: float = Field(ge=0, le=1)
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    timeframe: str
    reasoning: str
    technical_contribution: float = Field(ge=0, le=1, description="Weight of technical analysis")
    sentiment_contribution: float = Field(ge=0, le=1, description="Weight of sentiment analysis")


class RiskAssessment(BaseModel):
    risk_level: str = Field(description="LOW, MEDIUM, HIGH, or EXTREME")
    risk_score: float = Field(ge=0, le=1, description="0 low risk to 1 extreme risk")
    position_size_pct: float = Field(ge=0, le=100, description="Recommended position size as % of portfolio")
    max_drawdown_pct: float = Field(description="Expected maximum drawdown percentage")
    volatility_regime: str = Field(description="LOW, NORMAL, HIGH, or EXTREME volatility regime")
    warnings: list[str] = Field(description="Risk warnings and concerns")
    approved: bool = Field(description="Whether trade is approved given risk parameters")
    rejection_reason: Optional[str] = Field(default=None)


# ── API request / response models ───────────────────────────────────────────

class SignalRequest(BaseModel):
    symbol: str
    timeframe: Timeframe = Timeframe.H1


class SignalResponse(BaseModel):
    symbol: str
    timeframe: str
    timestamp: datetime
    direction: Direction
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    risk_level: str
    position_size_pct: float
    technical_score: float
    sentiment_score: float
    reasoning: str
    warnings: list[str]


class MarketDataResponse(BaseModel):
    symbol: str
    latest_price: float
    change_pct: float
    volume: float
    timestamp: datetime
    indicators: dict


class HealthResponse(BaseModel):
    status: str
    qdrant: bool
    redis: bool
    version: str = "1.0.0"


class NewsItem(BaseModel):
    title: str
    summary: str
    source: str
    url: str
    published_at: Optional[datetime] = None
    symbol: Optional[str] = None
