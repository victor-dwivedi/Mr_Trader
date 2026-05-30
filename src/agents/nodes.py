"""LangGraph node functions — each mutates a slice of TradingState."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from qdrant_client import QdrantClient

from src.config.settings import get_settings
from src.data.indicators import compute_indicators
from src.data.market_feed import fetch_market_data
from src.data.news_feed import fetch_all_news
from src.models.schemas import RiskAssessment, SentimentAnalysis, TechnicalAnalysis, TradeSignal
from src.models.state import TradingState
from src.rag.retriever import retrieve_news_context, retrieve_technical_context


# ── Shared singletons ────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_llm() -> ChatGoogleGenerativeAI:
    s = get_settings()
    return ChatGoogleGenerativeAI(
        model=s.llm_model,
        google_api_key=s.gemini_api_key,
        temperature=s.llm_temperature,
        max_output_tokens=4096,
    )


@lru_cache(maxsize=1)
def _get_qdrant() -> QdrantClient:
    s = get_settings()
    return QdrantClient(host=s.qdrant_host, port=s.qdrant_port)


def _fmt(d: Any) -> str:
    return json.dumps(d, indent=2, default=str)


# ── Node 1: Collect market data ──────────────────────────────────────────────

async def collect_market_data_node(state: TradingState) -> dict:
    settings = get_settings()
    symbol = state["symbol"]
    timeframe = state.get("timeframe", settings.default_timeframe)

    raw = await fetch_market_data(
        symbol,
        period=settings.market_data_period,
        interval=timeframe,
    )

    df = raw.pop("_df")
    indicators = compute_indicators(df)

    return {
        "market_data": raw,
        "indicators": indicators,
        "messages": [HumanMessage(content=f"Market data collected for {symbol} ({timeframe})")],
    }


# ── Node 2: Collect news ─────────────────────────────────────────────────────

async def collect_news_node(state: TradingState) -> dict:
    settings = get_settings()
    symbol = state["symbol"]

    news = await fetch_all_news(
        symbol,
        api_key=settings.news_api_key,
        limit=settings.news_per_symbol,
    )

    return {
        "news_items": news,
        "messages": [HumanMessage(content=f"Fetched {len(news)} news items for {symbol}")],
    }


# ── Node 3: RAG retrieval ────────────────────────────────────────────────────

async def retrieve_rag_context_node(state: TradingState) -> dict:
    settings = get_settings()
    client = _get_qdrant()
    symbol = state["symbol"]
    indicators = state.get("indicators") or {}

    indicators_summary = (
        f"RSI={indicators.get('rsi_14')}, "
        f"MACD={indicators.get('macd')}, "
        f"EMA9={indicators.get('ema_9')}, "
        f"close={indicators.get('close')}"
    )

    try:
        tech_ctx = retrieve_technical_context(
            client, settings.qdrant_collection, symbol, indicators_summary, limit=settings.rag_top_k
        )
    except Exception:
        tech_ctx = []

    try:
        news_ctx = retrieve_news_context(
            client, settings.qdrant_collection, symbol, limit=settings.rag_top_k
        )
    except Exception:
        news_ctx = []

    return {
        "technical_context": tech_ctx,
        "news_context": news_ctx,
        "messages": [HumanMessage(
            content=f"RAG retrieved {len(tech_ctx)} technical + {len(news_ctx)} news context items"
        )],
    }


# ── Node 4: Technical analysis ───────────────────────────────────────────────

async def analyze_technicals_node(state: TradingState) -> dict:
    llm = _get_llm()
    structured_llm = llm.with_structured_output(TechnicalAnalysis)

    symbol = state["symbol"]
    indicators = state.get("indicators") or {}
    market_data = state.get("market_data") or {}
    rag_context = state.get("technical_context") or []

    rag_text = "\n".join(f"- [{c['doc_type']}] {c['content']}" for c in rag_context) or "No prior context available."

    prompt = f"""You are an expert quantitative technical analyst. Analyze the following indicators for {symbol} and produce a structured analysis.

## Current Indicators
{_fmt(indicators)}

## Current Price Data
- Close: {market_data.get('latest_price')}
- Change: {market_data.get('change_pct')}%
- Volume: {market_data.get('volume')}

## Historical Context (RAG)
{rag_text}

Provide a thorough technical analysis. Be precise and quantitative.
"""

    result: TechnicalAnalysis = await structured_llm.ainvoke([
        SystemMessage(content="You are a professional quantitative technical analyst with 15 years of experience trading equities and futures."),
        HumanMessage(content=prompt),
    ])

    analysis_dict = result.model_dump()
    return {
        "technical_analysis": analysis_dict,
        "messages": [HumanMessage(content=f"Technical analysis: {result.trend} (score={result.score:.2f})")],
    }


# ── Node 5: Sentiment analysis ───────────────────────────────────────────────

async def analyze_sentiment_node(state: TradingState) -> dict:
    llm = _get_llm()
    structured_llm = llm.with_structured_output(SentimentAnalysis)

    symbol = state["symbol"]
    news_items = state.get("news_items") or []
    rag_context = state.get("news_context") or []

    news_text = "\n".join(
        f"{i+1}. [{item.get('source','?')}] {item['title']}: {item.get('summary','')}"
        for i, item in enumerate(news_items[:10])
    ) or "No recent news available."

    rag_text = "\n".join(f"- {c['content']}" for c in rag_context) or "No prior news context."

    prompt = f"""You are an expert financial sentiment analyst. Analyze the following news for {symbol}.

## Recent News
{news_text}

## Historical News Context (RAG)
{rag_text}

Analyze the overall market sentiment, identify key themes, and detect any major catalyst events.
"""

    result: SentimentAnalysis = await structured_llm.ainvoke([
        SystemMessage(content="You are a professional financial NLP specialist and sentiment analyst with expertise in market-moving news events."),
        HumanMessage(content=prompt),
    ])

    return {
        "sentiment_analysis": result.model_dump(),
        "messages": [HumanMessage(
            content=f"Sentiment: {result.overall_sentiment} (score={result.sentiment_score:.2f}, catalyst={result.catalyst_detected})"
        )],
    }


# ── Node 6: Signal generation ────────────────────────────────────────────────

async def generate_signal_node(state: TradingState) -> dict:
    llm = _get_llm()
    structured_llm = llm.with_structured_output(TradeSignal)

    symbol = state["symbol"]
    timeframe = state.get("timeframe", "1h")
    market_data = state.get("market_data") or {}
    tech = state.get("technical_analysis") or {}
    sentiment = state.get("sentiment_analysis") or {}
    indicators = state.get("indicators") or {}

    close = market_data.get("latest_price", 0)
    atr = indicators.get("atr_14") or (close * 0.02)

    prompt = f"""You are a professional algorithmic trading signal generator. Synthesize a precise trade signal for {symbol}.

## Technical Analysis
{_fmt(tech)}

## Sentiment Analysis
{_fmt(sentiment)}

## Current Market State
- Price: {close}
- ATR(14): {atr}
- Timeframe: {timeframe}

## Signal Guidelines
- Use ATR-based stop loss: entry ± 1.5×ATR
- Use 2:1 minimum risk/reward ratio for LONG/SHORT
- Set direction=HOLD if signals conflict or confidence < 0.55
- entry_price should be the current market price ({close})
- Assign technical_contribution + sentiment_contribution weights summing to 1.0
"""

    result: TradeSignal = await structured_llm.ainvoke([
        SystemMessage(content="You are a systematic quantitative trader. Generate disciplined, risk-adjusted trade signals based strictly on the provided analysis."),
        HumanMessage(content=prompt),
    ])

    return {
        "trade_signal": result.model_dump(),
        "messages": [HumanMessage(
            content=f"Signal: {result.direction} @ {result.entry_price} | confidence={result.confidence:.2f} | RR={result.risk_reward_ratio:.1f}"
        )],
    }


# ── Node 7: Risk assessment ──────────────────────────────────────────────────

async def assess_risk_node(state: TradingState) -> dict:
    llm = _get_llm()
    structured_llm = llm.with_structured_output(RiskAssessment)

    symbol = state["symbol"]
    trade_signal = state.get("trade_signal") or {}
    indicators = state.get("indicators") or {}
    market_data = state.get("market_data") or {}
    sentiment = state.get("sentiment_analysis") or {}

    atr = indicators.get("atr_14", 0)
    close = market_data.get("latest_price", 1)
    atr_pct = (atr / close * 100) if close else 0

    bb_bw = indicators.get("bb_bandwidth")

    prompt = f"""You are a risk management officer. Evaluate this trade signal for {symbol}.

## Trade Signal
{_fmt(trade_signal)}

## Volatility Metrics
- ATR(14): {atr} ({atr_pct:.2f}% of price)
- BB Bandwidth: {bb_bw}
- 24h Price Change: {market_data.get('change_pct')}%

## Market Sentiment Risk
- Catalyst detected: {sentiment.get('catalyst_detected')}
- Confidence: {sentiment.get('confidence')}

## Risk Rules
- HIGH volatility (ATR% > 3%): reduce position to max 1%
- EXTREME catalyst events: reduce to 0.5% or reject
- LOW confidence (< 0.6): reject or flag
- Recommended position sizing: Kelly / fixed-fraction between 0.5%–3%
"""

    result: RiskAssessment = await structured_llm.ainvoke([
        SystemMessage(content="You are a risk management specialist at a quantitative hedge fund. Your primary obligation is capital preservation."),
        HumanMessage(content=prompt),
    ])

    return {
        "risk_assessment": result.model_dump(),
        "messages": [HumanMessage(
            content=f"Risk: {result.risk_level} | position_size={result.position_size_pct:.1f}% | approved={result.approved}"
        )],
    }


# ── Node 8: Finalize output ──────────────────────────────────────────────────

async def finalize_output_node(state: TradingState) -> dict:
    from datetime import datetime

    symbol = state["symbol"]
    timeframe = state.get("timeframe", "1h")
    market_data = state.get("market_data") or {}
    tech = state.get("technical_analysis") or {}
    sentiment = state.get("sentiment_analysis") or {}
    signal = state.get("trade_signal") or {}
    risk = state.get("risk_assessment") or {}

    # Override signal if risk management rejects
    direction = signal.get("direction", "HOLD")
    if not risk.get("approved", True):
        direction = "HOLD"

    output = {
        "symbol": symbol,
        "timeframe": timeframe,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "direction": direction,
        "confidence": signal.get("confidence", 0.0),
        "entry_price": signal.get("entry_price", market_data.get("latest_price")),
        "stop_loss": signal.get("stop_loss"),
        "take_profit": signal.get("take_profit"),
        "risk_reward_ratio": signal.get("risk_reward_ratio", 0),
        "risk_level": risk.get("risk_level", "UNKNOWN"),
        "position_size_pct": risk.get("position_size_pct", 0),
        "technical_score": tech.get("score", 0),
        "sentiment_score": sentiment.get("sentiment_score", 0),
        "reasoning": signal.get("reasoning", ""),
        "warnings": risk.get("warnings", []),
        "risk_approved": risk.get("approved", False),
        "rejection_reason": risk.get("rejection_reason"),
    }

    return {
        "final_output": output,
        "messages": [HumanMessage(content=f"Final signal packaged: {direction} for {symbol}")],
    }
