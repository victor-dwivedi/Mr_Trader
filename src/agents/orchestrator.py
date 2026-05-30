"""LangGraph trading intelligence graph."""
from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from src.models.state import TradingState

from .nodes import (
    analyze_sentiment_node,
    analyze_technicals_node,
    assess_risk_node,
    collect_market_data_node,
    collect_news_node,
    finalize_output_node,
    generate_signal_node,
    retrieve_rag_context_node,
)


def _build_graph() -> StateGraph:
    graph = StateGraph(TradingState)

    # Register nodes
    graph.add_node("collect_market_data", collect_market_data_node)
    graph.add_node("collect_news", collect_news_node)
    graph.add_node("retrieve_rag_context", retrieve_rag_context_node)
    graph.add_node("analyze_technicals", analyze_technicals_node)
    graph.add_node("analyze_sentiment", analyze_sentiment_node)
    graph.add_node("generate_signal", generate_signal_node)
    graph.add_node("assess_risk", assess_risk_node)
    graph.add_node("finalize_output", finalize_output_node)

    # Flow edges
    graph.add_edge(START, "collect_market_data")
    graph.add_edge("collect_market_data", "collect_news")
    graph.add_edge("collect_news", "retrieve_rag_context")

    # Parallel fan-out: both analyses run after RAG retrieval
    graph.add_edge("retrieve_rag_context", "analyze_technicals")
    graph.add_edge("retrieve_rag_context", "analyze_sentiment")

    # Fan-in: signal generation waits for both analyses
    graph.add_edge("analyze_technicals", "generate_signal")
    graph.add_edge("analyze_sentiment", "generate_signal")

    graph.add_edge("generate_signal", "assess_risk")
    graph.add_edge("assess_risk", "finalize_output")
    graph.add_edge("finalize_output", END)

    return graph


@lru_cache(maxsize=1)
def get_compiled_graph():
    """Returns the compiled (and cached) LangGraph app."""
    return _build_graph().compile()


async def run_analysis(symbol: str, timeframe: str = "1h") -> dict:
    """Run the full multi-agent analysis pipeline for a symbol."""
    app = get_compiled_graph()

    initial_state: TradingState = {
        "messages": [],
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "market_data": None,
        "indicators": None,
        "news_items": None,
        "technical_context": None,
        "news_context": None,
        "technical_analysis": None,
        "sentiment_analysis": None,
        "trade_signal": None,
        "risk_assessment": None,
        "final_output": None,
    }

    final_state = await app.ainvoke(initial_state)
    return final_state.get("final_output") or {}
