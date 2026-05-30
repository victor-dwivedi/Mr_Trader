from __future__ import annotations

from typing import Annotated, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class TradingState(TypedDict):
    # Graph messages for LLM conversation history
    messages: Annotated[list[BaseMessage], add_messages]

    # Request params
    symbol: str
    timeframe: str

    # Raw data (populated by data-collection nodes)
    market_data: Optional[dict]
    indicators: Optional[dict]
    news_items: Optional[list[dict]]

    # RAG retrieved context
    technical_context: Optional[list[dict]]
    news_context: Optional[list[dict]]

    # LLM analysis outputs (structured Pydantic dicts)
    technical_analysis: Optional[dict]
    sentiment_analysis: Optional[dict]

    # Signal & risk (final outputs)
    trade_signal: Optional[dict]
    risk_assessment: Optional[dict]

    # Final packaged response
    final_output: Optional[dict]
