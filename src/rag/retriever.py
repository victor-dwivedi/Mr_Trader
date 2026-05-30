from __future__ import annotations

from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from .embeddings import embed_text


def retrieve_context(
    client: QdrantClient,
    collection_name: str,
    query: str,
    limit: int = 5,
    score_threshold: float = 0.5,
    symbol_filter: Optional[str] = None,
    doc_type_filter: Optional[str] = None,
) -> list[dict]:
    """Semantic search over the Qdrant collection, with optional metadata filters."""
    query_vector = embed_text(query)

    conditions = []
    if symbol_filter:
        conditions.append(FieldCondition(key="symbol", match=MatchValue(value=symbol_filter)))
    if doc_type_filter:
        conditions.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type_filter)))

    query_filter = Filter(must=conditions) if conditions else None

    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=limit,
        score_threshold=score_threshold,
        query_filter=query_filter,
    )

    return [
        {
            "content": r.payload.get("content", ""),
            "score": round(r.score, 4),
            "doc_type": r.payload.get("doc_type", "unknown"),
            "symbol": r.payload.get("symbol"),
            "source": r.payload.get("source"),
            "published_at": r.payload.get("published_at"),
        }
        for r in results
    ]


def retrieve_technical_context(
    client: QdrantClient,
    collection_name: str,
    symbol: str,
    indicators_summary: str,
    limit: int = 5,
) -> list[dict]:
    query = f"Technical analysis {symbol}: {indicators_summary}"
    return retrieve_context(
        client, collection_name, query,
        limit=limit, symbol_filter=symbol, doc_type_filter="analysis_technical",
    )


def retrieve_news_context(
    client: QdrantClient,
    collection_name: str,
    symbol: str,
    query_hint: str = "",
    limit: int = 5,
) -> list[dict]:
    query = f"Financial news sentiment {symbol} {query_hint}".strip()
    return retrieve_context(
        client, collection_name, query,
        limit=limit, symbol_filter=symbol, doc_type_filter="news",
    )
