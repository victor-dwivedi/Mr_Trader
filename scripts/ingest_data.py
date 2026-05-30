"""
Populate Qdrant with historical news and analysis documents.

Usage:
    python -m scripts.ingest_data --symbols AAPL TSLA SPY --days 30
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient

from src.config.settings import get_settings
from src.data.news_feed import fetch_all_news
from src.rag.ingestion import build_news_document, ensure_collection, ingest_documents


async def ingest_news_for_symbol(
    client: QdrantClient,
    collection: str,
    symbol: str,
    api_key: str,
    limit: int = 20,
) -> int:
    print(f"  Fetching news for {symbol}...")
    news = await fetch_all_news(symbol, api_key=api_key, limit=limit)

    docs = [
        build_news_document(
            title=item["title"],
            summary=item.get("summary", ""),
            source=item.get("source", "unknown"),
            url=item.get("url", ""),
            symbol=item.get("symbol", symbol),
            published_at=item.get("published_at"),
        )
        for item in news
        if item.get("title")
    ]

    count = ingest_documents(client, collection, docs)
    print(f"  Ingested {count} news documents for {symbol}")
    return count


async def main(symbols: list[str], recreate: bool = False) -> None:
    settings = get_settings()
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    print(f"Setting up Qdrant collection '{settings.qdrant_collection}' (recreate={recreate})...")
    ensure_collection(
        client,
        settings.qdrant_collection,
        vector_size=settings.embedding_dim,
        recreate=recreate,
    )
    print("Collection ready.")

    total = 0
    for symbol in symbols:
        print(f"\nProcessing {symbol}:")
        count = await ingest_news_for_symbol(
            client,
            settings.qdrant_collection,
            symbol,
            api_key=settings.news_api_key,
        )
        total += count

    print(f"\nIngestion complete. Total documents: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest financial data into Qdrant")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "TSLA", "SPY", "QQQ", "NVDA"],
        help="Stock symbols to ingest",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the Qdrant collection",
    )
    args = parser.parse_args()
    asyncio.run(main(args.symbols, args.recreate))
