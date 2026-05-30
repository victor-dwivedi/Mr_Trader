from __future__ import annotations

import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    PayloadSchemaType,
)

from .embeddings import embed_texts


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int = 384,
    recreate: bool = False,
) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if recreate and collection_name in existing:
        client.delete_collection(collection_name)
        existing.discard(collection_name)

    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        # Index commonly filtered payload fields
        for field, schema in [
            ("symbol", PayloadSchemaType.KEYWORD),
            ("doc_type", PayloadSchemaType.KEYWORD),
            ("timestamp", PayloadSchemaType.FLOAT),
        ]:
            try:
                client.create_payload_index(collection_name, field, schema)
            except Exception:
                pass


def ingest_documents(
    client: QdrantClient,
    collection_name: str,
    documents: list[dict],
    content_key: str = "content",
    batch_size: int = 64,
) -> int:
    """Embed and upsert documents into Qdrant. Returns the number of points inserted."""
    if not documents:
        return 0

    inserted = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        texts = [str(doc.get(content_key, "")) for doc in batch]
        embeddings = embed_texts(texts)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload=doc,
            )
            for doc, emb in zip(batch, embeddings)
        ]
        client.upsert(collection_name=collection_name, points=points)
        inserted += len(points)

    return inserted


def build_news_document(
    title: str,
    summary: str,
    source: str,
    url: str,
    symbol: Optional[str] = None,
    published_at: Optional[str] = None,
    sentiment_score: Optional[float] = None,
) -> dict:
    content = f"{title}. {summary}".strip()
    return {
        "content": content,
        "title": title,
        "summary": summary,
        "source": source,
        "url": url,
        "symbol": symbol or "MARKET",
        "doc_type": "news",
        "published_at": published_at,
        "sentiment_score": sentiment_score,
    }


def build_analysis_document(
    symbol: str,
    analysis_type: str,
    content: str,
    metadata: Optional[dict] = None,
) -> dict:
    return {
        "content": content,
        "symbol": symbol,
        "doc_type": f"analysis_{analysis_type}",
        **(metadata or {}),
    }
