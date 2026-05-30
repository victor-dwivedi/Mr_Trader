from __future__ import annotations

from fastapi import APIRouter

from src.config.settings import get_settings
from src.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    settings = get_settings()
    qdrant_ok = False
    redis_ok = False

    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=2)
        client.get_collections()
        qdrant_ok = True
    except Exception:
        pass

    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(settings.redis_url, socket_connect_timeout=2)
        await r.ping()
        await r.aclose()
        redis_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok" if (qdrant_ok and redis_ok) else "degraded",
        qdrant=qdrant_ok,
        redis=redis_ok,
    )
