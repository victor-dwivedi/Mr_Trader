from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    gemini_api_key: str = ""
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.1

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "trading_intelligence"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # Market Data
    default_symbols: list[str] = ["AAPL", "TSLA", "SPY", "QQQ", "NVDA"]
    default_timeframe: str = "1h"
    market_data_period: str = "5d"

    # News
    news_api_key: str = ""
    news_per_symbol: int = 10

    # Redis
    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 300

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_signal_history: int = 100

    # RAG
    rag_top_k: int = 5
    rag_score_threshold: float = 0.6


@lru_cache
def get_settings() -> Settings:
    return Settings()
