from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import Optional

import feedparser
import httpx
import yfinance as yf

# Free financial RSS feeds indexed by symbol or general market
_RSS_FEEDS = {
    "MARKET": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.reuters.com/reuters/technologyNews",
    ],
}

_SYMBOL_RSS = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"


def _parse_date(entry) -> Optional[str]:
    for attr in ("published_parsed", "updated_parsed"):
        val = getattr(entry, attr, None)
        if val:
            try:
                return datetime(*val[:6], tzinfo=timezone.utc).isoformat()
            except Exception:
                pass
    return None


def _clean_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


async def fetch_yfinance_news(symbol: str, limit: int = 10) -> list[dict]:
    loop = asyncio.get_event_loop()

    def _get():
        ticker = yf.Ticker(symbol)
        raw = ticker.news or []
        results = []
        for item in raw[:limit]:
            results.append({
                "title": item.get("title", ""),
                "summary": item.get("summary", item.get("title", "")),
                "source": item.get("publisher", "Yahoo Finance"),
                "url": item.get("link", ""),
                "published_at": datetime.fromtimestamp(
                    item.get("providerPublishTime", 0), tz=timezone.utc
                ).isoformat() if item.get("providerPublishTime") else None,
                "symbol": symbol,
            })
        return results

    return await loop.run_in_executor(None, _get)


async def fetch_rss_news(symbol: str, limit: int = 10) -> list[dict]:
    url = _SYMBOL_RSS.format(symbol=symbol)

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(url, follow_redirects=True)
            feed = feedparser.parse(response.text)
        except Exception:
            return []

    results = []
    for entry in feed.entries[:limit]:
        results.append({
            "title": _clean_html(entry.get("title", "")),
            "summary": _clean_html(entry.get("summary", "")),
            "source": feed.feed.get("title", "RSS"),
            "url": entry.get("link", ""),
            "published_at": _parse_date(entry),
            "symbol": symbol,
        })
    return results


async def fetch_newsapi(symbol: str, api_key: str, limit: int = 10) -> list[dict]:
    """Fetch from NewsAPI.org — requires a free API key."""
    if not api_key:
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": symbol,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": limit,
        "apiKey": api_key,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(url, params=params)
            data = resp.json()
        except Exception:
            return []

    articles = data.get("articles", [])
    return [
        {
            "title": a.get("title", ""),
            "summary": a.get("description", ""),
            "source": a.get("source", {}).get("name", "NewsAPI"),
            "url": a.get("url", ""),
            "published_at": a.get("publishedAt"),
            "symbol": symbol,
        }
        for a in articles
    ]


async def fetch_all_news(symbol: str, api_key: str = "", limit: int = 15) -> list[dict]:
    """Aggregate news from all available sources and deduplicate by title."""
    results = await asyncio.gather(
        fetch_yfinance_news(symbol, limit),
        fetch_rss_news(symbol, limit // 2),
        fetch_newsapi(symbol, api_key, limit // 2),
        return_exceptions=True,
    )

    seen_titles: set[str] = set()
    merged: list[dict] = []
    for batch in results:
        if isinstance(batch, Exception):
            continue
        for item in batch:
            key = item["title"].lower()[:80]
            if key not in seen_titles:
                seen_titles.add(key)
                merged.append(item)

    return merged[:limit]
