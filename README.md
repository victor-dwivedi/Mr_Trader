# Mr. Trader

A multi-agent trading intelligence system that pulls live market data, fetches recent news, runs technical + sentiment analysis in parallel, and produces a trade signal with a risk assessment — all in one pipeline call.

Built with LangGraph for orchestration, Qdrant for RAG context, Gemini as the LLM backbone, and FastAPI for the REST/WebSocket layer.

---

## How it works

The core is an 8-node LangGraph pipeline:

```
collect_market_data → collect_news → retrieve_rag_context
                                          ↓            ↓
                               analyze_technicals   analyze_sentiment
                                          ↓            ↓
                                      generate_signal
                                          ↓
                                      assess_risk
                                          ↓
                                      finalize_output
```

Technical and sentiment analysis run in parallel after the RAG retrieval step. Signal generation waits for both, then risk management either approves or overrides the direction to HOLD.

Each node uses Gemini with structured output (Pydantic models), so every stage returns typed, validated data rather than free-form text.

---

## Stack

- **LangGraph** — state graph orchestration
- **Gemini 2.5 Flash** (`langchain-google-genai`) — LLM for all analysis nodes
- **Qdrant** — vector store for historical news and technical context (RAG)
- **fastembed** (`BAAI/bge-small-en-v1.5`) — local embeddings, no API call needed
- **yfinance + pandas-ta** — market data and indicator computation
- **FastAPI + WebSocket** — HTTP API and real-time signal streaming
- **Redis** — signal caching (5 min TTL by default)

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/victor-dwivedi/Mr_Trader.git
cd Mr_Trader
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment variables

Create a `.env` file in the root:

```env
GEMINI_API_KEY=your_gemini_api_key
NEWS_API_KEY=your_newsapi_key

# optional overrides
QDRANT_HOST=localhost
QDRANT_PORT=6333
REDIS_URL=redis://localhost:6379
LLM_MODEL=gemini-2.5-flash
LLM_TEMPERATURE=0.1
```

### 3. Start infrastructure

```bash
docker-compose up qdrant redis -d
```

### 4. Ingest data into Qdrant

Run this once to populate the vector store with news and technical context:

```bash
python scripts/ingest_data.py
```

### 5. Start the API

```bash
uvicorn src.api.main:app --reload
```

API docs at `http://localhost:8000/docs`

---

## Usage

**Generate a signal:**

```bash
curl -X POST http://localhost:8000/signals/generate \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "timeframe": "1h"}'
```

**Batch signals for multiple symbols:**

```bash
curl "http://localhost:8000/signals/batch?symbols=AAPL,TSLA,NVDA&timeframe=1h"
```

**Signal history:**

```bash
curl http://localhost:8000/signals/history/AAPL
```

**Sample response:**

```json
{
  "symbol": "AAPL",
  "timeframe": "1h",
  "direction": "LONG",
  "confidence": 0.74,
  "entry_price": 213.45,
  "stop_loss": 209.80,
  "take_profit": 220.70,
  "risk_reward_ratio": 2.0,
  "risk_level": "MEDIUM",
  "position_size_pct": 1.5,
  "technical_score": 0.68,
  "sentiment_score": 0.55,
  "reasoning": "RSI recovering from oversold, MACD bullish cross forming...",
  "warnings": []
}
```

---

## Docker (full stack)

```bash
docker-compose up --build
```

This starts Qdrant, Redis, and the FastAPI app together.

---

## Project structure

```
src/
  agents/
    orchestrator.py   # LangGraph graph definition
    nodes.py          # all 8 node functions
  api/
    main.py           # FastAPI app
    routes/           # signals, market, health endpoints
    websocket.py      # real-time streaming
  config/
    settings.py       # pydantic-settings config
  data/
    market_feed.py    # yfinance wrapper
    news_feed.py      # NewsAPI + RSS feeds
    indicators.py     # pandas-ta computation
  models/
    schemas.py        # Pydantic models (LLM outputs + API I/O)
    state.py          # LangGraph TradingState TypedDict
  rag/
    embeddings.py     # fastembed setup
    ingestion.py      # ingest documents into Qdrant
    retriever.py      # similarity search helpers
scripts/
  ingest_data.py      # populate Qdrant
  backtest.py         # directional accuracy evaluation
```

---

## Tests

```bash
pytest tests/ -v
```

---

## Notes

- Timeframes supported: `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`
- Default symbols tracked: AAPL, TSLA, SPY, QQQ, NVDA
- If risk management rejects a signal, `direction` is overridden to `HOLD` regardless of what the signal node returned
- The graph is compiled once and cached via `lru_cache` — subsequent calls reuse the same compiled graph
