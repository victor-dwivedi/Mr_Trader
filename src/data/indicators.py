from __future__ import annotations

import numpy as np
import pandas as pd


def _safe(val) -> float | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(float(val), 4)


def compute_indicators(df: pd.DataFrame) -> dict:
    """Compute a comprehensive set of technical indicators from an OHLCV DataFrame."""
    try:
        import pandas_ta as ta  # lazy import — optional dependency
        return _compute_with_pandas_ta(df, ta)
    except ImportError:
        return _compute_manual(df)


def _compute_with_pandas_ta(df: pd.DataFrame, ta) -> dict:
    df = df.copy()

    # Momentum
    df.ta.rsi(length=14, append=True)
    df.ta.rsi(length=7, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.stoch(k=14, d=3, append=True)

    # Trend
    df.ta.ema(length=9, append=True)
    df.ta.ema(length=21, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.sma(length=20, append=True)

    # Volatility
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)

    # Volume
    df.ta.obv(append=True)
    df.ta.vwap(append=True)

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    close = float(latest["Close"])

    result = {
        # Price context
        "close": close,
        "prev_close": _safe(prev["Close"]),

        # RSI
        "rsi_14": _safe(latest.get("RSI_14")),
        "rsi_7": _safe(latest.get("RSI_7")),

        # MACD
        "macd": _safe(latest.get("MACD_12_26_9")),
        "macd_signal": _safe(latest.get("MACDs_12_26_9")),
        "macd_histogram": _safe(latest.get("MACDh_12_26_9")),

        # Stochastic
        "stoch_k": _safe(latest.get("STOCHk_14_3_3")),
        "stoch_d": _safe(latest.get("STOCHd_14_3_3")),

        # EMAs
        "ema_9": _safe(latest.get("EMA_9")),
        "ema_21": _safe(latest.get("EMA_21")),
        "ema_50": _safe(latest.get("EMA_50")),
        "ema_200": _safe(latest.get("EMA_200")),
        "sma_20": _safe(latest.get("SMA_20")),

        # Bollinger Bands
        "bb_upper": _safe(latest.get("BBU_20_2.0")),
        "bb_middle": _safe(latest.get("BBM_20_2.0")),
        "bb_lower": _safe(latest.get("BBL_20_2.0")),
        "bb_bandwidth": _safe(latest.get("BBB_20_2.0")),
        "bb_percent": _safe(latest.get("BBP_20_2.0")),

        # ATR
        "atr_14": _safe(latest.get("ATRr_14")),

        # Volume
        "obv": _safe(latest.get("OBV")),
        "vwap": _safe(latest.get("VWAP_D")),

        # Derived signals
        "price_vs_ema9": _safe((close - float(latest.get("EMA_9") or close)) / close * 100),
        "price_vs_ema21": _safe((close - float(latest.get("EMA_21") or close)) / close * 100),
        "price_vs_sma20": _safe((close - float(latest.get("SMA_20") or close)) / close * 100),
    }

    # EMA alignment (trend confirmation)
    ema9 = latest.get("EMA_9")
    ema21 = latest.get("EMA_21")
    ema50 = latest.get("EMA_50")
    if all(v is not None and not np.isnan(v) for v in [ema9, ema21, ema50]):
        result["ema_bullish_alignment"] = bool(float(ema9) > float(ema21) > float(ema50))
        result["ema_bearish_alignment"] = bool(float(ema9) < float(ema21) < float(ema50))

    # MACD crossover
    macd_val = latest.get("MACD_12_26_9")
    macd_sig = latest.get("MACDs_12_26_9")
    prev_macd = prev.get("MACD_12_26_9")
    prev_sig = prev.get("MACDs_12_26_9")
    if all(v is not None for v in [macd_val, macd_sig, prev_macd, prev_sig]):
        result["macd_bullish_cross"] = bool(float(macd_val) > float(macd_sig) and float(prev_macd) <= float(prev_sig))
        result["macd_bearish_cross"] = bool(float(macd_val) < float(macd_sig) and float(prev_macd) >= float(prev_sig))

    return result


def _compute_manual(df: pd.DataFrame) -> dict:
    """Fallback manual indicator computation if pandas-ta is not available."""
    df = df.copy()
    close = df["Close"]

    def ema(series, span):
        return series.ewm(span=span, adjust=False).mean()

    def rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    rsi14 = rsi(close)
    ema9 = ema(close, 9)
    ema21 = ema(close, 21)
    ema50 = ema(close, 50)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()

    latest_close = float(close.iloc[-1])

    return {
        "close": latest_close,
        "rsi_14": _safe(rsi14.iloc[-1]),
        "ema_9": _safe(ema9.iloc[-1]),
        "ema_21": _safe(ema21.iloc[-1]),
        "ema_50": _safe(ema50.iloc[-1]),
        "sma_20": _safe(sma20.iloc[-1]),
        "bb_upper": _safe(float(sma20.iloc[-1]) + 2 * float(std20.iloc[-1])),
        "bb_lower": _safe(float(sma20.iloc[-1]) - 2 * float(std20.iloc[-1])),
        "ema_bullish_alignment": bool(float(ema9.iloc[-1]) > float(ema21.iloc[-1]) > float(ema50.iloc[-1])),
    }
