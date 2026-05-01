from __future__ import annotations

import numpy as np
import pandas as pd


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def add_rsi(df: pd.DataFrame, period: int = 14, column: str = "close") -> pd.DataFrame:
    _require_columns(df, [column])
    out = df.copy()

    delta = out[column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    out[f"rsi_{period}"] = out[f"rsi_{period}"].fillna(50)

    return out


def add_roc(df: pd.DataFrame, period: int = 14, column: str = "close") -> pd.DataFrame:
    _require_columns(df, [column])
    out = df.copy()
    out[f"roc_{period}"] = ((out[column] - out[column].shift(period)) / out[column].shift(period)) * 100
    return out


def add_momentum(df: pd.DataFrame, period: int = 14, column: str = "close") -> pd.DataFrame:
    _require_columns(df, [column])
    out = df.copy()
    out[f"mom_{period}"] = out[column] - out[column].shift(period)
    return out


def add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    _require_columns(df, ["high", "low", "close"])
    out = df.copy()

    tp = (out["high"] + out["low"] + out["close"]) / 3.0
    sma_tp = tp.rolling(period, min_periods=period).mean()

    mean_dev = tp.rolling(period, min_periods=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))),
        raw=True,
    )

    out[f"cci_{period}"] = (tp - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))
    return out


def add_stochastics(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> pd.DataFrame:
    _require_columns(df, ["high", "low", "close"])
    out = df.copy()

    lowest_low = out["low"].rolling(k_period, min_periods=k_period).min()
    highest_high = out["high"].rolling(k_period, min_periods=k_period).max()

    raw_k = 100 * (out["close"] - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    k = raw_k.rolling(smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(d_period, min_periods=d_period).mean()

    out[f"stoch_k_{k_period}_{smooth_k}"] = k
    out[f"stoch_d_{k_period}_{smooth_k}_{d_period}"] = d

    return out


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "close",
) -> pd.DataFrame:
    _require_columns(df, [column])
    out = df.copy()

    ema_fast = out[column].ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = out[column].ewm(span=slow, adjust=False, min_periods=slow).mean()

    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    macd_hist = macd_line - macd_signal

    out[f"macd_line_{fast}_{slow}_{signal}"] = macd_line
    out[f"macd_signal_{fast}_{slow}_{signal}"] = macd_signal
    out[f"macd_hist_{fast}_{slow}_{signal}"] = macd_hist

    return out


def add_momentum_indicator_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a standard momentum indicator pack.

    Uses lowercase OHLCV columns:
    open, high, low, close, volume

    Does not mutate input df.
    """
    out = df.copy()

    # RSI
    for period in [7, 14, 21]:
        out = add_rsi(out, period=period)

    # ROC
    for period in [5, 10, 14, 20]:
        out = add_roc(out, period=period)

    # Momentum
    for period in [5, 10, 14, 20]:
        out = add_momentum(out, period=period)

    # CCI
    for period in [14, 20, 50]:
        out = add_cci(out, period=period)

    # Stochastics
    out = add_stochastics(out, k_period=14, smooth_k=3, d_period=3)
    out = add_stochastics(out, k_period=21, smooth_k=3, d_period=3)

    # MACD
    out = add_macd(out, fast=12, slow=26, signal=9)
    out = add_macd(out, fast=8, slow=21, signal=5)

    return out