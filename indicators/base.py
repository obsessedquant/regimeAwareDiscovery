"""
Base utilities for translated NinjaTrader indicators.
"""

from __future__ import annotations

import pandas as pd


REQUIRED_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def validate_ohlcv(df: pd.DataFrame, required=None) -> None:
    required = required or REQUIRED_OHLCV_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def true_range(df: pd.DataFrame) -> pd.Series:
    validate_ohlcv(df, ["high", "low", "close"])
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def wilder_smoothing(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1 / period, adjust=False).mean()
