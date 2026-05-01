# ============================================================
# Core Indicator Module
# Regime-Aware Strategy Discovery Pipeline
# ============================================================
#
# Recommended save location:
# C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery\indicators\core.py
#
# Purpose:
#   Production-grade Python equivalents for the first core indicators:
#     - SMA
#     - EMA
#     - WMA
#     - ZLEMA
#     - ATR
#     - Bollinger Bands
#     - Choppiness Index
#
# Design rules:
#   - Pure functions
#   - Do not mutate input df
#   - Return a dataframe copy with new columns appended
#   - Use lowercase OHLCV columns: open, high, low, close, volume
#   - Keep output names deterministic for strategy generation
#
# Notes on NinjaTrader parity:
#   - SMA/WMA/Bollinger/Choppiness are straightforward rolling calculations.
#   - EMA/ZLEMA/ATR parity can differ slightly depending on NT8 initialization.
#   - This module uses common NinjaTrader-compatible defaults, but final
#     validation should compare against exported NT8 values.
# ============================================================

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Validation helpers
# ------------------------------------------------------------

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _copy_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    return df.copy()


def validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _safe_period(period: int, name: str = "period") -> int:
    period = int(period)
    if period <= 0:
        raise ValueError(f"{name} must be > 0")
    return period


# ------------------------------------------------------------
# Core moving averages
# ------------------------------------------------------------

def compute_sma(
    df: pd.DataFrame,
    period: int = 14,
    price_col: str = "close",
    out_col: Optional[str] = None,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simple Moving Average.

    NinjaTrader-style behavior generally requires a full period before output.
    Therefore min_periods defaults to period.
    """
    period = _safe_period(period)
    validate_columns(df, [price_col])
    out = _copy_df(df)
    out_col = out_col or f"sma_{period}_{price_col}"
    min_periods = period if min_periods is None else int(min_periods)
    out[out_col] = out[price_col].rolling(window=period, min_periods=min_periods).mean()
    return out


def compute_ema(
    df: pd.DataFrame,
    period: int = 14,
    price_col: str = "close",
    out_col: Optional[str] = None,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    Exponential Moving Average.

    Uses alpha = 2 / (period + 1), matching the standard EMA formula.
    min_periods defaults to period to avoid early warmup values being used
    in strategy generation.
    """
    period = _safe_period(period)
    validate_columns(df, [price_col])
    out = _copy_df(df)
    out_col = out_col or f"ema_{period}_{price_col}"
    min_periods = period if min_periods is None else int(min_periods)
    out[out_col] = out[price_col].ewm(span=period, adjust=False, min_periods=min_periods).mean()
    return out


def compute_wma(
    df: pd.DataFrame,
    period: int = 14,
    price_col: str = "close",
    out_col: Optional[str] = None,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    Weighted Moving Average with linearly increasing weights.

    Most recent value gets the largest weight.
    """
    period = _safe_period(period)
    validate_columns(df, [price_col])
    out = _copy_df(df)
    out_col = out_col or f"wma_{period}_{price_col}"
    min_periods = period if min_periods is None else int(min_periods)

    weights = np.arange(1, period + 1, dtype=float)
    denom = weights.sum()

    def _wma(values: np.ndarray) -> float:
        return float(np.dot(values, weights) / denom)

    out[out_col] = out[price_col].rolling(window=period, min_periods=min_periods).apply(_wma, raw=True)
    return out


def compute_zlema(
    df: pd.DataFrame,
    period: int = 14,
    price_col: str = "close",
    out_col: Optional[str] = None,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    Zero-Lag EMA.

    NinjaTrader's common ZLEMA logic:
        lag = int((period - 1) / 2)
        adjusted_price = price + (price - price[lag])
        zlema = EMA(adjusted_price, period)
    """
    period = _safe_period(period)
    validate_columns(df, [price_col])
    out = _copy_df(df)
    out_col = out_col or f"zlema_{period}_{price_col}"
    min_periods = period if min_periods is None else int(min_periods)

    lag = int((period - 1) / 2)
    adjusted_price = out[price_col] + (out[price_col] - out[price_col].shift(lag))
    out[out_col] = adjusted_price.ewm(span=period, adjust=False, min_periods=min_periods).mean()
    return out


# ------------------------------------------------------------
# Volatility
# ------------------------------------------------------------

def true_range(df: pd.DataFrame, out_col: Optional[str] = None) -> pd.Series | pd.DataFrame:
    """
    True Range.

    If out_col is supplied, returns a dataframe copy with the TR column.
    Otherwise returns a Series.
    """
    validate_columns(df, ["high", "low", "close"])
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    if out_col is None:
        return tr

    out = _copy_df(df)
    out[out_col] = tr
    return out


def compute_atr(
    df: pd.DataFrame,
    period: int = 14,
    out_col: Optional[str] = None,
    method: str = "wilder",
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    Average True Range.

    method options:
      - "wilder": alpha = 1 / period, common NT/Wilder ATR behavior
      - "sma": simple rolling ATR
      - "ema": EMA-style ATR with alpha = 2 / (period + 1)

    For NT8 parity, start with method="wilder".
    """
    period = _safe_period(period)
    validate_columns(df, ["high", "low", "close"])
    out = _copy_df(df)
    out_col = out_col or f"atr_{period}"
    min_periods = period if min_periods is None else int(min_periods)

    tr = true_range(out)
    method = method.lower().strip()

    if method == "wilder":
        out[out_col] = tr.ewm(alpha=1 / period, adjust=False, min_periods=min_periods).mean()
    elif method == "sma":
        out[out_col] = tr.rolling(window=period, min_periods=min_periods).mean()
    elif method == "ema":
        out[out_col] = tr.ewm(span=period, adjust=False, min_periods=min_periods).mean()
    else:
        raise ValueError("method must be one of: 'wilder', 'sma', 'ema'")

    return out


# ------------------------------------------------------------
# Bands
# ------------------------------------------------------------

def compute_bollinger(
    df: pd.DataFrame,
    period: int = 20,
    num_std: float = 2.0,
    price_col: str = "close",
    out_prefix: Optional[str] = None,
    min_periods: Optional[int] = None,
    ddof: int = 0,
) -> pd.DataFrame:
    """
    Bollinger Bands.

    Outputs:
      {prefix}_mid
      {prefix}_upper
      {prefix}_lower
      {prefix}_width
      {prefix}_pct_b

    ddof defaults to 0. If NT8 parity requires sample standard deviation,
    change ddof to 1 during validation.
    """
    period = _safe_period(period)
    validate_columns(df, [price_col])
    out = _copy_df(df)
    out_prefix = out_prefix or f"bb_{period}_{str(num_std).replace('.', 'p')}_{price_col}"
    min_periods = period if min_periods is None else int(min_periods)

    mid = out[price_col].rolling(window=period, min_periods=min_periods).mean()
    std = out[price_col].rolling(window=period, min_periods=min_periods).std(ddof=ddof)
    upper = mid + num_std * std
    lower = mid - num_std * std

    out[f"{out_prefix}_mid"] = mid
    out[f"{out_prefix}_upper"] = upper
    out[f"{out_prefix}_lower"] = lower
    out[f"{out_prefix}_width"] = upper - lower
    out[f"{out_prefix}_pct_b"] = (out[price_col] - lower) / (upper - lower)
    return out


# ------------------------------------------------------------
# Choppiness Index
# ------------------------------------------------------------

def compute_choppiness_index(
    df: pd.DataFrame,
    period: int = 14,
    out_col: Optional[str] = None,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    Choppiness Index.

    Formula:
        100 * log10(sum(TR, n) / (max(high, n) - min(low, n))) / log10(n)

    This matches the common NT8 ChoppinessIndex implementation structure.
    """
    period = _safe_period(period)
    validate_columns(df, ["high", "low", "close"])
    out = _copy_df(df)
    out_col = out_col or f"choppiness_{period}"
    min_periods = period if min_periods is None else int(min_periods)

    tr = true_range(out)
    tr_sum = tr.rolling(window=period, min_periods=min_periods).sum()
    highest_high = out["high"].rolling(window=period, min_periods=min_periods).max()
    lowest_low = out["low"].rolling(window=period, min_periods=min_periods).min()
    price_range = highest_high - lowest_low

    # Avoid divide-by-zero and invalid log values.
    ratio = tr_sum / price_range.replace(0, np.nan)
    out[out_col] = 100.0 * np.log10(ratio) / np.log10(period)
    return out


# ------------------------------------------------------------
# Convenience batch builder
# ------------------------------------------------------------

def add_core_indicator_set(
    df: pd.DataFrame,
    price_col: str = "close",
    ma_periods: tuple[int, ...] = (10, 20, 40, 50, 100, 200),
    atr_periods: tuple[int, ...] = (14,),
    bollinger_periods: tuple[int, ...] = (20,),
    choppiness_periods: tuple[int, ...] = (14,),
) -> pd.DataFrame:
    """
    Adds a practical starter feature set for randomized strategy discovery.

    This is intentionally conservative. Expand later through the manifest-driven
    feature builder.
    """
    out = _copy_df(df)

    for p in ma_periods:
        out = compute_sma(out, period=p, price_col=price_col)
        out = compute_ema(out, period=p, price_col=price_col)
        out = compute_wma(out, period=p, price_col=price_col)
        out = compute_zlema(out, period=p, price_col=price_col)

    for p in atr_periods:
        out = compute_atr(out, period=p, method="wilder")

    for p in bollinger_periods:
        out = compute_bollinger(out, period=p, num_std=2.0, price_col=price_col)

    for p in choppiness_periods:
        out = compute_choppiness_index(out, period=p)

    return out


# ------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------

if __name__ == "__main__":
    sample = pd.DataFrame(
        {
            "open": np.linspace(100, 120, 300),
            "high": np.linspace(101, 121, 300),
            "low": np.linspace(99, 119, 300),
            "close": np.linspace(100.5, 120.5, 300) + np.sin(np.arange(300) / 7),
            "volume": np.random.default_rng(42).integers(100, 1000, 300),
        }
    )

    result = add_core_indicator_set(sample)
    print(result.tail())
    print("Columns added:", len(result.columns) - len(sample.columns))
