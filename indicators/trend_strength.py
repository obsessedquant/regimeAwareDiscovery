from __future__ import annotations

import numpy as np
import pandas as pd


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def add_adx_dmi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    _require_columns(df, ["high", "low", "close"])

    out = df.copy()

    high = out["high"]
    low = out["low"]
    close = out["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_dm_smoothed = pd.Series(plus_dm, index=out.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    minus_dm_smoothed = pd.Series(minus_dm, index=out.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = 100.0 * plus_dm_smoothed / atr.replace(0, np.nan)
    minus_di = 100.0 * minus_dm_smoothed / atr.replace(0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    out[f"plus_di_{period}"] = plus_di
    out[f"minus_di_{period}"] = minus_di
    out[f"adx_{period}"] = adx
    out[f"dmi_spread_{period}"] = plus_di - minus_di

    return out


def add_trend_strength_indicator_set(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for period in [14, 20]:
        out = add_adx_dmi(out, period=period)
    return out