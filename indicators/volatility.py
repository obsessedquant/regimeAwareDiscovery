"""
Translated NinjaTrader indicators: volatility.py

Add manually converted indicator functions here when several indicators belong
in the same category.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import validate_ohlcv, true_range, wilder_smoothing


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def add_atr_expansion_features(
    df: pd.DataFrame,
    atr_col: str = "atr_14",
    lookback: int = 20,
) -> pd.DataFrame:
    """
    Adds ATR expansion/contraction features using an existing ATR column.

    Assumes atr_col was already created by indicators.core.add_core_indicator_set().
    Does not mutate input df.
    """
    _require_columns(df, [atr_col])

    out = df.copy()

    atr_ma = out[atr_col].rolling(lookback, min_periods=lookback).mean()
    atr_std = out[atr_col].rolling(lookback, min_periods=lookback).std()

    out[f"{atr_col}_pct_change_1"] = out[atr_col].pct_change(1) * 100.0
    out[f"{atr_col}_pct_change_5"] = out[atr_col].pct_change(5) * 100.0
    out[f"{atr_col}_ma_{lookback}"] = atr_ma
    out[f"{atr_col}_zscore_{lookback}"] = (out[atr_col] - atr_ma) / atr_std.replace(0, np.nan)
    out[f"{atr_col}_expansion_{lookback}"] = (out[f"{atr_col}_zscore_{lookback}"] > 1.0).astype(int)
    out[f"{atr_col}_contraction_{lookback}"] = (out[f"{atr_col}_zscore_{lookback}"] < -1.0).astype(int)

    return out


def add_volatility_indicator_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds volatility-derived feature columns.

    This currently expects atr_14 to already exist from core indicators.
    """
    out = df.copy()

    if "atr_14" not in out.columns:
        raise ValueError("Expected atr_14 from core indicators before adding volatility features.")

    out = add_atr_expansion_features(out, atr_col="atr_14", lookback=20)
    out = add_atr_expansion_features(out, atr_col="atr_14", lookback=50)

    return out