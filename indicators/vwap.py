from __future__ import annotations

import numpy as np
import pandas as pd


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def add_session_vwap(
    df: pd.DataFrame,
    price_col: str = "close",
    volume_col: str = "volume",
    timestamp_col: str = "timestamp",
    session_start_hour: int = 17,
) -> pd.DataFrame:
    """
    Adds futures-session VWAP using session date anchored at session_start_hour.

    For NQ-style futures:
      session_start_hour=17 means a session begins at 17:00.
    """
    _require_columns(df, [price_col, volume_col])

    out = df.copy()

    if timestamp_col in out.columns:
        ts = pd.to_datetime(out[timestamp_col], errors="coerce")
    else:
        ts = pd.to_datetime(out.index, errors="coerce")

    session_date = (ts - pd.to_timedelta(session_start_hour, unit="h")).dt.date
    typical_price = (out["high"] + out["low"] + out["close"]) / 3.0 if {"high", "low", "close"}.issubset(out.columns) else out[price_col]

    pv = typical_price * out[volume_col]
    cum_pv = pv.groupby(session_date).cumsum()
    cum_vol = out[volume_col].groupby(session_date).cumsum().replace(0, np.nan)

    out["vwap_session"] = cum_pv / cum_vol
    out["vwap_distance"] = out[price_col] - out["vwap_session"]
    out["vwap_distance_pct"] = (out[price_col] / out["vwap_session"] - 1.0) * 100.0

    return out


def add_vwap_bands(
    df: pd.DataFrame,
    window: int = 20,
    mult: float = 2.0,
    price_col: str = "close",
) -> pd.DataFrame:
    _require_columns(df, [price_col, "vwap_session"])

    out = df.copy()

    spread = out[price_col] - out["vwap_session"]
    spread_std = spread.rolling(window, min_periods=window).std()

    suffix = f"{window}_{str(mult).replace('.', 'p')}"
    out[f"vwap_band_upper_{suffix}"] = out["vwap_session"] + mult * spread_std
    out[f"vwap_band_lower_{suffix}"] = out["vwap_session"] - mult * spread_std
    out[f"vwap_band_width_{suffix}"] = out[f"vwap_band_upper_{suffix}"] - out[f"vwap_band_lower_{suffix}"]

    return out


def add_vwap_indicator_set(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = add_session_vwap(out, session_start_hour=17)
    out = add_vwap_bands(out, window=20, mult=1.0)
    out = add_vwap_bands(out, window=20, mult=2.0)
    return out