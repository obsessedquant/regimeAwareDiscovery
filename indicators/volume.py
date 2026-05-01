from __future__ import annotations

import numpy as np
import pandas as pd


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def add_volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    _require_columns(df, ["volume"])

    out = df.copy()

    vol_sma = out["volume"].rolling(period, min_periods=period).mean()
    out[f"volume_sma_{period}"] = vol_sma
    out[f"volume_ratio_{period}"] = out["volume"] / vol_sma.replace(0, np.nan)

    return out


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, ["close", "volume"])

    out = df.copy()

    direction = np.sign(out["close"].diff()).fillna(0.0)
    obv = (direction * out["volume"]).cumsum()

    out["obv"] = obv
    out["obv_slope_10"] = out["obv"].diff(10)
    out["obv_slope_20"] = out["obv"].diff(20)

    return out


def add_volume_indicator_set(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for period in [20, 50]:
        out = add_volume_ratio(out, period=period)

    out = add_obv(out)

    return out