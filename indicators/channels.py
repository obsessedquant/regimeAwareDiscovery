from __future__ import annotations

import numpy as np
import pandas as pd


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def add_donchian_channels(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    _require_columns(df, ["high", "low", "close"])

    out = df.copy()

    out[f"donchian_high_{period}"] = out["high"].rolling(period, min_periods=period).max()
    out[f"donchian_low_{period}"] = out["low"].rolling(period, min_periods=period).min()
    out[f"donchian_mid_{period}"] = (
        out[f"donchian_high_{period}"] + out[f"donchian_low_{period}"]
    ) / 2.0
    out[f"donchian_width_{period}"] = (
        out[f"donchian_high_{period}"] - out[f"donchian_low_{period}"]
    )
    out[f"donchian_position_{period}"] = (
        (out["close"] - out[f"donchian_low_{period}"])
        / out[f"donchian_width_{period}"].replace(0, np.nan)
    )

    return out


def add_keltner_channels(
    df: pd.DataFrame,
    ema_period: int = 20,
    atr_col: str = "atr_14",
    mult: float = 2.0,
) -> pd.DataFrame:
    _require_columns(df, ["close", atr_col])

    out = df.copy()

    suffix = f"{ema_period}_{str(mult).replace('.', 'p')}"
    mid = out["close"].ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()

    out[f"keltner_mid_{suffix}"] = mid
    out[f"keltner_upper_{suffix}"] = mid + mult * out[atr_col]
    out[f"keltner_lower_{suffix}"] = mid - mult * out[atr_col]
    out[f"keltner_width_{suffix}"] = out[f"keltner_upper_{suffix}"] - out[f"keltner_lower_{suffix}"]
    out[f"keltner_position_{suffix}"] = (
        (out["close"] - out[f"keltner_lower_{suffix}"])
        / out[f"keltner_width_{suffix}"].replace(0, np.nan)
    )

    return out


def add_channel_indicator_set(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for period in [20, 50]:
        out = add_donchian_channels(out, period=period)

    out = add_keltner_channels(out, ema_period=20, atr_col="atr_14", mult=1.5)
    out = add_keltner_channels(out, ema_period=20, atr_col="atr_14", mult=2.0)
    out = add_keltner_channels(out, ema_period=50, atr_col="atr_14", mult=2.0)

    return out