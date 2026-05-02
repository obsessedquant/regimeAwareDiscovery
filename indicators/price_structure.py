from __future__ import annotations

import numpy as np
import pandas as pd


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _get_timestamp_series(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.Series:
    if timestamp_col in df.columns:
        ts = pd.to_datetime(df[timestamp_col], errors="coerce")
    else:
        ts = pd.to_datetime(df.index, errors="coerce")
        ts = pd.Series(ts, index=df.index)

    # Normalize timezone handling: convert tz-aware timestamps to tz-naive.
    try:
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_convert(None)
    except TypeError:
        try:
            ts = ts.dt.tz_localize(None)
        except TypeError:
            pass

    return ts


def _session_key(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    session_start_hour: int = 17,
) -> pd.Series:
    ts = _get_timestamp_series(df, timestamp_col=timestamp_col)
    return (ts - pd.to_timedelta(session_start_hour, unit="h")).dt.date


def add_candle_structure(df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(df, ["open", "high", "low", "close"])

    out = df.copy()

    out["bar_range"] = out["high"] - out["low"]
    out["bar_body"] = (out["close"] - out["open"]).abs()
    out["bar_body_pct_of_range"] = out["bar_body"] / out["bar_range"].replace(0, np.nan)

    out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
    out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]

    out["inside_bar"] = (
        (out["high"] < out["high"].shift(1))
        & (out["low"] > out["low"].shift(1))
    ).astype(int)

    out["outside_bar"] = (
        (out["high"] > out["high"].shift(1))
        & (out["low"] < out["low"].shift(1))
    ).astype(int)

    out["higher_high"] = (out["high"] > out["high"].shift(1)).astype(int)
    out["lower_low"] = (out["low"] < out["low"].shift(1)).astype(int)

    out["bullish_bar"] = (out["close"] > out["open"]).astype(int)
    out["bearish_bar"] = (out["close"] < out["open"]).astype(int)

    return out


def add_range_expansion(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    _require_columns(df, ["high", "low"])

    out = df.copy()

    if "bar_range" not in out.columns:
        out["bar_range"] = out["high"] - out["low"]

    range_ma = out["bar_range"].rolling(lookback, min_periods=lookback).mean()
    range_std = out["bar_range"].rolling(lookback, min_periods=lookback).std()

    out[f"bar_range_ma_{lookback}"] = range_ma
    out[f"bar_range_zscore_{lookback}"] = (
        (out["bar_range"] - range_ma) / range_std.replace(0, np.nan)
    )
    out[f"range_expansion_{lookback}"] = (
        out[f"bar_range_zscore_{lookback}"] > 1.0
    ).astype(int)
    out[f"range_compression_{lookback}"] = (
        out[f"bar_range_zscore_{lookback}"] < -1.0
    ).astype(int)

    return out


def add_prior_session_levels(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    session_start_hour: int = 17,
) -> pd.DataFrame:
    _require_columns(df, ["high", "low", "close"])

    out = df.copy()
    session = _session_key(
        out,
        timestamp_col=timestamp_col,
        session_start_hour=session_start_hour,
    )

    session_ohlc = (
        out.assign(_session=session)
        .groupby("_session", sort=False)
        .agg(
            session_high=("high", "max"),
            session_low=("low", "min"),
            session_close=("close", "last"),
        )
    )

    prior = session_ohlc.shift(1)
    prior.columns = ["prior_session_high", "prior_session_low", "prior_session_close"]

    prior_map = prior.reindex(session).reset_index(drop=True)

    out["prior_session_high"] = prior_map["prior_session_high"].to_numpy()
    out["prior_session_low"] = prior_map["prior_session_low"].to_numpy()
    out["prior_session_close"] = prior_map["prior_session_close"].to_numpy()

    out["close_above_prior_session_high"] = (
        out["close"] > out["prior_session_high"]
    ).astype(int)
    out["close_below_prior_session_low"] = (
        out["close"] < out["prior_session_low"]
    ).astype(int)
    out["prior_session_range"] = out["prior_session_high"] - out["prior_session_low"]

    return out


def add_opening_range(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    session_start_hour: int = 17,
    opening_minutes: int = 30,
) -> pd.DataFrame:
    _require_columns(df, ["high", "low", "close"])

    out = df.copy()

    ts = _get_timestamp_series(out, timestamp_col=timestamp_col)
    session = _session_key(
        out,
        timestamp_col=timestamp_col,
        session_start_hour=session_start_hour,
    )

    session_start = pd.to_datetime(session.astype(str)) + pd.to_timedelta(
        session_start_hour,
        unit="h",
    )

    minutes_from_session_start = (ts - session_start).dt.total_seconds() / 60.0

    in_opening_range = (
        (minutes_from_session_start >= 0)
        & (minutes_from_session_start < opening_minutes)
    )

    temp = out.assign(_session=session, _in_or=in_opening_range)

    or_levels = (
        temp[temp["_in_or"]]
        .groupby("_session", sort=False)
        .agg(
            opening_range_high=("high", "max"),
            opening_range_low=("low", "min"),
        )
    )

    mapped = or_levels.reindex(session).reset_index(drop=True)

    out[f"opening_range_high_{opening_minutes}"] = mapped[
        "opening_range_high"
    ].to_numpy()
    out[f"opening_range_low_{opening_minutes}"] = mapped[
        "opening_range_low"
    ].to_numpy()

    out[f"opening_range_width_{opening_minutes}"] = (
        out[f"opening_range_high_{opening_minutes}"]
        - out[f"opening_range_low_{opening_minutes}"]
    )

    out[f"close_above_opening_range_high_{opening_minutes}"] = (
        out["close"] > out[f"opening_range_high_{opening_minutes}"]
    ).astype(int)

    out[f"close_below_opening_range_low_{opening_minutes}"] = (
        out["close"] < out[f"opening_range_low_{opening_minutes}"]
    ).astype(int)

    return out


def add_price_structure_indicator_set(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out = add_candle_structure(out)
    out = add_range_expansion(out, lookback=20)
    out = add_range_expansion(out, lookback=50)
    out = add_prior_session_levels(out, session_start_hour=17)
    out = add_opening_range(out, session_start_hour=17, opening_minutes=30)
    out = add_opening_range(out, session_start_hour=17, opening_minutes=60)

    return out