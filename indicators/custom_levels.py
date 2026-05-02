from __future__ import annotations

import pandas as pd


def add_pivot_point(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pivot_point"] = (out["high"] + out["low"] + out["close"]) / 3.0
    return out


def add_standard_pivots(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    pp = (out["high"] + out["low"] + out["close"]) / 3.0
    rng = out["high"] - out["low"]

    out["pivot_point"] = pp

    # Standard pivots
    out["r1"] = (pp * 2.0) - out["low"]
    out["r2"] = pp + rng

    out["s1"] = (pp * 2.0) - out["high"]
    out["s2"] = pp - rng

    return out


def add_fib_levels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    pp = (out["high"] + out["low"] + out["close"]) / 3.0
    rng = out["high"] - out["low"]

    # IMPORTANT: replicate NT8 EXACTLY

    # FibR1 (non-standard from your code)
    out["fib_r1"] = (pp + (0.382 * 2.0)) - out["low"]

    # FibR2 (standard)
    out["fib_r2"] = pp + (0.618 * rng)

    # FibS1 (non-standard from your code)
    out["fib_s1"] = (pp - (0.382 * 2.0)) - out["high"]

    # FibS2 (standard)
    out["fib_s2"] = pp - (0.618 * rng)

    return out


def add_level_distances(df: pd.DataFrame) -> pd.DataFrame:
    """
    VERY HIGH VALUE FEATURES

    Converts levels into usable ML / rule features
    """
    out = df.copy()

    level_cols = [
        "pivot_point",
        "r1", "r2",
        "s1", "s2",
        "fib_r1", "fib_r2",
        "fib_s1", "fib_s2",
    ]

    for col in level_cols:
        if col in out.columns:
            out[f"dist_close_{col}"] = out["close"] - out[col]
            out[f"above_{col}"] = (out["close"] > out[col]).astype(int)

    return out


def add_custom_levels_indicator_set(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out = add_standard_pivots(out)
    out = add_fib_levels(out)
    out = add_level_distances(out)

    return out