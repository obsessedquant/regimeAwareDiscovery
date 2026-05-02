from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery")

INPUT_PARQUET = Path(
    r"C:\Users\srobi\OneDrive\Documents\Data\ninjatrader\regimeBuilder_NT8\nq_1min_with_nt8_regimes.parquet"
)

FEATURE_DIR = PROJECT_ROOT / "02_features"
OUTPUT_PARQUET = FEATURE_DIR / "feature_dataset_v5.parquet"
OUTPUT_SUMMARY = FEATURE_DIR / "feature_dataset_v5_summary.csv"

sys.path.insert(0, str(PROJECT_ROOT))

from indicators.core import add_core_indicator_set
from indicators.momentum import add_momentum_indicator_set
from indicators.vwap import add_vwap_indicator_set
from indicators.trend_strength import add_trend_strength_indicator_set
from indicators.volume import add_volume_indicator_set
from indicators.volatility import add_volatility_indicator_set
from indicators.channels import add_channel_indicator_set
from indicators.price_structure import add_price_structure_indicator_set
from indicators.custom_levels import add_custom_levels_indicator_set


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Timestamp": "timestamp",
        "DateTime": "timestamp",
        "datetime": "timestamp",
        "time": "timestamp",
    }

    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})
    return out


def add_regime_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "regime_family" not in out.columns and "frozen_broad_regime_family" in out.columns:
        out["regime_family"] = out["frozen_broad_regime_family"]

    if "regime_tuple" not in out.columns and "frozen_regime_tuple" in out.columns:
        out["regime_tuple"] = out["frozen_regime_tuple"]

    if "trend_state" not in out.columns and "frozen_trend_state" in out.columns:
        out["trend_state"] = out["frozen_trend_state"]

    if "intermediate_state" not in out.columns and "frozen_intermediate_state" in out.columns:
        out["intermediate_state"] = out["frozen_intermediate_state"]

    if "accel_state" not in out.columns and "frozen_accel_state" in out.columns:
        out["accel_state"] = out["frozen_accel_state"]

    return out


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col in df.columns:
        s = df[col]
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "non_null_count": int(s.notna().sum()),
                "null_count": int(s.isna().sum()),
                "null_pct": float(s.isna().mean()),
                "unique_count": int(s.nunique(dropna=True)),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Input parquet not found: {INPUT_PARQUET}")

    print("=" * 80)
    print("Building feature_dataset_v5.parquet")
    print("=" * 80)
    print(f"Input:  {INPUT_PARQUET}")
    print(f"Output: {OUTPUT_PARQUET}")

    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded rows: {len(df):,}")
    print(f"Loaded columns: {len(df.columns):,}")

    df = normalize_ohlcv_columns(df)
    df = add_regime_aliases(df)

    required = ["open", "high", "low", "close", "volume", "timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    print("Applying core indicators...")
    df = add_core_indicator_set(df)

    print("Applying momentum indicators...")
    df = add_momentum_indicator_set(df)

    print("Applying VWAP indicators...")
    df = add_vwap_indicator_set(df)

    print("Applying ADX/DMI trend-strength indicators...")
    df = add_trend_strength_indicator_set(df)

    print("Applying volume indicators...")
    df = add_volume_indicator_set(df)

    print("Applying volatility expansion/contraction indicators...")
    df = add_volatility_indicator_set(df)

    print("Applying channel indicators...")
    df = add_channel_indicator_set(df)

    print("Applying price-structure indicators...")
    df = add_price_structure_indicator_set(df)

    print("Applying custom NT8 level indicators...")
    df = add_custom_levels_indicator_set(df)

    print(f"Final rows: {len(df):,}")
    print(f"Final columns: {len(df.columns):,}")

    print("Writing parquet...")
    df.to_parquet(OUTPUT_PARQUET, index=False)

    print("Writing summary...")
    summary = build_summary(df)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    print("=" * 80)
    print("Done.")
    print(f"Wrote: {OUTPUT_PARQUET}")
    print(f"Wrote: {OUTPUT_SUMMARY}")
    print("=" * 80)


if __name__ == "__main__":
    main()