# ============================================================
# Build Core Feature Dataset
# Regime-Aware Strategy Discovery Pipeline
# ============================================================
#
# Recommended save location:
# C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery\tools\build_core_feature_dataset.py
#
# Purpose:
#   Read the canonical NT8 regime parquet, add core Python indicator features,
#   and write a Rust-ready feature dataset.
#
# Input default:
#   C:\Users\srobi\OneDrive\Documents\Data\ninjatrader\regimeBuilder_NT8\nq_1min_with_nt8_regimes.parquet
#
# Output default:
#   C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery\02_features\feature_dataset_core.parquet
#
# Run:
#   python build_core_feature_dataset.py
#
# Optional:
#   python build_core_feature_dataset.py --input "path\to\input.parquet" --output "path\to\output.parquet"
# ============================================================

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery")
DEFAULT_INPUT = Path(
    r"C:\Users\srobi\OneDrive\Documents\Data\ninjatrader\regimeBuilder_NT8\nq_1min_with_nt8_regimes.parquet"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "02_features" / "feature_dataset_core.parquet"
DEFAULT_SUMMARY_OUTPUT = PROJECT_ROOT / "02_features" / "feature_dataset_core_summary.csv"

# Make local package import work when script is run from tools\
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from indicators.core import add_core_indicator_set  # noqa: E402


REQUIRED_BASE_COLUMNS = ["open", "high", "low", "close", "volume"]

REGIME_COLUMN_CANDIDATES = [
    # NT8 frozen regime outputs (PRIMARY)
    "frozen_trend_state",
    "frozen_intermediate_state",
    "frozen_accel_state",
    "frozen_ppm3_state",
    "frozen_regime_score",
    "frozen_regime_code",
    "frozen_regime_tuple",
    "frozen_broad_regime_family",
    "frozen_ppm3_confirms_trend",
    "frozen_daily_long_bias",
    "frozen_daily_short_bias",

    # Generic aliases (SECONDARY)
    "regime_family",
    "regime_tuple",
    "trend_state",
    "intermediate_state",
    "accel_state",
    "trend",
    "intermediate",
    "accel",
]


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def validate_input_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_BASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Input parquet is missing required OHLCV columns: "
            f"{missing}. Available columns: {list(df.columns)}"
        )


def ensure_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps timestamp handling conservative.
    - If timestamp column exists, leave it.
    - If index looks datetime-like, reset it into timestamp.
    - Otherwise leave as-is and let downstream code use row order.
    """
    out = df.copy()
    if "timestamp" in out.columns:
        return out

    if isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index().rename(columns={out.index.name or "index": "timestamp"})
        return out

    return out


def normalize_regime_tuple(df: pd.DataFrame) -> pd.DataFrame:
    """
    If regime_tuple is absent but state columns exist, create it.
    Supports either:
      trend_state/intermediate_state/accel_state
    or:
      trend/intermediate/accel
    """
    out = df.copy()
    
    if "frozen_regime_tuple" in out.columns:
        out["regime_tuple"] = out["frozen_regime_tuple"]
        return out

    if "regime_tuple" in out.columns:
        return out

    if all(c in out.columns for c in ["trend_state", "intermediate_state", "accel_state"]):
        out["regime_tuple"] = (
            out["trend_state"].astype("Int64").astype(str)
            + "_"
            + out["intermediate_state"].astype("Int64").astype(str)
            + "_"
            + out["accel_state"].astype("Int64").astype(str)
        )
        return out

    if all(c in out.columns for c in ["trend", "intermediate", "accel"]):
        out["regime_tuple"] = (
            out["trend"].astype("Int64").astype(str)
            + "_"
            + out["intermediate"].astype("Int64").astype(str)
            + "_"
            + out["accel"].astype("Int64").astype(str)
        )
        return out
        
    if all(c in out.columns for c in [
        "frozen_trend_state",
        "frozen_intermediate_state",
        "frozen_accel_state"
    ]):
        out["regime_tuple"] = (
            out["frozen_trend_state"].astype("Int64").astype(str)
            + "_"
            + out["frozen_intermediate_state"].astype("Int64").astype(str)
            + "_"
            + out["frozen_accel_state"].astype("Int64").astype(str)
        )
        return out

    return out


def write_summary(df_before: pd.DataFrame, df_after: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    before_cols = set(df_before.columns)
    after_cols = set(df_after.columns)
    added_cols = sorted(after_cols - before_cols)

    rows = []
    for col in added_cols:
        s = df_after[col]
        rows.append(
            {
                "feature_column": col,
                "dtype": str(s.dtype),
                "non_null_count": int(s.notna().sum()),
                "null_count": int(s.isna().sum()),
                "null_pct": float(s.isna().mean()),
                "min": float(s.min()) if pd.api.types.is_numeric_dtype(s) and s.notna().any() else None,
                "max": float(s.max()) if pd.api.types.is_numeric_dtype(s) and s.notna().any() else None,
                "mean": float(s.mean()) if pd.api.types.is_numeric_dtype(s) and s.notna().any() else None,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_path, index=False)


def build_core_feature_dataset(
    input_path: Path,
    output_path: Path,
    summary_output_path: Path,
    compression: str = "zstd",
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    print(f"[{now_str()}] Reading input parquet...")
    print(f"INPUT: {input_path}")
    df = pd.read_parquet(input_path)

    print(f"[{now_str()}] Rows: {len(df):,}")
    print(f"[{now_str()}] Columns before: {len(df.columns):,}")

    df = ensure_timestamp_column(df)
    df = normalize_regime_tuple(df)
    
    # Create standardized column names for downstream Rust engine
    if "frozen_broad_regime_family" in df.columns:
        df["regime_family"] = df["frozen_broad_regime_family"]

    if "frozen_trend_state" in df.columns:
        df["trend_state"] = df["frozen_trend_state"]

    if "frozen_intermediate_state" in df.columns:
        df["intermediate_state"] = df["frozen_intermediate_state"]

    if "frozen_accel_state" in df.columns:
        df["accel_state"] = df["frozen_accel_state"]
    
    validate_input_columns(df)

    present_regime_cols = [c for c in REGIME_COLUMN_CANDIDATES if c in df.columns]
    print(f"[{now_str()}] Regime-related columns found: {present_regime_cols}")

    print(f"[{now_str()}] Adding core indicator features...")
    before = df.copy()
    features = add_core_indicator_set(df)

    added_cols = sorted(set(features.columns) - set(before.columns))
    print(f"[{now_str()}] Added feature columns: {len(added_cols):,}")
    for c in added_cols[:50]:
        print(f"  + {c}")
    if len(added_cols) > 50:
        print(f"  ... {len(added_cols) - 50} more")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{now_str()}] Writing output parquet...")
    print(f"OUTPUT: {output_path}")
    features.to_parquet(output_path, index=False, compression=compression)

    print(f"[{now_str()}] Writing feature summary...")
    print(f"SUMMARY: {summary_output_path}")
    write_summary(before, features, summary_output_path)

    print(f"[{now_str()}] Done.")
    print(f"Final rows: {len(features):,}")
    print(f"Final columns: {len(features.columns):,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build core feature dataset from NT8 regime parquet.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input canonical/regime parquet path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output feature parquet path")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help="Output feature summary CSV path",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        choices=["zstd", "snappy", "gzip", "brotli", "none"],
        help="Parquet compression codec",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compression = None if args.compression == "none" else args.compression
    build_core_feature_dataset(
        input_path=args.input,
        output_path=args.output,
        summary_output_path=args.summary_output,
        compression=compression,
    )
