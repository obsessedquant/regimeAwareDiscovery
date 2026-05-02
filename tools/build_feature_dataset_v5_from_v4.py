from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery")

INPUT_PARQUET = PROJECT_ROOT / "02_features" / "feature_dataset_v4.parquet"
OUTPUT_PARQUET = PROJECT_ROOT / "02_features" / "feature_dataset_v5.parquet"
OUTPUT_SUMMARY = PROJECT_ROOT / "02_features" / "feature_dataset_v5_summary.csv"

sys.path.insert(0, str(PROJECT_ROOT))

from indicators.custom_levels import add_custom_levels_indicator_set


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].notna().sum()),
                "null_count": int(df[col].isna().sum()),
                "null_pct": float(df[col].isna().mean()),
                "unique_count": int(df[col].nunique(dropna=True)),
            }
            for col in df.columns
        ]
    )


def main() -> None:
    print("=" * 80)
    print("Building feature_dataset_v5.parquet from existing v4")
    print("=" * 80)
    print(f"Input:  {INPUT_PARQUET}")
    print(f"Output: {OUTPUT_PARQUET}")

    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded rows: {len(df):,}")
    print(f"Loaded columns: {len(df.columns):,}")

    print("Applying custom NT8 level indicators...")
    df = add_custom_levels_indicator_set(df)

    print(f"Final rows: {len(df):,}")
    print(f"Final columns: {len(df.columns):,}")

    print("Writing parquet...")
    df.to_parquet(OUTPUT_PARQUET, index=False)

    print("Writing summary...")
    build_summary(df).to_csv(OUTPUT_SUMMARY, index=False)

    print("=" * 80)
    print("Done.")
    print(f"Wrote: {OUTPUT_PARQUET}")
    print(f"Wrote: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()