# ============================================================
# Analyze Rust Results
# Regime-Aware Strategy Discovery Pipeline
# ============================================================
#
# Recommended save location:
# C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery\tools\analyze_rust_results.py
#
# Purpose:
#   Read latest Rust engine output CSVs, filter top candidates, update the
#   strategy registry, and write ranked candidate files.
#
# Inputs:
#   05_rust_outputs\performance\performance_overall_*.csv
#   05_rust_outputs\regime_family\performance_regime_family_*.csv
#   05_rust_outputs\regime_tuple\performance_regime_tuple_*.csv
#   03_strategy_registry\strategy_combinations.csv
#
# Outputs:
#   06_candidates\top_strategy_candidates_overall.csv
#   06_candidates\top_strategy_candidates_regime_family.csv
#   06_candidates\top_strategy_candidates_regime_tuple.csv
#   06_candidates\top_strategy_candidates_all.csv
#   06_candidates\candidate_summary.txt
#   Updated 03_strategy_registry\strategy_combinations.csv
#
# Run:
#   python analyze_rust_results.py
#
# Optional:
#   python analyze_rust_results.py --min-np-dd 2 --overall-min-trades 100 --family-min-trades 25 --tuple-min-trades 10
# ============================================================

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery")
RUST_OUTPUT_ROOT = PROJECT_ROOT / "05_rust_outputs"
REGISTRY_PATH = PROJECT_ROOT / "03_strategy_registry" / "strategy_combinations.csv"
CANDIDATE_DIR = PROJECT_ROOT / "06_candidates"

OVERALL_DIR = RUST_OUTPUT_ROOT / "performance"
FAMILY_DIR = RUST_OUTPUT_ROOT / "regime_family"
TUPLE_DIR = RUST_OUTPUT_ROOT / "regime_tuple"

OVERALL_OUT = CANDIDATE_DIR / "top_strategy_candidates_overall.csv"
FAMILY_OUT = CANDIDATE_DIR / "top_strategy_candidates_regime_family.csv"
TUPLE_OUT = CANDIDATE_DIR / "top_strategy_candidates_regime_tuple.csv"
ALL_OUT = CANDIDATE_DIR / "top_strategy_candidates_all.csv"
SUMMARY_OUT = CANDIDATE_DIR / "candidate_summary.txt"


NUMERIC_COLUMNS = [
    "trade_count",
    "gross_profit",
    "gross_loss",
    "net_profit",
    "max_drawdown",
    "np_dd",
    "win_rate",
    "avg_trade",
    "profit_factor",
]


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def latest_file(folder: Path, pattern: str) -> Path:
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No files found in {folder} matching {pattern}")
    return files[0]


def read_perf(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in NUMERIC_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def filter_candidates(
    df: pd.DataFrame,
    min_np_dd: float,
    min_trades: int,
    require_net_profit_positive: bool = True,
    require_drawdown_positive: bool = True,
) -> pd.DataFrame:
    out = df.copy()

    mask = out["np_dd"].fillna(-999999) >= min_np_dd
    mask &= out["trade_count"].fillna(0) >= min_trades

    if require_net_profit_positive:
        mask &= out["net_profit"].fillna(0) > 0

    if require_drawdown_positive:
        mask &= out["max_drawdown"].fillna(0) > 0

    out = out.loc[mask].copy()
    out = out.sort_values(
        by=["np_dd", "net_profit", "trade_count"],
        ascending=[False, False, False],
    )
    return out


def add_candidate_metadata(df: pd.DataFrame, candidate_scope: str, min_trades: int, min_np_dd: float) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "candidate_scope", candidate_scope)
    out.insert(1, "candidate_min_trades", min_trades)
    out.insert(2, "candidate_min_np_dd", min_np_dd)
    out.insert(3, "selected_at", now_str())
    return out


def update_registry_with_overall(registry_path: Path, overall: pd.DataFrame) -> None:
    if not registry_path.exists():
        print(f"WARNING: Registry not found, skipping update: {registry_path}")
        return

    registry = pd.read_csv(registry_path, dtype=str).fillna("")
    perf = overall.copy()

    keep_cols = [
        "strategy_id",
        "trade_count",
        "net_profit",
        "max_drawdown",
        "np_dd",
        "profit_factor",
        "win_rate",
        "avg_trade",
    ]
    perf = perf[[c for c in keep_cols if c in perf.columns]].copy()

    for c in keep_cols:
        if c in perf.columns and c != "strategy_id":
            perf[c] = perf[c].astype(str)

    perf = perf.rename(
        columns={
            "trade_count": "trade_count_new",
            "net_profit": "net_profit_new",
            "max_drawdown": "max_drawdown_new",
            "np_dd": "np_dd_new",
            "profit_factor": "profit_factor_new",
            "win_rate": "win_rate_new",
            "avg_trade": "avg_trade_new",
        }
    )

    merged = registry.merge(perf, on="strategy_id", how="left")

    update_map = {
        "trade_count": "trade_count_new",
        "net_profit": "net_profit_new",
        "max_drawdown": "max_drawdown_new",
        "np_dd": "np_dd_new",
    }

    for old_col, new_col in update_map.items():
        if old_col not in merged.columns:
            merged[old_col] = ""
        if new_col in merged.columns:
            mask = merged[new_col].notna() & (merged[new_col].astype(str) != "")
            merged.loc[mask, old_col] = merged.loc[mask, new_col]

    if "status" not in merged.columns:
        merged["status"] = ""
    if "updated_at" not in merged.columns:
        merged["updated_at"] = ""

    completed_mask = merged.get("trade_count_new", pd.Series([""] * len(merged))).astype(str) != ""
    merged.loc[completed_mask, "status"] = "complete"
    merged.loc[completed_mask, "updated_at"] = now_str()

    drop_cols = [c for c in merged.columns if c.endswith("_new")]
    merged = merged.drop(columns=drop_cols)

    merged.to_csv(registry_path, index=False)
    print(f"Updated registry: {registry_path}")


def write_summary(
    summary_path: Path,
    overall_file: Path,
    family_file: Path,
    tuple_file: Path,
    overall: pd.DataFrame,
    family: pd.DataFrame,
    tuple_df: pd.DataFrame,
    cand_overall: pd.DataFrame,
    cand_family: pd.DataFrame,
    cand_tuple: pd.DataFrame,
    min_np_dd: float,
    overall_min_trades: int,
    family_min_trades: int,
    tuple_min_trades: int,
) -> None:
    lines = []
    lines.append("Regime-Aware Strategy Discovery Candidate Summary")
    lines.append("=" * 60)
    lines.append(f"Created: {now_str()}")
    lines.append("")
    lines.append("Input files:")
    lines.append(f"  Overall:       {overall_file}")
    lines.append(f"  Regime family: {family_file}")
    lines.append(f"  Regime tuple:  {tuple_file}")
    lines.append("")
    lines.append("Thresholds:")
    lines.append(f"  Min NP/DD:              {min_np_dd}")
    lines.append(f"  Overall min trades:     {overall_min_trades}")
    lines.append(f"  Regime-family min trades: {family_min_trades}")
    lines.append(f"  Regime-tuple min trades:  {tuple_min_trades}")
    lines.append("")
    lines.append("Raw rows:")
    lines.append(f"  Overall rows:       {len(overall):,}")
    lines.append(f"  Regime-family rows: {len(family):,}")
    lines.append(f"  Regime-tuple rows:  {len(tuple_df):,}")
    lines.append("")
    lines.append("Candidate rows:")
    lines.append(f"  Overall candidates:       {len(cand_overall):,}")
    lines.append(f"  Regime-family candidates: {len(cand_family):,}")
    lines.append(f"  Regime-tuple candidates:  {len(cand_tuple):,}")
    lines.append("")

    def add_top_block(title: str, df: pd.DataFrame, n: int = 10) -> None:
        lines.append(title)
        lines.append("-" * len(title))
        if df.empty:
            lines.append("  None")
            lines.append("")
            return
        cols = [c for c in ["strategy_id", "scope", "group_value", "trade_count", "net_profit", "max_drawdown", "np_dd", "profit_factor"] if c in df.columns]
        for _, r in df.head(n).iterrows():
            parts = [f"{c}={r[c]}" for c in cols]
            lines.append("  " + " | ".join(parts))
        lines.append("")

    add_top_block("Top Overall Candidates", cand_overall)
    add_top_block("Top Regime-Family Candidates", cand_family)
    add_top_block("Top Regime-Tuple Candidates", cand_tuple)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def analyze_results(
    min_np_dd: float,
    overall_min_trades: int,
    family_min_trades: int,
    tuple_min_trades: int,
    overall_file: Optional[Path] = None,
    family_file: Optional[Path] = None,
    tuple_file: Optional[Path] = None,
) -> None:
    overall_file = overall_file or latest_file(OVERALL_DIR, "performance_overall_*.csv")
    family_file = family_file or latest_file(FAMILY_DIR, "performance_regime_family_*.csv")
    tuple_file = tuple_file or latest_file(TUPLE_DIR, "performance_regime_tuple_*.csv")

    print(f"Reading overall:       {overall_file}")
    print(f"Reading regime family: {family_file}")
    print(f"Reading regime tuple:  {tuple_file}")

    overall = read_perf(overall_file)
    family = read_perf(family_file)
    tuple_df = read_perf(tuple_file)

    cand_overall = filter_candidates(overall, min_np_dd=min_np_dd, min_trades=overall_min_trades)
    cand_family = filter_candidates(family, min_np_dd=min_np_dd, min_trades=family_min_trades)
    cand_tuple = filter_candidates(tuple_df, min_np_dd=min_np_dd, min_trades=tuple_min_trades)

    cand_overall = add_candidate_metadata(cand_overall, "overall", overall_min_trades, min_np_dd)
    cand_family = add_candidate_metadata(cand_family, "regime_family", family_min_trades, min_np_dd)
    cand_tuple = add_candidate_metadata(cand_tuple, "regime_tuple", tuple_min_trades, min_np_dd)

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    cand_overall.to_csv(OVERALL_OUT, index=False)
    cand_family.to_csv(FAMILY_OUT, index=False)
    cand_tuple.to_csv(TUPLE_OUT, index=False)

    all_candidates = pd.concat([cand_overall, cand_family, cand_tuple], ignore_index=True)
    all_candidates = all_candidates.sort_values(
        by=["np_dd", "net_profit", "trade_count"],
        ascending=[False, False, False],
    )
    all_candidates.to_csv(ALL_OUT, index=False)

    update_registry_with_overall(REGISTRY_PATH, overall)

    write_summary(
        SUMMARY_OUT,
        overall_file,
        family_file,
        tuple_file,
        overall,
        family,
        tuple_df,
        cand_overall,
        cand_family,
        cand_tuple,
        min_np_dd,
        overall_min_trades,
        family_min_trades,
        tuple_min_trades,
    )

    print("Analysis complete.")
    print(f"Overall candidates:       {len(cand_overall):,} -> {OVERALL_OUT}")
    print(f"Regime-family candidates: {len(cand_family):,} -> {FAMILY_OUT}")
    print(f"Regime-tuple candidates:  {len(cand_tuple):,} -> {TUPLE_OUT}")
    print(f"All candidates:           {len(all_candidates):,} -> {ALL_OUT}")
    print(f"Summary:                  {SUMMARY_OUT}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze latest Rust backtest results and write top candidates.")
    parser.add_argument("--min-np-dd", type=float, default=2.0)
    parser.add_argument("--overall-min-trades", type=int, default=100)
    parser.add_argument("--family-min-trades", type=int, default=25)
    parser.add_argument("--tuple-min-trades", type=int, default=10)
    parser.add_argument("--overall-file", type=Path, default=None)
    parser.add_argument("--family-file", type=Path, default=None)
    parser.add_argument("--tuple-file", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze_results(
        min_np_dd=args.min_np_dd,
        overall_min_trades=args.overall_min_trades,
        family_min_trades=args.family_min_trades,
        tuple_min_trades=args.tuple_min_trades,
        overall_file=args.overall_file,
        family_file=args.family_file,
        tuple_file=args.tuple_file,
    )

# cd "C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery\tools"
# python analyze_rust_results.py