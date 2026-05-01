# ============================================================
# Score Portfolio Allocation
# Regime-Aware Strategy Discovery Pipeline
# ============================================================
#
# Recommended save location:
# C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery\tools\score_portfolio_allocation.py
#
# Purpose:
#   Convert regime-aware portfolio rows into allocation recommendations.
#
# Inputs:
#   07_portfolio\regime_family_portfolio.csv
#   07_portfolio\regime_tuple_portfolio.csv
#   07_portfolio\strategy_coverage_summary.csv
#
# Outputs:
#   08_allocation\strategy_allocation_all.csv
#   08_allocation\strategy_allocation_by_tuple.csv
#   08_allocation\strategy_allocation_by_family.csv
#   08_allocation\strategy_allocation_summary.txt
#
# Position sizing formula:
#   allowed_risk = account_size * risk_fraction
#   adjusted_dd = max_drawdown * drawdown_safety_factor
#   contracts = floor(allowed_risk / adjusted_dd)
#
# Default:
#   account_size = 10,000
#   risk_fraction = 0.50
#   drawdown_safety_factor = 2.0
#
# Run:
#   python score_portfolio_allocation.py
#
# Example:
#   python score_portfolio_allocation.py --account-size 25000 --risk-fraction 0.5 --max-rows-per-regime 2
# ============================================================

from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery")
PORTFOLIO_DIR = PROJECT_ROOT / "07_portfolio"
ALLOCATION_DIR = PROJECT_ROOT / "08_allocation"

FAMILY_PORTFOLIO = PORTFOLIO_DIR / "regime_family_portfolio.csv"
TUPLE_PORTFOLIO = PORTFOLIO_DIR / "regime_tuple_portfolio.csv"
COVERAGE_SUMMARY = PORTFOLIO_DIR / "strategy_coverage_summary.csv"

ALLOC_ALL_OUT = ALLOCATION_DIR / "strategy_allocation_all.csv"
ALLOC_FAMILY_OUT = ALLOCATION_DIR / "strategy_allocation_by_family.csv"
ALLOC_TUPLE_OUT = ALLOCATION_DIR / "strategy_allocation_by_tuple.csv"
SUMMARY_OUT = ALLOCATION_DIR / "strategy_allocation_summary.txt"


NUMERIC_COLS = [
    "trade_count",
    "gross_profit",
    "gross_loss",
    "net_profit",
    "max_drawdown",
    "np_dd",
    "win_rate",
    "avg_trade",
    "profit_factor",
    "portfolio_rank_score",
    "regime_rank",
    "coverage_score",
    "covered_regime_count",
    "family_coverage_count",
    "tuple_coverage_count",
    "best_np_dd",
    "median_np_dd",
    "total_net_profit",
    "total_trades",
    "avg_profit_factor",
]


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def safe_rank_pct(s: pd.Series, ascending: bool = True) -> pd.Series:
    if s.empty:
        return s
    return s.rank(pct=True, ascending=ascending).fillna(0.0)


def prepare_portfolio_rows(family: pd.DataFrame, tuple_df: pd.DataFrame) -> pd.DataFrame:
    pieces = []

    if not family.empty:
        f = family.copy()
        f["allocation_scope"] = "regime_family"
        f["active_regime_key"] = f.get("regime_family", "")
        pieces.append(f)

    if not tuple_df.empty:
        t = tuple_df.copy()
        t["allocation_scope"] = "regime_tuple"
        t["active_regime_key"] = t.get("regime_tuple", "")
        pieces.append(t)

    if not pieces:
        return pd.DataFrame()

    out = pd.concat(pieces, ignore_index=True, sort=False)
    return out


def attach_coverage(df: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    if df.empty or coverage.empty or "strategy_id" not in coverage.columns:
        df["coverage_score"] = 0.0
        df["covered_regime_count"] = 0
        return df

    keep = [
        "strategy_id",
        "coverage_score",
        "covered_regime_count",
        "family_coverage_count",
        "tuple_coverage_count",
        "best_np_dd",
        "median_np_dd",
        "total_net_profit",
        "total_trades",
        "avg_profit_factor",
    ]
    keep = [c for c in keep if c in coverage.columns]
    out = df.merge(coverage[keep], on="strategy_id", how="left", suffixes=("", "_coverage"))

    for c in keep:
        if c != "strategy_id" and c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    return out


def add_allocation_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    # Primary row-level quality.
    out["np_dd_score"] = safe_rank_pct(out["np_dd"].fillna(0).clip(lower=0), ascending=True)
    out["net_profit_score"] = safe_rank_pct(out["net_profit"].fillna(0).clip(lower=0), ascending=True)
    out["trade_count_score"] = safe_rank_pct(out["trade_count"].fillna(0).clip(lower=0), ascending=True)
    out["profit_factor_score"] = safe_rank_pct(out["profit_factor"].fillna(0).clip(lower=0, upper=10), ascending=True)
    out["coverage_component_score"] = safe_rank_pct(out.get("coverage_score", pd.Series([0] * len(out))).fillna(0), ascending=True)

    # Penalize very small sample sizes even if NP/DD is high.
    out["sample_quality_multiplier"] = (out["trade_count"].fillna(0) / 50.0).clip(lower=0.25, upper=1.0)

    out["allocation_score_raw"] = (
        0.35 * out["np_dd_score"]
        + 0.25 * out["net_profit_score"]
        + 0.15 * out["trade_count_score"]
        + 0.15 * out["profit_factor_score"]
        + 0.10 * out["coverage_component_score"]
    )

    out["allocation_score"] = out["allocation_score_raw"] * out["sample_quality_multiplier"]

    # Normalize weights across all selected rows.
    score_sum = out["allocation_score"].sum()
    if score_sum > 0:
        out["portfolio_weight"] = out["allocation_score"] / score_sum
    else:
        out["portfolio_weight"] = 0.0

    return out


def add_position_sizing(
    df: pd.DataFrame,
    account_size: float,
    risk_fraction: float,
    drawdown_safety_factor: float,
    max_contracts_per_row: int,
    min_contracts_if_qualified: int,
    micro_contract_ratio: int,
) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    allowed_risk = account_size * risk_fraction

    out["account_size"] = account_size
    out["risk_fraction"] = risk_fraction
    out["allowed_risk"] = allowed_risk
    out["drawdown_safety_factor"] = drawdown_safety_factor
    out["adjusted_drawdown"] = out["max_drawdown"].fillna(0) * drawdown_safety_factor

    def calc_contracts(adjusted_dd: float) -> int:
        if adjusted_dd <= 0 or not math.isfinite(adjusted_dd):
            return 0
        contracts = math.floor(allowed_risk / adjusted_dd)
        contracts = min(contracts, max_contracts_per_row)
        if contracts <= 0 and min_contracts_if_qualified > 0:
            return min_contracts_if_qualified
        return max(0, contracts)

    out["nq_contracts_raw"] = out["adjusted_drawdown"].map(calc_contracts)

    # Micro equivalent. 1 NQ = 10 MNQ by notional/point value.
    out["mnq_contracts_equivalent"] = out["nq_contracts_raw"] * micro_contract_ratio

    # If NQ raw is 0 but row qualifies, compute micros directly from risk.
    def calc_micros(adjusted_dd: float) -> int:
        if adjusted_dd <= 0 or not math.isfinite(adjusted_dd):
            return 0
        micro_adjusted_dd = adjusted_dd / micro_contract_ratio
        return max(0, math.floor(allowed_risk / micro_adjusted_dd))

    out["mnq_contracts_direct"] = out["adjusted_drawdown"].map(calc_micros)

    # Suggested display fields.
    out["suggested_nq_contracts"] = out["nq_contracts_raw"]
    out["suggested_mnq_contracts"] = out.apply(
        lambda r: int(r["mnq_contracts_equivalent"]) if r["nq_contracts_raw"] > 0 else int(r["mnq_contracts_direct"]),
        axis=1,
    )

    out["estimated_allocated_risk"] = out["suggested_nq_contracts"] * out["adjusted_drawdown"]
    out["estimated_allocated_risk_mnq"] = out["suggested_mnq_contracts"] * (out["adjusted_drawdown"] / micro_contract_ratio)

    return out


def limit_rows_per_regime(df: pd.DataFrame, max_rows_per_regime: int) -> pd.DataFrame:
    if df.empty or max_rows_per_regime <= 0:
        return df

    out = df.sort_values(
        by=["allocation_scope", "active_regime_key", "allocation_score", "np_dd", "net_profit"],
        ascending=[True, True, False, False, False],
    ).copy()

    out["allocation_regime_rank"] = out.groupby(["allocation_scope", "active_regime_key"]).cumcount() + 1
    return out[out["allocation_regime_rank"] <= max_rows_per_regime].copy()


def write_summary(df: pd.DataFrame, family: pd.DataFrame, tuple_df: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = []
    lines.append("Strategy Allocation Summary")
    lines.append("=" * 60)
    lines.append(f"Created: {now_str()}")
    lines.append("")
    lines.append("Settings:")
    lines.append(f"  account_size:              {args.account_size}")
    lines.append(f"  risk_fraction:             {args.risk_fraction}")
    lines.append(f"  drawdown_safety_factor:    {args.drawdown_safety_factor}")
    lines.append(f"  max_rows_per_regime:       {args.max_rows_per_regime}")
    lines.append(f"  max_contracts_per_row:     {args.max_contracts_per_row}")
    lines.append(f"  min_contracts_if_qualified:{args.min_contracts_if_qualified}")
    lines.append("")
    lines.append("Allocation sizes:")
    lines.append(f"  All allocation rows:    {len(df):,}")
    lines.append(f"  Family allocation rows: {len(family):,}")
    lines.append(f"  Tuple allocation rows:  {len(tuple_df):,}")
    lines.append("")

    if not df.empty:
        lines.append("Risk summary:")
        lines.append(f"  Total suggested NQ contracts across rows:  {int(df['suggested_nq_contracts'].sum())}")
        lines.append(f"  Total suggested MNQ contracts across rows: {int(df['suggested_mnq_contracts'].sum())}")
        lines.append(f"  Sum estimated allocated NQ risk:           {df['estimated_allocated_risk'].sum():,.2f}")
        lines.append(f"  Sum estimated allocated MNQ risk:          {df['estimated_allocated_risk_mnq'].sum():,.2f}")
        lines.append("")

        lines.append("Top Allocation Rows")
        lines.append("-" * 60)
        cols = [
            "allocation_scope", "active_regime_key", "strategy_id", "portfolio_weight",
            "allocation_score", "trade_count", "net_profit", "max_drawdown", "np_dd",
            "profit_factor", "suggested_nq_contracts", "suggested_mnq_contracts",
            "entry_type", "entry_left", "entry_right", "exit_type",
        ]
        cols = [c for c in cols if c in df.columns]
        for _, r in df.sort_values("allocation_score", ascending=False).head(25).iterrows():
            lines.append("  " + " | ".join(f"{c}={r[c]}" for c in cols))
    else:
        lines.append("No allocation rows produced.")

    SUMMARY_OUT.write_text("\n".join(lines), encoding="utf-8")


def build_allocation(args: argparse.Namespace) -> None:
    ALLOCATION_DIR.mkdir(parents=True, exist_ok=True)

    family = read_csv(FAMILY_PORTFOLIO)
    tuple_df = read_csv(TUPLE_PORTFOLIO)
    coverage = read_csv(COVERAGE_SUMMARY)

    combined = prepare_portfolio_rows(family, tuple_df)
    combined = attach_coverage(combined, coverage)

    # Limit before scoring if desired, to prevent very dense tuple allocations.
    combined = add_allocation_scores(combined)
    combined = limit_rows_per_regime(combined, args.max_rows_per_regime)
    combined = add_allocation_scores(combined)  # Re-normalize weights after limiting.

    combined = add_position_sizing(
        combined,
        account_size=args.account_size,
        risk_fraction=args.risk_fraction,
        drawdown_safety_factor=args.drawdown_safety_factor,
        max_contracts_per_row=args.max_contracts_per_row,
        min_contracts_if_qualified=args.min_contracts_if_qualified,
        micro_contract_ratio=args.micro_contract_ratio,
    )

    combined = combined.sort_values(
        by=["allocation_score", "np_dd", "net_profit", "trade_count"],
        ascending=[False, False, False, False],
    )

    family_alloc = combined[combined["allocation_scope"] == "regime_family"].copy() if not combined.empty else pd.DataFrame()
    tuple_alloc = combined[combined["allocation_scope"] == "regime_tuple"].copy() if not combined.empty else pd.DataFrame()

    combined.to_csv(ALLOC_ALL_OUT, index=False)
    family_alloc.to_csv(ALLOC_FAMILY_OUT, index=False)
    tuple_alloc.to_csv(ALLOC_TUPLE_OUT, index=False)
    write_summary(combined, family_alloc, tuple_alloc, args)

    print("Portfolio allocation scoring complete.")
    print(f"All allocation rows:    {len(combined):,} -> {ALLOC_ALL_OUT}")
    print(f"Family allocation rows: {len(family_alloc):,} -> {ALLOC_FAMILY_OUT}")
    print(f"Tuple allocation rows:  {len(tuple_alloc):,} -> {ALLOC_TUPLE_OUT}")
    print(f"Summary:                {SUMMARY_OUT}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score regime-aware portfolio allocation.")
    parser.add_argument("--account-size", type=float, default=10_000.0)
    parser.add_argument("--risk-fraction", type=float, default=0.50)
    parser.add_argument("--drawdown-safety-factor", type=float, default=2.0)
    parser.add_argument("--max-rows-per-regime", type=int, default=2)
    parser.add_argument("--max-contracts-per-row", type=int, default=10)
    parser.add_argument("--min-contracts-if-qualified", type=int, default=0)
    parser.add_argument("--micro-contract-ratio", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    build_allocation(parse_args())

# cd "C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery\tools"
# python score_portfolio_allocation.py
# python score_portfolio_allocation.py --account-size 25000 --risk-fraction 0.5 --max-rows-per-regime 2