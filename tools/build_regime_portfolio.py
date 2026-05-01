# ============================================================
# Build Regime Portfolio
# Regime-Aware Strategy Discovery Pipeline
# ============================================================
#
# Recommended save location:
# C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery\tools\build_regime_portfolio.py
#
# Purpose:
#   Convert candidate strategy results into a regime-aware portfolio map.
#
# Inputs:
#   06_candidates\top_strategy_candidates_overall.csv
#   06_candidates\top_strategy_candidates_regime_family.csv
#   06_candidates\top_strategy_candidates_regime_tuple.csv
#   06_candidates\top_strategy_candidates_all.csv
#   03_strategy_registry\strategy_combinations.csv
#
# Outputs:
#   07_portfolio\regime_family_portfolio.csv
#   07_portfolio\regime_tuple_portfolio.csv
#   07_portfolio\strategy_coverage_summary.csv
#   07_portfolio\regime_portfolio_summary.txt
#
# Run:
#   python build_regime_portfolio.py
#
# Optional:
#   python build_regime_portfolio.py --top-n-per-regime 3 --min-np-dd 2 --min-trades-family 25 --min-trades-tuple 10
# ============================================================

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery")
CANDIDATE_DIR = PROJECT_ROOT / "06_candidates"
PORTFOLIO_DIR = PROJECT_ROOT / "07_portfolio"
REGISTRY_PATH = PROJECT_ROOT / "03_strategy_registry" / "strategy_combinations.csv"

OVERALL_CANDIDATES = CANDIDATE_DIR / "top_strategy_candidates_overall.csv"
FAMILY_CANDIDATES = CANDIDATE_DIR / "top_strategy_candidates_regime_family.csv"
TUPLE_CANDIDATES = CANDIDATE_DIR / "top_strategy_candidates_regime_tuple.csv"
ALL_CANDIDATES = CANDIDATE_DIR / "top_strategy_candidates_all.csv"

FAMILY_PORTFOLIO_OUT = PORTFOLIO_DIR / "regime_family_portfolio.csv"
TUPLE_PORTFOLIO_OUT = PORTFOLIO_DIR / "regime_tuple_portfolio.csv"
STRATEGY_COVERAGE_OUT = PORTFOLIO_DIR / "strategy_coverage_summary.csv"
SUMMARY_OUT = PORTFOLIO_DIR / "regime_portfolio_summary.txt"


REGIME_FAMILIES = [
    "Bull_Continuation",
    "Bull_Pullback",
    "Bull_Recovery",
    "Bull_Stress",
    "Bear_Continuation",
    "Bear_Bounce",
    "Bear_Reversal_Risk",
    "Bear_Stress",
    "Neutral_Balanced",
    "Neutral_Up_Bias",
    "Neutral_Down_Bias",
    "Mixed_Transition",
]

NUMERIC_COLS = [
    "candidate_min_trades",
    "candidate_min_np_dd",
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


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_registry(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str).fillna("")


def rank_score(df: pd.DataFrame) -> pd.Series:
    """
    Composite score for ranking candidates within a regime.

    Intent:
      - NP/DD is primary
      - Net profit matters
      - Trade count matters, but we use log-like scaling so giant trade counts
        do not dominate
      - Profit factor helps break ties
    """
    np_dd = df["np_dd"].fillna(0).clip(lower=0)
    net_profit = df["net_profit"].fillna(0).clip(lower=0)
    trade_count = df["trade_count"].fillna(0).clip(lower=0)
    profit_factor = df["profit_factor"].fillna(0).clip(lower=0, upper=10)

    # Normalize using ranks to avoid scale problems.
    np_dd_rank = np_dd.rank(pct=True)
    profit_rank = net_profit.rank(pct=True)
    trade_rank = trade_count.rank(pct=True)
    pf_rank = profit_factor.rank(pct=True)

    return (
        0.45 * np_dd_rank
        + 0.25 * profit_rank
        + 0.20 * trade_rank
        + 0.10 * pf_rank
    )


def extract_config_fields(registry: pd.DataFrame) -> pd.DataFrame:
    """
    Parse config_json from strategy registry into readable columns.
    """
    if registry.empty or "config_json" not in registry.columns:
        return pd.DataFrame(columns=["strategy_id"])

    rows = []
    for _, r in registry.iterrows():
        sid = r.get("strategy_id", "")
        raw = r.get("config_json", "")
        try:
            cfg = json.loads(raw)
        except Exception:
            rows.append({"strategy_id": sid})
            continue

        entry = cfg.get("entry_rule", {})
        exit_rule = cfg.get("exit_rule", {})
        filters = cfg.get("filters", [])

        rows.append(
            {
                "strategy_id": sid,
                "entry_type": entry.get("template", entry.get("type", "")),
                "entry_side": entry.get("side", ""),
                "entry_left": entry.get("left", ""),
                "entry_right": entry.get("right", ""),
                "exit_type": exit_rule.get("type", ""),
                "exit_max_bars": exit_rule.get("max_bars_in_trade", ""),
                "exit_atr_column": exit_rule.get("atr_column", ""),
                "exit_stop_atr_multiple": exit_rule.get("stop_atr_multiple", ""),
                "exit_reward_risk": exit_rule.get("reward_risk", ""),
                "filter_count": len(filters),
                "filters_json": json.dumps(filters, sort_keys=True),
            }
        )

    return pd.DataFrame(rows)


def attach_strategy_details(candidates: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    details = extract_config_fields(registry)
    if details.empty:
        return candidates
    return candidates.merge(details, on="strategy_id", how="left")


# ------------------------------------------------------------
# Portfolio builders
# ------------------------------------------------------------

def build_group_portfolio(
    candidates: pd.DataFrame,
    group_scope: str,
    top_n_per_regime: int,
    min_np_dd: float,
    min_trades: int,
    min_profit_factor: float,
    require_positive_net: bool = True,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()

    df = candidates.copy()
    df = df[df["scope"].astype(str) == group_scope].copy()

    if df.empty:
        return df

    mask = df["np_dd"].fillna(-999) >= min_np_dd
    mask &= df["trade_count"].fillna(0) >= min_trades
    mask &= df["profit_factor"].fillna(0) >= min_profit_factor
    if require_positive_net:
        mask &= df["net_profit"].fillna(0) > 0

    df = df.loc[mask].copy()
    if df.empty:
        return df

    df["portfolio_rank_score"] = rank_score(df)

    df = df.sort_values(
        by=["group_value", "portfolio_rank_score", "np_dd", "net_profit", "trade_count"],
        ascending=[True, False, False, False, False],
    )

    df["regime_rank"] = df.groupby("group_value").cumcount() + 1
    df = df[df["regime_rank"] <= top_n_per_regime].copy()

    # More user-friendly naming.
    if group_scope == "regime_family":
        df = df.rename(columns={"group_value": "regime_family"})
    elif group_scope == "regime_tuple":
        df = df.rename(columns={"group_value": "regime_tuple"})

    df.insert(0, "portfolio_built_at", now_str())
    return df


def build_strategy_coverage(family_portfolio: pd.DataFrame, tuple_portfolio: pd.DataFrame) -> pd.DataFrame:
    pieces = []

    if not family_portfolio.empty:
        tmp = family_portfolio.copy()
        tmp["coverage_type"] = "regime_family"
        tmp["covered_regime"] = tmp["regime_family"]
        pieces.append(tmp)

    if not tuple_portfolio.empty:
        tmp = tuple_portfolio.copy()
        tmp["coverage_type"] = "regime_tuple"
        tmp["covered_regime"] = tmp["regime_tuple"]
        pieces.append(tmp)

    if not pieces:
        return pd.DataFrame()

    all_rows = pd.concat(pieces, ignore_index=True)

    summary = (
        all_rows.groupby("strategy_id")
        .agg(
            covered_regime_count=("covered_regime", "nunique"),
            family_coverage_count=("coverage_type", lambda s: int((s == "regime_family").sum())),
            tuple_coverage_count=("coverage_type", lambda s: int((s == "regime_tuple").sum())),
            total_candidate_rows=("strategy_id", "size"),
            best_np_dd=("np_dd", "max"),
            median_np_dd=("np_dd", "median"),
            total_net_profit=("net_profit", "sum"),
            total_trades=("trade_count", "sum"),
            avg_profit_factor=("profit_factor", "mean"),
        )
        .reset_index()
    )

    summary["coverage_score"] = (
        0.40 * summary["covered_regime_count"].rank(pct=True)
        + 0.25 * summary["best_np_dd"].rank(pct=True)
        + 0.20 * summary["total_net_profit"].rank(pct=True)
        + 0.15 * summary["total_trades"].rank(pct=True)
    )

    summary = summary.sort_values(
        by=["coverage_score", "covered_regime_count", "best_np_dd", "total_net_profit"],
        ascending=[False, False, False, False],
    )
    return summary


def write_summary(
    family_portfolio: pd.DataFrame,
    tuple_portfolio: pd.DataFrame,
    coverage: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines = []
    lines.append("Regime-Aware Portfolio Summary")
    lines.append("=" * 60)
    lines.append(f"Created: {now_str()}")
    lines.append("")
    lines.append("Settings:")
    lines.append(f"  top_n_per_regime: {args.top_n_per_regime}")
    lines.append(f"  min_np_dd:        {args.min_np_dd}")
    lines.append(f"  min_trades_family:{args.min_trades_family}")
    lines.append(f"  min_trades_tuple: {args.min_trades_tuple}")
    lines.append(f"  min_profit_factor:{args.min_profit_factor}")
    lines.append("")
    lines.append("Portfolio sizes:")
    lines.append(f"  Regime-family rows: {len(family_portfolio):,}")
    lines.append(f"  Regime-tuple rows:  {len(tuple_portfolio):,}")
    lines.append(f"  Strategy coverage rows: {len(coverage):,}")
    lines.append("")

    if not family_portfolio.empty and "regime_family" in family_portfolio.columns:
        covered_families = sorted(family_portfolio["regime_family"].dropna().unique().tolist())
        missing_families = [f for f in REGIME_FAMILIES if f not in covered_families]
        lines.append("Regime-family coverage:")
        lines.append(f"  Covered: {len(covered_families)} / {len(REGIME_FAMILIES)}")
        lines.append(f"  Missing: {missing_families if missing_families else 'None'}")
        lines.append("")

    def add_top(title: str, df: pd.DataFrame, group_col: str, n: int = 10) -> None:
        lines.append(title)
        lines.append("-" * len(title))
        if df.empty:
            lines.append("  None")
            lines.append("")
            return
        cols = [group_col, "regime_rank", "strategy_id", "trade_count", "net_profit", "max_drawdown", "np_dd", "profit_factor", "entry_type", "entry_left", "entry_right", "exit_type"]
        cols = [c for c in cols if c in df.columns]
        for _, r in df.head(n).iterrows():
            lines.append("  " + " | ".join(f"{c}={r[c]}" for c in cols))
        lines.append("")

    add_top("Top Family Portfolio Rows", family_portfolio, "regime_family")
    add_top("Top Tuple Portfolio Rows", tuple_portfolio, "regime_tuple")

    lines.append("Top Strategy Coverage")
    lines.append("-" * len("Top Strategy Coverage"))
    if coverage.empty:
        lines.append("  None")
    else:
        cols = ["strategy_id", "coverage_score", "covered_regime_count", "family_coverage_count", "tuple_coverage_count", "best_np_dd", "total_net_profit", "total_trades"]
        for _, r in coverage.head(15).iterrows():
            lines.append("  " + " | ".join(f"{c}={r[c]}" for c in cols if c in coverage.columns))

    SUMMARY_OUT.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def build_portfolio(args: argparse.Namespace) -> None:
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)

    family_candidates = read_csv_if_exists(FAMILY_CANDIDATES)
    tuple_candidates = read_csv_if_exists(TUPLE_CANDIDATES)
    registry = read_registry(REGISTRY_PATH)

    family_candidates = attach_strategy_details(family_candidates, registry)
    tuple_candidates = attach_strategy_details(tuple_candidates, registry)

    family_portfolio = build_group_portfolio(
        family_candidates,
        group_scope="regime_family",
        top_n_per_regime=args.top_n_per_regime,
        min_np_dd=args.min_np_dd,
        min_trades=args.min_trades_family,
        min_profit_factor=args.min_profit_factor,
    )

    tuple_portfolio = build_group_portfolio(
        tuple_candidates,
        group_scope="regime_tuple",
        top_n_per_regime=args.top_n_per_regime,
        min_np_dd=args.min_np_dd,
        min_trades=args.min_trades_tuple,
        min_profit_factor=args.min_profit_factor,
    )

    coverage = build_strategy_coverage(family_portfolio, tuple_portfolio)

    family_portfolio.to_csv(FAMILY_PORTFOLIO_OUT, index=False)
    tuple_portfolio.to_csv(TUPLE_PORTFOLIO_OUT, index=False)
    coverage.to_csv(STRATEGY_COVERAGE_OUT, index=False)
    write_summary(family_portfolio, tuple_portfolio, coverage, args)

    print("Regime portfolio build complete.")
    print(f"Family portfolio rows: {len(family_portfolio):,} -> {FAMILY_PORTFOLIO_OUT}")
    print(f"Tuple portfolio rows:  {len(tuple_portfolio):,} -> {TUPLE_PORTFOLIO_OUT}")
    print(f"Coverage rows:         {len(coverage):,} -> {STRATEGY_COVERAGE_OUT}")
    print(f"Summary:               {SUMMARY_OUT}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build regime-aware strategy portfolio from candidate files.")
    parser.add_argument("--top-n-per-regime", type=int, default=3)
    parser.add_argument("--min-np-dd", type=float, default=2.0)
    parser.add_argument("--min-trades-family", type=int, default=25)
    parser.add_argument("--min-trades-tuple", type=int, default=10)
    parser.add_argument("--min-profit-factor", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    build_portfolio(parse_args())

# cd "C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery\tools"
# python build_regime_portfolio.py