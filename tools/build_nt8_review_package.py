# ============================================================
# Build NT8 Review Package
# Regime-Aware Strategy Discovery Pipeline
# ============================================================
#
# Recommended save location:
# C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery\tools\build_nt8_review_package.py
#
# Purpose:
#   Package generated NinjaTrader strategy files for manual review/testing.
#
# Inputs:
#   09_nt8_deployment\strategy_specs\*.json
#   09_nt8_deployment\generated_strategies\*.cs
#   09_nt8_deployment\generated_strategy_index.csv
#   09_nt8_deployment\nt8_strategy_spec_index.csv
#
# Outputs:
#   09_nt8_deployment\review_packages\<strategy_class>\
#       strategy.cs
#       spec.json
#       review_summary.txt
#
# Run:
#   python build_nt8_review_package.py --max-packages 10
#   python build_nt8_review_package.py --class-name RA_strat_...
# ============================================================

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery")
DEPLOY_ROOT = PROJECT_ROOT / "09_nt8_deployment"
SPEC_DIR = DEPLOY_ROOT / "strategy_specs"
GENERATED_DIR = DEPLOY_ROOT / "generated_strategies"
REVIEW_DIR = DEPLOY_ROOT / "review_packages"
GENERATED_INDEX = DEPLOY_ROOT / "generated_strategy_index.csv"
SPEC_INDEX = DEPLOY_ROOT / "nt8_strategy_spec_index.csv"


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str).fillna("")


def find_spec_for_class(class_name: str) -> Path | None:
    direct = SPEC_DIR / f"{class_name}.json"
    if direct.exists():
        return direct

    idx = read_csv_if_exists(SPEC_INDEX)
    if not idx.empty and "nt8_class_name" in idx.columns:
        rows = idx[idx["nt8_class_name"] == class_name]
        if not rows.empty:
            p = Path(rows.iloc[0].get("spec_path", ""))
            if p.exists():
                return p

    matches = list(SPEC_DIR.glob(f"*{class_name}*.json"))
    return matches[0] if matches else None


def find_cs_for_class(class_name: str) -> Path | None:
    direct = GENERATED_DIR / f"{class_name}.cs"
    if direct.exists():
        return direct

    idx = read_csv_if_exists(GENERATED_INDEX)
    if not idx.empty and "nt8_class_name" in idx.columns:
        rows = idx[idx["nt8_class_name"] == class_name]
        if not rows.empty:
            p = Path(rows.iloc[0].get("cs_path", ""))
            if p.exists():
                return p

    matches = list(GENERATED_DIR.glob(f"*{class_name}*.cs"))
    return matches[0] if matches else None


def summarize_rule(rule: dict) -> str:
    if not rule:
        return "None"
    rtype = rule.get("template", rule.get("type", "unknown"))
    left = rule.get("left", "")
    right = rule.get("right", "")
    side = rule.get("side", "")
    parts = [f"type={rtype}"]
    if side:
        parts.append(f"side={side}")
    if left or right:
        parts.append(f"left={left}")
        parts.append(f"right={right}")
    return " | ".join(parts)


def summarize_filters(filters: list[dict]) -> str:
    if not filters:
        return "None"
    lines = []
    for i, f in enumerate(filters, start=1):
        ftype = f.get("type", "unknown")
        if ftype == "threshold":
            lines.append(f"{i}. {f.get('column')} {f.get('operator')} {f.get('value')}")
        elif ftype == "comparison":
            lines.append(f"{i}. {f.get('left')} {f.get('operator')} {f.get('right')}")
        elif ftype == "regime_family_in":
            lines.append(f"{i}. regime_family in {f.get('values')}")
        else:
            lines.append(f"{i}. {json.dumps(f, sort_keys=True)}")
    return "\n".join(lines)


def summarize_exit(rule: dict) -> str:
    if not rule:
        return "None"
    rtype = rule.get("type", "unknown")
    if rtype == "time_stop":
        return f"time_stop | max_bars_in_trade={rule.get('max_bars_in_trade')}"
    if rtype == "opposite_cross":
        return f"opposite_cross | left={rule.get('left')} | right={rule.get('right')} | max_bars_in_trade={rule.get('max_bars_in_trade')}"
    if rtype == "fixed_rr_atr_stop":
        return (
            f"fixed_rr_atr_stop | atr_column={rule.get('atr_column')} | "
            f"stop_atr_multiple={rule.get('stop_atr_multiple')} | "
            f"reward_risk={rule.get('reward_risk')} | "
            f"max_bars_in_trade={rule.get('max_bars_in_trade')}"
        )
    return json.dumps(rule, sort_keys=True)


def build_review_summary(spec: dict, cs_path: Path, spec_path: Path) -> str:
    deployment = spec.get("deployment", {})
    regime = spec.get("regime_filter", {})
    perf = spec.get("performance", {})
    alloc = spec.get("allocation", {})
    costs = spec.get("costs", {})

    lines = []
    lines.append("NT8 Strategy Review Package")
    lines.append("=" * 70)
    lines.append(f"Created: {now_str()}")
    lines.append("")

    lines.append("Files")
    lines.append("-" * 70)
    lines.append(f"Strategy .cs: {cs_path}")
    lines.append(f"Spec JSON:    {spec_path}")
    lines.append("")

    lines.append("Identity")
    lines.append("-" * 70)
    lines.append(f"Strategy ID:     {spec.get('strategy_id')}")
    lines.append(f"Parameter Hash:  {spec.get('parameter_hash')}")
    lines.append(f"NT8 Class Name:  {spec.get('nt8_class_name')}")
    lines.append("")

    lines.append("Deployment Recommendation")
    lines.append("-" * 70)
    lines.append(f"Enabled:         {deployment.get('enabled')}")
    lines.append(f"Instrument Type: {deployment.get('instrument_type')}")
    lines.append(f"Quantity:        {deployment.get('quantity')}")
    lines.append(f"Scope:           {deployment.get('allocation_scope')}")
    lines.append(f"Active Regime:   {deployment.get('active_regime_key')}")
    lines.append("")

    lines.append("Regime Filter")
    lines.append("-" * 70)
    lines.append(f"Scope:           {regime.get('scope')}")
    lines.append(f"Key:             {regime.get('key')}")
    lines.append(f"Entry-time only: {regime.get('entry_time_only')}")
    lines.append(f"NT8 Indicator:   {regime.get('nt8_indicator')}")
    lines.append("")

    lines.append("Strategy Logic")
    lines.append("-" * 70)
    lines.append("Entry:")
    lines.append(f"  {summarize_rule(spec.get('entry_rule', {}))}")
    lines.append("")
    lines.append("Filters:")
    lines.append(summarize_filters(spec.get("filters", [])))
    lines.append("")
    lines.append("Exit:")
    lines.append(f"  {summarize_exit(spec.get('exit_rule', {}))}")
    lines.append("")

    lines.append("Research Performance")
    lines.append("-" * 70)
    lines.append(f"Trade Count:     {perf.get('trade_count')}")
    lines.append(f"Net Profit:      {perf.get('net_profit')}")
    lines.append(f"Max Drawdown:    {perf.get('max_drawdown')}")
    lines.append(f"NP/DD:           {perf.get('np_dd')}")
    lines.append(f"Profit Factor:   {perf.get('profit_factor')}")
    lines.append(f"Win Rate:        {perf.get('win_rate')}")
    lines.append(f"Avg Trade:       {perf.get('avg_trade')}")
    lines.append("")

    lines.append("Allocation")
    lines.append("-" * 70)
    lines.append(f"Portfolio Weight:       {alloc.get('portfolio_weight')}")
    lines.append(f"Allocation Score:       {alloc.get('allocation_score')}")
    lines.append(f"Account Size:           {alloc.get('account_size')}")
    lines.append(f"Risk Fraction:          {alloc.get('risk_fraction')}")
    lines.append(f"Allowed Risk:           {alloc.get('allowed_risk')}")
    lines.append(f"Adjusted Drawdown:      {alloc.get('adjusted_drawdown')}")
    lines.append(f"Suggested NQ Contracts: {alloc.get('suggested_nq_contracts')}")
    lines.append(f"Suggested MNQ Contracts:{alloc.get('suggested_mnq_contracts')}")
    lines.append("")

    lines.append("Cost Assumptions Used in Research")
    lines.append("-" * 70)
    lines.append(f"Commission RT:   {costs.get('commission_round_trip')}")
    lines.append(f"Slippage Ticks:  {costs.get('slippage_ticks')}")
    lines.append(f"Tick Size:       {costs.get('tick_size')}")
    lines.append(f"Point Value:     {costs.get('point_value')}")
    lines.append("")

    lines.append("Manual NT8 Validation Checklist")
    lines.append("-" * 70)
    lines.append("[ ] Copy strategy.cs into NinjaTrader Strategies folder")
    lines.append("[ ] Compile successfully")
    lines.append("[ ] Confirm instrument matches recommendation")
    lines.append("[ ] Confirm Quantity matches recommendation")
    lines.append("[ ] Confirm Slippage = 3 ticks in strategy defaults")
    lines.append("[ ] Run Strategy Analyzer over same historical range")
    lines.append("[ ] Compare trade count directionally vs research result")
    lines.append("[ ] Compare net profit / max drawdown directionally vs research result")
    lines.append("[ ] If acceptable, mark as approved manually")
    lines.append("")

    lines.append("Notes")
    lines.append("-" * 70)
    for note in spec.get("notes", []):
        lines.append(f"- {note}")

    return "\n".join(lines)


def build_package_for_class(class_name: str, overwrite: bool = True) -> Path:
    spec_path = find_spec_for_class(class_name)
    cs_path = find_cs_for_class(class_name)

    if spec_path is None:
        raise FileNotFoundError(f"Spec JSON not found for class: {class_name}")
    if cs_path is None:
        raise FileNotFoundError(f"Generated .cs not found for class: {class_name}")

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    package_dir = REVIEW_DIR / class_name

    if package_dir.exists() and overwrite:
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(cs_path, package_dir / "strategy.cs")
    shutil.copy2(spec_path, package_dir / "spec.json")

    summary = build_review_summary(spec, package_dir / "strategy.cs", package_dir / "spec.json")
    (package_dir / "review_summary.txt").write_text(summary, encoding="utf-8")

    return package_dir


def get_generated_classes(max_packages: int, class_name: str | None) -> list[str]:
    if class_name:
        return [class_name]

    idx = read_csv_if_exists(GENERATED_INDEX)
    if not idx.empty and "nt8_class_name" in idx.columns:
        classes = idx["nt8_class_name"].tolist()
    else:
        classes = [p.stem for p in sorted(GENERATED_DIR.glob("*.cs"), key=lambda p: p.stat().st_mtime, reverse=True)]

    if max_packages > 0:
        classes = classes[:max_packages]
    return classes


def build_packages(args: argparse.Namespace) -> None:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    classes = get_generated_classes(args.max_packages, args.class_name)

    rows = []
    for cls in classes:
        package_dir = build_package_for_class(cls, overwrite=not args.no_overwrite)
        rows.append({"nt8_class_name": cls, "review_package_path": str(package_dir)})
        print(f"Built review package: {package_dir}")

    index_path = REVIEW_DIR / "review_package_index.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False)

    print("Review package build complete.")
    print(f"Packages: {len(rows):,}")
    print(f"Index:    {index_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NT8 manual review packages from generated strategies.")
    parser.add_argument("--max-packages", type=int, default=10, help="Max packages to build. Use 0 for all.")
    parser.add_argument("--class-name", type=str, default=None, help="Specific NT8 class name to package")
    parser.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing package folders")
    return parser.parse_args()


if __name__ == "__main__":
    build_packages(parse_args())

# cd "C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery\tools"
# python build_nt8_review_package.py --max-packages 10