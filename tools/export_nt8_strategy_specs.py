# ============================================================
# Export NT8 Strategy Specs
# Regime-Aware Strategy Discovery Pipeline
# ============================================================
#
# Recommended save location:
# C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery\tools\export_nt8_strategy_specs.py
#
# Purpose:
#   Convert allocation winners into deployment-ready JSON specs that can later
#   be used to generate NinjaTrader 8 strategy code.
#
# Inputs:
#   08_allocation\strategy_allocation_all.csv
#   03_strategy_registry\strategy_combinations.csv
#
# Outputs:
#   09_nt8_deployment\strategy_specs\*.json
#   09_nt8_deployment\nt8_strategy_spec_index.csv
#
# Run:
#   python export_nt8_strategy_specs.py
#
# Example:
#   python export_nt8_strategy_specs.py --max-specs 25 --prefer-micro
# ============================================================

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery")
ALLOCATION_PATH = PROJECT_ROOT / "08_allocation" / "strategy_allocation_all.csv"
REGISTRY_PATH = PROJECT_ROOT / "03_strategy_registry" / "strategy_combinations.csv"
DEPLOY_ROOT = PROJECT_ROOT / "09_nt8_deployment"
SPEC_DIR = DEPLOY_ROOT / "strategy_specs"
SPEC_INDEX_PATH = DEPLOY_ROOT / "nt8_strategy_spec_index.csv"


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def nt8_safe_name(text: str) -> str:
    text = str(text)
    text = text.replace("-", "m")
    text = text.replace("+", "p")
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "Unknown"
    if text[0].isdigit():
        text = "S_" + text
    return text


def regime_suffix(row: pd.Series) -> str:
    scope = str(row.get("allocation_scope", ""))
    key = str(row.get("active_regime_key", ""))
    if scope == "regime_family":
        return nt8_safe_name(key)
    if scope == "regime_tuple":
        return "Tuple_" + nt8_safe_name(key)
    return "All"


def choose_contract(row: pd.Series, prefer_micro: bool) -> tuple[str, int]:
    nq = int(float(row.get("suggested_nq_contracts", 0) or 0))
    mnq = int(float(row.get("suggested_mnq_contracts", 0) or 0))

    if prefer_micro and mnq > 0:
        return "MNQ", mnq
    if nq > 0:
        return "NQ", nq
    if mnq > 0:
        return "MNQ", mnq
    return "NONE", 0


def load_registry_config_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Registry not found: {path}")

    df = pd.read_csv(path, dtype=str).fillna("")
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        sid = row.get("strategy_id", "")
        raw = row.get("config_json", "")
        if not sid or not raw:
            continue
        try:
            out[sid] = json.loads(raw)
        except Exception:
            continue
    return out


def build_spec(row: pd.Series, config: dict[str, Any], prefer_micro: bool) -> dict[str, Any]:
    contract_type, quantity = choose_contract(row, prefer_micro=prefer_micro)
    suffix = regime_suffix(row)
    strategy_id = str(row["strategy_id"])
    class_name = nt8_safe_name(f"RA_{strategy_id}_{suffix}_{contract_type}")

    spec = {
        "generated_at": now_str(),
        "strategy_id": strategy_id,
        "parameter_hash": str(row.get("parameter_hash", "")),
        "nt8_class_name": class_name,
        "nt8_namespace": "NinjaTrader.NinjaScript.Strategies",
        "deployment": {
            "instrument_type": contract_type,
            "quantity": quantity,
            "enabled": quantity > 0,
            "allocation_scope": str(row.get("allocation_scope", "")),
            "active_regime_key": str(row.get("active_regime_key", "")),
        },
        "regime_filter": {
            "scope": str(row.get("allocation_scope", "")),
            "key": str(row.get("active_regime_key", "")),
            "entry_time_only": True,
            "nt8_indicator": "RegimeClassifierRB1",
        },
        "strategy_config": config,
        "entry_rule": config.get("entry_rule", {}),
        "filters": config.get("filters", []),
        "exit_rule": config.get("exit_rule", {}),
        "costs": config.get("costs", {}),
        "performance": {
            "trade_count": float(row.get("trade_count", 0) or 0),
            "net_profit": float(row.get("net_profit", 0) or 0),
            "max_drawdown": float(row.get("max_drawdown", 0) or 0),
            "np_dd": float(row.get("np_dd", 0) or 0),
            "profit_factor": float(row.get("profit_factor", 0) or 0),
            "win_rate": float(row.get("win_rate", 0) or 0),
            "avg_trade": float(row.get("avg_trade", 0) or 0),
        },
        "allocation": {
            "portfolio_weight": float(row.get("portfolio_weight", 0) or 0),
            "allocation_score": float(row.get("allocation_score", 0) or 0),
            "account_size": float(row.get("account_size", 0) or 0),
            "risk_fraction": float(row.get("risk_fraction", 0) or 0),
            "allowed_risk": float(row.get("allowed_risk", 0) or 0),
            "adjusted_drawdown": float(row.get("adjusted_drawdown", 0) or 0),
            "suggested_nq_contracts": int(float(row.get("suggested_nq_contracts", 0) or 0)),
            "suggested_mnq_contracts": int(float(row.get("suggested_mnq_contracts", 0) or 0)),
        },
        "notes": [
            "Generated from research pipeline output.",
            "Validate in NT8 Strategy Analyzer before live/sim deployment.",
            "Regime filter should be checked at entry only.",
        ],
    }
    return spec


def export_specs(args: argparse.Namespace) -> None:
    if not ALLOCATION_PATH.exists():
        raise FileNotFoundError(f"Allocation file not found: {ALLOCATION_PATH}")

    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    DEPLOY_ROOT.mkdir(parents=True, exist_ok=True)

    allocation = pd.read_csv(ALLOCATION_PATH)
    config_map = load_registry_config_map(REGISTRY_PATH)

    # Sort best first.
    for col in ["allocation_score", "np_dd", "net_profit", "trade_count"]:
        if col in allocation.columns:
            allocation[col] = pd.to_numeric(allocation[col], errors="coerce")

    allocation = allocation.sort_values(
        by=["allocation_score", "np_dd", "net_profit", "trade_count"],
        ascending=[False, False, False, False],
    )

    if args.only_enabled:
        allocation = allocation[
            (pd.to_numeric(allocation.get("suggested_nq_contracts", 0), errors="coerce").fillna(0) > 0)
            | (pd.to_numeric(allocation.get("suggested_mnq_contracts", 0), errors="coerce").fillna(0) > 0)
        ].copy()

    if args.max_specs > 0:
        allocation = allocation.head(args.max_specs).copy()

    index_rows = []

    for _, row in allocation.iterrows():
        sid = str(row["strategy_id"])
        config = config_map.get(sid)
        if not config:
            print(f"WARNING: Missing config_json for {sid}; skipping")
            continue

        spec = build_spec(row, config, prefer_micro=args.prefer_micro)
        spec_path = SPEC_DIR / f"{spec['nt8_class_name']}.json"
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

        index_rows.append(
            {
                "strategy_id": sid,
                "nt8_class_name": spec["nt8_class_name"],
                "spec_path": str(spec_path),
                "instrument_type": spec["deployment"]["instrument_type"],
                "quantity": spec["deployment"]["quantity"],
                "enabled": spec["deployment"]["enabled"],
                "allocation_scope": spec["deployment"]["allocation_scope"],
                "active_regime_key": spec["deployment"]["active_regime_key"],
                "trade_count": spec["performance"]["trade_count"],
                "net_profit": spec["performance"]["net_profit"],
                "max_drawdown": spec["performance"]["max_drawdown"],
                "np_dd": spec["performance"]["np_dd"],
                "profit_factor": spec["performance"]["profit_factor"],
                "allocation_score": spec["allocation"]["allocation_score"],
            }
        )

    index = pd.DataFrame(index_rows)
    index.to_csv(SPEC_INDEX_PATH, index=False)

    print("NT8 strategy spec export complete.")
    print(f"Specs written: {len(index_rows):,}")
    print(f"Spec folder:   {SPEC_DIR}")
    print(f"Spec index:    {SPEC_INDEX_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export NT8 deployment strategy specs from allocation output.")
    parser.add_argument("--max-specs", type=int, default=25, help="Max specs to export. Use 0 for all.")
    parser.add_argument("--prefer-micro", action="store_true", help="Prefer MNQ sizing even if NQ sizing is available.")
    parser.add_argument("--only-enabled", action="store_true", help="Export only specs with quantity > 0.")
    return parser.parse_args()


if __name__ == "__main__":
    export_specs(parse_args())

# cd "C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery\tools"
# python export_nt8_strategy_specs.py --max-specs 10 --only-enabled
# python export_nt8_strategy_specs.py --max-specs 10 --only-enabled --prefer-micro