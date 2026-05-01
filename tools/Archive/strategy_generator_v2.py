# ============================================================
# Strategy Generator
# Regime-Aware Strategy Discovery Pipeline
# v2 momentum-enabled, v1-safe
# ============================================================

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery")

CORE_FEATURE_DATASET_PATH = PROJECT_ROOT / "02_features" / "feature_dataset_core.parquet"
V2_FEATURE_DATASET_PATH = PROJECT_ROOT / "02_features" / "feature_dataset_v2.parquet"

REGISTRY_PATH = PROJECT_ROOT / "03_strategy_registry" / "strategy_combinations.csv"
RUST_INPUT_DIR = PROJECT_ROOT / "04_rust_inputs"


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


MA_TYPES = ["sma", "ema", "wma", "zlema"]
MA_PERIODS = [10, 20, 40, 50, 100, 200]

BOLLINGER_PREFIX = "bb_20_2p0_close"
CHOP_COL = "choppiness_14"
ATR_COL = "atr_14"


BASE_ENTRY_RULE_TYPES = [
    "cross_above",
    "cross_below",
    "price_cross_above",
    "price_cross_below",
    "bollinger_breakout_long",
    "bollinger_breakout_short",
    "bollinger_mean_revert_long",
    "bollinger_mean_revert_short",
]

MOMENTUM_ENTRY_RULE_TYPES = [
    "rsi_oversold_long",
    "rsi_overbought_short",
    "rsi_cross_above",
    "rsi_cross_below",
    "macd_cross_above",
    "macd_cross_below",
    "stoch_cross_above",
    "stoch_cross_below",
    "cci_reversal_long",
    "cci_reversal_short",
    "cci_trend_long",
    "cci_trend_short",
]

BASE_FILTER_TYPES = [
    "none",
    "choppiness_below",
    "choppiness_above",
    "trend_above_ma",
    "trend_below_ma",
]

MOMENTUM_FILTER_TYPES = [
    "roc_positive",
    "roc_negative",
    "rsi_above",
    "rsi_below",
    "macd_hist_positive",
    "macd_hist_negative",
    "cci_above_zero",
    "cci_below_zero",
]

EXIT_RULE_TYPES = [
    "fixed_rr_atr_stop",
    "opposite_cross",
    "time_stop",
]


REGISTRY_FIELDS = [
    "strategy_id",
    "parameter_hash",
    "status",
    "created_at",
    "updated_at",
    "strategy_family",
    "entry_rule_type",
    "exit_rule_type",
    "regime_filter_type",
    "regime_filter_value",
    "config_json",
    "trade_count",
    "net_profit",
    "max_drawdown",
    "np_dd",
    "error_message",
]


@dataclass
class GeneratedStrategy:
    strategy_id: str
    parameter_hash: str
    status: str
    created_at: str
    updated_at: str
    strategy_family: str
    entry_rule_type: str
    exit_rule_type: str
    regime_filter_type: str
    regime_filter_value: str
    config_json: str
    trade_count: str = ""
    net_profit: str = ""
    max_drawdown: str = ""
    np_dd: str = ""
    error_message: str = ""


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def batch_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def stable_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(obj: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json(obj).encode("utf-8")).hexdigest()[:16]


def ma_col(ma_type: str, period: int) -> str:
    return f"{ma_type}_{period}_close"


def read_existing_registry(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=REGISTRY_FIELDS)
    return pd.read_csv(path, dtype=str).fillna("")


def write_registry(path: Path, registry: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for field in REGISTRY_FIELDS:
        if field not in registry.columns:
            registry[field] = ""
    registry = registry[REGISTRY_FIELDS]
    registry.to_csv(path, index=False)


def load_feature_columns(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {path}")

    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    return parquet_file.schema.names


def validate_required_columns(columns: list[str], use_v2_features: bool) -> None:
    required = [
        "open",
        "high",
        "low",
        "close",
        ATR_COL,
        CHOP_COL,
        f"{BOLLINGER_PREFIX}_upper",
        f"{BOLLINGER_PREFIX}_lower",
    ]

    has_regime_family = (
        "regime_family" in columns
        or "frozen_broad_regime_family" in columns
    )

    has_regime_tuple = (
        "regime_tuple" in columns
        or "frozen_regime_tuple" in columns
    )

    if not has_regime_family:
        required.append("regime_family")

    if not has_regime_tuple:
        required.append("regime_tuple")

    if use_v2_features:
        required += [
            "rsi_14",
            "roc_14",
            "cci_20",
            "stoch_k_14_3",
            "stoch_d_14_3_3",
            "macd_line_12_26_9",
            "macd_signal_12_26_9",
            "macd_hist_12_26_9",
        ]

    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(f"Feature dataset missing required columns: {missing}")


def random_ma_pair(rng: random.Random) -> tuple[str, str]:
    ma_type_1 = rng.choice(MA_TYPES)
    ma_type_2 = rng.choice(MA_TYPES)
    p1, p2 = rng.sample(MA_PERIODS, 2)
    fast, slow = sorted([p1, p2])
    return ma_col(ma_type_1, fast), ma_col(ma_type_2, slow)


def random_entry_rule(rng: random.Random, use_v2_features: bool) -> dict[str, Any]:
    rule_types = BASE_ENTRY_RULE_TYPES.copy()
    if use_v2_features:
        rule_types += MOMENTUM_ENTRY_RULE_TYPES

    rule_type = rng.choice(rule_types)

    if rule_type in {"cross_above", "cross_below"}:
        left, right = random_ma_pair(rng)
        side = "long" if rule_type == "cross_above" else "short"
        return {
            "type": rule_type,
            "side": side,
            "left": left,
            "right": right,
        }

    if rule_type == "price_cross_above":
        return {
            "type": "cross_above",
            "side": "long",
            "left": "close",
            "right": ma_col(rng.choice(MA_TYPES), rng.choice(MA_PERIODS)),
            "template": rule_type,
        }

    if rule_type == "price_cross_below":
        return {
            "type": "cross_below",
            "side": "short",
            "left": "close",
            "right": ma_col(rng.choice(MA_TYPES), rng.choice(MA_PERIODS)),
            "template": rule_type,
        }

    if rule_type == "bollinger_breakout_long":
        return {
            "type": "cross_above",
            "side": "long",
            "left": "close",
            "right": f"{BOLLINGER_PREFIX}_upper",
            "template": rule_type,
        }

    if rule_type == "bollinger_breakout_short":
        return {
            "type": "cross_below",
            "side": "short",
            "left": "close",
            "right": f"{BOLLINGER_PREFIX}_lower",
            "template": rule_type,
        }

    if rule_type == "bollinger_mean_revert_long":
        return {
            "type": "cross_above",
            "side": "long",
            "left": "close",
            "right": f"{BOLLINGER_PREFIX}_lower",
            "template": rule_type,
        }

    if rule_type == "bollinger_mean_revert_short":
        return {
            "type": "cross_below",
            "side": "short",
            "left": "close",
            "right": f"{BOLLINGER_PREFIX}_upper",
            "template": rule_type,
        }

    if rule_type == "rsi_oversold_long":
        threshold = rng.choice([25.0, 30.0, 35.0])
        return {
            "type": "cross_below",
            "side": "long",
            "left": "rsi_14",
            "right": threshold,
            "template": rule_type,
        }

    if rule_type == "rsi_overbought_short":
        threshold = rng.choice([65.0, 70.0, 75.0])
        return {
            "type": "cross_above",
            "side": "short",
            "left": "rsi_14",
            "right": threshold,
            "template": rule_type,
        }

    if rule_type == "rsi_cross_above":
        return {
            "type": "cross_above",
            "side": "long",
            "left": "rsi_14",
            "right": rng.choice([30.0, 40.0, 50.0]),
            "template": rule_type,
        }

    if rule_type == "rsi_cross_below":
        return {
            "type": "cross_below",
            "side": "short",
            "left": "rsi_14",
            "right": rng.choice([50.0, 60.0, 70.0]),
            "template": rule_type,
        }

    if rule_type == "macd_cross_above":
        return {
            "type": "cross_above",
            "side": "long",
            "left": "macd_line_12_26_9",
            "right": "macd_signal_12_26_9",
            "template": rule_type,
        }

    if rule_type == "macd_cross_below":
        return {
            "type": "cross_below",
            "side": "short",
            "left": "macd_line_12_26_9",
            "right": "macd_signal_12_26_9",
            "template": rule_type,
        }

    if rule_type == "stoch_cross_above":
        return {
            "type": "cross_above",
            "side": "long",
            "left": "stoch_k_14_3",
            "right": "stoch_d_14_3_3",
            "template": rule_type,
        }

    if rule_type == "stoch_cross_below":
        return {
            "type": "cross_below",
            "side": "short",
            "left": "stoch_k_14_3",
            "right": "stoch_d_14_3_3",
            "template": rule_type,
        }

    if rule_type == "cci_reversal_long":
        threshold = rng.choice([-200.0, -150.0, -100.0])
        return {
            "type": "cross_below",
            "side": "long",
            "left": "cci_20",
            "right": threshold,
            "template": rule_type,
        }

    if rule_type == "cci_reversal_short":
        threshold = rng.choice([100.0, 150.0, 200.0])
        return {
            "type": "cross_above",
            "side": "short",
            "left": "cci_20",
            "right": threshold,
            "template": rule_type,
        }

    if rule_type == "cci_trend_long":
        return {
            "type": "threshold",
            "side": "long",
            "column": "cci_20",
            "operator": ">",
            "value": rng.choice([0.0, 50.0, 100.0]),
            "template": rule_type,
        }

    if rule_type == "cci_trend_short":
        return {
            "type": "threshold",
            "side": "short",
            "column": "cci_20",
            "operator": "<",
            "value": rng.choice([0.0, -50.0, -100.0]),
            "template": rule_type,
        }

    raise ValueError(f"Unhandled entry rule type: {rule_type}")


def random_filter(rng: random.Random, use_v2_features: bool) -> Optional[dict[str, Any]]:
    filter_types = BASE_FILTER_TYPES.copy()
    if use_v2_features:
        filter_types += MOMENTUM_FILTER_TYPES

    filter_type = rng.choice(filter_types)

    if filter_type == "none":
        return None

    if filter_type == "choppiness_below":
        return {
            "type": "threshold",
            "column": CHOP_COL,
            "operator": "<",
            "value": rng.choice([38.2, 45.0, 50.0, 55.0]),
        }

    if filter_type == "choppiness_above":
        return {
            "type": "threshold",
            "column": CHOP_COL,
            "operator": ">",
            "value": rng.choice([50.0, 55.0, 61.8, 62.8]),
        }

    if filter_type == "trend_above_ma":
        return {
            "type": "comparison",
            "left": "close",
            "operator": ">",
            "right": ma_col(rng.choice(MA_TYPES), rng.choice([40, 50, 100, 200])),
        }

    if filter_type == "trend_below_ma":
        return {
            "type": "comparison",
            "left": "close",
            "operator": "<",
            "right": ma_col(rng.choice(MA_TYPES), rng.choice([40, 50, 100, 200])),
        }

    if filter_type == "roc_positive":
        return {
            "type": "threshold",
            "column": "roc_14",
            "operator": ">",
            "value": rng.choice([0.0, 0.05, 0.10, 0.20]),
        }

    if filter_type == "roc_negative":
        return {
            "type": "threshold",
            "column": "roc_14",
            "operator": "<",
            "value": rng.choice([0.0, -0.05, -0.10, -0.20]),
        }

    if filter_type == "rsi_above":
        return {
            "type": "threshold",
            "column": "rsi_14",
            "operator": ">",
            "value": rng.choice([50.0, 55.0, 60.0]),
        }

    if filter_type == "rsi_below":
        return {
            "type": "threshold",
            "column": "rsi_14",
            "operator": "<",
            "value": rng.choice([50.0, 45.0, 40.0]),
        }

    if filter_type == "macd_hist_positive":
        return {
            "type": "threshold",
            "column": "macd_hist_12_26_9",
            "operator": ">",
            "value": 0.0,
        }

    if filter_type == "macd_hist_negative":
        return {
            "type": "threshold",
            "column": "macd_hist_12_26_9",
            "operator": "<",
            "value": 0.0,
        }

    if filter_type == "cci_above_zero":
        return {
            "type": "threshold",
            "column": "cci_20",
            "operator": ">",
            "value": 0.0,
        }

    if filter_type == "cci_below_zero":
        return {
            "type": "threshold",
            "column": "cci_20",
            "operator": "<",
            "value": 0.0,
        }

    raise ValueError(f"Unhandled filter type: {filter_type}")


def random_exit_rule(rng: random.Random, entry_rule: dict[str, Any]) -> dict[str, Any]:
    exit_type = rng.choice(EXIT_RULE_TYPES)

    if exit_type == "fixed_rr_atr_stop":
        return {
            "type": "fixed_rr_atr_stop",
            "atr_column": ATR_COL,
            "stop_atr_multiple": rng.choice([0.75, 1.0, 1.25, 1.5, 2.0, 2.5]),
            "reward_risk": rng.choice([1.0, 1.25, 1.5, 2.0, 2.5, 3.0]),
            "max_bars_in_trade": rng.choice([30, 60, 120, 240, 390, 780]),
        }

    if exit_type == "opposite_cross" and entry_rule.get("type") in {"cross_above", "cross_below"}:
        return {
            "type": "opposite_cross",
            "left": entry_rule["left"],
            "right": entry_rule["right"],
            "max_bars_in_trade": rng.choice([60, 120, 240, 390, 780]),
        }

    return {
        "type": "time_stop",
        "max_bars_in_trade": rng.choice([30, 60, 120, 240, 390, 780]),
    }


def random_regime_filter(
    rng: random.Random,
    include_specialists: bool,
) -> tuple[str, str, Optional[dict[str, Any]]]:
    if not include_specialists:
        return "all", "ALL", None

    mode = rng.choices(["all", "family"], weights=[0.35, 0.65], k=1)[0]

    if mode == "all":
        return "all", "ALL", None

    family = rng.choice(REGIME_FAMILIES)
    return (
        "regime_family",
        family,
        {
            "type": "regime_family_in",
            "column": "regime_family",
            "values": [family],
        },
    )


def build_strategy_config(
    rng: random.Random,
    include_regime_specialists: bool,
    use_v2_features: bool,
    instrument: str = "NQ",
    timeframe: str = "1min",
) -> dict[str, Any]:
    entry = random_entry_rule(rng, use_v2_features=use_v2_features)
    filt = random_filter(rng, use_v2_features=use_v2_features)
    exit_rule = random_exit_rule(rng, entry)
    regime_filter_type, regime_filter_value, regime_filter = random_regime_filter(
        rng,
        include_regime_specialists,
    )

    filters = []
    if filt is not None:
        filters.append(filt)
    if regime_filter is not None:
        filters.append(regime_filter)

    config = {
        "instrument": instrument,
        "timeframe": timeframe,
        "engine_version": "strategy_generator_v2_momentum" if use_v2_features else "strategy_generator_v1",
        "feature_version": "v2_momentum" if use_v2_features else "core",
        "entry_timing": "next_bar_open_after_signal",
        "regime_assignment": "entry_time_only",
        "costs": {
            "commission_round_trip": 5.0,
            "slippage_ticks": 3,
            "tick_size": 0.25,
            "point_value": 20.0,
        },
        "entry_rule": entry,
        "filters": filters,
        "exit_rule": exit_rule,
        "position": {
            "direction": entry.get("side", "long"),
            "contracts": 1,
        },
        "metadata": {
            "regime_filter_type": regime_filter_type,
            "regime_filter_value": regime_filter_value,
        },
    }

    parameter_hash = stable_hash(config)
    config["strategy_id"] = f"strat_{parameter_hash}"
    config["parameter_hash"] = parameter_hash
    return config


def make_registry_row(config: dict[str, Any]) -> GeneratedStrategy:
    ts = now_str()
    metadata = config.get("metadata", {})
    entry_rule = config.get("entry_rule", {})
    exit_rule = config.get("exit_rule", {})

    return GeneratedStrategy(
        strategy_id=config["strategy_id"],
        parameter_hash=config["parameter_hash"],
        status="pending",
        created_at=ts,
        updated_at=ts,
        strategy_family="momentum_random_v2" if config.get("feature_version") == "v2_momentum" else "core_random_v1",
        entry_rule_type=entry_rule.get("template", entry_rule.get("type", "")),
        exit_rule_type=exit_rule.get("type", ""),
        regime_filter_type=metadata.get("regime_filter_type", "all"),
        regime_filter_value=metadata.get("regime_filter_value", "ALL"),
        config_json=json.dumps(config, sort_keys=True),
    )


def generate_batch(
    batch_size: int,
    seed: Optional[int],
    include_regime_specialists: bool,
    use_v2_features: bool,
    max_attempts_multiplier: int = 50,
) -> tuple[list[dict[str, Any]], pd.DataFrame, Path]:
    rng = random.Random(seed)

    feature_dataset_path = V2_FEATURE_DATASET_PATH if use_v2_features else CORE_FEATURE_DATASET_PATH

    columns = load_feature_columns(feature_dataset_path)
    validate_required_columns(columns, use_v2_features=use_v2_features)

    registry = read_existing_registry(REGISTRY_PATH)
    existing_hashes = set(registry["parameter_hash"].astype(str).tolist()) if not registry.empty else set()

    new_configs: list[dict[str, Any]] = []
    new_rows: list[dict[str, str]] = []

    attempts = 0
    max_attempts = max(batch_size * max_attempts_multiplier, batch_size)

    while len(new_configs) < batch_size and attempts < max_attempts:
        attempts += 1
        config = build_strategy_config(
            rng,
            include_regime_specialists=include_regime_specialists,
            use_v2_features=use_v2_features,
        )
        h = config["parameter_hash"]

        if h in existing_hashes:
            continue

        existing_hashes.add(h)
        new_configs.append(config)
        new_rows.append(asdict(make_registry_row(config)))

    if len(new_configs) < batch_size:
        print(
            f"WARNING: Requested {batch_size:,} strategies but only generated "
            f"{len(new_configs):,} unique configs after {attempts:,} attempts."
        )

    if new_rows:
        registry = pd.concat([registry, pd.DataFrame(new_rows)], ignore_index=True)
        write_registry(REGISTRY_PATH, registry)

    RUST_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    batch_path = RUST_INPUT_DIR / f"strategy_batch_{batch_timestamp()}.json"

    batch_payload = {
        "created_at": now_str(),
        "feature_dataset_path": str(feature_dataset_path),
        "feature_version": "v2_momentum" if use_v2_features else "core",
        "strategy_count": len(new_configs),
        "strategies": new_configs,
    }

    batch_path.write_text(json.dumps(batch_payload, indent=2), encoding="utf-8")

    return new_configs, registry, batch_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate randomized strategy configs for Rust evaluation.")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of new strategies to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--include-regime-specialists",
        action="store_true",
        help="Include regime-family specialist strategy variants",
    )
    parser.add_argument(
        "--use-v2-features",
        action="store_true",
        help="Use feature_dataset_v2.parquet and enable momentum strategy templates",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    configs, registry, batch_path = generate_batch(
        batch_size=args.batch_size,
        seed=args.seed,
        include_regime_specialists=args.include_regime_specialists,
        use_v2_features=args.use_v2_features,
    )

    print("Strategy batch generated.")
    print(f"Feature version:   {'v2_momentum' if args.use_v2_features else 'core'}")
    print(f"Batch size:        {len(configs):,}")
    print(f"Registry path:     {REGISTRY_PATH}")
    print(f"Registry rows:     {len(registry):,}")
    print(f"Rust batch path:   {batch_path}")

    if configs:
        print("First strategy:")
        print(json.dumps(configs[0], indent=2))
        
# python strategy_generator.py --batch-size 50
# python strategy_generator.py --batch-size 50 --use-v2-features