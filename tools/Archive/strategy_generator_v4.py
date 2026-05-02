# ============================================================
# Strategy Generator
# Regime-Aware Strategy Discovery Pipeline
# v1 core + v2 momentum + v3 VWAP/ADX/volume/volatility
# + v4 channels / price structure
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
V3_FEATURE_DATASET_PATH = PROJECT_ROOT / "02_features" / "feature_dataset_v3.parquet"
V4_FEATURE_DATASET_PATH = PROJECT_ROOT / "02_features" / "feature_dataset_v4.parquet"

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

V3_ENTRY_RULE_TYPES = [
    "price_cross_above_vwap",
    "price_cross_below_vwap",
    "vwap_mean_revert_long",
    "vwap_mean_revert_short",
    "vwap_band_breakout_long",
    "vwap_band_breakout_short",
]

V4_ENTRY_RULE_TYPES = [
    "donchian_breakout_long",
    "donchian_breakout_short",
    "donchian_mean_revert_long",
    "donchian_mean_revert_short",
    "keltner_breakout_long",
    "keltner_breakout_short",
    "keltner_mean_revert_long",
    "keltner_mean_revert_short",
    "prior_session_high_breakout_long",
    "prior_session_low_breakout_short",
    "opening_range_breakout_long",
    "opening_range_breakout_short",
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

V3_FILTER_TYPES = [
    "adx_above",
    "adx_below",
    "dmi_bullish",
    "dmi_bearish",
    "volume_ratio_above",
    "obv_slope_positive",
    "obv_slope_negative",
    "atr_expansion",
    "atr_contraction",
    "price_above_vwap",
    "price_below_vwap",
]

V4_FILTER_TYPES = [
    "inside_bar_filter",
    "outside_bar_filter",
    "range_expansion_filter",
    "range_compression_filter",
    "close_above_prior_session_high",
    "close_below_prior_session_low",
    "close_above_opening_range_high",
    "close_below_opening_range_low",
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


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

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


def validate_required_columns(
    columns: list[str],
    use_v2_features: bool,
    use_v3_features: bool,
    use_v4_features: bool,
) -> None:
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

    if use_v2_features or use_v3_features or use_v4_features:
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

    if use_v3_features or use_v4_features:
        required += [
            "vwap_session",
            "vwap_distance",
            "vwap_distance_pct",
            "vwap_band_upper_20_1p0",
            "vwap_band_lower_20_1p0",
            "vwap_band_upper_20_2p0",
            "vwap_band_lower_20_2p0",
            "adx_14",
            "plus_di_14",
            "minus_di_14",
            "dmi_spread_14",
            "volume_ratio_20",
            "obv_slope_10",
            "atr_14_zscore_20",
            "atr_14_expansion_20",
            "atr_14_contraction_20",
        ]

    if use_v4_features:
        required += [
            "donchian_high_20",
            "donchian_low_20",
            "donchian_mid_20",
            "donchian_high_50",
            "donchian_low_50",
            "keltner_upper_20_1p5",
            "keltner_lower_20_1p5",
            "keltner_upper_20_2p0",
            "keltner_lower_20_2p0",
            "prior_session_high",
            "prior_session_low",
            "prior_session_close",
            "opening_range_high_30",
            "opening_range_low_30",
            "opening_range_high_60",
            "opening_range_low_60",
            "inside_bar",
            "outside_bar",
            "bar_range_zscore_20",
            "range_expansion_20",
            "range_compression_20",
            "close_above_prior_session_high",
            "close_below_prior_session_low",
            "close_above_opening_range_high_30",
            "close_below_opening_range_low_30",
        ]

    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(f"Feature dataset missing required columns: {missing}")


def feature_version_label(
    use_v2_features: bool,
    use_v3_features: bool,
    use_v4_features: bool,
) -> str:
    if use_v4_features:
        return "v4_channels_price_structure"
    if use_v3_features:
        return "v3_vwap_adx_volume_volatility"
    if use_v2_features:
        return "v2_momentum"
    return "core"


def engine_version_label(
    use_v2_features: bool,
    use_v3_features: bool,
    use_v4_features: bool,
) -> str:
    if use_v4_features:
        return "strategy_generator_v4"
    if use_v3_features:
        return "strategy_generator_v3"
    if use_v2_features:
        return "strategy_generator_v2_momentum"
    return "strategy_generator_v1"


# ------------------------------------------------------------
# Random strategy construction
# ------------------------------------------------------------

def random_ma_pair(rng: random.Random) -> tuple[str, str]:
    ma_type_1 = rng.choice(MA_TYPES)
    ma_type_2 = rng.choice(MA_TYPES)
    p1, p2 = rng.sample(MA_PERIODS, 2)

    fast, slow = sorted([p1, p2])
    left = ma_col(ma_type_1, fast)
    right = ma_col(ma_type_2, slow)
    return left, right


def random_entry_rule(
    rng: random.Random,
    use_v2_features: bool,
    use_v3_features: bool,
    use_v4_features: bool,
) -> dict[str, Any]:
    rule_types = BASE_ENTRY_RULE_TYPES.copy()

    if use_v2_features or use_v3_features or use_v4_features:
        rule_types += MOMENTUM_ENTRY_RULE_TYPES

    if use_v3_features or use_v4_features:
        rule_types += V3_ENTRY_RULE_TYPES

    if use_v4_features:
        rule_types += V4_ENTRY_RULE_TYPES

    rule_type = rng.choice(rule_types)

    # -----------------------------
    # Core entries
    # -----------------------------

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

    # -----------------------------
    # v2 momentum entries
    # -----------------------------

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
        
    # -----------------------------
    # v3 VWAP entries
    # -----------------------------

    if rule_type == "price_cross_above_vwap":
        return {
            "type": "cross_above",
            "side": "long",
            "left": "close",
            "right": "vwap_session",
            "template": rule_type,
        }

    if rule_type == "price_cross_below_vwap":
        return {
            "type": "cross_below",
            "side": "short",
            "left": "close",
            "right": "vwap_session",
            "template": rule_type,
        }

    if rule_type == "vwap_mean_revert_long":
        return {
            "type": "cross_above",
            "side": "long",
            "left": "close",
            "right": rng.choice(["vwap_band_lower_20_1p0", "vwap_band_lower_20_2p0"]),
            "template": rule_type,
        }

    if rule_type == "vwap_mean_revert_short":
        return {
            "type": "cross_below",
            "side": "short",
            "left": "close",
            "right": rng.choice(["vwap_band_upper_20_1p0", "vwap_band_upper_20_2p0"]),
            "template": rule_type,
        }

    if rule_type == "vwap_band_breakout_long":
        return {
            "type": "cross_above",
            "side": "long",
            "left": "close",
            "right": rng.choice(["vwap_band_upper_20_1p0", "vwap_band_upper_20_2p0"]),
            "template": rule_type,
        }

    if rule_type == "vwap_band_breakout_short":
        return {
            "type": "cross_below",
            "side": "short",
            "left": "close",
            "right": rng.choice(["vwap_band_lower_20_1p0", "vwap_band_lower_20_2p0"]),
            "template": rule_type,
        }

    # -----------------------------
    # v4 channel / structure entries
    # -----------------------------

    if rule_type == "donchian_breakout_long":
        period = rng.choice([20, 50])
        return {
            "type": "cross_above",
            "side": "long",
            "left": "close",
            "right": f"donchian_high_{period}",
            "template": rule_type,
        }

    if rule_type == "donchian_breakout_short":
        period = rng.choice([20, 50])
        return {
            "type": "cross_below",
            "side": "short",
            "left": "close",
            "right": f"donchian_low_{period}",
            "template": rule_type,
        }

    if rule_type == "donchian_mean_revert_long":
        period = rng.choice([20, 50])
        return {
            "type": "cross_above",
            "side": "long",
            "left": "close",
            "right": f"donchian_low_{period}",
            "template": rule_type,
        }

    if rule_type == "donchian_mean_revert_short":
        period = rng.choice([20, 50])
        return {
            "type": "cross_below",
            "side": "short",
            "left": "close",
            "right": f"donchian_high_{period}",
            "template": rule_type,
        }

    if rule_type == "keltner_breakout_long":
        band = rng.choice(["keltner_upper_20_1p5", "keltner_upper_20_2p0"])
        return {
            "type": "cross_above",
            "side": "long",
            "left": "close",
            "right": band,
            "template": rule_type,
        }

    if rule_type == "keltner_breakout_short":
        band = rng.choice(["keltner_lower_20_1p5", "keltner_lower_20_2p0"])
        return {
            "type": "cross_below",
            "side": "short",
            "left": "close",
            "right": band,
            "template": rule_type,
        }

    if rule_type == "keltner_mean_revert_long":
        band = rng.choice(["keltner_lower_20_1p5", "keltner_lower_20_2p0"])
        return {
            "type": "cross_above",
            "side": "long",
            "left": "close",
            "right": band,
            "template": rule_type,
        }

    if rule_type == "keltner_mean_revert_short":
        band = rng.choice(["keltner_upper_20_1p5", "keltner_upper_20_2p0"])
        return {
            "type": "cross_below",
            "side": "short",
            "left": "close",
            "right": band,
            "template": rule_type,
        }

    if rule_type == "prior_session_high_breakout_long":
        return {
            "type": "cross_above",
            "side": "long",
            "left": "close",
            "right": "prior_session_high",
            "template": rule_type,
        }

    if rule_type == "prior_session_low_breakout_short":
        return {
            "type": "cross_below",
            "side": "short",
            "left": "close",
            "right": "prior_session_low",
            "template": rule_type,
        }

    if rule_type == "opening_range_breakout_long":
        minutes = rng.choice([30, 60])
        return {
            "type": "cross_above",
            "side": "long",
            "left": "close",
            "right": f"opening_range_high_{minutes}",
            "template": rule_type,
        }

    if rule_type == "opening_range_breakout_short":
        minutes = rng.choice([30, 60])
        return {
            "type": "cross_below",
            "side": "short",
            "left": "close",
            "right": f"opening_range_low_{minutes}",
            "template": rule_type,
        }

    raise ValueError(f"Unhandled entry rule type: {rule_type}")


def random_filter(
    rng: random.Random,
    use_v2_features: bool,
    use_v3_features: bool,
    use_v4_features: bool,
) -> Optional[dict[str, Any]]:
    filter_types = BASE_FILTER_TYPES.copy()

    if use_v2_features or use_v3_features or use_v4_features:
        filter_types += MOMENTUM_FILTER_TYPES

    if use_v3_features or use_v4_features:
        filter_types += V3_FILTER_TYPES

    if use_v4_features:
        filter_types += V4_FILTER_TYPES

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

    # -----------------------------
    # v2 momentum filters
    # -----------------------------

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

    # -----------------------------
    # v3 filters
    # -----------------------------

    if filter_type == "adx_above":
        return {
            "type": "threshold",
            "column": "adx_14",
            "operator": ">",
            "value": rng.choice([20.0, 25.0, 30.0]),
        }

    if filter_type == "adx_below":
        return {
            "type": "threshold",
            "column": "adx_14",
            "operator": "<",
            "value": rng.choice([15.0, 20.0, 25.0]),
        }

    if filter_type == "dmi_bullish":
        return {
            "type": "comparison",
            "left": "plus_di_14",
            "operator": ">",
            "right": "minus_di_14",
        }

    if filter_type == "dmi_bearish":
        return {
            "type": "comparison",
            "left": "minus_di_14",
            "operator": ">",
            "right": "plus_di_14",
        }

    if filter_type == "volume_ratio_above":
        return {
            "type": "threshold",
            "column": "volume_ratio_20",
            "operator": ">",
            "value": rng.choice([1.2, 1.5, 2.0]),
        }

    if filter_type == "obv_slope_positive":
        return {
            "type": "threshold",
            "column": "obv_slope_10",
            "operator": ">",
            "value": 0.0,
        }

    if filter_type == "obv_slope_negative":
        return {
            "type": "threshold",
            "column": "obv_slope_10",
            "operator": "<",
            "value": 0.0,
        }

    if filter_type == "atr_expansion":
        return {
            "type": "threshold",
            "column": "atr_14_zscore_20",
            "operator": ">",
            "value": rng.choice([0.5, 1.0, 1.5]),
        }

    if filter_type == "atr_contraction":
        return {
            "type": "threshold",
            "column": "atr_14_zscore_20",
            "operator": "<",
            "value": rng.choice([-0.5, -1.0, -1.5]),
        }

    if filter_type == "price_above_vwap":
        return {
            "type": "comparison",
            "left": "close",
            "operator": ">",
            "right": "vwap_session",
        }

    if filter_type == "price_below_vwap":
        return {
            "type": "comparison",
            "left": "close",
            "operator": "<",
            "right": "vwap_session",
        }

    # -----------------------------
    # v4 price-structure filters
    # -----------------------------

    if filter_type == "inside_bar_filter":
        return {
            "type": "threshold",
            "column": "inside_bar",
            "operator": ">",
            "value": 0.5,
        }

    if filter_type == "outside_bar_filter":
        return {
            "type": "threshold",
            "column": "outside_bar",
            "operator": ">",
            "value": 0.5,
        }

    if filter_type == "range_expansion_filter":
        return {
            "type": "threshold",
            "column": "bar_range_zscore_20",
            "operator": ">",
            "value": rng.choice([0.5, 1.0, 1.5]),
        }

    if filter_type == "range_compression_filter":
        return {
            "type": "threshold",
            "column": "bar_range_zscore_20",
            "operator": "<",
            "value": rng.choice([-0.5, -1.0, -1.5]),
        }

    if filter_type == "close_above_prior_session_high":
        return {
            "type": "comparison",
            "left": "close",
            "operator": ">",
            "right": "prior_session_high",
        }

    if filter_type == "close_below_prior_session_low":
        return {
            "type": "comparison",
            "left": "close",
            "operator": "<",
            "right": "prior_session_low",
        }

    if filter_type == "close_above_opening_range_high":
        minutes = rng.choice([30, 60])
        return {
            "type": "comparison",
            "left": "close",
            "operator": ">",
            "right": f"opening_range_high_{minutes}",
        }

    if filter_type == "close_below_opening_range_low":
        minutes = rng.choice([30, 60])
        return {
            "type": "comparison",
            "left": "close",
            "operator": "<",
            "right": f"opening_range_low_{minutes}",
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
    use_v3_features: bool,
    use_v4_features: bool,
    instrument: str = "NQ",
    timeframe: str = "1min",
) -> dict[str, Any]:
    entry = random_entry_rule(
        rng,
        use_v2_features=use_v2_features,
        use_v3_features=use_v3_features,
        use_v4_features=use_v4_features,
    )

    filt = random_filter(
        rng,
        use_v2_features=use_v2_features,
        use_v3_features=use_v3_features,
        use_v4_features=use_v4_features,
    )

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

    feature_version = feature_version_label(use_v2_features, use_v3_features, use_v4_features)
    engine_version = engine_version_label(use_v2_features, use_v3_features, use_v4_features)

    config = {
        "instrument": instrument,
        "timeframe": timeframe,
        "engine_version": engine_version,
        "feature_version": feature_version,
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


# ------------------------------------------------------------
# Batch generation
# ------------------------------------------------------------

def make_registry_row(config: dict[str, Any]) -> GeneratedStrategy:
    ts = now_str()
    metadata = config.get("metadata", {})
    entry_rule = config.get("entry_rule", {})
    exit_rule = config.get("exit_rule", {})

    feature_version = config.get("feature_version", "core")

    if feature_version.startswith("v4"):
        strategy_family = "v4_channels_structure_random"
    elif feature_version.startswith("v3"):
        strategy_family = "v3_vwap_adx_volume_random"
    elif feature_version == "v2_momentum":
        strategy_family = "momentum_random_v2"
    else:
        strategy_family = "core_random_v1"

    return GeneratedStrategy(
        strategy_id=config["strategy_id"],
        parameter_hash=config["parameter_hash"],
        status="pending",
        created_at=ts,
        updated_at=ts,
        strategy_family=strategy_family,
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
    use_v3_features: bool,
    use_v4_features: bool,
    max_attempts_multiplier: int = 50,
) -> tuple[list[dict[str, Any]], pd.DataFrame, Path]:
    rng = random.Random(seed)

    if use_v4_features:
        feature_dataset_path = V4_FEATURE_DATASET_PATH
    elif use_v3_features:
        feature_dataset_path = V3_FEATURE_DATASET_PATH
    elif use_v2_features:
        feature_dataset_path = V2_FEATURE_DATASET_PATH
    else:
        feature_dataset_path = CORE_FEATURE_DATASET_PATH

    columns = load_feature_columns(feature_dataset_path)

    validate_required_columns(
        columns,
        use_v2_features=use_v2_features,
        use_v3_features=use_v3_features,
        use_v4_features=use_v4_features,
    )

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
            use_v3_features=use_v3_features,
            use_v4_features=use_v4_features,
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

    feature_version = feature_version_label(
        use_v2_features,
        use_v3_features,
        use_v4_features,
    )

    batch_payload = {
        "created_at": now_str(),
        "feature_dataset_path": str(feature_dataset_path),
        "feature_version": feature_version,
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

    parser.add_argument(
        "--use-v3-features",
        action="store_true",
        help="Use feature_dataset_v3.parquet and enable VWAP/ADX/volume/volatility templates",
    )

    parser.add_argument(
        "--use-v4-features",
        action="store_true",
        help="Use feature_dataset_v4.parquet and enable channel/price-structure templates",
    )

    args = parser.parse_args()

    feature_flags = [
        args.use_v2_features,
        args.use_v3_features,
        args.use_v4_features,
    ]

    if sum(bool(x) for x in feature_flags) > 1:
        raise ValueError("Use only one of --use-v2-features, --use-v3-features, or --use-v4-features.")

    return args


if __name__ == "__main__":
    args = parse_args()

    configs, registry, batch_path = generate_batch(
        batch_size=args.batch_size,
        seed=args.seed,
        include_regime_specialists=args.include_regime_specialists,
        use_v2_features=args.use_v2_features,
        use_v3_features=args.use_v3_features,
        use_v4_features=args.use_v4_features,
    )

    feature_version = feature_version_label(
        args.use_v2_features,
        args.use_v3_features,
        args.use_v4_features,
    )

    print("Strategy batch generated.")
    print(f"Feature version:   {feature_version}")
    print(f"Batch size:        {len(configs):,}")
    print(f"Registry path:     {REGISTRY_PATH}")
    print(f"Registry rows:     {len(registry):,}")
    print(f"Rust batch path:   {batch_path}")

    if configs:
        print("First strategy:")
        print(json.dumps(configs[0], indent=2))