# ============================================================
# Generate NT8 Strategy Code - Fully Updated
# Regime-Aware Strategy Discovery Pipeline
# ============================================================
#
# Recommended save location:
# C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery\tools\generate_nt8_strategy_code.py
#
# Purpose:
#   Read deployment JSON specs and generate NinjaTrader 8 strategy .cs files.
#
# Inputs:
#   09_nt8_deployment\strategy_specs\*.json
#
# Outputs:
#   09_nt8_deployment\generated_strategies\*.cs
#   09_nt8_deployment\generated_strategy_index.csv
#
# Run:
#   python generate_nt8_strategy_code.py --max-files 1
#   python generate_nt8_strategy_code.py --max-files 10
#   python generate_nt8_strategy_code.py --spec "path\to\specific_spec.json"
#
# Notes:
#   - Includes Slippage = 3 in NT8 SetDefaults.
#   - Includes full RegimeClassifierRB1 constructor parameters.
#   - Tuple-based strategies use FrozenTrendStateSeries,
#     FrozenIntermediateStateSeries, and FrozenAccelStateSeries.
#   - Family-based strategies require exact RegimeCodeToFamily mapping before use.
#   - Opposite-cross exit logic intentionally mirrors the Rust engine:
#       long entry  -> exit when same pair crosses below
#       short entry -> exit when same pair crosses above
#
# IMPORTANT:
#   Generated strategies are intended for NT8 Strategy Analyzer validation first.
#   Do not deploy live until compile + historical parity + sim validation pass.
# ============================================================

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(r"C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery")
DEPLOY_ROOT = PROJECT_ROOT / "09_nt8_deployment"
SPEC_DIR = DEPLOY_ROOT / "strategy_specs"
GENERATED_DIR = DEPLOY_ROOT / "generated_strategies"
INDEX_PATH = DEPLOY_ROOT / "generated_strategy_index.csv"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

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


def parse_tuple_key(key: str) -> tuple[int, int, int]:
    nums = re.findall(r"-?\d+", str(key))
    if len(nums) < 3:
        raise ValueError(f"Could not parse regime tuple key: {key}")
    return int(nums[0]), int(nums[1]), int(nums[2])


def csharp_string(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def csharp_bool(value: bool) -> str:
    return "true" if bool(value) else "false"


def csharp_float(value: float | int | str, default: float = 0.0) -> str:
    try:
        v = float(value)
    except Exception:
        v = default
    if v == int(v):
        return f"{v:.1f}"
    return repr(v)


# ------------------------------------------------------------
# Indicator expression mapping
# ------------------------------------------------------------

def indicator_expr(col: str) -> str:
    """
    Convert generated feature column names into NT8 expression snippets.

    Supported examples:
      close
      sma_20_close
      ema_40_close
      wma_100_close
      zlema_50_close
      bb_20_2p0_close_lower
      bb_20_2p0_close_upper
      bb_20_2p0_close_mid
      choppiness_14
      atr_14
    """
    col = str(col)

    if col == "close":
        return "Close[0]"
    if col == "open":
        return "Open[0]"
    if col == "high":
        return "High[0]"
    if col == "low":
        return "Low[0]"

    m = re.match(r"sma_(\d+)_close", col)
    if m:
        return f"SMA(Close, {m.group(1)})[0]"

    m = re.match(r"ema_(\d+)_close", col)
    if m:
        return f"EMA(Close, {m.group(1)})[0]"

    m = re.match(r"wma_(\d+)_close", col)
    if m:
        return f"WMA(Close, {m.group(1)})[0]"

    m = re.match(r"zlema_(\d+)_close", col)
    if m:
        return f"ZLEMA({m.group(1)}, 0)"

    m = re.match(r"atr_(\d+)", col)
    if m:
        return f"ATR({m.group(1)})[0]"

    m = re.match(r"choppiness_(\d+)", col)
    if m:
        return f"ChoppinessIndexValue({m.group(1)}, 0)"

    m = re.match(r"bb_(\d+)_([0-9p]+)_close_(upper|lower|mid|width|pct_b)", col)
    if m:
        period = int(m.group(1))
        std = float(m.group(2).replace("p", "."))
        band = m.group(3)
        if band == "upper":
            return f"BollingerUpper({period}, {std}, 0)"
        if band == "lower":
            return f"BollingerLower({period}, {std}, 0)"
        if band == "mid":
            return f"SMA(Close, {period})[0]"
        if band == "width":
            return f"(BollingerUpper({period}, {std}, 0) - BollingerLower({period}, {std}, 0))"
        if band == "pct_b":
            return f"((Close[0] - BollingerLower({period}, {std}, 0)) / Math.Max(TickSize, BollingerUpper({period}, {std}, 0) - BollingerLower({period}, {std}, 0)))"

    raise ValueError(f"Unsupported indicator/column expression: {col}")


def indicator_expr_prev(col: str) -> str:
    col = str(col)

    if col == "close":
        return "Close[1]"
    if col == "open":
        return "Open[1]"
    if col == "high":
        return "High[1]"
    if col == "low":
        return "Low[1]"

    m = re.match(r"sma_(\d+)_close", col)
    if m:
        return f"SMA(Close, {m.group(1)})[1]"

    m = re.match(r"ema_(\d+)_close", col)
    if m:
        return f"EMA(Close, {m.group(1)})[1]"

    m = re.match(r"wma_(\d+)_close", col)
    if m:
        return f"WMA(Close, {m.group(1)})[1]"

    m = re.match(r"zlema_(\d+)_close", col)
    if m:
        return f"ZLEMA({m.group(1)}, 1)"

    m = re.match(r"atr_(\d+)", col)
    if m:
        return f"ATR({m.group(1)})[1]"

    m = re.match(r"choppiness_(\d+)", col)
    if m:
        return f"ChoppinessIndexValue({m.group(1)}, 1)"

    m = re.match(r"bb_(\d+)_([0-9p]+)_close_(upper|lower|mid|width|pct_b)", col)
    if m:
        period = int(m.group(1))
        std = float(m.group(2).replace("p", "."))
        band = m.group(3)
        if band == "upper":
            return f"BollingerUpper({period}, {std}, 1)"
        if band == "lower":
            return f"BollingerLower({period}, {std}, 1)"
        if band == "mid":
            return f"SMA(Close, {period})[1]"
        if band == "width":
            return f"(BollingerUpper({period}, {std}, 1) - BollingerLower({period}, {std}, 1))"
        if band == "pct_b":
            return f"((Close[1] - BollingerLower({period}, {std}, 1)) / Math.Max(TickSize, BollingerUpper({period}, {std}, 1) - BollingerLower({period}, {std}, 1)))"

    raise ValueError(f"Unsupported previous indicator expression: {col}")


# ------------------------------------------------------------
# Rule generation
# ------------------------------------------------------------

def cross_code(rule: dict[str, Any]) -> str:
    left = rule["left"]
    right = rule["right"]
    l0 = indicator_expr(left)
    r0 = indicator_expr(right)
    l1 = indicator_expr_prev(left)
    r1 = indicator_expr_prev(right)

    if rule["type"] in {"cross_above", "price_cross_above"}:
        return f"(({l1}) <= ({r1}) && ({l0}) > ({r0}))"

    if rule["type"] in {"cross_below", "price_cross_below"}:
        return f"(({l1}) >= ({r1}) && ({l0}) < ({r0}))"

    raise ValueError(f"Unsupported cross type: {rule['type']}")


def filter_code(filter_rule: dict[str, Any]) -> str:
    ftype = filter_rule.get("type")

    if ftype == "threshold":
        col = indicator_expr(filter_rule["column"])
        op = filter_rule["operator"]
        val = float(filter_rule["value"])
        return f"(({col}) {op} {csharp_float(val)})"

    if ftype == "comparison":
        left = indicator_expr(filter_rule["left"])
        right = indicator_expr(filter_rule["right"])
        op = filter_rule["operator"]
        return f"(({left}) {op} ({right}))"

    # Regime filters are handled separately.
    if ftype == "regime_family_in":
        return "true"

    raise ValueError(f"Unsupported filter type for NT8 generation: {ftype}")


def max_bars_required(spec: dict[str, Any]) -> int:
    text = json.dumps(spec)
    periods = [int(x) for x in re.findall(r"(?:sma|ema|wma|zlema|atr|choppiness)_(\d+)", text)]
    periods += [int(x) for x in re.findall(r"bb_(\d+)_", text)]
    return max(periods + [20]) + 10


def generate_regime_code(regime_filter: dict[str, Any]) -> str:
    scope = regime_filter.get("scope", "")
    key = regime_filter.get("key", "")

    if scope == "regime_tuple":
        trend, interm, accel = parse_tuple_key(key)
        return f"(CurrentTrendState() == {trend} && CurrentIntermediateState() == {interm} && CurrentAccelState() == {accel})"

    if scope == "regime_family":
        return f"(CurrentRegimeFamily() == \"{csharp_string(key)}\")"

    return "true"


def generate_exit_code(exit_rule: dict[str, Any], entry_rule: dict[str, Any], side: str) -> tuple[str, str]:
    """
    Returns:
      on_entry_init_code, on_position_exit_code
    """
    exit_type = exit_rule.get("type", "time_stop")

    if exit_type == "time_stop":
        return "", '''
            if (Position.MarketPosition != MarketPosition.Flat && CurrentBar - entryBar >= MaxBarsInTrade)
            {
                if (Position.MarketPosition == MarketPosition.Long)
                    ExitLong("TimeStop", EntrySignalName);
                else if (Position.MarketPosition == MarketPosition.Short)
                    ExitShort("TimeStop", EntrySignalName);
            }'''

    if exit_type == "opposite_cross":
        # Mirrors Rust engine behavior:
        #   Long exits when the same left/right pair crosses below.
        #   Short exits when the same left/right pair crosses above.
        opp_rule = {
            "type": "cross_below" if side == "long" else "cross_above",
            "left": exit_rule.get("left", entry_rule.get("left")),
            "right": exit_rule.get("right", entry_rule.get("right")),
        }
        opp_signal = cross_code(opp_rule)
        return "", f'''
            if (Position.MarketPosition != MarketPosition.Flat)
            {{
                bool oppositeCross = {opp_signal};
                if (oppositeCross || CurrentBar - entryBar >= MaxBarsInTrade)
                {{
                    if (Position.MarketPosition == MarketPosition.Long)
                        ExitLong(oppositeCross ? "OppositeCross" : "TimeStop", EntrySignalName);
                    else if (Position.MarketPosition == MarketPosition.Short)
                        ExitShort(oppositeCross ? "OppositeCross" : "TimeStop", EntrySignalName);
                }}
            }}'''

    if exit_type == "fixed_rr_atr_stop":
        atr_col = exit_rule.get("atr_column", "atr_14")
        atr_expr = indicator_expr(atr_col)
        stop_mult = csharp_float(exit_rule.get("stop_atr_multiple", 1.5), default=1.5)
        rr = csharp_float(exit_rule.get("reward_risk", 2.0), default=2.0)

        on_entry_init_code = f'''
                double atr = {atr_expr};
                double stopDist = atr * {stop_mult};
                double targetDist = stopDist * {rr};
                if (isLongEntry)
                {{
                    SetStopLoss(EntrySignalName, CalculationMode.Price, Close[0] - stopDist, false);
                    SetProfitTarget(EntrySignalName, CalculationMode.Price, Close[0] + targetDist);
                }}
                else
                {{
                    SetStopLoss(EntrySignalName, CalculationMode.Price, Close[0] + stopDist, false);
                    SetProfitTarget(EntrySignalName, CalculationMode.Price, Close[0] - targetDist);
                }}'''

        on_position_exit_code = '''
            if (Position.MarketPosition != MarketPosition.Flat && CurrentBar - entryBar >= MaxBarsInTrade)
            {
                if (Position.MarketPosition == MarketPosition.Long)
                    ExitLong("TimeStop", EntrySignalName);
                else if (Position.MarketPosition == MarketPosition.Short)
                    ExitShort("TimeStop", EntrySignalName);
            }'''
        return on_entry_init_code, on_position_exit_code

    raise ValueError(f"Unsupported exit type: {exit_type}")


# ------------------------------------------------------------
# C# generation
# ------------------------------------------------------------

def generate_csharp(spec: dict[str, Any]) -> str:
    class_name = spec["nt8_class_name"]
    strategy_id = spec["strategy_id"]
    deployment = spec["deployment"]
    regime_filter = spec["regime_filter"]
    entry_rule = spec["entry_rule"]
    filters = [f for f in spec.get("filters", []) if f.get("type") != "regime_family_in"]
    exit_rule = spec["exit_rule"]
    performance = spec.get("performance", {})
    allocation = spec.get("allocation", {})
    costs = spec.get("costs", {})

    quantity = int(deployment.get("quantity", 1))
    instrument_type = deployment.get("instrument_type", "NQ")
    side = entry_rule.get("side", "long").lower()
    max_bars = int(exit_rule.get("max_bars_in_trade", 390))
    bars_required = max(max_bars_required(spec), 20)
    slippage_ticks = int(float(costs.get("slippage_ticks", 3)))

    entry_signal = cross_code(entry_rule)
    filter_parts = [filter_code(f) for f in filters]
    filter_signal = " && ".join(filter_parts) if filter_parts else "true"
    regime_code = generate_regime_code(regime_filter)
    on_entry_init_code, on_position_exit_code = generate_exit_code(exit_rule, entry_rule, side)

    if side == "short":
        enter_call = "EnterShort(Quantity, EntrySignalName);"
    else:
        enter_call = "EnterLong(Quantity, EntrySignalName);"

    scope = regime_filter.get("scope", "")
    key = regime_filter.get("key", "")

    return f'''#region Using declarations
using System;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.Gui.Tools;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{{
    public class {class_name} : Strategy
    {{
        private RegimeClassifierRB1 rb1;
        private int entryBar = -1;
        private const string EntrySignalName = "{csharp_string(strategy_id)}";

        [NinjaScriptProperty]
        public int Quantity {{ get; set; }} = {quantity};

        [NinjaScriptProperty]
        public int MaxBarsInTrade {{ get; set; }} = {max_bars};

        [NinjaScriptProperty]
        public bool EnableLongs {{ get; set; }} = {csharp_bool(side == 'long')};

        [NinjaScriptProperty]
        public bool EnableShorts {{ get; set; }} = {csharp_bool(side == 'short')};

        // ------------------------------------------------------------
        // RegimeClassifierRB1 parameters
        // Defaults should match your validated NT8/Python regime parity baseline.
        // ------------------------------------------------------------
        [NinjaScriptProperty]
        public int SessionStartHour {{ get; set; }} = 17;

        [NinjaScriptProperty]
        public int PPMD1Len {{ get; set; }} = 1;

        [NinjaScriptProperty]
        public int PPMD2Len {{ get; set; }} = 2;

        [NinjaScriptProperty]
        public double PPMScale {{ get; set; }} = 100000.0;

        [NinjaScriptProperty]
        public bool UseCustomThresholds {{ get; set; }} = true;

        [NinjaScriptProperty]
        public double TrendQ10 {{ get; set; }} = -3.358386407960507;
        [NinjaScriptProperty]
        public double TrendQ35 {{ get; set; }} = 4.390053241846475;
        [NinjaScriptProperty]
        public double TrendQ65 {{ get; set; }} = 8.56895233999072;
        [NinjaScriptProperty]
        public double TrendQ90 {{ get; set; }} = 12.97583963222094;

        [NinjaScriptProperty]
        public double IntermediateQ10 {{ get; set; }} = -28.23291921593665;
        [NinjaScriptProperty]
        public double IntermediateQ35 {{ get; set; }} = 0.8379808724681627;
        [NinjaScriptProperty]
        public double IntermediateQ65 {{ get; set; }} = 16.213599436996578;
        [NinjaScriptProperty]
        public double IntermediateQ90 {{ get; set; }} = 32.778799330650294;

        [NinjaScriptProperty]
        public double AccelQ10 {{ get; set; }} = -24.14494403825941;
        [NinjaScriptProperty]
        public double AccelQ35 {{ get; set; }} = 1.8245415257303608;
        [NinjaScriptProperty]
        public double AccelQ65 {{ get; set; }} = 15.094298122346729;
        [NinjaScriptProperty]
        public double AccelQ90 {{ get; set; }} = 29.31547118263314;

        [NinjaScriptProperty]
        public double Ppm3Q10 {{ get; set; }} = -17.60865921550788;
        [NinjaScriptProperty]
        public double Ppm3Q35 {{ get; set; }} = 2.211054075311929;
        [NinjaScriptProperty]
        public double Ppm3Q65 {{ get; set; }} = 12.828722120514438;
        [NinjaScriptProperty]
        public double Ppm3Q90 {{ get; set; }} = 24.66872505913484;

        [NinjaScriptProperty]
        public bool EnableDebugPrints {{ get; set; }} = false;

        protected override void OnStateChange()
        {{
            if (State == State.SetDefaults)
            {{
                Name = "{class_name}";
                Description = "Auto-generated regime-aware strategy. Instrument={instrument_type}; Scope={scope}; Key={csharp_string(key)}; NPDD={performance.get('np_dd', '')}; Trades={performance.get('trade_count', '')}; AllocationScore={allocation.get('allocation_score', '')}";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 30;
                IsInstantiatedOnEachOptimizationIteration = false;
                BarsRequiredToTrade = {bars_required};
                Slippage = {slippage_ticks};
            }}
            else if (State == State.DataLoaded)
            {{
                rb1 = RegimeClassifierRB1(
                    SessionStartHour,
                    PPMD1Len,
                    PPMD2Len,
                    PPMScale,
                    UseCustomThresholds,
                    TrendQ10,
                    TrendQ35,
                    TrendQ65,
                    TrendQ90,
                    IntermediateQ10,
                    IntermediateQ35,
                    IntermediateQ65,
                    IntermediateQ90,
                    AccelQ10,
                    AccelQ35,
                    AccelQ65,
                    AccelQ90,
                    Ppm3Q10,
                    Ppm3Q35,
                    Ppm3Q65,
                    Ppm3Q90,
                    EnableDebugPrints
                );
            }}
        }}

        protected override void OnBarUpdate()
        {{
            if (CurrentBar < BarsRequiredToTrade)
                return;

{on_position_exit_code}

            if (Position.MarketPosition != MarketPosition.Flat)
                return;

            bool regimeOk = {regime_code};
            bool entrySignal = {entry_signal};
            bool filtersOk = {filter_signal};
            bool isLongEntry = "{side}" == "long";
            bool directionOk = (isLongEntry && EnableLongs) || (!isLongEntry && EnableShorts);

            if (regimeOk && entrySignal && filtersOk && directionOk)
            {{
                entryBar = CurrentBar;
{on_entry_init_code}
                {enter_call}
            }}
        }}

        // ------------------------------------------------------------
        // Regime accessors
        // ------------------------------------------------------------
        private int CurrentTrendState()
        {{
            return Convert.ToInt32(rb1.FrozenTrendStateSeries[0]);
        }}

        private int CurrentIntermediateState()
        {{
            return Convert.ToInt32(rb1.FrozenIntermediateStateSeries[0]);
        }}

        private int CurrentAccelState()
        {{
            return Convert.ToInt32(rb1.FrozenAccelStateSeries[0]);
        }}

        private string CurrentRegimeFamily()
        {{
            int code = Convert.ToInt32(rb1.FrozenRegimeCodeSeries[0]);
            return RegimeCodeToFamily(code);
        }}

        private string RegimeCodeToFamily(int code)
        {{
            // TODO: Replace this mapping with your exact RB1 code-to-family mapping before using family-based strategies.
            // Tuple-based generated strategies do not use this method.
            return code.ToString();
        }}

        // ------------------------------------------------------------
        // Helper indicators aligned with Python core.py
        // ------------------------------------------------------------
        private double ZLEMA(int period, int barsAgo)
        {{
            int lag = (int)((period - 1) / 2.0);
            int idx = barsAgo;
            int lagIdx = barsAgo + lag;
            if (CurrentBar < lagIdx)
                return EMA(Close, period)[barsAgo];

            // ZLEMA approximation: adjusted price = price + (price - price[lag])
            // For exact parity, replace with your validated NT8 ZLEMA indicator if available.
            double adjusted = Close[idx] + (Close[idx] - Close[lagIdx]);
            double emaBase = EMA(Close, period)[barsAgo];
            double priceBase = Close[barsAgo];
            return emaBase + (adjusted - priceBase);
        }}

        private double BollingerUpper(int period, double stdDev, int barsAgo)
        {{
            return Bollinger(Close, stdDev, period).Upper[barsAgo];
        }}

        private double BollingerLower(int period, double stdDev, int barsAgo)
        {{
            return Bollinger(Close, stdDev, period).Lower[barsAgo];
        }}

        private double ChoppinessIndexValue(int period, int barsAgo)
        {{
            double trSum = 0.0;
            double highestHigh = High[barsAgo];
            double lowestLow = Low[barsAgo];

            for (int i = barsAgo; i < barsAgo + period; i++)
            {{
                if (CurrentBar <= i + 1)
                    break;

                double prevClose = Close[i + 1];
                double tr = Math.Max(High[i] - Low[i], Math.Max(Math.Abs(High[i] - prevClose), Math.Abs(Low[i] - prevClose)));
                trSum += tr;
                highestHigh = Math.Max(highestHigh, High[i]);
                lowestLow = Math.Min(lowestLow, Low[i]);
            }}

            double range = highestHigh - lowestLow;
            if (range <= 0 || trSum <= 0)
                return 0;

            return 100.0 * Math.Log10(trSum / range) / Math.Log10(period);
        }}
    }}
}}
'''


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def load_specs(args: argparse.Namespace) -> list[Path]:
    if args.spec:
        return [args.spec]

    specs = sorted(SPEC_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if args.max_files > 0:
        specs = specs[: args.max_files]
    return specs


def generate(args: argparse.Namespace) -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    spec_paths = load_specs(args)

    if not spec_paths:
        raise FileNotFoundError(f"No specs found in {SPEC_DIR}")

    rows = []
    for spec_path in spec_paths:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        cs = generate_csharp(spec)
        class_name = spec["nt8_class_name"]
        out_path = GENERATED_DIR / f"{class_name}.cs"
        out_path.write_text(cs, encoding="utf-8")

        rows.append(
            {
                "strategy_id": spec.get("strategy_id", ""),
                "nt8_class_name": class_name,
                "spec_path": str(spec_path),
                "cs_path": str(out_path),
                "instrument_type": spec.get("deployment", {}).get("instrument_type", ""),
                "quantity": spec.get("deployment", {}).get("quantity", ""),
                "allocation_scope": spec.get("deployment", {}).get("allocation_scope", ""),
                "active_regime_key": spec.get("deployment", {}).get("active_regime_key", ""),
            }
        )
        print(f"Generated: {out_path}")

    with INDEX_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("NT8 strategy code generation complete.")
    print(f"Files generated: {len(rows):,}")
    print(f"Output folder:   {GENERATED_DIR}")
    print(f"Index:           {INDEX_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate NT8 .cs strategies from deployment specs.")
    parser.add_argument("--max-files", type=int, default=1, help="Max files to generate from latest specs. Use 0 for all.")
    parser.add_argument("--spec", type=Path, default=None, help="Specific JSON spec to generate")
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())

# python generate_nt8_strategy_code.py --max-files 1