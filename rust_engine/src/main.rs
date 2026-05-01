// ============================================================
// Rust Backtest Engine v1
// Regime-Aware Strategy Discovery Pipeline
// ============================================================
//
// Recommended project location:
// C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery\rust_engine
//
// Save this file as:
// rust_engine\src\main.rs
//
// Cargo.toml dependencies needed:
//
// [package]
// name = "regime_backtest_engine"
// version = "0.1.0"
// edition = "2021"
//
// [dependencies]
// anyhow = "1.0"
// chrono = { version = "0.4", features = ["serde"] }
// clap = { version = "4.5", features = ["derive"] }
// polars = { version = "0.43", features = ["lazy", "parquet", "csv", "strings", "dtype-categorical"] }
// serde = { version = "1.0", features = ["derive"] }
// serde_json = "1.0"
//
// Build:
//   cargo build --release
//
// Run:
//   cargo run --release -- --batch-json "C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery\04_rust_inputs\strategy_batch_20260430_094630.json"
//
// v1 supports:
//   Entry:
//     - cross_above
//     - cross_below
//   Filters:
//     - threshold
//     - comparison
//     - regime_family_in
//   Exits:
//     - time_stop
//     - opposite_cross
//     - fixed_rr_atr_stop
//
// Assumptions:
//   - Entry occurs at next bar open after signal confirmation.
//   - Regime is stamped from the entry bar only.
//   - One contract per strategy config initially.
//   - Long and short supported.
//   - Trades do not overlap within the same strategy.
// ============================================================

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(long)]
    batch_json: PathBuf,

    #[arg(long, default_value = "C:\\Users\\srobi\\OneDrive\\Documents\\Data\\regimeAwareDiscovery\\05_rust_outputs")]
    output_root: PathBuf,
}

#[derive(Debug, Deserialize)]
struct BatchPayload {
    feature_dataset_path: String,
    strategy_count: usize,
    strategies: Vec<StrategyConfig>,
}

#[derive(Debug, Deserialize, Clone)]
struct StrategyConfig {
    strategy_id: String,
    parameter_hash: String,
    instrument: String,
    timeframe: String,
    entry_rule: EntryRule,
    filters: Vec<FilterRule>,
    exit_rule: ExitRule,
    position: PositionConfig,
    costs: CostConfig,
    metadata: Option<HashMap<String, Value>>,
}

#[derive(Debug, Deserialize, Clone)]
struct EntryRule {
    #[serde(rename = "type")]
    rule_type: String,
    side: String,
    left: String,
    right: String,
    template: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
struct FilterRule {
    #[serde(rename = "type")]
    filter_type: String,
    column: Option<String>,
    operator: Option<String>,
    value: Option<f64>,
    left: Option<String>,
    right: Option<String>,
    values: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Clone)]
struct ExitRule {
    #[serde(rename = "type")]
    exit_type: String,
    max_bars_in_trade: Option<usize>,
    left: Option<String>,
    right: Option<String>,
    atr_column: Option<String>,
    stop_atr_multiple: Option<f64>,
    reward_risk: Option<f64>,
}

#[derive(Debug, Deserialize, Clone)]
struct PositionConfig {
    direction: String,
    contracts: i32,
}

#[derive(Debug, Deserialize, Clone)]
struct CostConfig {
    commission_round_trip: f64,
    slippage_ticks: f64,
    tick_size: f64,
    point_value: f64,
}

#[derive(Debug, Serialize, Clone)]
struct TradeRecord {
    strategy_id: String,
    parameter_hash: String,
    instrument: String,
    timeframe: String,
    side: String,
    entry_bar_index: i64,
    exit_bar_index: i64,
    entry_timestamp: String,
    exit_timestamp: String,
    entry_price: f64,
    exit_price: f64,
    contracts: i32,
    gross_pnl: f64,
    commission: f64,
    slippage_cost: f64,
    net_pnl: f64,
    bars_held: i64,
    exit_reason: String,
    entry_regime_family: String,
    entry_regime_tuple: String,
    entry_trend_state: i64,
    entry_intermediate_state: i64,
    entry_accel_state: i64,
}

#[derive(Debug, Serialize, Clone)]
struct PerformanceRecord {
    strategy_id: String,
    parameter_hash: String,
    scope: String,
    group_value: String,
    trade_count: usize,
    gross_profit: f64,
    gross_loss: f64,
    net_profit: f64,
    max_drawdown: f64,
    np_dd: f64,
    win_rate: f64,
    avg_trade: f64,
    profit_factor: f64,
}

struct MarketData {
    timestamp: Vec<String>,
    bar_index: Vec<i64>,
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    regime_family: Vec<String>,
    regime_tuple: Vec<String>,
    trend_state: Vec<i64>,
    intermediate_state: Vec<i64>,
    accel_state: Vec<i64>,
    numeric_cols: HashMap<String, Vec<f64>>,
    string_cols: HashMap<String, Vec<String>>,
}

impl MarketData {
    fn len(&self) -> usize {
        self.close.len()
    }

    fn f64_col(&self, name: &str) -> Result<&Vec<f64>> {
        self.numeric_cols
            .get(name)
            .ok_or_else(|| anyhow!("Missing numeric column: {}", name))
    }

    fn string_col(&self, name: &str) -> Result<&Vec<String>> {
        self.string_cols
            .get(name)
            .ok_or_else(|| anyhow!("Missing string column: {}", name))
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Reading batch JSON: {}", args.batch_json.display());
    let batch = read_batch(&args.batch_json)?;
    println!("Strategies in batch: {}", batch.strategy_count);
    println!("Feature dataset: {}", batch.feature_dataset_path);

    println!("Loading market data...");
    let market = load_market_data(Path::new(&batch.feature_dataset_path))?;
    println!("Rows loaded: {}", market.len());

    let trades_dir = args.output_root.join("trades");
    let perf_dir = args.output_root.join("performance");
    let family_dir = args.output_root.join("regime_family");
    let tuple_dir = args.output_root.join("regime_tuple");
    fs::create_dir_all(&trades_dir)?;
    fs::create_dir_all(&perf_dir)?;
    fs::create_dir_all(&family_dir)?;
    fs::create_dir_all(&tuple_dir)?;

    let mut all_trades: Vec<TradeRecord> = Vec::new();
    let mut all_overall_perf: Vec<PerformanceRecord> = Vec::new();
    let mut all_family_perf: Vec<PerformanceRecord> = Vec::new();
    let mut all_tuple_perf: Vec<PerformanceRecord> = Vec::new();

    for (idx, strategy) in batch.strategies.iter().enumerate() {
        if idx % 10 == 0 {
            println!("Evaluating strategy {}/{}: {}", idx + 1, batch.strategies.len(), strategy.strategy_id);
        }

        match evaluate_strategy(&market, strategy) {
            Ok(trades) => {
                let overall = summarize_performance(strategy, &trades, "overall", "ALL");
                let family = summarize_by_group(strategy, &trades, "regime_family");
                let tuple = summarize_by_group(strategy, &trades, "regime_tuple");

                all_overall_perf.push(overall);
                all_family_perf.extend(family);
                all_tuple_perf.extend(tuple);
                all_trades.extend(trades);
            }
            Err(e) => {
                eprintln!("FAILED strategy {}: {:?}", strategy.strategy_id, e);
            }
        }
    }

    let stamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();

    write_csv(&trades_dir.join(format!("trades_{}.csv", stamp)), &all_trades)?;
    write_csv(&perf_dir.join(format!("performance_overall_{}.csv", stamp)), &all_overall_perf)?;
    write_csv(&family_dir.join(format!("performance_regime_family_{}.csv", stamp)), &all_family_perf)?;
    write_csv(&tuple_dir.join(format!("performance_regime_tuple_{}.csv", stamp)), &all_tuple_perf)?;

    println!("Done.");
    println!("Trades: {}", all_trades.len());
    println!("Output root: {}", args.output_root.display());

    Ok(())
}

fn read_batch(path: &Path) -> Result<BatchPayload> {
    let file = File::open(path).with_context(|| format!("Could not open batch file: {}", path.display()))?;
    let reader = BufReader::new(file);
    let batch: BatchPayload = serde_json::from_reader(reader)?;
    Ok(batch)
}

fn load_market_data(path: &Path) -> Result<MarketData> {
    let file = File::open(path).with_context(|| format!("Could not open parquet: {}", path.display()))?;
    let df = ParquetReader::new(file).finish()?;

    let names: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();
    let mut numeric_cols: HashMap<String, Vec<f64>> = HashMap::new();
    let mut string_cols: HashMap<String, Vec<String>> = HashMap::new();

    for name in names.iter() {
        let s = df.column(name)?;
        match s.dtype() {
            DataType::Float64 | DataType::Float32 | DataType::Int64 | DataType::Int32 | DataType::UInt64 | DataType::UInt32 => {
                let casted = s.cast(&DataType::Float64)?;
                let ca = casted.f64()?;
                let vals: Vec<f64> = ca.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect();
                numeric_cols.insert(name.clone(), vals);
            }
            DataType::String => {
                let ca = s.str()?;
                let vals: Vec<String> = ca.into_iter().map(|v| v.unwrap_or("").to_string()).collect();
                string_cols.insert(name.clone(), vals);
            }
            _ => {}
        }
    }

    let timestamp = get_string_or_numeric_as_string(&df, "timestamp")?;
    let bar_index = get_i64_col(&df, "bar_index").unwrap_or_else(|_| (0..df.height() as i64).collect());
    let open = get_f64_col(&df, "open")?;
    let high = get_f64_col(&df, "high")?;
    let low = get_f64_col(&df, "low")?;
    let close = get_f64_col(&df, "close")?;
    let regime_family = get_string_col(&df, "regime_family")?;
    let regime_tuple = get_string_or_numeric_as_string(&df, "regime_tuple")?;
    let trend_state = get_i64_col(&df, "trend_state")?;
    let intermediate_state = get_i64_col(&df, "intermediate_state")?;
    let accel_state = get_i64_col(&df, "accel_state")?;

    Ok(MarketData {
        timestamp,
        bar_index,
        open,
        high,
        low,
        close,
        regime_family,
        regime_tuple,
        trend_state,
        intermediate_state,
        accel_state,
        numeric_cols,
        string_cols,
    })
}

fn get_f64_col(df: &DataFrame, name: &str) -> Result<Vec<f64>> {
    let s = df.column(name)?.cast(&DataType::Float64)?;
    let ca = s.f64()?;
    Ok(ca.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect())
}

fn get_i64_col(df: &DataFrame, name: &str) -> Result<Vec<i64>> {
    let s = df.column(name)?.cast(&DataType::Int64)?;
    let ca = s.i64()?;
    Ok(ca.into_iter().map(|v| v.unwrap_or(0)).collect())
}

fn get_string_col(df: &DataFrame, name: &str) -> Result<Vec<String>> {
    let s = df.column(name)?;
    let ca = s.str()?;
    Ok(ca.into_iter().map(|v| v.unwrap_or("").to_string()).collect())
}

fn get_string_or_numeric_as_string(df: &DataFrame, name: &str) -> Result<Vec<String>> {
    let s = df.column(name)?;
    match s.dtype() {
        DataType::String => get_string_col(df, name),
        _ => {
            let f = s.cast(&DataType::Float64)?;
            let ca = f.f64()?;
            Ok(ca
                .into_iter()
                .map(|v| match v {
                    Some(x) if x.fract() == 0.0 => format!("{}", x as i64),
                    Some(x) => format!("{}", x),
                    None => "".to_string(),
                })
                .collect())
        }
    }
}

fn evaluate_strategy(market: &MarketData, strategy: &StrategyConfig) -> Result<Vec<TradeRecord>> {
    let mut trades = Vec::new();
    let n = market.len();
    if n < 3 {
        return Ok(trades);
    }

    let side = strategy.entry_rule.side.to_lowercase();
    let contracts = strategy.position.contracts.max(1);
    let mut i: usize = 1;

    while i + 1 < n {
        if !passes_entry_signal(market, &strategy.entry_rule, i)? || !passes_filters(market, &strategy.filters, i)? {
            i += 1;
            continue;
        }

        let entry_i = i + 1;
        if entry_i >= n {
            break;
        }

        let entry_price = market.open[entry_i];
        if !entry_price.is_finite() {
            i += 1;
            continue;
        }

        let exit = find_exit(market, strategy, entry_i, entry_price, &side)?;
        let exit_i = exit.0;
        let exit_price = exit.1;
        let exit_reason = exit.2;

        let gross_pnl = calc_gross_pnl(entry_price, exit_price, &side, contracts, strategy.costs.point_value);
        let commission = strategy.costs.commission_round_trip * contracts as f64;
        let slippage_cost = strategy.costs.slippage_ticks * strategy.costs.tick_size * strategy.costs.point_value * contracts as f64 * 2.0;
        let net_pnl = gross_pnl - commission - slippage_cost;

        trades.push(TradeRecord {
            strategy_id: strategy.strategy_id.clone(),
            parameter_hash: strategy.parameter_hash.clone(),
            instrument: strategy.instrument.clone(),
            timeframe: strategy.timeframe.clone(),
            side: side.clone(),
            entry_bar_index: market.bar_index[entry_i],
            exit_bar_index: market.bar_index[exit_i],
            entry_timestamp: market.timestamp[entry_i].clone(),
            exit_timestamp: market.timestamp[exit_i].clone(),
            entry_price,
            exit_price,
            contracts,
            gross_pnl,
            commission,
            slippage_cost,
            net_pnl,
            bars_held: exit_i as i64 - entry_i as i64,
            exit_reason,
            entry_regime_family: market.regime_family[entry_i].clone(),
            entry_regime_tuple: market.regime_tuple[entry_i].clone(),
            entry_trend_state: market.trend_state[entry_i],
            entry_intermediate_state: market.intermediate_state[entry_i],
            entry_accel_state: market.accel_state[entry_i],
        });

        // No overlapping positions. Resume after exit.
        i = exit_i + 1;
    }

    Ok(trades)
}

fn passes_entry_signal(market: &MarketData, entry: &EntryRule, i: usize) -> Result<bool> {
    let left = market.f64_col(&entry.left)?;
    let right = market.f64_col(&entry.right)?;

    let l_prev = left[i - 1];
    let r_prev = right[i - 1];
    let l_now = left[i];
    let r_now = right[i];

    if !(l_prev.is_finite() && r_prev.is_finite() && l_now.is_finite() && r_now.is_finite()) {
        return Ok(false);
    }

    match entry.rule_type.as_str() {
        "cross_above" | "price_cross_above" => Ok(l_prev <= r_prev && l_now > r_now),
        "cross_below" | "price_cross_below" => Ok(l_prev >= r_prev && l_now < r_now),
        other => Err(anyhow!("Unsupported entry rule type: {}", other)),
    }
}

fn passes_filters(market: &MarketData, filters: &[FilterRule], i: usize) -> Result<bool> {
    for filter in filters {
        let ok = match filter.filter_type.as_str() {
            "threshold" => {
                let column = filter.column.as_ref().ok_or_else(|| anyhow!("threshold filter missing column"))?;
                let op = filter.operator.as_ref().ok_or_else(|| anyhow!("threshold filter missing operator"))?;
                let value = filter.value.ok_or_else(|| anyhow!("threshold filter missing value"))?;
                let col = market.f64_col(column)?;
                compare_f64(col[i], op, value)
            }
            "comparison" => {
                let left_name = filter.left.as_ref().ok_or_else(|| anyhow!("comparison filter missing left"))?;
                let right_name = filter.right.as_ref().ok_or_else(|| anyhow!("comparison filter missing right"))?;
                let op = filter.operator.as_ref().ok_or_else(|| anyhow!("comparison filter missing operator"))?;
                let left = market.f64_col(left_name)?;
                let right = market.f64_col(right_name)?;
                compare_f64(left[i], op, right[i])
            }
            "regime_family_in" => {
                let column = filter.column.as_deref().unwrap_or("regime_family");
                let values = filter.values.as_ref().ok_or_else(|| anyhow!("regime filter missing values"))?;
                let col = market.string_col(column)?;
                values.iter().any(|v| v == &col[i])
            }
            other => return Err(anyhow!("Unsupported filter type: {}", other)),
        };

        if !ok {
            return Ok(false);
        }
    }
    Ok(true)
}

fn compare_f64(left: f64, op: &str, right: f64) -> bool {
    if !(left.is_finite() && right.is_finite()) {
        return false;
    }
    match op {
        ">" => left > right,
        ">=" => left >= right,
        "<" => left < right,
        "<=" => left <= right,
        "==" => (left - right).abs() < 1e-12,
        "!=" => (left - right).abs() >= 1e-12,
        _ => false,
    }
}

fn find_exit(
    market: &MarketData,
    strategy: &StrategyConfig,
    entry_i: usize,
    entry_price: f64,
    side: &str,
) -> Result<(usize, f64, String)> {
    let n = market.len();
    let max_bars = strategy.exit_rule.max_bars_in_trade.unwrap_or(390).max(1);
    let max_exit_i = (entry_i + max_bars).min(n - 1);

    match strategy.exit_rule.exit_type.as_str() {
        "time_stop" => Ok((max_exit_i, market.close[max_exit_i], "time_stop".to_string())),
        "opposite_cross" => {
            let left_name = strategy.exit_rule.left.as_ref().ok_or_else(|| anyhow!("opposite_cross missing left"))?;
            let right_name = strategy.exit_rule.right.as_ref().ok_or_else(|| anyhow!("opposite_cross missing right"))?;
            let left = market.f64_col(left_name)?;
            let right = market.f64_col(right_name)?;

            for j in (entry_i + 1)..=max_exit_i {
                let l_prev = left[j - 1];
                let r_prev = right[j - 1];
                let l_now = left[j];
                let r_now = right[j];
                if !(l_prev.is_finite() && r_prev.is_finite() && l_now.is_finite() && r_now.is_finite()) {
                    continue;
                }

                let crossed = if side == "long" {
                    l_prev >= r_prev && l_now < r_now
                } else {
                    l_prev <= r_prev && l_now > r_now
                };

                if crossed {
                    let exit_i = (j + 1).min(n - 1);
                    return Ok((exit_i, market.open[exit_i], "opposite_cross".to_string()));
                }
            }
            Ok((max_exit_i, market.close[max_exit_i], "time_stop_after_no_opposite_cross".to_string()))
        }
        "fixed_rr_atr_stop" => {
            let atr_col = strategy.exit_rule.atr_column.as_deref().unwrap_or("atr_14");
            let atr = market.f64_col(atr_col)?[entry_i];
            if !atr.is_finite() || atr <= 0.0 {
                return Ok((max_exit_i, market.close[max_exit_i], "time_stop_invalid_atr".to_string()));
            }

            let stop_mult = strategy.exit_rule.stop_atr_multiple.unwrap_or(1.5);
            let rr = strategy.exit_rule.reward_risk.unwrap_or(2.0);
            let stop_dist = atr * stop_mult;
            let target_dist = stop_dist * rr;

            let (stop_price, target_price) = if side == "long" {
                (entry_price - stop_dist, entry_price + target_dist)
            } else {
                (entry_price + stop_dist, entry_price - target_dist)
            };

            for j in (entry_i + 1)..=max_exit_i {
                let high = market.high[j];
                let low = market.low[j];

                if side == "long" {
                    let hit_stop = low <= stop_price;
                    let hit_target = high >= target_price;
                    if hit_stop && hit_target {
                        // Conservative same-bar tie-break: assume stop first.
                        return Ok((j, stop_price, "stop_same_bar_tie".to_string()));
                    } else if hit_stop {
                        return Ok((j, stop_price, "stop".to_string()));
                    } else if hit_target {
                        return Ok((j, target_price, "target".to_string()));
                    }
                } else {
                    let hit_stop = high >= stop_price;
                    let hit_target = low <= target_price;
                    if hit_stop && hit_target {
                        return Ok((j, stop_price, "stop_same_bar_tie".to_string()));
                    } else if hit_stop {
                        return Ok((j, stop_price, "stop".to_string()));
                    } else if hit_target {
                        return Ok((j, target_price, "target".to_string()));
                    }
                }
            }

            Ok((max_exit_i, market.close[max_exit_i], "time_stop_after_no_stop_target".to_string()))
        }
        other => Err(anyhow!("Unsupported exit rule type: {}", other)),
    }
}

fn calc_gross_pnl(entry_price: f64, exit_price: f64, side: &str, contracts: i32, point_value: f64) -> f64 {
    let points = if side == "long" {
        exit_price - entry_price
    } else {
        entry_price - exit_price
    };
    points * point_value * contracts as f64
}

fn summarize_by_group(strategy: &StrategyConfig, trades: &[TradeRecord], group: &str) -> Vec<PerformanceRecord> {
    let mut map: HashMap<String, Vec<TradeRecord>> = HashMap::new();

    for t in trades {
        let key = match group {
            "regime_family" => t.entry_regime_family.clone(),
            "regime_tuple" => t.entry_regime_tuple.clone(),
            _ => "UNKNOWN".to_string(),
        };
        map.entry(key).or_default().push(t.clone());
    }

    let mut out = Vec::new();
    for (key, group_trades) in map {
        out.push(summarize_performance(strategy, &group_trades, group, &key));
    }
    out
}

fn summarize_performance(
    strategy: &StrategyConfig,
    trades: &[TradeRecord],
    scope: &str,
    group_value: &str,
) -> PerformanceRecord {
    let trade_count = trades.len();
    let mut gross_profit = 0.0;
    let mut gross_loss = 0.0;
    let mut net_profit = 0.0;
    let mut wins = 0usize;

    let mut equity = 0.0;
    let mut peak = 0.0;
    let mut max_drawdown = 0.0;

    for t in trades {
        let pnl = t.net_pnl;
        net_profit += pnl;
        equity += pnl;
        if pnl > 0.0 {
            gross_profit += pnl;
            wins += 1;
        } else if pnl < 0.0 {
            gross_loss += pnl.abs();
        }
        if equity > peak {
            peak = equity;
        }
        let dd = peak - equity;
        if dd > max_drawdown {
            max_drawdown = dd;
        }
    }

    let np_dd = if max_drawdown > 0.0 { net_profit / max_drawdown } else { 0.0 };
    let win_rate = if trade_count > 0 { wins as f64 / trade_count as f64 } else { 0.0 };
    let avg_trade = if trade_count > 0 { net_profit / trade_count as f64 } else { 0.0 };
    let profit_factor = if gross_loss > 0.0 { gross_profit / gross_loss } else if gross_profit > 0.0 { 999.0 } else { 0.0 };

    PerformanceRecord {
        strategy_id: strategy.strategy_id.clone(),
        parameter_hash: strategy.parameter_hash.clone(),
        scope: scope.to_string(),
        group_value: group_value.to_string(),
        trade_count,
        gross_profit,
        gross_loss,
        net_profit,
        max_drawdown,
        np_dd,
        win_rate,
        avg_trade,
        profit_factor,
    }
}

fn write_csv<T: Serialize>(path: &Path, records: &[T]) -> Result<()> {
    let mut file = File::create(path)?;
    let mut writer = csv::Writer::from_writer(&mut file);
    for r in records {
        writer.serialize(r)?;
    }
    writer.flush()?;
    println!("Wrote: {}", path.display());
    Ok(())
}
