# Regime-Aware Strategy Discovery and NT8 Review Pipeline

## 1. Overall Explanation of What This Pipeline Does

### Explanation

This pipeline is a research, strategy-discovery, portfolio-construction, and NinjaTrader review-preparation system.

The objective is to automatically generate many trading strategy ideas, evaluate them at scale using precomputed Python indicators and a Rust backtest engine, measure performance by regime, identify strategies that work well in specific market regimes, score them for allocation, and optionally generate NinjaTrader 8 strategy code and review packages for manual Strategy Analyzer testing.

The core principle is:

```text
NinjaTrader is the source of truth for regimes.
Python/Rust is the research and evaluation engine.
NinjaTrader Strategy Analyzer remains the manual validation gate before deployment.
```

The pipeline is designed to answer questions like:

```text
Which strategy performs best in Bull_Pullback?
Which strategy performs best in tuple (-2,-2,-2)?
Which strategies have Net Profit / Max Drawdown >= 2?
Which strategies should be traded with NQ vs MNQ sizing?
Which generated strategies deserve manual NinjaTrader validation?
```

It does **not** automatically deploy strategies live. Instead, it creates review-ready NinjaScript strategy files and review packages so the user can manually test them in NinjaTrader Strategy Analyzer.

### Future ChatGPT Prompt

```text
I am working with a Regime-Aware Strategy Discovery and NT8 Review Pipeline. The pipeline generates randomized trading strategies, evaluates them using Python-computed indicators and a Rust backtest engine, measures performance overall and by regime family/tuple, filters candidates by Net Profit / Max Drawdown, builds a regime-aware portfolio, scores allocation sizing, and optionally exports NinjaTrader 8 strategy code and review packages for manual Strategy Analyzer validation. NinjaTrader is the source of truth for regimes, while Python/Rust is the research engine. The system does not auto-deploy live; generated NinjaScript is manually reviewed and tested in NT8 before any deployment decision.
```

---

## 2. How the Pipeline Works

### Explanation

The pipeline works in stages:

```text
1. NT8 regime parquet + OHLCV data
2. Python feature generation
3. Random strategy generation
4. Rust backtesting
5. Candidate analysis
6. Regime portfolio construction
7. Allocation scoring
8. Optional NT8 strategy spec export
9. Optional NT8 C# generation
10. Optional NT8 manual review package build
```

The daily discovery process starts with a Rust-ready feature dataset:

```text
C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery\02_features\feature_dataset_core.parquet
```

This file contains:

```text
OHLCV
NT8 frozen regime columns
standard regime aliases
Python indicator features
```

The strategy generator creates randomized strategy configurations and stores them in a registry to avoid duplicate testing.

The Rust engine evaluates those strategies against the feature dataset. It outputs trades, overall performance, regime-family performance, and regime-tuple performance.

The analysis script filters candidates that pass thresholds such as:

```text
NP/DD >= 2
minimum trade count
positive net profit
positive max drawdown
```

The portfolio builder ranks the best strategies per regime family and regime tuple. The allocation script then calculates suggested NQ/MNQ sizing using:

```text
allowed_risk = account_size * risk_fraction
adjusted_drawdown = max_drawdown * drawdown_safety_factor
contracts = floor(allowed_risk / adjusted_drawdown)
```

Finally, optional NT8 scripts export deployment specs, generate NinjaScript `.cs` files, and build review packages.

### Future ChatGPT Prompt

```text
The pipeline works as follows: first, a canonical NT8-derived regime/market dataset is converted into a feature dataset with Python indicator columns. Then strategy_generator.py creates randomized strategy JSON configs while avoiding duplicates using a registry. The Rust engine evaluates those configs against the feature parquet and writes trade/performance outputs. analyze_rust_results.py filters strategies by Net Profit / Max Drawdown and trade count. build_regime_portfolio.py ranks the best strategies by regime family and tuple. score_portfolio_allocation.py calculates NQ/MNQ sizing. Optional NT8 scripts then export specs, generate NinjaScript strategies, and build review packages for manual Strategy Analyzer testing.
```

---

## 3. Which Scripts Are Run Each Day?

### Daily Core Script

The primary daily script is:

```text
C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery\tools\run_daily_discovery.py
```

A typical daily run is:

```powershell
cd "C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery\tools"
python run_daily_discovery.py --batch-size 500 --include-regime-specialists --account-size 25000 --risk-fraction 0.5 --max-rows-per-regime 2
```

The daily runner should execute the core workflow:

```text
1. strategy_generator.py
2. Rust engine
3. analyze_rust_results.py
4. build_regime_portfolio.py
5. score_portfolio_allocation.py
6. update daily_discovery_run_log.csv
```

The daily run is used when the user wants to discover more strategies and update the candidate/portfolio/allocation outputs.

### Daily Outputs

The daily process updates:

```text
03_strategy_registry\strategy_combinations.csv
03_strategy_registry\daily_discovery_run_log.csv
04_rust_inputs\strategy_batch_*.json
05_rust_outputs\trades\trades_*.csv
05_rust_outputs\performance\performance_overall_*.csv
05_rust_outputs\regime_family\performance_regime_family_*.csv
05_rust_outputs\regime_tuple\performance_regime_tuple_*.csv
06_candidates\top_strategy_candidates_*.csv
07_portfolio\regime_family_portfolio.csv
07_portfolio\regime_tuple_portfolio.csv
07_portfolio\strategy_coverage_summary.csv
08_allocation\strategy_allocation_*.csv
```

### Future ChatGPT Prompt

```text
The daily script is run_daily_discovery.py located at C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery\tools. It runs the daily research workflow: strategy generation, Rust backtest, result analysis, regime portfolio construction, allocation scoring, and run logging. A typical command is: python run_daily_discovery.py --batch-size 500 --include-regime-specialists --account-size 25000 --risk-fraction 0.5 --max-rows-per-regime 2. This updates strategy_combinations.csv, daily_discovery_run_log.csv, Rust output files, candidate files, portfolio files, and allocation files.
```

---

## 4. Which Scripts Are Run Only in Special Circumstances?

### Data/Feature Preparation Scripts

These are not run every day unless the base dataset or indicators change.

#### `build_core_feature_dataset.py`

Run when:

```text
The canonical NT8 parquet changes
New OHLCV/regime data is added
Core indicator logic changes
Feature columns need to be rebuilt
```

Input:

```text
C:\Users\srobi\OneDrive\Documents\Data\ninjatrader\regimeBuilder_NT8\nq_1min_with_nt8_regimes.parquet
```

Output:

```text
02_features\feature_dataset_core.parquet
02_features\feature_dataset_core_summary.csv
```

#### `plan_indicator_conversion.py`

Run when:

```text
The indicator manifest changes
New NinjaTrader indicators are added
The user wants to reprioritize indicator conversion
```

#### `indicator_conversion_scaffold.py`

Run when:

```text
Scanning NT8 indicators for the first time
Refreshing the indicator manifest
Adding new indicator source files
```

### NT8 Review Preparation Scripts

These are run only when the user wants to prepare specific candidates for manual NT8 Strategy Analyzer testing.

#### `export_nt8_strategy_specs.py`

Run when:

```text
The user wants to convert top allocation candidates into NT8 deployment specs
```

Example:

```powershell
python export_nt8_strategy_specs.py --max-specs 10 --only-enabled
```

#### `generate_nt8_strategy_code.py`

Run when:

```text
The user wants to generate NinjaScript .cs strategy files from specs
```

Example:

```powershell
python generate_nt8_strategy_code.py --max-files 10
```

#### `build_nt8_review_package.py`

Run when:

```text
The user wants a review-ready folder containing strategy.cs, spec.json, and review_summary.txt
```

Example:

```powershell
python build_nt8_review_package.py --max-packages 10
```

### Future ChatGPT Prompt

```text
Special-circumstance scripts include build_core_feature_dataset.py, which is only run when the base NT8 regime parquet or feature logic changes; indicator_conversion_scaffold.py and plan_indicator_conversion.py, which are run when managing indicator conversion; and the NT8 review scripts export_nt8_strategy_specs.py, generate_nt8_strategy_code.py, and build_nt8_review_package.py, which are only run when I want to prepare top candidates for manual NinjaTrader Strategy Analyzer testing. These are not part of every daily research run unless explicitly enabled.
```

---

## 5. Process Flow: How Data Flows From One Script to the Next

### Step 0 — Canonical NT8 Regime Dataset

Primary source:

```text
C:\Users\srobi\OneDrive\Documents\Data\ninjatrader\regimeBuilder_NT8\nq_1min_with_nt8_regimes.parquet
```

Contains:

```text
timestamp
open/high/low/close/volume
frozen_trend_state
frozen_intermediate_state
frozen_accel_state
frozen_regime_tuple
frozen_broad_regime_family
```

Objective:

```text
Use NinjaTrader-derived regimes as the source of truth.
```

---

### Step 1 — Build Core Feature Dataset

Script:

```text
tools\build_core_feature_dataset.py
```

Input:

```text
nq_1min_with_nt8_regimes.parquet
```

Output:

```text
02_features\feature_dataset_core.parquet
02_features\feature_dataset_core_summary.csv
```

Objective:

```text
Add Python indicator features such as SMA, EMA, WMA, ZLEMA, ATR, Bollinger Bands, and Choppiness Index.
```

---

### Step 2 — Generate Strategy Batch

Script:

```text
tools\strategy_generator.py
```

Inputs:

```text
02_features\feature_dataset_core.parquet
03_strategy_registry\strategy_combinations.csv
```

Outputs:

```text
04_rust_inputs\strategy_batch_YYYYMMDD_HHMMSS.json
03_strategy_registry\strategy_combinations.csv
```

Objective:

```text
Generate duplicate-safe randomized strategy configs.
```

---

### Step 3 — Run Rust Backtest Engine

Rust project:

```text
rust_engine
```

Input:

```text
04_rust_inputs\strategy_batch_*.json
02_features\feature_dataset_core.parquet
```

Outputs:

```text
05_rust_outputs\trades\trades_*.csv
05_rust_outputs\performance\performance_overall_*.csv
05_rust_outputs\regime_family\performance_regime_family_*.csv
05_rust_outputs\regime_tuple\performance_regime_tuple_*.csv
```

Objective:

```text
Evaluate strategies quickly across millions of bars and stamp each trade with entry-time regime.
```

---

### Step 4 — Analyze Rust Results

Script:

```text
tools\analyze_rust_results.py
```

Inputs:

```text
05_rust_outputs\performance\performance_overall_*.csv
05_rust_outputs\regime_family\performance_regime_family_*.csv
05_rust_outputs\regime_tuple\performance_regime_tuple_*.csv
```

Outputs:

```text
06_candidates\top_strategy_candidates_overall.csv
06_candidates\top_strategy_candidates_regime_family.csv
06_candidates\top_strategy_candidates_regime_tuple.csv
06_candidates\top_strategy_candidates_all.csv
06_candidates\candidate_summary.txt
```

Objective:

```text
Filter strategies using NP/DD and minimum trade count thresholds.
```

---

### Step 5 — Build Regime Portfolio

Script:

```text
tools\build_regime_portfolio.py
```

Inputs:

```text
06_candidates\top_strategy_candidates_regime_family.csv
06_candidates\top_strategy_candidates_regime_tuple.csv
03_strategy_registry\strategy_combinations.csv
```

Outputs:

```text
07_portfolio\regime_family_portfolio.csv
07_portfolio\regime_tuple_portfolio.csv
07_portfolio\strategy_coverage_summary.csv
07_portfolio\regime_portfolio_summary.txt
```

Objective:

```text
Rank the best strategies per regime family and regime tuple.
```

---

### Step 6 — Score Allocation

Script:

```text
tools\score_portfolio_allocation.py
```

Inputs:

```text
07_portfolio\regime_family_portfolio.csv
07_portfolio\regime_tuple_portfolio.csv
07_portfolio\strategy_coverage_summary.csv
```

Outputs:

```text
08_allocation\strategy_allocation_all.csv
08_allocation\strategy_allocation_by_family.csv
08_allocation\strategy_allocation_by_tuple.csv
08_allocation\strategy_allocation_summary.txt
```

Objective:

```text
Calculate allocation scores and suggested NQ/MNQ contract sizing.
```

---

### Step 7 — Optional NT8 Spec Export

Script:

```text
tools\export_nt8_strategy_specs.py
```

Inputs:

```text
08_allocation\strategy_allocation_all.csv
03_strategy_registry\strategy_combinations.csv
```

Outputs:

```text
09_nt8_deployment\strategy_specs\*.json
09_nt8_deployment\nt8_strategy_spec_index.csv
```

Objective:

```text
Create deployment specs from top allocation candidates.
```

---

### Step 8 — Optional NT8 Code Generation

Script:

```text
tools\generate_nt8_strategy_code.py
```

Input:

```text
09_nt8_deployment\strategy_specs\*.json
```

Output:

```text
09_nt8_deployment\generated_strategies\*.cs
09_nt8_deployment\generated_strategy_index.csv
```

Objective:

```text
Generate NinjaTrader 8 Strategy .cs files from specs.
```

---

### Step 9 — Optional NT8 Review Package

Script:

```text
tools\build_nt8_review_package.py
```

Inputs:

```text
09_nt8_deployment\strategy_specs\*.json
09_nt8_deployment\generated_strategies\*.cs
```

Outputs:

```text
09_nt8_deployment\review_packages\<strategy_class>\strategy.cs
09_nt8_deployment\review_packages\<strategy_class>\spec.json
09_nt8_deployment\review_packages\<strategy_class>\review_summary.txt
```

Objective:

```text
Create review-ready strategy packages for manual NT8 validation.
```

### Future ChatGPT Prompt

```text
The data flow is: NT8 regime parquet -> build_core_feature_dataset.py -> feature_dataset_core.parquet -> strategy_generator.py -> strategy_batch JSON + registry -> Rust engine -> trades/performance CSVs -> analyze_rust_results.py -> top candidate CSVs -> build_regime_portfolio.py -> regime portfolio CSVs -> score_portfolio_allocation.py -> allocation CSVs -> optional export_nt8_strategy_specs.py -> JSON specs -> optional generate_nt8_strategy_code.py -> NinjaScript .cs files -> optional build_nt8_review_package.py -> review folders for manual NT8 Strategy Analyzer testing.
```

---

## 6. Process for Converting Strategies to NinjaScript Code

### Explanation

The conversion process begins only after the research pipeline identifies a strategy worth reviewing. The user does not deploy automatically.

The conversion flow is:

```text
1. Use allocation output to select top candidates
2. Export JSON specs
3. Generate NT8 strategy code
4. Build review packages
5. Manually copy strategy.cs into NinjaTrader
6. Compile
7. Run Strategy Analyzer
8. Approve/reject manually
```

### Step 1 — Export NT8 Specs

Command:

```powershell
python export_nt8_strategy_specs.py --max-specs 10 --only-enabled
```

Optional safer MNQ preference:

```powershell
python export_nt8_strategy_specs.py --max-specs 10 --only-enabled --prefer-micro
```

This creates:

```text
09_nt8_deployment\strategy_specs\*.json
```

Each spec includes:

```text
strategy_id
regime filter
entry rule
filters
exit rule
performance metrics
allocation recommendation
NQ/MNQ quantity
```

---

### Step 2 — Generate NinjaScript Code

Command:

```powershell
python generate_nt8_strategy_code.py --max-files 10
```

This creates:

```text
09_nt8_deployment\generated_strategies\*.cs
```

Each generated strategy includes:

```text
RegimeClassifierRB1 integration
Regime tuple/family filter
Entry logic
Exit logic
Quantity
Slippage = 3
RB1 threshold parameters
```

---

### Step 3 — Build Review Packages

Command:

```powershell
python build_nt8_review_package.py --max-packages 10
```

This creates:

```text
09_nt8_deployment\review_packages\<strategy_class>\
```

Each folder contains:

```text
strategy.cs
spec.json
review_summary.txt
```

---

### Step 4 — Manual NinjaTrader Strategy Analyzer Process

For each package:

```text
1. Open review_summary.txt
2. Copy strategy.cs into NinjaTrader strategy folder
3. Compile in NinjaTrader
4. Confirm Quantity and Slippage
5. Run Strategy Analyzer on same instrument/data range
6. Compare results directionally against review_summary.txt
7. Approve/reject manually
```

### Future ChatGPT Prompt

```text
To convert winning strategies to NinjaScript, I first run export_nt8_strategy_specs.py to convert allocation rows into JSON specs. Then I run generate_nt8_strategy_code.py to generate NT8 .cs strategy files. Then I run build_nt8_review_package.py to create a folder for each strategy containing strategy.cs, spec.json, and review_summary.txt. I manually copy strategy.cs into NinjaTrader, compile it, run Strategy Analyzer, and compare trade count, net profit, drawdown, and NP/DD against the research summary. I do not automatically deploy live strategies from this pipeline.
```

---

## 7. Step-by-Step Instructions for the User

### A. First-Time or Data Refresh Workflow

Run when base NT8 data or feature logic changes.

```powershell
cd "C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery\tools"
python build_core_feature_dataset.py
```

Reason:

```text
Rebuild the feature dataset used by Rust.
```

---

### B. Daily Discovery Workflow

Run daily or whenever the user wants to discover more strategies.

```powershell
cd "C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery\tools"
python run_daily_discovery.py --batch-size 500 --include-regime-specialists --account-size 25000 --risk-fraction 0.5 --max-rows-per-regime 2
```

Reason:

```text
Generate new random strategies, evaluate them, update candidates, build portfolio, and score allocation.
```

---

### C. Smaller Test Run

Use when testing pipeline changes.

```powershell
python run_daily_discovery.py --batch-size 25 --include-regime-specialists --account-size 10000 --risk-fraction 0.5 --max-rows-per-regime 2
```

Reason:

```text
Quick validation after code changes.
```

---

### D. NT8 Review Preparation

Run when the user wants to manually test top candidates in NinjaTrader.

```powershell
python export_nt8_strategy_specs.py --max-specs 10 --only-enabled
python generate_nt8_strategy_code.py --max-files 10
python build_nt8_review_package.py --max-packages 10
```

Reason:

```text
Prepare review-ready NT8 strategy packages.
```

---

### E. Manual NT8 Testing

For each review package:

```text
1. Open 09_nt8_deployment\review_packages\<strategy_class>\review_summary.txt
2. Copy strategy.cs into NinjaTrader
3. Compile
4. Run Strategy Analyzer
5. Compare to expected research metrics
6. Approve/reject manually
```

Reason:

```text
Manual validation is the final gate before any simulated or live deployment.
```

### Future ChatGPT Prompt

```text
My user workflow is: run build_core_feature_dataset.py only when base data/features change; run run_daily_discovery.py daily to generate/evaluate strategies and update candidates/portfolio/allocation; run export_nt8_strategy_specs.py, generate_nt8_strategy_code.py, and build_nt8_review_package.py only when I want to prepare top candidates for manual NinjaTrader Strategy Analyzer testing. I use small batch sizes like 25 for testing pipeline changes and larger batch sizes like 500+ for daily discovery.
```

---

## 8. Recommended Current Operating Procedure

### Daily Research Command

```powershell
cd "C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery\tools"
python run_daily_discovery.py --batch-size 500 --include-regime-specialists --account-size 25000 --risk-fraction 0.5 --max-rows-per-regime 2
```

### Weekly or When Needed NT8 Review Package Command

```powershell
python export_nt8_strategy_specs.py --max-specs 10 --only-enabled
python generate_nt8_strategy_code.py --max-files 10
python build_nt8_review_package.py --max-packages 10
```

### Manual Validation

```text
Use the review packages to manually test top candidates in NinjaTrader Strategy Analyzer.
```

### Future ChatGPT Prompt

```text
My recommended current operating procedure is to run the daily discovery runner with a batch size such as 500, including regime specialists and using my chosen account/risk parameters. I do not generate NinjaScript for every daily result. Periodically, I export the top allocation candidates into NT8 specs, generate .cs files, and build review packages. Then I manually test those strategies in NinjaTrader Strategy Analyzer before deciding whether to deploy them.
```

---

## 9. Key Design Decisions

### Decision 1 — Separate Strategies Rather Than One Mega-Strategy

The pipeline is designed to generate one NinjaTrader strategy per winning strategy/regime combination.

Reason:

```text
Easier debugging
Cleaner Strategy Analyzer testing
Independent enable/disable
Clear attribution
Safer live/sim review process
```

### Decision 2 — Entry-Time Regime Only

All performance grouping and strategy filters use regime at entry time.

Reason:

```text
Matches the research logic and avoids changing trade attribution after entry.
```

### Decision 3 — Manual NT8 Validation Before Deployment

The pipeline does not auto-deploy.

Reason:

```text
Generated code must be manually compiled, tested, and reviewed in NinjaTrader Strategy Analyzer.
```

### Decision 4 — Prefer MNQ When NQ Sizing Is Too Large

Allocation output determines whether the strategy should be tested as NQ or MNQ.

Reason:

```text
Many strategies have too much drawdown for one full NQ contract at smaller account sizes, but may be appropriate using MNQ.
```

### Future ChatGPT Prompt

```text
Important design decisions: generated strategies should be separate NT8 strategy files rather than one mega-strategy; regime filtering is entry-time only; generated NT8 code is manually validated before any deployment; allocation output determines whether NQ or MNQ sizing is appropriate; and NinjaTrader remains the source of truth for regimes while Python/Rust handles research and evaluation.
```

---

## 10. Future Improvements

Potential future improvements:

```text
1. Add more converted indicators beyond the core set.
2. Add strategy templates beyond crosses/Bollinger/choppiness.
3. Add correlation filtering across strategies.
4. Add walk-forward validation.
5. Add out-of-sample testing periods.
6. Add a manual approval registry for NT8-tested strategies.
7. Add an approved-strategy dashboard.
8. Add a comparison script for NT8 Strategy Analyzer exports vs Rust outputs.
9. Add family-code mapping for generated family-based strategies.
10. Add automatic generation of review packages only for top N allocation rows.
```

### Future ChatGPT Prompt

```text
Potential future improvements for this pipeline include converting more NinjaTrader indicators into Python, adding more strategy templates, adding correlation-aware portfolio selection, adding walk-forward/out-of-sample validation, maintaining a manual approval registry for NT8-tested strategies, building an approved-strategy dashboard, comparing NT8 Strategy Analyzer exports against Rust outputs, completing family-code mapping for family-based generated strategies, and optionally generating NT8 review packages only for the top N allocation rows.
```
