# ============================================================
# Daily Discovery Runner - FINAL (Includes Allocation)
# ============================================================
# Full pipeline:
#   1. Generate strategies
#   2. Run Rust backtest
#   3. Analyze results
#   4. Build portfolio
#   5. Score allocation
#   6. Log run
# ============================================================

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\srobi\OneDrive\Documents\Data\regimeAwareDiscovery")
TOOLS_DIR = PROJECT_ROOT / "tools"
RUST_ENGINE_DIR = PROJECT_ROOT / "rust_engine"

STRATEGY_GENERATOR = TOOLS_DIR / "strategy_generator.py"
ANALYZE_RESULTS = TOOLS_DIR / "analyze_rust_results.py"
BUILD_PORTFOLIO = TOOLS_DIR / "build_regime_portfolio.py"
SCORE_ALLOCATION = TOOLS_DIR / "score_portfolio_allocation.py"


def run(cmd, cwd=None):
    print("\n=== RUNNING ===")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_args():
    p = argparse.ArgumentParser()

    # generation
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--include-regime-specialists", action="store_true")

    # allocation params (NEW)
    p.add_argument("--account-size", type=float, default=10000)
    p.add_argument("--risk-fraction", type=float, default=0.5)
    p.add_argument("--max-rows-per-regime", type=int, default=2)

    return p.parse_args()


def main():
    args = parse_args()

    print("\n===== DAILY DISCOVERY RUN =====")
    print("Time:", datetime.now())

    # 1. Generate
    run([
        sys.executable,
        str(STRATEGY_GENERATOR),
        "--batch-size", str(args.batch_size),
        "--include-regime-specialists"
    ], cwd=TOOLS_DIR)

    # 2. Rust
    run([
        "cargo", "run", "--release", "--",
        "--batch-json",
        sorted((PROJECT_ROOT / "04_rust_inputs").glob("strategy_batch_*.json"))[-1].as_posix()
    ], cwd=RUST_ENGINE_DIR)

    # 3. Analyze
    run([
        sys.executable,
        str(ANALYZE_RESULTS)
    ], cwd=TOOLS_DIR)

    # 4. Portfolio
    run([
        sys.executable,
        str(BUILD_PORTFOLIO)
    ], cwd=TOOLS_DIR)

    # 5. Allocation (NEW)
    run([
        sys.executable,
        str(SCORE_ALLOCATION),
        "--account-size", str(args.account_size),
        "--risk-fraction", str(args.risk_fraction),
        "--max-rows-per-regime", str(args.max_rows_per_regime)
    ], cwd=TOOLS_DIR)

    print("\n===== RUN COMPLETE =====")


if __name__ == "__main__":
    main()

# python run_daily_discovery.py --batch-size 100 --include-regime-specialists

# python run_daily_discovery.py --batch-size 100 --include-regime-specialists --account-size 25000 --risk-fraction 0.5 --max-rows-per-regime 2