"""
scripts/daily_refresh.py
------------------------
Ordered daily data pipeline. Run this once per day (via cron or Docker).
Each step is idempotent: re-running on the same day is safe and fast because
each ingestion module checks file freshness before hitting an API.

Order matters:
  1. FRED macro      — no upstream dependencies
  2. Prices (Massive)— no upstream dependencies
  3. ETF holdings    — no upstream dependencies (pulls from SSGA)
  4. Live scoring    — depends on prices + cached CRSP/Compustat/IBES
  5. Score holdings  — depends on prices + cached CRSP/Compustat/IBES
  6. ETF scores      — depends on holdings + scores

WRDS (CRSP / Compustat / IBES) is pulled monthly — run refresh_2025.py
manually or add a separate monthly cron entry.

Usage:
    python scripts/daily_refresh.py
    python scripts/daily_refresh.py --force   # ignore freshness checks
"""

import sys
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STEPS = [
    {
        "name":    "FRED macro data",
        "module":  "ingestion.fred_macro",
        "fn":      "run",
        "kwargs":  {},                          # force kwarg added below if --force
        "skip_if": lambda: _parquet_fresh("data/macro/fred_raw.parquet", hours=20),
    },
    {
        "name":    "Massive daily prices",
        "module":  "ingestion.massive_prices",
        "fn":      "run",
        "kwargs":  {},
        "skip_if": None,                        # has its own per-ticker check
    },
    {
        "name":    "ETF holdings (SSGA)",
        "module":  "ingestion.etf_holdings",
        "fn":      "run",
        "kwargs":  {},
        "skip_if": lambda: _parquet_fresh("data/etf_holdings/etf_summary.parquet", hours=20),
    },
    {
        "name":    "Live scoring",
        "module":  "ingestion.live_scoring",
        "fn":      "run",
        "kwargs":  {},
        "skip_if": lambda: _csv_fresh("data/results/five_factor_scores_latest.csv", hours=20),
    },
    {
        "name":    "Score holdings",
        "module":  "ingestion.score_holdings",
        "fn":      "run",
        "kwargs":  {},
        "skip_if": None,
    },
]


def _file_age_hours(path: str) -> float:
    import time
    p = ROOT / path
    if not p.exists():
        return float("inf")
    return (time.time() - p.stat().st_mtime) / 3600


def _parquet_fresh(path: str, hours: float = 20) -> bool:
    return _file_age_hours(path) < hours


def _csv_fresh(path: str, hours: float = 20) -> bool:
    return _file_age_hours(path) < hours


def run_step(step: dict, force: bool = False) -> bool:
    name = step["name"]

    if not force and step.get("skip_if") and step["skip_if"]():
        log.info("%-30s  SKIP  (data is fresh)", name)
        return True

    log.info("%-30s  START", name)
    try:
        mod  = __import__(step["module"], fromlist=[step["fn"]])
        fn   = getattr(mod, step["fn"])
        kw   = dict(step["kwargs"])
        if force and "force" in fn.__code__.co_varnames:
            kw["force"] = True
        fn(**kw)
        log.info("%-30s  OK", name)
        return True
    except Exception as exc:
        log.error("%-30s  FAILED — %s", name, exc)
        return False


def main():
    parser = argparse.ArgumentParser(description="SectorScope daily data refresh")
    parser.add_argument("--force", action="store_true",
                        help="Ignore freshness checks and re-pull everything")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("SectorScope daily refresh  —  %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
    log.info("=" * 60)

    results = {}
    for step in STEPS:
        results[step["name"]] = run_step(step, force=args.force)

    log.info("=" * 60)
    ok  = sum(v for v in results.values())
    log.info("Complete: %d/%d steps succeeded", ok, len(results))
    for name, success in results.items():
        log.info("  %-30s  %s", name, "OK" if success else "FAILED")
    log.info("=" * 60)

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
