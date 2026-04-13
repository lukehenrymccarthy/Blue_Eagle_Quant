"""
ingestion/fred_macro.py
-----------------------
Pulls macro time series from FRED for the macro factor.
Saves each series to data/macro/ and computes candidate signals.

We pull a wide basket first, then select the best signals in phase 2
using IC (information coefficient) testing against future returns.

FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html
fredapi docs: https://github.com/mortada/fredapi
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY")
OUTPUT_DIR   = Path("data/macro")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Candidate FRED series ─────────────────────────────────────────────────────
# Pull all of these, then IC-test to find the best 2–3 for your macro factor.
# Series ID     : Description
FRED_SERIES = {
    # Yield curve
    "T10Y2Y"      : "10Y-2Y Treasury spread (yield curve)",
    "T10Y3M"      : "10Y-3M Treasury spread",
    "DGS10"       : "10-year Treasury yield",
    "DFF"         : "Effective Fed Funds rate",

    # Growth / activity
    "INDPRO"      : "Industrial Production Index",
    "PAYEMS"      : "Total nonfarm payrolls",
    "UNRATE"      : "Unemployment rate",
    "ISRATIO"     : "Inventory-to-sales ratio (retail)",

    # Inflation
    "CPIAUCSL"    : "CPI all items (seasonally adjusted)",
    "CPILFESL"    : "Core CPI (ex food & energy)",
    "PCEPI"       : "PCE price index",

    # Leading indicators
    "UMCSENT"     : "University of Michigan consumer sentiment",
    "PERMIT"      : "Building permits (housing starts proxy)",
    "ICSA"        : "Initial jobless claims (weekly, use MoM change)",

    # Credit / risk
    "BAMLH0A0HYM2": "ICE BofA High Yield OAS (credit spreads)",
    "TEDRATE"     : "TED spread (LIBOR - T-bill, banking stress)",
}


def pull_all_series(start: str = "2000-01-01") -> pd.DataFrame:
    """
    Pull all candidate FRED series and align to a monthly frequency.
    Returns a DataFrame: index = month-end dates, columns = series IDs.
    """
    fred   = Fred(api_key=FRED_API_KEY)
    frames = {}

    for series_id, description in FRED_SERIES.items():
        print(f"  Pulling {series_id} — {description}")
        try:
            s = fred.get_series(series_id, observation_start=start)
            frames[series_id] = s
        except Exception as e:
            print(f"    [WARN] {series_id}: {e}")

    if not frames:
        raise RuntimeError("No FRED series could be pulled. Check your API key.")

    raw = pd.DataFrame(frames)
    raw.index = pd.to_datetime(raw.index)

    # Resample to month-end (use last available observation in each month)
    monthly = raw.resample("ME").last()
    return monthly


def compute_macro_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw FRED levels into stationary signals suitable for factor use.

    Transformations applied:
      - Month-over-month change (for levels like unemployment, CPI)
      - 3-month rate of change (for smoother trend signals)
      - Z-score over trailing 36-month window (normalizes across regimes)

    The output signals are what get tested against future returns in phase 2.
    """
    signals = pd.DataFrame(index=df.index)

    for col in df.columns:
        s = df[col].dropna()

        # MoM change
        signals[f"{col}_mom"]   = df[col].diff(1)

        # 3-month rate of change (% for levels, pp for rates)
        signals[f"{col}_3m"]    = df[col].pct_change(3) \
                                   if df[col].mean() > 5 \
                                   else df[col].diff(3)

        # Rolling z-score (how extreme is the current reading vs trailing 3yr?)
        roll_mean = df[col].rolling(36, min_periods=18).mean()
        roll_std  = df[col].rolling(36, min_periods=18).std()
        signals[f"{col}_zscore"] = (df[col] - roll_mean) / roll_std.replace(0, np.nan)

    # Yield curve: level is already a spread — add a regime flag
    if "T10Y2Y" in df.columns:
        signals["yield_curve_inverted"] = (df["T10Y2Y"] < 0).astype(int)

    # Credit spread: rising spreads = risk-off (negative for equities)
    if "BAMLH0A0HYM2" in df.columns:
        signals["hy_spread_widening"] = df["BAMLH0A0HYM2"].diff(3)

    return signals.sort_index()


def is_fresh(max_age_hours: float = 20) -> bool:
    """Return True if both output files exist and were written within max_age_hours."""
    import time
    for p in [OUTPUT_DIR / "fred_raw.parquet", OUTPUT_DIR / "fred_signals.parquet"]:
        if not p.exists():
            return False
        age_h = (time.time() - p.stat().st_mtime) / 3600
        if age_h > max_age_hours:
            return False
    return True


def run(start: str = "2000-01-01", force: bool = False):
    if not force and is_fresh():
        print("FRED data is already fresh (< 20 h old). Skipping pull. Use force=True to override.")
        return pd.read_parquet(OUTPUT_DIR / "fred_raw.parquet"), \
               pd.read_parquet(OUTPUT_DIR / "fred_signals.parquet")

    print("Pulling FRED macro series...")
    raw     = pull_all_series(start)
    signals = compute_macro_signals(raw)

    raw.to_parquet(OUTPUT_DIR / "fred_raw.parquet")
    signals.to_parquet(OUTPUT_DIR / "fred_signals.parquet")

    print(f"\nSaved raw series   → {OUTPUT_DIR / 'fred_raw.parquet'}")
    print(f"Saved signals      → {OUTPUT_DIR / 'fred_signals.parquet'}")
    print(f"\n{len(signals.columns)} candidate signals ready for IC testing.")
    return raw, signals


def load_macro_signals() -> pd.DataFrame:
    path = OUTPUT_DIR / "fred_signals.parquet"
    if not path.exists():
        raise FileNotFoundError("Run run() first to pull FRED data.")
    return pd.read_parquet(path)


def get_macro_at_date(rebalance_date: pd.Timestamp) -> pd.Series:
    """
    Return the most recent macro signal values available as of a rebalance date.
    FRED data has publication lags — we use the last available month-end
    observation strictly before the rebalance date.
    """
    signals = load_macro_signals()
    available = signals[signals.index <= rebalance_date]
    if available.empty:
        return pd.Series(dtype=float)
    return available.iloc[-1]  # most recent available row


if __name__ == "__main__":
    run()
