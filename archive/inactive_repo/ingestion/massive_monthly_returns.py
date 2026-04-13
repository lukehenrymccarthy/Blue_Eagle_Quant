"""
ingestion/massive_monthly_returns.py
-------------------------------------
Computes monthly returns from Massive daily price files and appends them
to data/fundamentals/crsp_monthly_returns.parquet, bridging the gap between
the last CRSP pull and the current date.

For each ticker in data/prices/, computes:
  monthly ret = last_close / first_close_of_month - 1

Maps ticker → permno using the most recent mapping in the existing CRSP file.
Skips months already present. Safe to re-run.

Usage:
    python ingestion/massive_monthly_returns.py                    # all available months
    python ingestion/massive_monthly_returns.py --start 2026-01    # from Jan 2026
    python ingestion/massive_monthly_returns.py --end   2026-02    # through Feb 2026
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

CRSP_PATH  = Path("data/fundamentals/crsp_monthly_returns.parquet")
PRICES_DIR = Path("data/prices")


def build_ticker_permno_map(crsp: pd.DataFrame) -> dict:
    """Most recent permno for each ticker from the CRSP file."""
    return (crsp.dropna(subset=["ticker"])
                .sort_values("date")
                .groupby("ticker")["permno"]
                .last()
                .to_dict())


def compute_monthly_returns(start_month: str, end_month: str) -> pd.DataFrame:
    """
    For each price file in data/prices/, compute monthly close-to-close returns
    for every full (or partial-final) month in [start_month, end_month].
    Returns a DataFrame with columns matching crsp_monthly_returns.
    """
    crsp     = pd.read_parquet(CRSP_PATH)
    crsp_max = pd.to_datetime(crsp["date"]).max()
    t2p      = build_ticker_permno_map(crsp)

    months = pd.period_range(start_month, end_month, freq="M")
    rows   = []

    for fpath in PRICES_DIR.glob("*.parquet"):
        ticker = fpath.stem
        permno = t2p.get(ticker)
        if permno is None:
            continue

        try:
            df = pd.read_parquet(fpath)
        except Exception:
            continue

        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        for period in months:
            month_start = period.start_time
            month_end   = period.end_time
            slice_      = df.loc[month_start:month_end, "close"].dropna()

            if len(slice_) < 5:          # skip sparse / missing months
                continue

            first_close = slice_.iloc[0]
            last_close  = slice_.iloc[-1]
            if first_close <= 0:
                continue

            ret        = last_close / first_close - 1
            date_label = slice_.index[-1] + pd.offsets.MonthEnd(0)

            # Skip if already in CRSP
            if date_label <= crsp_max:
                continue

            rows.append({
                "permno": int(permno),
                "date":   date_label,
                "ret":    round(ret, 6),
                "retx":   round(ret, 6),   # no dividend split available; use total ret
                "price":  round(last_close, 4),
                "shrout": np.nan,
                "ticker": ticker,
                "exchcd": np.nan,
                "shrcd":  10,
                "siccd":  np.nan,
            })

    if not rows:
        return pd.DataFrame()

    new_df = pd.DataFrame(rows)
    new_df["date"] = pd.to_datetime(new_df["date"])
    return new_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-01", help="First month to compute (YYYY-MM)")
    parser.add_argument("--end",   default="2026-03", help="Last month to compute (YYYY-MM)")
    args = parser.parse_args()

    print(f"\n  Computing monthly returns from Massive prices: {args.start} → {args.end}")

    new_df = compute_monthly_returns(args.start, args.end)
    if new_df.empty:
        print("  Nothing new to append (all months already in CRSP file or no price data).")
        return

    print(f"  Computed {len(new_df):,} rows | "
          f"{new_df['permno'].nunique():,} permnos | "
          f"{new_df['date'].min().date()} → {new_df['date'].max().date()}")

    existing = pd.read_parquet(CRSP_PATH)
    combined = (pd.concat([existing, new_df], ignore_index=True)
                  .sort_values(["permno", "date"])
                  .drop_duplicates(subset=["permno", "date"], keep="last"))
    combined.to_parquet(CRSP_PATH, index=False)
    added = len(combined) - len(existing)
    print(f"  +{added:,} rows appended → {CRSP_PATH}")
    print(f"  CRSP now covers: {pd.to_datetime(combined['date']).min().date()} "
          f"→ {pd.to_datetime(combined['date']).max().date()}")


if __name__ == "__main__":
    main()
