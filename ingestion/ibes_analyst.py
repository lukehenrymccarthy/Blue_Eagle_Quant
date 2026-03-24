"""
ingestion/ibes_analyst.py
--------------------------
Pulls IBES consensus analyst recommendations via WRDS and computes
momentum/revision signals for use as the analyst factor.

Source table: ibes.recdsum
  - Monthly consensus across all analysts covering each stock
  - IBES rating scale: 1=Strong Buy, 2=Buy, 3=Hold, 4=Underperform, 5=Sell

Signals computed:
  neg_meanrec      - negated mean recommendation (high = more bullish consensus)
  buy_pct          - % analysts with buy/strong buy rating
  net_upgrades     - numup - numdown (net upgrades in the period)
  rev_3m           - 3-month change in meanrec, negated (improvement = negative delta)
  buy_pct_rev_3m   - 3-month change in buy%, positive = getting more bullish
  coverage         - number of analysts covering (attention proxy)
  coverage_chg_3m  - 3-month change in coverage (growing attention = positive)
  neg_dispersion   - negated stdev of ratings (high disagreement = uncertainty = bad)

Usage:
    python ingestion/ibes_analyst.py
"""

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

WRDS_USERNAME = os.getenv("WRDS_USERNAME")
CRSP_PATH     = Path("data/fundamentals/crsp_monthly_returns.parquet")
OUTPUT_DIR    = Path("data/analyst")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 2004
END_YEAR   = 2024


# ─────────────────────────────────────────────────────────────────────────────
# PULL FROM WRDS
# ─────────────────────────────────────────────────────────────────────────────

def pull_ibes_recdsum(start_year: int = START_YEAR, end_year: int = END_YEAR) -> pd.DataFrame:
    """
    Pull monthly consensus recommendations from ibes.recdsum.
    Links to CRSP permno via 8-digit CUSIP match to crsp.msenames.ncusip.

    IBES rating scale: 1=Strong Buy → 5=Strong Sell
    numup/numdown: upgrades and downgrades issued in that month's summary period.
    """
    import wrds
    db = wrds.Connection(wrds_username=WRDS_USERNAME)

    query = f"""
        SELECT
            a.ticker,
            a.cusip,
            a.statpers,
            a.meanrec,
            a.medrec,
            a.stdev,
            a.numrec,
            a.numup,
            a.numdown,
            a.buypct,
            a.holdpct,
            a.sellpct,
            b.permno
        FROM ibes.recdsum AS a
        LEFT JOIN (
            SELECT DISTINCT
                SUBSTRING(ncusip, 1, 8) AS cusip8,
                permno
            FROM crsp.msenames
            WHERE shrcd IN (10, 11)
        ) AS b
            ON a.cusip = b.cusip8
        WHERE a.statpers BETWEEN '{start_year}-01-01' AND '{end_year}-12-31'
          AND a.usfirm  = 1
          AND a.numrec  >= 2
        ORDER BY a.cusip, a.statpers
    """

    print("Querying IBES consensus recommendations...")
    df = db.raw_sql(query, date_cols=["statpers"])
    db.close()

    print(f"  Rows pulled      : {len(df):,}")
    print(f"  Unique IBES tickers: {df['ticker'].nunique():,}")
    print(f"  Matched to permno: {df['permno'].notna().sum():,} rows ({df['permno'].notna().mean():.1%})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all analyst factor signals from raw IBES consensus data.
    Works at the (permno, statpers) level after deduplication.
    """
    # Drop rows with no permno link
    df = df.dropna(subset=["permno"]).copy()
    df["permno"]   = df["permno"].astype(int)
    df["statpers"] = pd.to_datetime(df["statpers"]) + pd.offsets.MonthEnd(0)

    # Cast numeric cols (SQL NULLs arrive as pd.NA)
    num_cols = ["meanrec", "medrec", "stdev", "numrec", "numup", "numdown",
                "buypct", "holdpct", "sellpct"]
    df[num_cols] = df[num_cols].astype(float)

    # One row per (permno, month) — some CUSIPs map to multiple permnos,
    # take the one with the most recent / most analyst coverage
    df = (
        df.sort_values(["permno", "statpers", "numrec"], ascending=[True, True, False])
        .drop_duplicates(subset=["permno", "statpers"])
    )

    df = df.sort_values(["permno", "statpers"])

    # ── Level signals ─────────────────────────────────────────────────────────
    # Negate meanrec: IBES 1=SB (bullish), 5=SS (bearish)
    # → high neg_meanrec = bullish consensus
    df["neg_meanrec"]    = -df["meanrec"]
    df["buy_pct"]        = df["buypct"]
    df["coverage"]       = df["numrec"]
    df["neg_dispersion"] = -df["stdev"]   # lower disagreement = more conviction

    # Net upgrades: positive = more upgrades than downgrades this month
    df["net_upgrades"] = df["numup"] - df["numdown"]

    # ── Revision / momentum signals (3-month change) ─────────────────────────
    grp = df.groupby("permno")

    # Change in mean rec over 3 months: negative = got more bullish (negate → positive signal)
    df["rev_3m"] = -grp["meanrec"].transform(lambda x: x.diff(3))

    # Change in buy% over 3 months: positive = analysts getting more bullish
    df["buy_pct_rev_3m"] = grp["buy_pct"].transform(lambda x: x.diff(3))

    # Change in analyst coverage over 3 months
    df["coverage_chg_3m"] = grp["coverage"].transform(lambda x: x.diff(3))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────────────────────────────────────

def run(start_year: int = START_YEAR, end_year: int = END_YEAR):
    raw = pull_ibes_recdsum(start_year, end_year)

    print("\nComputing analyst signals...")
    signals = compute_signals(raw)

    signal_cols = [
        "neg_meanrec", "buy_pct", "net_upgrades",
        "rev_3m", "buy_pct_rev_3m", "coverage", "coverage_chg_3m", "neg_dispersion",
    ]
    print(f"\n  Signal availability (non-null %):")
    for col in signal_cols:
        pct = signals[col].notna().mean()
        print(f"    {col:<22s} {pct:.1%}")

    out_path = OUTPUT_DIR / "ibes_signals.parquet"
    signals.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path}")
    print(f"  Rows   : {len(signals):,}")
    print(f"  Permnos: {signals['permno'].nunique():,}")
    print(f"  Dates  : {signals['statpers'].min().date()} → {signals['statpers'].max().date()}")
    return signals


def load_analyst_signals() -> pd.DataFrame:
    path = OUTPUT_DIR / "ibes_signals.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "Run ingestion/ibes_analyst.py first to pull IBES data."
        )
    return pd.read_parquet(path)


if __name__ == "__main__":
    run()
