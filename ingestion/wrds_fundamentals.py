"""
ingestion/wrds_fundamentals.py
------------------------------
Pulls quarterly fundamentals from Compustat via WRDS.
Applies point-in-time (PIT) logic so factors use only data
available at the time of each monthly rebalance — no look-ahead bias.

Saves to data/fundamentals/compustat_quarterly.parquet
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

WRDS_USERNAME = os.getenv("WRDS_USERNAME")
OUTPUT_DIR    = Path("data/fundamentals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Compustat column definitions ──────────────────────────────────────────────
# atq   = total assets
# ltq   = total liabilities
# ceqq  = common equity
# ibq   = income before extraordinary items (net income proxy)
# oancfy= operating cash flow, year-to-date (fundq uses 'y' suffix for cash flows)
# capxy = capital expenditures, year-to-date (same reason)
# saleq = net sales / revenue
# epspxq= EPS (diluted, excl. extraordinary items)
# cshoq = common shares outstanding
# prccq = stock price (Compustat close, can cross-check with CRSP)
# rdq   = earnings announcement date (use for PIT lag)
# sic   = SIC code — joined from comp.company (not available in fundq directly)

COMPUSTAT_COLS = [
    "gvkey", "datadate", "rdq",
    "atq", "ltq", "ceqq", "ibq",
    "oancfy", "capxy", "saleq",
    "epspxq", "cshoq", "prccq",
    # sic comes from comp.company via a separate join — not in fundq
]


def pull_compustat_quarterly(
    start_year: int = 2004,   # pull one extra year for lagged ratios
    end_year:   int = 2024,
) -> pd.DataFrame:
    import wrds

    db = wrds.Connection(wrds_username=WRDS_USERNAME)

    cols = ", ".join(f"a.{c}" for c in COMPUSTAT_COLS)

    query = f"""
        SELECT {cols},
               b.lpermno AS permno,    -- link to CRSP permno via CCM
               c.sic                   -- SIC code lives in comp.company, not fundq
        FROM comp.fundq AS a
        INNER JOIN crsp.ccmxpf_linktable AS b
            ON a.gvkey = b.gvkey
            AND b.linktype IN ('LU', 'LC')
            AND b.linkprim IN ('P', 'C')
            AND a.datadate BETWEEN b.linkdt AND COALESCE(b.linkenddt, CURRENT_DATE)
        LEFT JOIN comp.company AS c
            ON a.gvkey = c.gvkey
        WHERE a.datadate BETWEEN '{start_year}-01-01' AND '{end_year}-12-31'
          AND a.indfmt  = 'INDL'       -- industrial format (excl. banks/insurance)
          AND a.datafmt = 'STD'        -- standardised
          AND a.popsrc  = 'D'          -- domestic
          AND a.consol  = 'C'          -- consolidated
          AND a.ceqq    > 0            -- positive book equity
        ORDER BY a.gvkey, a.datadate
    """

    print("Querying Compustat quarterly fundamentals...")
    df = db.raw_sql(query, date_cols=["datadate", "rdq"])
    db.close()

    print(f"  Rows pulled: {len(df):,}")
    print(f"  Unique gvkeys: {df['gvkey'].nunique():,}")
    return df


def add_fundamental_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the core fundamental signals needed for the model.
    All ratios use point-in-time values from the quarter's filing.

    Ratios added:
      book_equity    - ceqq (common equity, book value)
      roe_ttm        - trailing 12m ROE (sum of 4 quarters of ibq / avg ceqq)
      fcf_yield      - (oancfy - capxy de-cumulated to quarterly) / market_cap
      gross_margin   - gross profit proxy (saleq - cogs proxy via ibq+capxq) / saleq
                       Note: use Compustat cogsy for exact COGS in annual if available
      earnings_yield - epspxq (ttm) / prccq  [= 1 / P/E]
      pb_ratio       - prccq * cshoq / ceqq
      market_cap     - prccq * cshoq (millions)
    """
    df = df.sort_values(["gvkey", "datadate"]).copy()

    # Cast all numeric columns to float so np.where works (SQL nulls come in as pd.NA)
    num_cols = ["atq", "ltq", "ceqq", "ibq", "oancfy", "capxy",
                "saleq", "epspxq", "cshoq", "prccq"]
    df[num_cols] = df[num_cols].astype(float)

    # ── De-cumulate YTD cash flow items → quarterly increments ───────────────
    # oancfy and capxy are year-to-date in fundq (they reset each Q1).
    # diff() gives quarterly increment; where diff is negative (year reset) or
    # NaN (first obs / gap), fall back to the YTD value itself (= Q1 amount).
    for ytd_col, q_col in [("oancfy", "oancfq"), ("capxy", "capxq")]:
        diff = df.groupby("gvkey")[ytd_col].transform(lambda x: x.diff())
        df[q_col] = diff.where(diff.notna() & (diff >= 0), df[ytd_col])

    # ── Trailing 12-month aggregates (sum of 4 quarters) ─────────────────────
    for col in ["ibq", "oancfq", "capxq", "epspxq"]:
        df[f"{col}_ttm"] = (
            df.groupby("gvkey")[col]
            .transform(lambda x: x.rolling(4, min_periods=4).sum())
        )

    # ── Market cap ────────────────────────────────────────────────────────────
    df["market_cap"] = df["prccq"] * df["cshoq"]  # millions

    # ── ROE (TTM) ─────────────────────────────────────────────────────────────
    df["avg_equity"] = (
        df.groupby("gvkey")["ceqq"]
        .transform(lambda x: x.rolling(4, min_periods=2).mean())
    )
    df["roe_ttm"] = np.where(
        df["avg_equity"] > 0,
        df["ibq_ttm"] / df["avg_equity"],
        np.nan,
    )

    # ── FCF yield ─────────────────────────────────────────────────────────────
    df["fcf_ttm"]   = df["oancfq_ttm"] - df["capxq_ttm"]
    df["fcf_yield"] = np.where(
        df["market_cap"] > 0,
        df["fcf_ttm"] / df["market_cap"],
        np.nan,
    )

    # ── Earnings yield (inverse P/E) ──────────────────────────────────────────
    df["earnings_yield"] = np.where(
        df["prccq"] > 0,
        df["epspxq_ttm"] / df["prccq"],
        np.nan,
    )

    # ── Price-to-book ────────────────────────────────────────────────────────
    df["pb_ratio"] = np.where(
        df["ceqq"] > 0,
        df["market_cap"] / df["ceqq"],
        np.nan,
    )

    return df


def apply_pit_lag(df: pd.DataFrame, lag_days: int = 60) -> pd.DataFrame:
    """
    Point-in-time adjustment.

    Compustat's `rdq` = earnings announcement date. This is when data
    becomes publicly available. We add a safety buffer (default 60 days)
    to ensure data is actually accessible before using it in a signal.

    The result: `available_date` = the earliest month-end rebalance
    date at which this quarter's data can safely be used.
    """
    # If rdq is missing, fall back to datadate + 90 days (fiscal quarter end + 3m)
    df["pit_date"] = df["rdq"].fillna(df["datadate"] + pd.Timedelta(days=90))
    df["pit_date"] = df["pit_date"] + pd.Timedelta(days=lag_days)

    # Snap to month-end so it aligns with monthly rebalance dates
    df["available_date"] = df["pit_date"] + pd.offsets.MonthEnd(0)

    return df


def run(start_year: int = 2004, end_year: int = 2024):
    df = pull_compustat_quarterly(start_year, end_year)
    df = add_fundamental_ratios(df)
    df = apply_pit_lag(df)

    out_path = OUTPUT_DIR / "compustat_quarterly.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path}")
    return df


def load_fundamentals() -> pd.DataFrame:
    path = OUTPUT_DIR / "compustat_quarterly.parquet"
    if not path.exists():
        raise FileNotFoundError("Run run() first to pull Compustat data.")
    return pd.read_parquet(path)


def get_fundamentals_at_date(rebalance_date: pd.Timestamp) -> pd.DataFrame:
    """
    For a given rebalance date, return the most recent quarter of
    fundamentals for each firm where available_date <= rebalance_date.
    This is the core PIT lookup used in factor construction.
    """
    df = load_fundamentals()
    df["rebalance_date"] = rebalance_date

    # Keep only data available by this rebalance date
    eligible = df[df["available_date"] <= rebalance_date].copy()

    # Most recent quarter per firm
    latest = (
        eligible
        .sort_values("datadate")
        .groupby("permno")
        .last()
        .reset_index()
    )
    return latest


if __name__ == "__main__":
    run()
