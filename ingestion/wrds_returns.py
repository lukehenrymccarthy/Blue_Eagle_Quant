"""
ingestion/wrds_returns.py
-------------------------
Pulls monthly total returns from CRSP via WRDS.
Saves to data/fundamentals/crsp_monthly_returns.parquet

WRDS Python docs: https://wrds-www.wharton.upenn.edu/pages/support/programming-wrds/programming-python/
"""

import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

WRDS_USERNAME = os.getenv("WRDS_USERNAME")
OUTPUT_DIR    = Path("data/fundamentals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def pull_crsp_monthly(
    start_year: int = 2005,
    end_year:   int = 2024,
) -> pd.DataFrame:
    """
    Pull monthly stock returns from CRSP msf (Monthly Stock File).

    Key columns returned:
      permno  - CRSP permanent identifier (stable across ticker changes)
      date    - month-end date
      ret     - total return (dividend-inclusive), decimal (0.05 = 5%)
      retx    - ex-dividend return
      prc     - price (negative = average of bid/ask when no trade)
      shrout  - shares outstanding (thousands)
      ticker  - ticker symbol at that date
      exchcd  - exchange code (1=NYSE, 2=AMEX, 3=NASDAQ)
      shrcd   - share code (use 10, 11 for common equity only)
    """
    import wrds  # import here so the file is importable without wrds installed

    db = wrds.Connection(wrds_username=WRDS_USERNAME)

    query = f"""
        SELECT
            a.permno,
            a.date,
            a.ret,
            a.retx,
            ABS(a.prc)         AS price,
            a.shrout,
            b.ticker,
            b.exchcd,
            b.shrcd,
            b.siccd
        FROM crsp.msf AS a
        LEFT JOIN crsp.msenames AS b
            ON a.permno = b.permno
            AND b.namedt <= a.date
            AND a.date  <= b.nameendt
        WHERE a.date BETWEEN '{start_year}-01-01' AND '{end_year}-12-31'
          AND b.exchcd IN (1, 2, 3)     -- NYSE, AMEX, NASDAQ only
          AND b.shrcd  IN (10, 11)       -- common equity only (exclude ADRs, ETFs, etc.)
          AND a.ret IS NOT NULL
        ORDER BY a.permno, a.date
    """

    print("Querying CRSP monthly stock file...")
    df = db.raw_sql(query, date_cols=["date"])
    db.close()

    print(f"  Rows pulled: {len(df):,}")
    print(f"  Unique permnos: {df['permno'].nunique():,}")
    print(f"  Date range: {df['date'].min()} → {df['date'].max()}")

    return df


def pull_crsp_delistings() -> pd.DataFrame:
    """
    Pull delisting returns from CRSP mse.
    Critical for avoiding survivorship bias — merge these into msf returns.
    When a stock is delisted, ret is often missing; this fills it in.
    """
    import wrds

    db = wrds.Connection(wrds_username=WRDS_USERNAME)

    query = """
        SELECT permno, dlstdt AS date, dlret AS ret_delist, dlstcd
        FROM crsp.msedelist
        WHERE dlret IS NOT NULL
    """
    df = db.raw_sql(query, date_cols=["date"])
    db.close()
    return df


def merge_delist_returns(returns: pd.DataFrame, delistings: pd.DataFrame) -> pd.DataFrame:
    """
    Merge delisting returns into the main return series.
    Standard academic practice: if dlret exists, replace missing ret with dlret.
    If both exist, compound them: (1+ret)*(1+dlret) - 1.
    """
    # Normalize delisting date to month-end to match msf
    delistings["date"] = delistings["date"] + pd.offsets.MonthEnd(0)

    merged = returns.merge(
        delistings[["permno", "date", "ret_delist"]],
        on=["permno", "date"],
        how="left",
    )

    mask_both    = merged["ret"].notna() & merged["ret_delist"].notna()
    mask_delist  = merged["ret"].isna()  & merged["ret_delist"].notna()

    merged.loc[mask_both,   "ret"] = (1 + merged.loc[mask_both, "ret"]) * \
                                     (1 + merged.loc[mask_both, "ret_delist"]) - 1
    merged.loc[mask_delist, "ret"] = merged.loc[mask_delist, "ret_delist"]
    merged.drop(columns="ret_delist", inplace=True)

    return merged


def run(start_year: int = 2005, end_year: int = 2024):
    returns    = pull_crsp_monthly(start_year, end_year)
    delistings = pull_crsp_delistings()
    returns    = merge_delist_returns(returns, delistings)

    out_path = OUTPUT_DIR / "crsp_monthly_returns.parquet"
    returns.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path}")
    return returns


def load_returns() -> pd.DataFrame:
    path = OUTPUT_DIR / "crsp_monthly_returns.parquet"
    if not path.exists():
        raise FileNotFoundError("Run run() first to pull CRSP data.")
    return pd.read_parquet(path)


def returns_wide(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Pivot to wide format: index=date, columns=permno, values=ret.
    Useful for cross-sectional factor calculations.
    """
    if df is None:
        df = load_returns()
    return df.pivot(index="date", columns="permno", values="ret")


if __name__ == "__main__":
    run()
