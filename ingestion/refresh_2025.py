"""
ingestion/refresh_2025.py
-------------------------
Pulls 2025 incremental data from WRDS and appends to existing parquets:
  - CRSP monthly returns    → data/fundamentals/crsp_monthly_returns.parquet
  - Compustat quarterly     → data/fundamentals/compustat_quarterly.parquet
  - IBES analyst signals    → data/analyst/ibes_signals.parquet

Uses the latest date already in each file as the start point so this
is safe to re-run (deduplicates on permno/date before saving).

Usage:
    python ingestion/refresh_2025.py
    python ingestion/refresh_2025.py --skip-compustat  # fastest if fundmentals unchanged
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WRDS_USERNAME = os.getenv("WRDS_USERNAME")

CRSP_PATH       = Path("data/fundamentals/crsp_monthly_returns.parquet")
COMPUSTAT_PATH  = Path("data/fundamentals/compustat_quarterly.parquet")
IBES_PATH       = Path("data/analyst/ibes_signals.parquet")


# ── helpers ───────────────────────────────────────────────────────────────────

def _latest_date(path: Path, date_col: str) -> str:
    if not path.exists():
        return "2005-01-01"
    df = pd.read_parquet(path, columns=[date_col])
    latest = pd.to_datetime(df[date_col]).max()
    # pull from the month after the latest we already have
    next_month = (latest + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d")
    return next_month


# ══════════════════════════════════════════════════════════════════════════════
# CRSP
# ══════════════════════════════════════════════════════════════════════════════

def refresh_crsp(start: str, end: str = "2025-12-31") -> int:
    """
    Pull 2025 CRSP returns from crsp.msf_v2 (new CIZ format, available through 2025).
    msf_v2 columns differ from legacy msf: mthret/mthcaldt/mthcap/shrout/siccd/ticker.
    Only pulls common equity on major exchanges (mirrors legacy msf filters).
    """
    import wrds

    print(f"\n  Pulling CRSP msf_v2: {start} → {end}")
    db = wrds.Connection(wrds_username=WRDS_USERNAME)

    query = f"""
        SELECT
            permno,
            mthcaldt          AS date,
            mthret            AS ret,
            mthretx           AS retx,
            mthprc            AS price,
            shrout,
            ticker,
            primaryexch       AS exchcd,
            securitysubtype   AS shrcd,
            siccd,
            mthcap            AS market_cap_v2
        FROM crsp.msf_v2
        WHERE mthcaldt BETWEEN '{start}' AND '{end}'
          AND primaryexch   IN ('N','A','Q')        -- NYSE, AMEX, NASDAQ
          AND securitytype  = 'EQTY'
          AND securitysubtype IN ('COM','ADRC')
          AND issuertype    = 'CORP'
          AND usincflg      = 'Y'
          AND mthret IS NOT NULL
        ORDER BY permno, mthcaldt
    """
    new_df = db.raw_sql(query, date_cols=["date"])

    # Map exchange letters to legacy exchcd integers for compatibility
    exch_map = {"N": 1, "A": 2, "Q": 3}
    new_df["exchcd"] = new_df["exchcd"].map(exch_map).fillna(new_df["exchcd"])
    # shrcd: flag common shares (COM=10, ADRC=11)
    new_df["shrcd"] = new_df["shrcd"].map({"COM": 10, "ADRC": 11}).fillna(10).astype(int)
    new_df.drop(columns=["market_cap_v2"], errors="ignore", inplace=True)

    # Delisting returns from msedelist (schema unchanged in v2)
    delist_q = f"""
        SELECT permno, dlstdt AS date, dlret AS ret_delist
        FROM crsp.msedelist
        WHERE dlret IS NOT NULL
          AND dlstdt BETWEEN '{start}' AND '{end}'
    """
    delistings = db.raw_sql(delist_q, date_cols=["date"])
    db.close()

    if not delistings.empty:
        delistings["date"] = pd.to_datetime(delistings["date"]) + pd.offsets.MonthEnd(0)
        new_df = new_df.merge(delistings[["permno","date","ret_delist"]],
                              on=["permno","date"], how="left")
        mask_both   = new_df["ret"].notna() & new_df["ret_delist"].notna()
        mask_delist = new_df["ret"].isna()  & new_df["ret_delist"].notna()
        new_df.loc[mask_both,   "ret"] = (1 + new_df.loc[mask_both,   "ret"]) * \
                                         (1 + new_df.loc[mask_both,   "ret_delist"]) - 1
        new_df.loc[mask_delist, "ret"] = new_df.loc[mask_delist, "ret_delist"]
        new_df.drop(columns="ret_delist", inplace=True)

    if new_df.empty:
        print("  Nothing new — msf_v2 returned no rows for this period.")
        return 0

    print(f"  Pulled {len(new_df):,} rows | "
          f"{new_df['permno'].nunique():,} permnos | "
          f"{new_df['date'].min().date()} → {new_df['date'].max().date()}")

    if CRSP_PATH.exists():
        existing = pd.read_parquet(CRSP_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.sort_values(["permno","date"]).drop_duplicates(
            subset=["permno","date"], keep="last")
        combined.to_parquet(CRSP_PATH, index=False)
        added = len(combined) - len(existing)
    else:
        new_df.to_parquet(CRSP_PATH, index=False)
        added = len(new_df)

    print(f"  +{added:,} new rows appended → {CRSP_PATH}")
    return added


# ══════════════════════════════════════════════════════════════════════════════
# COMPUSTAT
# ══════════════════════════════════════════════════════════════════════════════

def refresh_compustat(start: str, end: str = "2025-12-31") -> int:
    import wrds

    print(f"\n  Pulling Compustat fundq: {start} → {end}")
    db = wrds.Connection(wrds_username=WRDS_USERNAME)

    query = f"""
        SELECT
            a.gvkey, a.datadate, a.rdq,
            a.atq, a.ltq, a.ceqq, a.ibq,
            a.oancfy, a.capxy, a.saleq,
            a.epspxq, a.cshoq, a.prccq,
            b.lpermno AS permno,
            c.sic
        FROM comp.fundq AS a
        LEFT JOIN crsp.ccmxpf_lnkhist AS b
            ON a.gvkey = b.gvkey
            AND b.linktype IN ('LU','LC','LS')
            AND b.linkprim IN ('P','C')
            AND b.linkdt   <= a.datadate
            AND (b.linkenddt >= a.datadate OR b.linkenddt IS NULL)
        LEFT JOIN comp.company AS c
            ON a.gvkey = c.gvkey
        WHERE a.datafmt = 'STD'
          AND a.indfmt = 'INDL'
          AND a.consol = 'C'
          AND a.popsrc = 'D'
          AND a.datadate BETWEEN '{start}' AND '{end}'
          AND b.lpermno IS NOT NULL
        ORDER BY a.gvkey, a.datadate
    """
    new_df = db.raw_sql(query, date_cols=["datadate", "rdq"])
    db.close()

    if new_df.empty:
        print("  Nothing new — Compustat already up to date.")
        return 0

    # Rebuild derived columns (same logic as wrds_fundamentals.py)
    new_df["permno"] = pd.to_numeric(new_df["permno"], errors="coerce").dropna().astype(int)
    new_df = new_df.dropna(subset=["permno"])

    new_df["ibq_ttm"]    = new_df.groupby("gvkey")["ibq"].transform(lambda x: x.rolling(4).sum())
    new_df["oancfq_ttm"] = new_df.groupby("gvkey")["oancfy"].transform(lambda x: x.rolling(4).sum())
    new_df["roe_ttm"]    = new_df["ibq_ttm"] / new_df["ceqq"].replace(0, np.nan)

    mkt_cap              = new_df["prccq"].abs() * new_df["cshoq"] * 1e3
    new_df["market_cap"] = mkt_cap
    new_df["fcf_yield"]  = (new_df["ibq_ttm"] - (new_df["capxy"].fillna(0))) / mkt_cap.replace(0, np.nan)

    # Point-in-time available_date: rdq + 2 business days (SEC filing lag)
    new_df["available_date"] = new_df["rdq"].fillna(
        new_df["datadate"] + pd.offsets.MonthEnd(3)   # fallback: 3m after quarter end
    ) + pd.offsets.BDay(2)

    # rename sic to siccd for consistency with existing schema
    if "sic" in new_df.columns and "siccd" not in new_df.columns:
        new_df = new_df.rename(columns={"sic": "siccd"})

    print(f"  Pulled {len(new_df):,} rows | "
          f"{new_df['permno'].nunique():,} permnos | "
          f"{new_df['datadate'].min().date()} → {new_df['datadate'].max().date()}")

    if COMPUSTAT_PATH.exists():
        existing = pd.read_parquet(COMPUSTAT_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.sort_values(["gvkey","datadate"]).drop_duplicates(
            subset=["gvkey","datadate"], keep="last")
        combined.to_parquet(COMPUSTAT_PATH, index=False)
        added = len(combined) - len(existing)
    else:
        new_df.to_parquet(COMPUSTAT_PATH, index=False)
        added = len(new_df)

    print(f"  +{added:,} new rows appended → {COMPUSTAT_PATH}")
    return added


# ══════════════════════════════════════════════════════════════════════════════
# IBES
# ══════════════════════════════════════════════════════════════════════════════

def refresh_ibes(start: str, end: str = "2025-12-31") -> int:
    import wrds

    print(f"\n  Pulling IBES: {start} → {end}")
    db = wrds.Connection(wrds_username=WRDS_USERNAME)

    # Get consensus recommendations and compute 3-month revision
    query = f"""
        SELECT
            a.ticker   AS ibes_ticker,
            a.statpers,
            a.meanrec,
            a.numrec,
            b.lpermno  AS permno
        FROM ibes.recdsum AS a
        LEFT JOIN crsp.ccmxpf_lnkhist AS c
            ON c.lpermno = b.lpermno   -- placeholder; use iclink below
        LEFT JOIN wrdsapps.ibcrsphist AS b
            ON a.ticker = b.ticker
            AND b.sdate <= a.statpers
            AND (b.edate >= a.statpers OR b.edate IS NULL)
        WHERE a.statpers BETWEEN '{start}' AND '{end}'
          AND b.lpermno IS NOT NULL
        ORDER BY a.ticker, a.statpers
    """

    query = f"""
        SELECT
            b.permno,
            a.statpers,
            a.meanrec
        FROM ibes.recdsum AS a
        INNER JOIN wrdsapps.ibcrsphist AS b
            ON a.ticker = b.ticker
            AND b.sdate <= a.statpers
            AND (b.edate >= a.statpers OR b.edate IS NULL)
        WHERE a.statpers BETWEEN '{start}' AND '{end}'
          AND b.permno IS NOT NULL
        ORDER BY b.permno, a.statpers
    """
    new_df = db.raw_sql(query, date_cols=["statpers"])
    db.close()

    if new_df.empty:
        print("  Nothing new — IBES already up to date.")
        return 0

    new_df["permno"] = pd.to_numeric(new_df["permno"], errors="coerce")
    new_df = new_df.dropna(subset=["permno"])
    new_df["permno"] = new_df["permno"].astype(int)

    # Compute 3-month revision (change in mean recommendation, inverted: lower = upgrade)
    new_df = new_df.sort_values(["permno","statpers"])
    new_df["rev_3m"] = -(new_df.groupby("permno")["meanrec"]
                         .transform(lambda x: x - x.shift(3)))

    print(f"  Pulled {len(new_df):,} rows | "
          f"{new_df['permno'].nunique():,} permnos | "
          f"{new_df['statpers'].min().date()} → {new_df['statpers'].max().date()}")

    if IBES_PATH.exists():
        existing = pd.read_parquet(IBES_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.sort_values(["permno","statpers"]).drop_duplicates(
            subset=["permno","statpers"], keep="last")
        combined.to_parquet(IBES_PATH, index=False)
        added = len(combined) - len(existing)
    else:
        new_df.to_parquet(IBES_PATH, index=False)
        added = len(new_df)

    print(f"  +{added:,} new rows appended → {IBES_PATH}")
    return added


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-compustat", action="store_true")
    parser.add_argument("--skip-ibes",      action="store_true")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  WRDS 2025 INCREMENTAL REFRESH")
    print("═" * 60)

    crsp_start = _latest_date(CRSP_PATH, "date")
    print(f"\n  CRSP last date in file:       {crsp_start} (pulling from here)")
    refresh_crsp(crsp_start, args.end)

    if not args.skip_compustat:
        comp_start = _latest_date(COMPUSTAT_PATH, "datadate")
        print(f"\n  Compustat last date in file:  {comp_start}")
        refresh_compustat(comp_start, args.end)

    if not args.skip_ibes:
        ibes_start = _latest_date(IBES_PATH, "statpers")
        print(f"\n  IBES last date in file:       {ibes_start}")
        refresh_ibes(ibes_start, args.end)

    print("\n" + "═" * 60)
    print("  Refresh complete. Now verify ranges:")
    for path, col in [(CRSP_PATH,"date"), (COMPUSTAT_PATH,"datadate"), (IBES_PATH,"statpers")]:
        if path.exists():
            df = pd.read_parquet(path, columns=[col])
            mn = pd.to_datetime(df[col]).min().date()
            mx = pd.to_datetime(df[col]).max().date()
            print(f"  {path.name:<45} {mn} → {mx}")
    print("═" * 60)


if __name__ == "__main__":
    main()
