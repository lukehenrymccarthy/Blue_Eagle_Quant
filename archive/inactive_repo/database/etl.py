"""
database/etl.py
----------------
ETL pipeline — loads all SectorScope source data into the DuckDB warehouse.

Run order (idempotent — safe to re-run; uses INSERT OR REPLACE):
  python database/etl.py [--table TABLE] [--reset]

Flags
-----
  --table  <name>   Load only one specific table (e.g. --table fact_returns)
  --reset           DROP and recreate all tables before loading

Database file:  data/sectorscope.duckdb   (~120 MB after full load)

Tables loaded (in dependency order)
------------------------------------
  dim_stock              from CRSP + Compustat SIC
  dim_date               generated from date range
  fact_returns           from CRSP monthly returns parquet
  fact_fundamentals      from Compustat quarterly parquet
  fact_analyst           from IBES signals parquet
  fact_macro             from FRED signals parquet
  fact_factor_scores     from five_factor_scores_latest.csv  (+ history)
  fact_etf_prices        from data/prices/*.parquet
  fact_etf_holdings      from data/etf_holdings/*_holdings.parquet
  fact_etf_scores        from data/etf_holdings/etf_summary.parquet
  fact_backtest_returns  from twenty_year_raw_returns.csv
  fact_backtest_metrics  from twenty_year_summary.csv + subperiod.csv
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
from datetime import date

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DB_PATH     = Path("data/sectorscope.duckdb")
SCHEMA_PATH = Path("database/schema.sql")


# ── Connection & schema ───────────────────────────────────────────────────────

def get_conn(reset: bool = False) -> duckdb.DuckDBPyConnection:
    if reset and DB_PATH.exists():
        DB_PATH.unlink()
        print("  [reset] Deleted existing database.")
    conn = duckdb.connect(str(DB_PATH))
    conn.execute("PRAGMA threads=4")
    conn.execute("PRAGMA memory_limit='1GB'")
    # Apply schema (CREATE TABLE IF NOT EXISTS — safe to re-run)
    sql = SCHEMA_PATH.read_text()
    # Strip single-line comments before splitting on ";" so that semicolons
    # inside comment text don't break CREATE TABLE statements.
    import re
    sql_clean = re.sub(r'--[^\n]*', '', sql)
    statements = [s.strip() for s in sql_clean.split(";") if s.strip()]
    for stmt in statements:
        try:
            conn.execute(stmt)
        except Exception as e:
            # Skip harmless errors (e.g. INSERT OR IGNORE on already-present rows)
            if "already exists" not in str(e).lower() and "duplicate" not in str(e).lower():
                print(f"  [WARN] schema: {e}")
    return conn


# ── Helpers ───────────────────────────────────────────────────────────────────

def upsert(conn: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame,
           pk: list[str]) -> int:
    """
    Insert-or-replace rows into `table` using DuckDB's INSERT OR REPLACE.
    df columns must match table columns exactly (extra columns are dropped,
    missing non-PK columns are filled with NULL).
    """
    if df.empty:
        return 0
    # Register df as a temporary view
    conn.register("_upsert_src", df)
    cols = ", ".join(df.columns)
    conn.execute(f"""
        INSERT OR REPLACE INTO {table} ({cols})
        SELECT {cols} FROM _upsert_src
    """)
    conn.unregister("_upsert_src")
    return len(df)


def run_step(name: str, fn, conn, only: str | None):
    if only and only != name:
        return
    print(f"\n  [{name}]")
    try:
        n = fn(conn)
        print(f"    ✓  {n:,} rows loaded")
    except Exception as e:
        print(f"    ✗  ERROR: {e}")
        import traceback; traceback.print_exc()


# ── Step loaders ─────────────────────────────────────────────────────────────

def load_dim_stock(conn) -> int:
    from ingestion.wrds_returns import load_returns
    from backtest.five_factor_model import _sic_to_etf

    print("    Loading CRSP returns for dim_stock...", end=" ", flush=True)
    crsp = load_returns()
    print(f"{len(crsp):,} rows")

    # Latest record per permno
    crsp["date"] = pd.to_datetime(crsp["date"])
    latest = crsp.sort_values("date").groupby("permno").last().reset_index()
    first  = crsp.sort_values("date").groupby("permno")["date"].min().reset_index()
    first.columns = ["permno", "first_obs_date"]

    dim = latest.merge(first, on="permno")
    dim["sector_etf"] = dim["siccd"].apply(
        lambda s: _sic_to_etf(s) if pd.notna(s) else None
    )
    dim = dim.rename(columns={
        "ticker":  "ticker",
        "exchcd":  "exchange_code",
        "shrcd":   "share_class",
        "siccd":   "sic_code",
        "date":    "last_obs_date",
    })
    dim["company_name"] = None    # CRSP msf doesn't carry names
    cols = ["permno","ticker","company_name","sic_code","sector_etf",
            "exchange_code","share_class","first_obs_date","last_obs_date"]
    return upsert(conn, "dim_stock", dim[cols], pk=["permno"])


def load_dim_date(conn) -> int:
    dates = pd.date_range("2005-01-01", "2025-12-31", freq="D")
    df = pd.DataFrame({"date_id": dates})
    df["year"]          = df["date_id"].dt.year.astype("int16")
    df["month"]         = df["date_id"].dt.month.astype("int16")
    df["quarter"]       = df["date_id"].dt.quarter.astype("int16")
    df["week_of_year"]  = df["date_id"].dt.isocalendar().week.astype("int16")
    df["is_month_end"]  = df["date_id"].isin(pd.date_range("2005-01-01","2025-12-31", freq="ME"))
    df["is_quarter_end"]= df["date_id"].isin(pd.date_range("2005-01-01","2025-12-31", freq="QE"))
    df["is_year_end"]   = df["date_id"].isin(pd.date_range("2005-01-01","2025-12-31", freq="YE"))
    return upsert(conn, "dim_date", df, pk=["date_id"])


def load_fact_returns(conn) -> int:
    """Load CRSP monthly returns — directly from parquet via DuckDB."""
    parquet = Path("data/fundamentals/crsp_monthly_returns.parquet")
    if not parquet.exists():
        # Fall back to Python loader if flat file not present
        from ingestion.wrds_returns import load_returns
        crsp = load_returns()
    else:
        crsp = pd.read_parquet(parquet)

    crsp = crsp.rename(columns={"shrout": "shares_out"})
    crsp["date"]       = pd.to_datetime(crsp["date"])
    crsp["market_cap"] = crsp["price"].abs() * crsp["shares_out"]

    cols = ["permno","date","ret","retx","price","shares_out","market_cap"]
    missing = [c for c in cols if c not in crsp.columns]
    for c in missing:
        crsp[c] = None

    return upsert(conn, "fact_returns", crsp[cols], pk=["permno","date"])


def load_fact_fundamentals(conn) -> int:
    p = Path("data/fundamentals/compustat_quarterly.parquet")
    fund = pd.read_parquet(p, columns=[
        "permno","datadate","rdq",
        "atq","ltq","ceqq","ibq","ibq_ttm","oancfq_ttm","capxq_ttm",
        "saleq","epspxq_ttm","roe_ttm","fcf_ttm","fcf_yield",
        "earnings_yield","pb_ratio","market_cap","avg_equity",
    ])
    fund = fund.rename(columns={
        "atq":        "total_assets",
        "ltq":        "total_liabilities",
        "ceqq":       "common_equity",
        "ibq":        "net_income_q",
        "ibq_ttm":    "net_income_ttm",
        "oancfq_ttm": "oper_cf_ttm",
        "capxq_ttm":  "capex_ttm",
        "saleq":      "revenue_q",
        "epspxq_ttm": "eps_ttm",
    })
    fund["datadate"] = pd.to_datetime(fund["datadate"])
    fund["rdq"]      = pd.to_datetime(fund["rdq"])
    fund = fund.dropna(subset=["permno","datadate"])
    fund["permno"]   = fund["permno"].astype(int)
    cols = ["permno","datadate","rdq","total_assets","total_liabilities",
            "common_equity","net_income_q","net_income_ttm","oper_cf_ttm",
            "capex_ttm","revenue_q","eps_ttm","roe_ttm","fcf_ttm",
            "fcf_yield","earnings_yield","pb_ratio","market_cap"]
    return upsert(conn, "fact_fundamentals", fund[cols], pk=["permno","datadate"])


def load_fact_analyst(conn) -> int:
    p = Path("data/analyst/ibes_signals.parquet")
    ibes = pd.read_parquet(p)
    # Note: ibes already has a computed 'buy_pct'; drop raw 'buypct' to avoid dup
    if "buypct" in ibes.columns and "buy_pct" in ibes.columns:
        ibes = ibes.drop(columns=["buypct"])
    ibes = ibes.rename(columns={
        "meanrec":          "mean_rec",
        "medrec":           "median_rec",
        "numrec":           "num_analysts",
        "numup":            "num_upgrades",
        "numdown":          "num_downgrades",
        "buypct":           "buy_pct",     # only used if computed buy_pct absent
        "holdpct":          "hold_pct",
        "sellpct":          "sell_pct",
        "net_upgrades":     "net_upgrades",
        "rev_3m":           "rev_3m",
        "buy_pct_rev_3m":   "buy_pct_rev_3m",
        "coverage_chg_3m":  "coverage_chg_3m",
    })
    ibes["statpers"]  = pd.to_datetime(ibes["statpers"])
    ibes["permno"]    = ibes["permno"].astype("Int64")
    ibes = ibes.dropna(subset=["permno","statpers"])
    cols = ["permno","statpers","ticker","mean_rec","median_rec",
            "num_analysts","num_upgrades","num_downgrades",
            "buy_pct","hold_pct","sell_pct","net_upgrades",
            "rev_3m","buy_pct_rev_3m","coverage_chg_3m"]
    missing = [c for c in cols if c not in ibes.columns]
    for c in missing:
        ibes[c] = None
    return upsert(conn, "fact_analyst", ibes[cols], pk=["permno","statpers"])


def load_fact_macro(conn) -> int:
    import numpy as np
    p       = Path("data/macro/fred_signals.parquet")
    macro   = pd.read_parquet(p)
    macro.index = pd.to_datetime(macro.index) + pd.offsets.MonthEnd(0)
    macro   = macro.reset_index().rename(columns={"index": "date",
                                                    "__index_level_0__": "date"})
    # Compute model macro factor z (rolling 36m, sign-flipped)
    if "hy_spread_widening" in macro.columns:
        s         = macro.set_index("date")["hy_spread_widening"]
        roll_mean = s.rolling(36, min_periods=12).mean()
        roll_std  = s.rolling(36, min_periods=12).std().replace(0, np.nan)
        macro_z   = -((s - roll_mean) / roll_std)
        macro["macro_factor_z"] = macro_z.values
    else:
        macro["macro_factor_z"] = np.nan

    col_map = {
        "T10Y2Y_3m":          "t10y2y_3m_chg",
        "T10Y2Y_zscore":      "t10y2y_zscore",
        "T10Y2Y_mom":         "t10y2y_spread",
        "DFF_mom":            "fed_funds_rate",
        "DFF_3m":             "fed_funds_3m_chg",
        "DGS10_mom":          "dgs10",
        "DGS10_3m":           "dgs10_3m_chg",
        "CPIAUCSL_mom":       "cpi_mom",
        "CPIAUCSL_3m":        "cpi_3m_chg",
        "CPIAUCSL_zscore":    "cpi_zscore",
        "UNRATE_mom":         "unrate",
        "UNRATE_3m":          "unrate_mom",
        "PAYEMS_mom":         "payems_mom",
        "BAMLH0A0HYM2_mom":   "hy_spread",
        "BAMLH0A0HYM2_3m":    "hy_spread_3m_chg",
        "BAMLH0A0HYM2_zscore":"hy_spread_zscore",
        "hy_spread_widening": "hy_spread_widening",
        "yield_curve_inverted":"yield_curve_inverted",
    }
    macro = macro.rename(columns=col_map)
    cols  = ["date","t10y2y_spread","t10y2y_3m_chg","t10y2y_zscore",
             "yield_curve_inverted","fed_funds_rate","fed_funds_3m_chg",
             "dgs10","dgs10_3m_chg","cpi_mom","cpi_3m_chg","cpi_zscore",
             "unrate","unrate_mom","payems_mom",
             "hy_spread","hy_spread_3m_chg","hy_spread_zscore",
             "hy_spread_widening","macro_factor_z"]
    for c in cols:
        if c not in macro.columns:
            macro[c] = None
    return upsert(conn, "fact_macro", macro[cols], pk=["date"])


def load_fact_factor_scores(conn) -> int:
    today = pd.Timestamp.today().normalize()
    p     = Path("data/results/five_factor_scores_latest.csv")
    df    = pd.read_csv(p, index_col=0)

    if df.index.dtype == object:
        df["ticker"]    = df.index.str.upper()
        df["score_date"] = today
    else:
        # backtest format with permno index
        df["permno"]    = df.index.astype(int)
        ibes = pd.read_parquet("data/analyst/ibes_signals.parquet",
                               columns=["permno","ticker","statpers"])
        ibes["statpers"] = pd.to_datetime(ibes["statpers"])
        tm = (ibes.sort_values("statpers").groupby("permno")["ticker"].last().str.upper())
        df["ticker"]    = tm.reindex(df["permno"]).values
        df["score_date"] = today

    df = df.dropna(subset=["ticker","composite"])
    cols = ["ticker","score_date","permno","momentum_18m","roe_ttm",
            "sector_rs_3m","analyst_rev_3m","macro_hy","composite"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return upsert(conn, "fact_factor_scores", df[cols], pk=["ticker","score_date"])


def load_fact_etf_prices(conn) -> int:
    price_dir = Path("data/prices")
    total = 0
    parquets = list(price_dir.glob("*.parquet"))
    print(f"    Found {len(parquets)} price parquet files", end="", flush=True)
    chunks = []
    for p in parquets:
        try:
            df = pd.read_parquet(p)
            df.index = pd.to_datetime(df.index)
            df = df.reset_index().rename(columns={"date": "date",
                                                   "index": "date",
                                                   "close": "close",
                                                   "volume": "volume"})
            if "ticker" not in df.columns:
                df["ticker"] = p.stem.upper()
            df = df[df["ticker"].str.match(r"^[A-Z]{1,5}$", na=False)]
            keep = ["ticker","date","close","volume"]
            for c in keep:
                if c not in df.columns:
                    df[c] = None
            chunks.append(df[keep])
        except Exception:
            pass
    if chunks:
        all_df = pd.concat(chunks, ignore_index=True)
        all_df["volume"] = pd.to_numeric(all_df["volume"], errors="coerce").astype(float)
        total = upsert(conn, "fact_etf_prices", all_df, pk=["ticker","date"])
    print()
    return total


def load_fact_etf_holdings(conn) -> int:
    etf_dir = Path("data/etf_holdings")
    today   = pd.Timestamp.today().normalize()
    chunks  = []
    factor_cols = ["momentum_18m","roe_ttm","sector_rs_3m","analyst_rev_3m","macro_hy"]

    for p in sorted(etf_dir.glob("*_holdings.parquet")):
        etf = p.stem.replace("_holdings","").upper()
        try:
            df = pd.read_parquet(p)
            df["etf_ticker"]    = etf
            df["as_of_date"]    = today
            df = df.rename(columns={"ticker":    "holding_ticker",
                                     "name":      "holding_name",
                                     "composite": "composite_score"})
            for c in factor_cols:
                if c not in df.columns:
                    df[c] = None
            cols = ["etf_ticker","as_of_date","holding_ticker","holding_name",
                    "weight","price","composite_score"] + factor_cols
            for c in cols:
                if c not in df.columns:
                    df[c] = None
            chunks.append(df[cols])
        except Exception as e:
            print(f"      [WARN] {etf}: {e}")

    if not chunks:
        return 0
    all_df = pd.concat(chunks, ignore_index=True)
    return upsert(conn, "fact_etf_holdings", all_df,
                  pk=["etf_ticker","as_of_date","holding_ticker"])


def load_fact_etf_scores(conn) -> int:
    p     = Path("data/etf_holdings/etf_summary.parquet")
    df    = pd.read_parquet(p)
    today = pd.Timestamp.today().normalize()
    df["score_date"] = today
    df = df.rename(columns={
        "etf":        "etf_ticker",
        "etf_score":  "weighted_composite",
        "coverage_w": "coverage_weight_pct",
        "n_holdings": "n_holdings",
        "n_scored":   "n_scored",
    })
    cols = ["etf_ticker","score_date","weighted_composite","score_z",
            "signal","n_holdings","n_scored","coverage_weight_pct"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return upsert(conn, "fact_etf_scores", df[cols],
                  pk=["etf_ticker","score_date"])


def load_fact_backtest_returns(conn) -> int:
    p  = Path("data/results/twenty_year_raw_returns.csv")
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df.index.name = "rebalance_date"
    long = df.reset_index().melt(id_vars="rebalance_date",
                                  var_name="strategy",
                                  value_name="period_return")
    long = long.dropna(subset=["period_return"])
    return upsert(conn, "fact_backtest_returns", long,
                  pk=["strategy","rebalance_date"])


def load_fact_backtest_metrics(conn) -> int:
    rows = []

    # Full-period summary
    p_sum = Path("data/results/twenty_year_summary.csv")
    if p_sum.exists():
        df = pd.read_csv(p_sum)
        df["period_label"] = "full"
        df["period_start"] = pd.Timestamp("2005-01-01")
        df["period_end"]   = pd.Timestamp("2024-12-31")
        rows.append(df)

    # Sub-period breakdown
    p_sub = Path("data/results/twenty_year_subperiod.csv")
    if p_sub.exists():
        sub = pd.read_csv(p_sub)
        if not sub.empty:
            sub = sub.rename(columns={
                "period":         "period_label",
                "model_ann_ret":  "ann_return",
                "model_sharpe":   "sharpe",
                "model_max_dd":   "max_drawdown",
            })
            sub["strategy"] = "ML-Optimised Weights"
            rows.append(sub)

    if not rows:
        return 0

    all_df = pd.concat(rows, ignore_index=True)
    metric_cols = ["ann_return","ann_vol","sharpe","max_drawdown","calmar",
                   "hit_rate","t_stat","total_return","n_periods"]
    for c in metric_cols:
        if c not in all_df.columns:
            all_df[c] = None

    cols = ["strategy","period_label","period_start","period_end"] + metric_cols
    for c in cols:
        if c not in all_df.columns:
            all_df[c] = None
    all_df = all_df[all_df["strategy"].notna()]
    return upsert(conn, "fact_backtest_metrics", all_df[cols],
                  pk=["strategy","period_label"])


# ── Main ─────────────────────────────────────────────────────────────────────

STEPS = [
    ("dim_stock",              load_dim_stock),
    ("dim_date",               load_dim_date),
    ("fact_returns",           load_fact_returns),
    ("fact_fundamentals",      load_fact_fundamentals),
    ("fact_analyst",           load_fact_analyst),
    ("fact_macro",             load_fact_macro),
    ("fact_factor_scores",     load_fact_factor_scores),
    ("fact_etf_prices",        load_fact_etf_prices),
    ("fact_etf_holdings",      load_fact_etf_holdings),
    ("fact_etf_scores",        load_fact_etf_scores),
    ("fact_backtest_returns",  load_fact_backtest_returns),
    ("fact_backtest_metrics",  load_fact_backtest_metrics),
]


def main():
    parser = argparse.ArgumentParser(description="SectorScope ETL → DuckDB")
    parser.add_argument("--table",  default=None,  help="Load only this table")
    parser.add_argument("--reset",  action="store_true",
                        help="Drop all data and reload from scratch")
    args = parser.parse_args()

    W = 60
    print("\n" + "═" * W)
    print("  SECTORSCOPE  —  ETL → DuckDB")
    print(f"  DB path : {DB_PATH}")
    print(f"  Schema  : {SCHEMA_PATH}")
    print("═" * W)

    conn = get_conn(reset=args.reset)

    for name, fn in STEPS:
        run_step(name, fn, conn, only=args.table)

    # ── Final stats ──────────────────────────────────────────────────────────
    print("\n" + "═" * W)
    print("  TABLE ROW COUNTS")
    print("═" * W)
    tables = [
        "dim_stock","dim_date","fact_returns","fact_fundamentals",
        "fact_analyst","fact_macro","fact_factor_scores","fact_etf_prices",
        "fact_etf_holdings","fact_etf_scores",
        "fact_backtest_returns","fact_backtest_metrics",
    ]
    for tbl in tables:
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            print(f"  {tbl:<35}  {n:>10,}")
        except Exception:
            print(f"  {tbl:<35}  {'(error)':>10}")

    import os
    size_mb = os.path.getsize(DB_PATH) / 1e6
    print(f"\n  Database size: {size_mb:.1f} MB  →  {DB_PATH}")
    print("═" * W + "\n")
    conn.close()


if __name__ == "__main__":
    main()
