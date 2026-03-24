"""
ingestion/score_holdings.py
----------------------------
Re-score the full CRSP universe + any ETF holding tickers missing from CRSP
using the five-factor model at the most recent available date.

Improvements over a bare backtest run:
  • Fixes SIC mapping gaps (extended _sic_to_etf)
  • Fetches momentum from cached yfinance prices for tickers not in CRSP
  • Uses partial scoring (2+ factors) instead of strict 4-factor intersection
  • Refreshes ETF holdings parquet + DuckDB tables

Run:
    python ingestion/score_holdings.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.five_factor_model import (
    load_crsp_wide,
    build_compustat_panels,
    build_sector_rs_panel,
    build_analyst_panel,
    build_macro_factor,
    _zscore,
    _sic_to_etf,
)

RESULTS_DIR = Path("data/results")
PRICES_DIR  = Path("data/prices")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SECTOR_ETFS = ["XLC","XLY","XLP","XLE","XLF","XLV","XLI","XLB","XLRE","XLK","XLU","SPY"]
MIN_FACTORS = 2    # require at least this many non-macro factors to compute composite

# Factor weights: momentum de-emphasised to 5%; remaining 95% split equally across the other four
FACTOR_WEIGHTS = {
    "momentum_18m":   0.05,
    "roe_ttm":        0.2375,
    "sector_rs_3m":   0.2375,
    "analyst_rev_3m": 0.2375,
    "macro_hy":       0.2375,
}


def _momentum_from_yf_cache(tickers: list[str], last_date: pd.Timestamp) -> pd.Series:
    """
    Compute 18m-1m momentum from cached yfinance price parquet files
    in data/prices/ for tickers not in CRSP.
    """
    results = {}
    cutoff_18m = last_date - pd.DateOffset(months=19)   # 18m + 1m skip
    cutoff_1m  = last_date - pd.DateOffset(months=1)

    for t in tickers:
        path = PRICES_DIR / f"{t.upper()}.parquet"
        if not path.exists():
            continue
        try:
            px = pd.read_parquet(path)
            # Normalise column names
            px.columns = [c.lower() for c in px.columns]
            close_col = next((c for c in ["adj close","close","adjclose"] if c in px.columns), None)
            if close_col is None:
                continue
            px.index = pd.to_datetime(px.index)
            px = px.sort_index()[close_col].dropna()

            # Prices at start of 18m window and 1m ago
            px_18m = px.asof(cutoff_18m)
            px_1m  = px.asof(cutoff_1m)
            if pd.isna(px_18m) or pd.isna(px_1m) or px_18m == 0:
                continue
            results[t] = px_1m / px_18m - 1
        except Exception:
            pass
    return pd.Series(results, name="momentum_18m")


def _roe_for_tickers(tickers: list[str]) -> pd.Series:
    """Get latest ROE TTM from Compustat for a list of tickers."""
    crsp = pd.read_parquet(
        "data/fundamentals/crsp_monthly_returns.parquet",
        columns=["permno","ticker","date"]
    )
    crsp["ticker"] = crsp["ticker"].str.upper()
    crsp["date"] = pd.to_datetime(crsp["date"])
    ticker_to_permno = crsp.sort_values("date").groupby("ticker")["permno"].last()

    fund = pd.read_parquet(
        "data/fundamentals/compustat_quarterly.parquet",
        columns=["permno","datadate","roe_ttm","available_date"]
    )
    fund["available_date"] = pd.to_datetime(fund["available_date"])
    # Latest available ROE per permno
    latest_roe = (fund.sort_values("available_date")
                  .groupby("permno")["roe_ttm"].last())

    results = {}
    for t in tickers:
        perm = ticker_to_permno.get(t)
        if perm is None:
            continue
        roe = latest_roe.get(perm)
        if roe is not None and not pd.isna(roe):
            results[t] = roe
    return pd.Series(results, name="roe_ttm")


def _analyst_for_tickers(tickers: list[str]) -> pd.Series:
    """Get latest analyst rev_3m from IBES for a list of tickers."""
    ibes = pd.read_parquet(
        "data/analyst/ibes_signals.parquet",
        columns=["ticker","statpers","rev_3m"]
    )
    ibes["ticker"]   = ibes["ticker"].str.upper()
    ibes["statpers"] = pd.to_datetime(ibes["statpers"])
    latest_rev = (ibes.sort_values("statpers")
                  .groupby("ticker")["rev_3m"].last())
    return latest_rev.reindex(tickers).dropna().rename("analyst_rev_3m")


def _sector_rs_for_tickers(tickers: list[str], last_date: pd.Timestamp) -> pd.Series:
    """
    Compute sector RS signal for tickers by mapping their ticker to a sector ETF
    using yfinance-fetched ETF returns.
    """
    # Get SIC codes from Compustat
    crsp = pd.read_parquet(
        "data/fundamentals/crsp_monthly_returns.parquet",
        columns=["permno","ticker","date"]
    )
    crsp["ticker"] = crsp["ticker"].str.upper()
    crsp["date"] = pd.to_datetime(crsp["date"])
    ticker_to_permno = crsp.sort_values("date").groupby("ticker")["permno"].last()

    fund = pd.read_parquet(
        "data/fundamentals/compustat_quarterly.parquet",
        columns=["permno","datadate","sic"]
    )
    latest_sic = fund.sort_values("datadate").groupby("permno")["sic"].last()

    # ETF 3m RS vs SPY
    raw = yf.download(SECTOR_ETFS, period="6mo", interval="1mo",
                      auto_adjust=True, progress=False)["Close"]
    raw.index = pd.to_datetime(raw.index) + pd.offsets.MonthEnd(0)
    monthly_ret = raw.pct_change()
    spy_3m = (1 + monthly_ret["SPY"].fillna(0)).rolling(3).apply(np.prod, raw=True) - 1
    etf_rs = {}
    for etf in SECTOR_ETFS:
        if etf == "SPY" or etf not in monthly_ret.columns:
            continue
        etf_3m = (1 + monthly_ret[etf].fillna(0)).rolling(3).apply(np.prod, raw=True) - 1
        rs_val = (etf_3m - spy_3m).asof(last_date)
        if not pd.isna(rs_val):
            etf_rs[etf] = float(rs_val)

    results = {}
    for t in tickers:
        perm = ticker_to_permno.get(t)
        sic = latest_sic.get(perm) if perm else None
        etf = _sic_to_etf(sic) if sic else None
        if etf and etf in etf_rs:
            results[t] = etf_rs[etf]
    return pd.Series(results, name="sector_rs_3m")


def score_latest() -> pd.DataFrame:
    """Compute five-factor composite scores for all CRSP stocks + ETF holdings."""

    W = 65
    print("\n" + "═" * W)
    print("  FIVE-FACTOR SCORING  —  Full CRSP Universe")
    print("═" * W + "\n")

    # ── CRSP-based panels ─────────────────────────────────────────────────────
    print("  Loading CRSP monthly returns...")
    ret_wide = load_crsp_wide()
    last_date = ret_wide.index[-1]
    print(f"  {ret_wide.shape[1]:,} stocks | latest date: {last_date.date()}\n")

    print("  Loading Compustat (ROE TTM + SIC)...")
    roe_panel, sic_map = build_compustat_panels(ret_wide)
    print(f"  ROE panel: {roe_panel.shape[1]:,} stocks\n")

    print("  Computing sector relative-strength (yfinance)...")
    sector_rs = build_sector_rs_panel(ret_wide, sic_map)
    print(f"  Sector RS: {sector_rs.shape[1]:,} stocks mapped\n")

    print("  Loading IBES analyst revisions...")
    analyst_rev = build_analyst_panel(ret_wide)
    print(f"  Analyst Rev: {analyst_rev.shape[1]:,} stocks\n")

    macro_factor = build_macro_factor(ret_wide)
    macro_val = None
    if macro_factor is not None:
        macro_val = macro_factor.asof(last_date)
        sign = "bullish" if macro_val >= 0 else "bearish"
        print(f"  Macro HY z-score: {macro_val:+.3f} ({sign})\n")

    # ── Compute z-scores at latest date ───────────────────────────────────────
    mom_raw = (1 + ret_wide.fillna(0)).rolling(18).apply(np.prod, raw=True) - 1
    panels = {
        "momentum_18m":   mom_raw.shift(1),
        "roe_ttm":        roe_panel,
        "sector_rs_3m":   sector_rs,
        "analyst_rev_3m": analyst_rev,
    }

    print("  Computing cross-sectional z-scores at latest date...")
    raw_scores = {}    # fname → Series(permno → z)
    for fname, panel in panels.items():
        if panel.empty or last_date not in panel.index:
            continue
        row = panel.loc[last_date].dropna()
        # Cast index to plain int64 for consistent joining
        row.index = row.index.astype(np.int64)
        z = _zscore(row)
        if not z.empty:
            raw_scores[fname] = z
            print(f"    {fname:<20s} {len(z):,} stocks scored")

    # ── Partial-factor composite (permno-indexed) ─────────────────────────────
    all_permnos = set()
    for s in raw_scores.values():
        all_permnos.update(s.index)

    rows = []
    factor_names = list(raw_scores.keys())
    for perm in all_permnos:
        vals = {f: raw_scores[f].get(perm) for f in factor_names}
        present = {f: v for f, v in vals.items() if v is not None and not np.isnan(v)}
        if len(present) < MIN_FACTORS:
            continue
        rows.append({"permno": perm, **present})

    scores_df = pd.DataFrame(rows).set_index("permno")

    # Add macro as 5th factor
    if macro_val is not None and not pd.isna(macro_val):
        scores_df["macro_hy"] = float(macro_val)

    all_cols = [c for c in FACTOR_WEIGHTS if c in scores_df.columns]
    weights   = np.array([FACTOR_WEIGHTS[c] for c in all_cols])
    weights   = weights / weights.sum()   # re-normalise in case some factors absent
    scores_df["composite"] = scores_df[all_cols].fillna(0).values @ weights

    scores_df = scores_df.sort_values("composite", ascending=False)
    scores_df.index.name = "permno"
    print(f"\n  {len(scores_df):,} CRSP stocks scored (by permno)\n")

    # ── Convert permno index → ticker using CRSP (fallback to IBES) ──────────
    crsp_lk = pd.read_parquet(
        "data/fundamentals/crsp_monthly_returns.parquet",
        columns=["permno", "ticker", "date"]
    )
    crsp_lk["ticker"] = crsp_lk["ticker"].str.upper()
    crsp_lk["date"]   = pd.to_datetime(crsp_lk["date"])
    # Latest ticker per permno from CRSP
    crsp_ticker_map = (crsp_lk.sort_values("date")
                       .groupby("permno")["ticker"].last()
                       .str.upper())

    # IBES ticker as fallback for permnos CRSP doesn't have a recent ticker for
    ibes_lk = pd.read_parquet(
        "data/analyst/ibes_signals.parquet",
        columns=["permno", "ticker", "statpers"]
    )
    ibes_lk["ticker"]  = ibes_lk["ticker"].str.upper()
    ibes_lk["statpers"] = pd.to_datetime(ibes_lk["statpers"])
    ibes_ticker_map = (ibes_lk.sort_values("statpers")
                       .groupby("permno")["ticker"].last()
                       .str.upper())

    # Merge: CRSP first, IBES as fallback
    scores_df.index = scores_df.index.astype(np.int64)
    ticker_series = crsp_ticker_map.reindex(scores_df.index)
    missing_mask  = ticker_series.isna()
    ticker_series[missing_mask] = ibes_ticker_map.reindex(
        scores_df.index[missing_mask]
    ).values

    scores_df.index = ticker_series.values
    scores_df.index.name = "ticker"
    scores_df = scores_df[scores_df.index.notna()]
    # Drop duplicate tickers — keep highest composite
    scores_df = scores_df[~scores_df.index.duplicated(keep="first")]
    print(f"  Mapped to {len(scores_df):,} unique tickers\n")

    # ── Score ETF holdings not in CRSP via yfinance cache ─────────────────────
    print("  Scoring ETF-only holdings from yfinance price cache...")
    # Find all ETF holding tickers not covered by the CRSP-based scores
    all_etf_tickers = []
    for p in Path("data/etf_holdings").glob("*_holdings.parquet"):
        df_h = pd.read_parquet(p, columns=["ticker"])
        all_etf_tickers.extend(df_h["ticker"].str.upper().tolist())
    all_etf_tickers = list(set(all_etf_tickers))
    crsp_missing = [t for t in all_etf_tickers if t not in scores_df.index]
    print(f"  {len(crsp_missing)} ETF tickers still missing after CRSP scoring — trying yfinance cache\n")

    if crsp_missing:
        mom_yf    = _momentum_from_yf_cache(crsp_missing, last_date)
        roe_yf    = _roe_for_tickers(crsp_missing)
        ana_yf    = _analyst_for_tickers(crsp_missing)
        sec_yf    = _sector_rs_for_tickers(crsp_missing, last_date)

        extra_rows = []
        for t in crsp_missing:
            vals = {
                "momentum_18m":   mom_yf.get(t),
                "roe_ttm":        roe_yf.get(t),
                "sector_rs_3m":   sec_yf.get(t),
                "analyst_rev_3m": ana_yf.get(t),
            }
            present = {f: v for f, v in vals.items() if v is not None and not np.isnan(v)}
            if len(present) >= MIN_FACTORS:
                extra_rows.append({"ticker": t, **present})

        if extra_rows:
            extra_df = pd.DataFrame(extra_rows).set_index("ticker")
            # Z-score across the extra universe
            for col in ["momentum_18m","roe_ttm","sector_rs_3m","analyst_rev_3m"]:
                if col in extra_df.columns:
                    extra_df[col] = _zscore(extra_df[col])
            if macro_val is not None:
                extra_df["macro_hy"] = float(macro_val)
            ex_cols = [c for c in FACTOR_WEIGHTS if c in extra_df.columns]
            ex_w    = np.array([FACTOR_WEIGHTS[c] for c in ex_cols])
            ex_w    = ex_w / ex_w.sum()
            extra_df["composite"] = extra_df[ex_cols].fillna(0).values @ ex_w
            print(f"  Scored {len(extra_df)} CRSP-missing tickers from yfinance cache")

            # Save separately for etf_holdings.py (ticker-indexed)
            extra_path = RESULTS_DIR / "five_factor_scores_etf_extra.csv"
            extra_df.to_csv(extra_path)
        else:
            print("  No CRSP-missing tickers could be scored from cache")

    # ── Merge extra yfinance-cached scores and save (ticker-indexed) ──────────
    extra_path = RESULTS_DIR / "five_factor_scores_etf_extra.csv"
    # (extra_rows populated above — only if crsp_missing non-empty)
    if "extra_rows" in dir() and extra_rows:
        extra_df2 = pd.DataFrame(extra_rows).set_index("ticker")
        for col in ["momentum_18m","roe_ttm","sector_rs_3m","analyst_rev_3m"]:
            if col in extra_df2.columns:
                extra_df2[col] = _zscore(extra_df2[col])
        if macro_val is not None:
            extra_df2["macro_hy"] = float(macro_val)
        ex2_cols = [c for c in FACTOR_WEIGHTS if c in extra_df2.columns]
        ex2_w    = np.array([FACTOR_WEIGHTS[c] for c in ex2_cols])
        ex2_w    = ex2_w / ex2_w.sum()
        extra_df2["composite"] = extra_df2[ex2_cols].fillna(0).values @ ex2_w
        scores_df = pd.concat([scores_df, extra_df2[~extra_df2.index.isin(scores_df.index)]])
        print(f"  Added {len(extra_df2)} extra tickers → {len(scores_df):,} total")

    out = RESULTS_DIR / "five_factor_scores_latest.csv"
    scores_df.to_csv(out)
    print(f"\n  Saved → {out} ({len(scores_df):,} stocks)\n")

    return scores_df


def refresh_etf_holdings():
    """Re-run ETF holdings ingestion with updated scores."""
    print("  Re-running ETF holdings ingestion with updated scores...")
    from ingestion import etf_holdings
    etf_holdings.main()


def refresh_db_tables():
    """Re-load the three affected DuckDB tables."""
    print("\n  Refreshing DuckDB warehouse tables...")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from database.etl import (
        get_conn, run_step,
        load_fact_factor_scores,
        load_fact_etf_holdings,
        load_fact_etf_scores,
    )
    conn = get_conn()
    run_step("fact_factor_scores", load_fact_factor_scores, conn, only=None)
    run_step("fact_etf_holdings",  load_fact_etf_holdings,  conn, only=None)
    run_step("fact_etf_scores",    load_fact_etf_scores,    conn, only=None)
    conn.close()
    print("  Done.\n")


if __name__ == "__main__":
    score_latest()
    refresh_etf_holdings()
    refresh_db_tables()
