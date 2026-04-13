"""
ingestion/live_scoring.py
--------------------------
Pull fresh price data from Massive (Polygon) and recompute the five-factor
composite scores using the most current information available.

Factors:
  F1  Momentum 18m-1m     — computed from Massive daily prices
  F2  ROE TTM             — Compustat (most recent available)
  F3  Sector RS vs SPY 3m — computed from Massive sector ETF prices
  F4  Analyst Rev 3m      — IBES (most recent available)
  F5  Macro HY            — FRED HY Spread Widening z-score

Output: data/results/five_factor_scores_latest.csv  (indexed by permno)

Usage:
    python ingestion/live_scoring.py
    python ingestion/live_scoring.py --skip-download   # use cached prices only
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.five_factor_model import _sic_to_etf, _zscore

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY     = os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")
PRICES_DIR  = Path("data/prices")
RESULTS_DIR = Path("data/results")
PRICES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SECTOR_ETFS = ["XLC", "XLY", "XLP", "XLE", "XLF",
               "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU", "SPY"]

# Pull 20 months of history (18m momentum + 1m skip + buffer)
PRICE_START = (date.today() - timedelta(days=620)).isoformat()
PRICE_END   = date.today().isoformat()

SLEEP_SEC   = 0.15   # between Massive requests


# ── Massive price fetcher ─────────────────────────────────────────────────────

def _fetch_one(client, ticker: str) -> pd.DataFrame:
    """Pull daily adjusted closes from Massive for one ticker."""
    from massive import RESTClient  # noqa — already imported via client
    aggs = []
    try:
        for agg in client.list_aggs(
            ticker=ticker, multiplier=1, timespan="day",
            from_=PRICE_START, to=PRICE_END,
            adjusted=True, sort="asc", limit=50000,
        ):
            aggs.append({
                "date":   pd.Timestamp(agg.timestamp, unit="ms").normalize(),
                "close":  agg.close,
                "volume": agg.volume,
            })
    except Exception as e:
        print(f"    [WARN] {ticker}: {e}")
        return pd.DataFrame()

    if not aggs:
        return pd.DataFrame()

    df = pd.DataFrame(aggs).set_index("date").sort_index()
    df["ticker"] = ticker
    return df


def pull_prices(tickers: list[str], skip_if_fresh: bool = True) -> None:
    """
    Download prices for every ticker in `tickers`, saving each to
    data/prices/{TICKER}.parquet.  Skips if the file was updated today.
    """
    from massive import RESTClient
    client = RESTClient(api_key=API_KEY)

    today = date.today()
    skipped = 0
    pulled  = 0
    failed  = 0

    for i, ticker in enumerate(tickers):
        out = PRICES_DIR / f"{ticker}.parquet"

        if skip_if_fresh and out.exists():
            mtime = pd.Timestamp(out.stat().st_mtime, unit="s").date()
            if mtime >= today - timedelta(days=1):   # skip if pulled yesterday or today
                skipped += 1
                continue

        df = _fetch_one(client, ticker)
        if not df.empty:
            df.to_parquet(out)
            pulled += 1
        else:
            failed += 1

        time.sleep(SLEEP_SEC)

        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{len(tickers)}]  pulled={pulled}  "
                  f"skipped={skipped}  failed={failed}")

    print(f"  Done: pulled={pulled}  skipped={skipped}  failed={failed}")


# ── Load prices → monthly returns ─────────────────────────────────────────────

def load_monthly_returns(tickers: list[str]) -> pd.DataFrame:
    """
    Load daily close prices from parquet cache, resample to month-end,
    compute monthly returns.  Returns wide DataFrame: index=month-end, cols=ticker.
    """
    frames = {}
    for ticker in tickers:
        p = PRICES_DIR / f"{ticker}.parquet"
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p, columns=["close"])
            df.index = pd.to_datetime(df.index)
            # Last trading day of each month
            monthly = df["close"].resample("ME").last().dropna()
            if len(monthly) >= 3:
                frames[ticker] = monthly
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()

    prices = pd.DataFrame(frames).sort_index()
    returns = prices.pct_change().dropna(how="all")
    return returns


# ── Factor builders ───────────────────────────────────────────────────────────

def compute_momentum(monthly_ret: pd.DataFrame) -> pd.Series:
    """
    Momentum 18m-1m: product of returns from t-18 to t-1, skip last month.
    Returns a Series of current momentum scores (latest month).
    """
    if len(monthly_ret) < 19:
        return pd.Series(dtype=float)

    # 18-month rolling cumulative return, shifted by 1 to skip last month
    cum18 = (1 + monthly_ret.fillna(0)).rolling(18).apply(np.prod, raw=True) - 1
    mom   = cum18.shift(1)   # skip most recent month
    latest = mom.iloc[-1].dropna()
    return latest


def compute_sector_rs(monthly_ret: pd.DataFrame) -> pd.Series:
    """
    Sector RS vs SPY 3m: each stock gets its sector ETF's 3-month return
    minus SPY's 3-month return.  Mapping via Compustat SIC codes.
    """
    if "SPY" not in monthly_ret.columns:
        return pd.Series(dtype=float)

    spy_3m = (1 + monthly_ret["SPY"].fillna(0)).rolling(3).apply(np.prod, raw=True) - 1

    # Build ETF RS scores
    etf_rs = {}
    for etf in SECTOR_ETFS:
        if etf == "SPY" or etf not in monthly_ret.columns:
            continue
        etf_3m = (1 + monthly_ret[etf].fillna(0)).rolling(3).apply(np.prod, raw=True) - 1
        etf_rs[etf] = (etf_3m - spy_3m).iloc[-1]   # latest value

    # Load SIC → permno → ETF mapping
    fund_path = Path("data/fundamentals/compustat_quarterly.parquet")
    if not fund_path.exists():
        return pd.Series(dtype=float)

    import pyarrow.parquet as pq
    schema_names = pq.read_schema(fund_path).names
    sic_col = "siccd" if "siccd" in schema_names else ("sic" if "sic" in schema_names else None)
    if not sic_col:
        return pd.Series(dtype=float)

    fund = pd.read_parquet(fund_path, columns=["permno", "datadate", sic_col])
    fund["datadate"] = pd.to_datetime(fund["datadate"])
    sic_latest = fund.sort_values("datadate").groupby("permno")[sic_col].last()
    etf_map = sic_latest.apply(_sic_to_etf).dropna()   # permno → ETF

    # We need permno → ticker to assign ETF RS scores to tickers
    ibes = pd.read_parquet("data/analyst/ibes_signals.parquet",
                           columns=["permno", "ticker", "statpers"])
    ibes["statpers"] = pd.to_datetime(ibes["statpers"])
    ticker_map = (ibes.sort_values("statpers")
                  .groupby("permno")["ticker"].last()
                  .str.upper())   # permno → ticker

    # Build ticker → ETF → RS value
    permno_to_etf = etf_map
    ticker_to_etf = {}
    for perm, ticker in ticker_map.items():
        if perm in permno_to_etf:
            ticker_to_etf[ticker] = permno_to_etf[perm]

    sector_scores = pd.Series({
        t: etf_rs[e]
        for t, e in ticker_to_etf.items()
        if e in etf_rs and t in monthly_ret.columns
    })
    return sector_scores.dropna()


def load_roe_latest(ticker_to_permno: pd.Series) -> pd.Series:
    """
    Load the most recent ROE TTM from Compustat for each permno,
    then map to tickers.  Returns Series indexed by ticker.
    """
    fund_path = Path("data/fundamentals/compustat_quarterly.parquet")
    if not fund_path.exists():
        return pd.Series(dtype=float)

    fund = pd.read_parquet(fund_path, columns=["permno", "available_date", "roe_ttm"])
    fund = fund.dropna(subset=["roe_ttm"])
    fund["available_date"] = pd.to_datetime(fund["available_date"])

    roe_latest = (fund.sort_values("available_date")
                  .groupby("permno")["roe_ttm"].last())

    # Map permno → ticker
    roe_by_ticker = {}
    for ticker, permno in ticker_to_permno.items():
        if permno in roe_latest.index:
            roe_by_ticker[ticker] = roe_latest[permno]

    return pd.Series(roe_by_ticker).dropna()


def load_analyst_latest(ticker_to_permno: pd.Series) -> pd.Series:
    """
    Load the most recent analyst rec revision (rev_3m) from IBES,
    mapped to tickers.  Returns Series indexed by ticker.
    """
    analyst_path = Path("data/analyst/ibes_signals.parquet")
    if not analyst_path.exists():
        return pd.Series(dtype=float)

    ibes = pd.read_parquet(analyst_path, columns=["permno", "statpers", "rev_3m"])
    ibes = ibes.dropna(subset=["rev_3m"])
    ibes["statpers"] = pd.to_datetime(ibes["statpers"])

    rev_latest = (ibes.sort_values("statpers")
                  .groupby("permno")["rev_3m"].last())

    rev_by_ticker = {}
    for ticker, permno in ticker_to_permno.items():
        if permno in rev_latest.index:
            rev_by_ticker[ticker] = rev_latest[permno]

    return pd.Series(rev_by_ticker).dropna()


def load_macro_score() -> float | None:
    """Current HY spread widening z-score (sign-flipped, same as model)."""
    macro_path = Path("data/macro/fred_signals.parquet")
    if not macro_path.exists():
        return None
    signals = pd.read_parquet(macro_path)
    if "hy_spread_widening" not in signals.columns:
        return None
    s = signals["hy_spread_widening"].dropna()
    roll_mean = s.rolling(36, min_periods=12).mean()
    roll_std  = s.rolling(36, min_periods=12).std().replace(0, float("nan"))
    macro_z   = -((s - roll_mean) / roll_std)
    val = macro_z.dropna().iloc[-1]
    return float(val)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Live five-factor scoring via Massive")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip Massive download; use cached prices only")
    args = parser.parse_args()

    print("\n" + "═" * 70)
    print("  LIVE SCORING  —  Five-Factor Model (Massive data)")
    print(f"  Price window: {PRICE_START} → {PRICE_END}")
    print("═" * 70)

    # ── Step 1: Build ticker universe from IBES ────────────────────────────────
    print("\nSTEP 1 — Building ticker universe")
    ibes = pd.read_parquet("data/analyst/ibes_signals.parquet",
                           columns=["permno", "ticker", "statpers"])
    ibes["statpers"] = pd.to_datetime(ibes["statpers"])
    recent = ibes[ibes["statpers"] >= "2023-01-01"]
    ticker_map = (recent.sort_values("statpers")
                  .groupby("permno")["ticker"].last()
                  .str.upper()
                  .dropna())
    # Restrict to clean tickers (1–5 capital letters)
    ticker_map = ticker_map[ticker_map.str.match(r"^[A-Z]{1,5}$")]
    # Build ticker → permno; deduplicate by keeping the most recent permno per ticker
    _df = pd.DataFrame({"permno": ticker_map.index, "ticker": ticker_map.values})
    _df = _df.sort_values("permno").drop_duplicates(subset="ticker", keep="last")
    ticker_to_permno = pd.Series(_df["permno"].values, index=_df["ticker"].values)

    stock_tickers = sorted(ticker_map.values.tolist())
    all_tickers   = stock_tickers + SECTOR_ETFS
    print(f"  Stock universe : {len(stock_tickers)} tickers")
    print(f"  + Sector ETFs  : {len(SECTOR_ETFS)}")

    # ── Step 2: Pull prices from Massive ──────────────────────────────────────
    if not args.skip_download:
        print(f"\nSTEP 2 — Pulling prices from Massive (~{len(all_tickers)} tickers)")
        print("  (This may take a few minutes — skips tickers pulled yesterday/today)")
        pull_prices(all_tickers, skip_if_fresh=True)
    else:
        print("\nSTEP 2 — Skipped (--skip-download)")

    # ── Step 3: Load prices → monthly returns ─────────────────────────────────
    print("\nSTEP 3 — Loading prices and computing monthly returns")
    monthly_ret = load_monthly_returns(all_tickers)
    if monthly_ret.empty:
        print("  [ERROR] No price data found.")
        return
    print(f"  Monthly returns: {monthly_ret.shape[1]} tickers × {len(monthly_ret)} months")
    print(f"  Date range     : {monthly_ret.index.min().date()} → {monthly_ret.index.max().date()}")

    # ── Step 4: Compute factors ────────────────────────────────────────────────
    print("\nSTEP 4 — Computing factors")

    print("  F1  Momentum 18m-1m...")
    f1_mom = compute_momentum(monthly_ret)
    print(f"       {len(f1_mom)} tickers scored")

    print("  F2  ROE TTM (Compustat)...")
    f2_roe = load_roe_latest(ticker_to_permno)
    print(f"       {len(f2_roe)} tickers scored")

    print("  F3  Sector RS vs SPY 3m...")
    f3_rs = compute_sector_rs(monthly_ret)
    print(f"       {len(f3_rs)} tickers scored")

    print("  F4  Analyst Rev 3m (IBES)...")
    f4_analyst = load_analyst_latest(ticker_to_permno)
    print(f"       {len(f4_analyst)} tickers scored")

    print("  F5  Macro HY (FRED)...")
    macro_val = load_macro_score()
    if macro_val is not None:
        label = "bullish" if macro_val >= 0 else "bearish"
        print(f"       z-score = {macro_val:+.3f} ({label})")
    else:
        print("       [SKIP] FRED signals not found")

    # ── Step 5: Build composite ────────────────────────────────────────────────
    print("\nSTEP 5 — Building composite scores")

    factors = {
        "momentum_18m":   f1_mom,
        "roe_ttm":        f2_roe,
        "sector_rs_3m":   f3_rs,
        "analyst_rev_3m": f4_analyst,
    }

    # Z-score each factor, intersect common tickers
    z_scores = {}
    for fname, s in factors.items():
        z = _zscore(s)
        if not z.empty:
            z_scores[fname] = z
            print(f"  {fname:20s}: {len(z)} tickers after z-score")

    if not z_scores:
        print("  [ERROR] No factor scores computed.")
        return

    if len(z_scores) < 2:
        print("  [ERROR] Need at least 2 factors. Check data.")
        return

    common = list(z_scores.values())[0].index
    for z in z_scores.values():
        common = common.intersection(z.index)
    print(f"  Common universe ({len(z_scores)} factors): {len(common)} tickers")

    if len(common) < 50:
        print(f"  [WARN] Only {len(common)} tickers in common universe — "
              "run without --skip-download to pull full price history.")

    scores_df = pd.concat(
        {k: v.loc[common] for k, v in z_scores.items()}, axis=1
    )

    # Add macro as 5th equal-weight factor (scalar broadcast)
    n_cs = len(z_scores)
    if macro_val is not None:
        scores_df["macro_hy"] = macro_val
        scores_df["composite"] = (
            scores_df[[c for c in scores_df.columns if c != "macro_hy"]].mean(axis=1) * n_cs
            + macro_val
        ) / (n_cs + 1)
    else:
        scores_df["composite"] = scores_df.mean(axis=1)

    scores_df = scores_df.sort_values("composite", ascending=False)

    # Map ticker index back to permno for compatibility with existing dashboard
    scores_df.index.name = "ticker"
    permno_index = ticker_to_permno.reindex(scores_df.index)
    scores_df.insert(0, "permno", permno_index)

    # ── Step 6: Save ──────────────────────────────────────────────────────────
    out = RESULTS_DIR / "five_factor_scores_latest.csv"
    scores_df.to_csv(out)
    print(f"\n  Saved → {out}")
    print(f"  Rows  : {len(scores_df)} tickers scored")

    print(f"\n  Top 15 as of {date.today()}:")
    print(f"  {'Rank':>4}  {'Ticker':>6}  {'Composite':>9}  "
          f"{'Momentum':>9}  {'ROE':>9}  {'SectorRS':>9}  {'Analyst':>9}  "
          + (f"{'MacroHY':>9}" if macro_val is not None else ""))
    active_cols = [c for c in ["momentum_18m", "roe_ttm", "sector_rs_3m",
                               "analyst_rev_3m", "macro_hy"] if c in scores_df.columns]
    for rank, (ticker, row) in enumerate(scores_df.head(15).iterrows(), 1):
        vals = "  ".join(f"{row[c]:>9.3f}" for c in active_cols)
        print(f"  {rank:>4}  {ticker:>6}  {row['composite']:>9.3f}  {vals}")


if __name__ == "__main__":
    main()
