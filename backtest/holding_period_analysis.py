"""
backtest/holding_period_analysis.py
-------------------------------------
Holding-period sensitivity test for the equal-weight five-factor model.

From the SAME monthly rebalance date and SAME top-50 composite score
selection, measures forward returns at:
  1W  =  5 trading days
  1M  = 21 trading days
  3M  = 63 trading days

Macro overlay is excluded here — this test isolates signal decay,
asking: "how long does the composite score remain predictive?"

Method
------
  1. Rebuild monthly composite scores from existing factor panels
     (same factors: Momentum 18m-1m, ROE TTM, Sector RS, Analyst Rev)
  2. At each month-end, select top-50 permnos, map to tickers via IBES
  3. Bulk-download daily prices from yfinance for all needed tickers
  4. For each rebalance, compute equal-weight portfolio returns at each horizon
  5. Report per-horizon metrics and a side-by-side summary

Period: 2021-04-01 – 2024-12-31 (constrained by daily price availability)

Output
------
  data/results/holding_period_summary.csv
"""

import sys, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RISK_FREE_ANN  = 0.05
BASKET_SIZE    = 50
ANALYSIS_START = "2021-04-01"   # daily data available from 2021-03
ANALYSIS_END   = "2024-12-31"
TC             = 0.001

# Forward horizons in trading days
HORIZONS = {"1W": 5, "1M": 21, "3M": 63}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _zscore(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 10:
        return pd.Series(dtype=float)
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    s = s.clip(lo, hi)
    std = s.std()
    return (s - s.mean()) / std if std > 0 else pd.Series(0.0, index=s.index)


def metrics(r: pd.Series, trading_days_per_period: int) -> dict:
    """Annualised metrics from a series of holding-period returns."""
    r = r.dropna()
    if len(r) < 5:
        return {}
    tdays_per_year = 252
    ppy  = tdays_per_year / trading_days_per_period
    rf   = (1 + RISK_FREE_ANN) ** (trading_days_per_period / tdays_per_year) - 1
    mu   = r.mean()
    sig  = r.std()
    ann_ret = (1 + mu) ** ppy - 1
    ann_vol = sig * np.sqrt(ppy)
    sharpe  = (mu - rf) / sig * np.sqrt(ppy) if sig > 0 else np.nan
    nav     = (1 + r).cumprod()
    max_dd  = ((nav - nav.cummax()) / nav.cummax()).min()
    calmar  = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    hit     = (r > 0).mean()
    t_stat  = mu / (sig / np.sqrt(len(r))) if sig > 0 else np.nan
    return dict(
        ann_return   = round(ann_ret * 100, 2),
        ann_vol      = round(ann_vol  * 100, 2),
        sharpe       = round(sharpe,   3),
        max_drawdown = round(max_dd   * 100, 2),
        calmar       = round(calmar,   3),
        hit_rate     = round(hit      * 100, 1),
        t_stat       = round(t_stat,   2),
        n_periods    = len(r),
    )


# ── Build monthly composite scores ────────────────────────────────────────────

def build_composite_panel() -> pd.DataFrame:
    """
    Returns a DataFrame: index=month-end dates, columns=permno,
    values=composite z-score. Only the analysis window is returned.
    Reuses the same factor-building logic as five_factor_model.py.
    """
    from backtest.five_factor_model import (
        load_crsp_wide, build_compustat_panels,
        build_sector_rs_panel, build_analyst_panel,
        build_all_factor_panels, _zscore as _z,
    )

    print("  Loading CRSP...")
    ret_wide = load_crsp_wide().loc["2010-01-01":ANALYSIS_END]
    crsp_set = set(ret_wide.columns)

    print("  Loading Compustat...")
    try:
        roe_panel, sic_map = build_compustat_panels(ret_wide)
    except Exception as e:
        print(f"    [SKIP] {e}")
        roe_panel, sic_map = pd.DataFrame(), pd.Series(dtype=str)

    print("  Loading Sector RS (yfinance)...")
    try:
        sector_rs = build_sector_rs_panel(ret_wide, sic_map)
    except Exception as e:
        print(f"    [SKIP] {e}")
        sector_rs = pd.DataFrame()

    print("  Loading IBES analyst revisions...")
    try:
        analyst_rev = build_analyst_panel(ret_wide)
    except Exception as e:
        print(f"    [SKIP] {e}")
        analyst_rev = pd.DataFrame()

    panels = build_all_factor_panels(ret_wide, roe_panel, sector_rs, analyst_rev)
    factor_names = list(panels.keys())
    print(f"  Active factors: {', '.join(factor_names)}\n")

    # Build composite score at each month-end in the analysis window
    analysis_dates = ret_wide.loc[ANALYSIS_START:].index
    composite_rows = {}

    for rdate in analysis_dates:
        factor_scores = []
        for fname in factor_names:
            panel = panels[fname]
            if rdate not in panel.index:
                continue
            row = panel.loc[rdate].dropna()
            row = row[row.index.isin(crsp_set)]
            z   = _z(row)
            if len(z) >= BASKET_SIZE:
                factor_scores.append(z)

        if not factor_scores:
            continue

        common = factor_scores[0].index
        for s in factor_scores[1:]:
            common = common.intersection(s.index)

        if len(common) < BASKET_SIZE:
            continue

        composite = pd.concat([s.loc[common] for s in factor_scores], axis=1).mean(axis=1)
        composite_rows[rdate] = composite

    composite_df = pd.DataFrame(composite_rows).T   # date × permno
    print(f"  Composite panel: {len(composite_df)} months × up to {composite_df.shape[1]:,} permnos")
    return composite_df


# ── Permno → Ticker mapping ───────────────────────────────────────────────────

def build_ticker_map() -> pd.Series:
    """Series: permno (float) → ticker (str), using latest IBES entry."""
    ibes = pd.read_parquet("data/analyst/ibes_signals.parquet",
                           columns=["permno", "ticker", "statpers"])
    ibes["statpers"] = pd.to_datetime(ibes["statpers"])
    return (ibes.sort_values("statpers")
            .groupby("permno")["ticker"].last()
            .str.upper())


# ── Download daily prices for needed tickers ──────────────────────────────────

def get_daily_returns(tickers: list[str]) -> pd.DataFrame:
    """
    Download daily adjusted close prices from yfinance.
    Returns a wide DataFrame of daily returns: index=date, columns=ticker.
    """
    print(f"  Downloading daily prices for {len(tickers)} tickers via yfinance...")
    chunk_size = 100
    all_closes = []
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            raw = yf.download(
                chunk, start="2021-01-01", end="2025-06-01",
                auto_adjust=True, progress=False,
            )["Close"]
            if isinstance(raw, pd.Series):
                raw = raw.to_frame(name=chunk[0])
            all_closes.append(raw)
        except Exception as e:
            print(f"    [WARN] chunk {i//chunk_size}: {e}")

    if not all_closes:
        return pd.DataFrame()

    closes = pd.concat(all_closes, axis=1)
    closes = closes.loc[:, ~closes.columns.duplicated()]
    daily_ret = closes.pct_change().dropna(how="all")
    print(f"  Daily returns: {daily_ret.shape[1]} tickers, "
          f"{daily_ret.index.min().date()} – {daily_ret.index.max().date()}")
    return daily_ret


# ── Holding-period return computation ─────────────────────────────────────────

def forward_return(daily_ret: pd.DataFrame, tickers: list[str],
                   from_date: pd.Timestamp, n_days: int) -> float | None:
    """
    Equal-weight forward return for `tickers` starting the trading day
    AFTER `from_date`, measured over `n_days` trading days.
    """
    avail = [t for t in tickers if t in daily_ret.columns]
    if not avail:
        return None

    future = daily_ret.loc[daily_ret.index > from_date]
    if len(future) < n_days:
        return None

    window = future.iloc[:n_days][avail].dropna(how="all", axis=1)
    if window.empty or window.shape[1] < max(5, len(avail) // 3):
        return None

    ew_daily = window.mean(axis=1)
    return float((1 + ew_daily).prod() - 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 70)
    print("  HOLDING-PERIOD SENSITIVITY ANALYSIS")
    print(f"  Top-{BASKET_SIZE} composite basket | No macro overlay")
    print(f"  Horizons: 1W (5d), 1M (21d), 3M (63d)")
    print(f"  Period: {ANALYSIS_START} – {ANALYSIS_END}")
    print("═" * 70 + "\n")

    # ── Step 1: Build composite panel ─────────────────────────────────────────
    print("STEP 1 — Building monthly composite scores")
    print("─" * 50)
    composite_df = build_composite_panel()
    if composite_df.empty:
        print("[ERROR] No composite scores. Check data paths.")
        return

    # ── Step 2: Build permno→ticker map ───────────────────────────────────────
    print("\nSTEP 2 — Building permno → ticker mapping")
    print("─" * 50)
    ticker_map = build_ticker_map()
    print(f"  IBES ticker map: {len(ticker_map):,} permnos")

    # ── Step 3: Identify top-50 at each rebalance; collect needed tickers ─────
    print("\nSTEP 3 — Selecting top-50 each month")
    print("─" * 50)

    monthly_selections = {}   # date → list of tickers
    needed_tickers     = set()

    for rdate, row in composite_df.iterrows():
        scored = row.dropna().sort_values(ascending=False)
        # Take top-150 permnos, map to tickers, keep first 50 mappable ones
        top_permnos = scored.head(150).index
        top_tickers = (ticker_map.reindex(top_permnos)
                       .dropna()
                       .values[:BASKET_SIZE])
        if len(top_tickers) < BASKET_SIZE // 2:
            continue
        monthly_selections[rdate] = list(top_tickers)
        needed_tickers.update(top_tickers)

    print(f"  Rebalance dates : {len(monthly_selections)}")
    print(f"  Unique tickers  : {len(needed_tickers)}")

    # ── Step 4: Download daily prices ─────────────────────────────────────────
    print("\nSTEP 4 — Downloading daily prices")
    print("─" * 50)
    daily_ret = get_daily_returns(sorted(needed_tickers))

    # ── Step 5: Compute forward returns at each horizon ───────────────────────
    print("\nSTEP 5 — Computing forward returns")
    print("─" * 50)

    records = []   # one row per (rebalance_date, horizon)
    prev_tickers = []

    for rdate in sorted(monthly_selections.keys()):
        tickers = monthly_selections[rdate]

        # Turnover-based TC (applied once regardless of horizon)
        if prev_tickers:
            prev_set = set(prev_tickers)
            curr_set = set(tickers)
            turnover = len(curr_set.symmetric_difference(prev_set)) / (2 * BASKET_SIZE)
        else:
            turnover = 1.0
        tc_drag = turnover * TC
        prev_tickers = tickers

        for horizon_label, n_days in HORIZONS.items():
            ret = forward_return(daily_ret, tickers, rdate, n_days)
            if ret is not None:
                records.append({
                    "date":    rdate,
                    "horizon": horizon_label,
                    "n_days":  n_days,
                    "ret":     ret - tc_drag,
                })

    df = pd.DataFrame(records)
    print(f"  Total observations: {len(df)} ({len(df)//len(HORIZONS)} months × {len(HORIZONS)} horizons)")

    # ── Step 6: Metrics per horizon ───────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  RESULTS — Invested-only metrics by holding period")
    print(f"  Top-{BASKET_SIZE} basket | Equal-weight | 10bps TC/side")
    print("═" * 80)

    hdr = (f"  {'Horizon':>8}  {'N days':>7}  {'Ann Ret':>8}  {'Ann Vol':>8}  "
           f"{'Sharpe':>7}  {'Max DD':>8}  {'Hit%':>6}  {'t-stat':>7}  {'N':>5}")
    print(hdr)
    print("  " + "─" * 76)

    results = []
    for h_label, n_days in HORIZONS.items():
        sub = df[df["horizon"] == h_label]["ret"]
        m   = metrics(sub, n_days)
        if not m:
            continue
        results.append({"horizon": h_label, "n_trading_days": n_days, **m})
        print(
            f"  {h_label:>8}  {n_days:>7}  "
            f"  {m['ann_return']:>+6.1f}%"
            f"  {m['ann_vol']:>7.1f}%"
            f"  {m['sharpe']:>7.3f}"
            f"  {m['max_drawdown']:>+7.1f}%"
            f"  {m['hit_rate']:>5.1f}%"
            f"  {m['t_stat']:>7.2f}"
            f"  {m['n_periods']:>5}"
        )

    print("═" * 80)
    print("  Note: 3M returns overlap (each period shares ~2 months with the next).")
    print("        t-stats for 3M are conservative due to fewer independent periods.\n")

    # ── Signal-decay chart (mean return by horizon) ───────────────────────────
    print("  Mean return by horizon (raw, before annualisation):")
    for h_label, n_days in HORIZONS.items():
        sub = df[df["horizon"] == h_label]["ret"]
        ppy = 252 / n_days
        ann = (1 + sub.mean()) ** ppy - 1
        bar_len = max(0, int(sub.mean() * 1000))
        bar = "█" * bar_len
        print(f"    {h_label:>3}  {sub.mean()*100:>+6.2f}% per period  "
              f"({ann*100:>+6.1f}% ann)  {bar}")

    # ── Save ──────────────────────────────────────────────────────────────────
    summary = pd.DataFrame(results)
    out = RESULTS_DIR / "holding_period_summary.csv"
    summary.to_csv(out, index=False)
    print(f"\n  Saved → {out}")

    # Also save raw returns for each horizon
    df.to_csv(RESULTS_DIR / "holding_period_raw_returns.csv", index=False)
    print(f"  Saved → {RESULTS_DIR / 'holding_period_raw_returns.csv'}")


if __name__ == "__main__":
    main()
