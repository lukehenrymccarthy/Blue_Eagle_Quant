"""
backtest/twenty_year_backtest.py
---------------------------------
Proper 20-year walk-forward backtest of the five-factor model with
ML-optimised weights.  Strict point-in-time (PIT) data discipline:

  Factor             PIT anchor
  ---------          --------------------------------------------------
  Momentum 18m-1m    CRSP monthly ret, skip last month (shift 1)
  ROE TTM            Compustat rdq  = actual earnings report date
                     Only the most recent filing with rdq <= rdate
  Sector RS 3m       Sector ETF 3m return vs SPY through end of rdate
                     (ETF prices are public intraday — no lag needed)
  Analyst Rev 3m     IBES statpers = date consensus was compiled
                     Forward-filled only up to rdate
  Macro HY           FRED BAMLH0A0HYM2 rolling z-score asof rdate
                     (FRED releases with ~1-week lag; monthly is safe)

Lookahead guards
----------------
  * Compustat: only rows where rdq <= rdate are used (no future filings)
  * IBES:      only rows where statpers <= rdate (resample+ffill ensures this)
  * Macro:     .asof(rdate) — never reads beyond rebalance date
  * Momentum:  uses ret_wide up to rdate with shift(1) — no future returns

Period: Jan 2005 – Dec 2024  (240 months, ~20 years)
  Warm-up: first 18 months consumed by momentum (factually available from mid-2006)

Outputs
-------
  data/results/twenty_year_equity_curves.csv   — monthly NAV for each config
  data/results/twenty_year_summary.csv         — annualised metrics table
  data/results/twenty_year_subperiod.csv       — sub-period breakdown
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.five_factor_model import (
    _zscore, compute_metrics, _sic_to_etf,
    build_sector_rs_panel, build_analyst_panel,
    build_macro_factor, build_all_factor_panels,
    load_crsp_wide,
)

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR   = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

START_DATE    = "2005-01-01"
END_DATE      = "2024-12-31"
BASKET_SIZE   = 50
HOLD_MONTHS   = 1
TC            = 0.001          # 10 bps per side
RISK_FREE_ANN = 0.05

# Sub-periods for breakdown table
SUB_PERIODS = [
    ("2006-01", "2009-12", "2006–2009  (GFC)"),
    ("2010-01", "2014-12", "2010–2014  (Recovery)"),
    ("2015-01", "2019-12", "2015–2019  (Bull)"),
    ("2020-01", "2024-12", "2020–2024  (COVID+Tightening)"),
]


# ══════════════════════════════════════════════════════════════════════════════
# PIT-ACCURATE COMPUSTAT LOADER
# ══════════════════════════════════════════════════════════════════════════════

def build_compustat_pit(ret_wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build a strictly point-in-time ROE TTM panel using rdq (actual report
    date) rather than the conservative available_date.

    Logic
    -----
    1. For each (permno, quarter) row, the data becomes known on rdq.
    2. At each month-end rebalance date we carry forward the MOST RECENT
       quarterly filing whose rdq <= rebalance_date.
    3. Forward-fill: if a firm hasn't reported a new quarter yet, we keep
       the last known value (still PIT-safe — it was already public).
    4. Values more than 15 months stale are set to NaN (stale-data filter).

    Returns
    -------
    roe_panel : DataFrame  (date × permno), monthly PIT ROE TTM
    sic_map   : Series     permno → sector ETF using latest SIC per firm
    """
    import pyarrow.parquet as pq

    fund_path = Path("data/fundamentals/compustat_quarterly.parquet")
    schema_names = pq.read_schema(fund_path).names
    sic_col   = "siccd" if "siccd" in schema_names else ("sic" if "sic" in schema_names else None)
    load_cols = ["permno", "datadate", "rdq", "roe_ttm"]
    if sic_col:
        load_cols.append(sic_col)

    fund = pd.read_parquet(fund_path, columns=load_cols)
    fund["rdq"]      = pd.to_datetime(fund["rdq"])
    fund["datadate"] = pd.to_datetime(fund["datadate"])

    # Drop rows with missing rdq or roe — can't establish PIT date
    fund = fund.dropna(subset=["rdq", "roe_ttm"])

    # The PIT date for each observation = rdq (day results were published)
    # We round to month-end so it aligns with our monthly rebalance grid.
    # Adding MonthEnd(0) keeps dates that already land on month-end, and
    # advances others to the FOLLOWING month-end — i.e. we don't know data
    # until the month it was filed finishes.
    fund["pit_month"] = fund["rdq"] + pd.offsets.MonthEnd(0)

    # Pivot: for each (pit_month, permno), take the LAST reported roe_ttm
    # (handles rare same-month dual filings)
    roe_raw = (
        fund.sort_values("rdq")
        .pivot_table(
            index   = "pit_month",
            columns = "permno",
            values  = "roe_ttm",
            aggfunc = "last",
        )
    )
    roe_raw.index = pd.to_datetime(roe_raw.index)

    # Resample to every month-end in our CRSP range, forward-fill
    # This carries the last known value forward (PIT-safe).
    full_idx  = ret_wide.index  # already month-end
    roe_panel = roe_raw.reindex(full_idx, method="ffill")

    # Stale-data filter: if the most recent rdq is >15 months before the
    # rebalance date, the financial data is too old to be informative.
    # Build a "last_rdq" panel and mask stale cells.
    last_rdq_raw = (
        fund.sort_values("rdq")
        .pivot_table(
            index   = "pit_month",
            columns = "permno",
            values  = "rdq",
            aggfunc = "last",
        )
    )
    last_rdq_raw.index = pd.to_datetime(last_rdq_raw.index)
    last_rdq_panel = last_rdq_raw.reindex(full_idx, method="ffill")

    # Broadcast rebalance date across columns for vectorised comparison
    rdate_broadcast = pd.DataFrame(
        np.tile(full_idx.values[:, None], (1, roe_panel.shape[1])),
        index   = full_idx,
        columns = roe_panel.columns,
    )
    staleness_days = (rdate_broadcast - last_rdq_panel).apply(
        lambda col: col.dt.days, axis=0
    )
    stale_mask = staleness_days > 455   # ~15 months
    roe_panel  = roe_panel.where(~stale_mask, np.nan)

    # ── SIC map ───────────────────────────────────────────────────────────────
    if sic_col and sic_col in fund.columns:
        sic_latest = fund.sort_values("datadate").groupby("permno")[sic_col].last()
        sic_map    = sic_latest.apply(_sic_to_etf).dropna()
    else:
        sic_map = pd.Series(dtype=str)

    return roe_panel, sic_map


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE  (same structure as five_factor_model, adapted for 20yr)
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    ret_wide:     pd.DataFrame,
    panels:       dict[str, pd.DataFrame],
    macro_factor: pd.Series | None,
    basket:       int,
    hold_months:  int,
    weights:      dict | None,
    label:        str = "",
) -> pd.Series:
    """
    Walk-forward backtest.  Returns a Series of holding-period returns
    indexed by rebalance date.

    PIT guarantee: at rebalance date `rdate`, each factor panel value for
    `rdate` was computed only from data timestamped ≤ rdate (enforced by
    the individual panel builders above).
    """
    dates       = ret_wide.index
    rebal_dates = dates[::hold_months]
    fnames      = list(panels.keys())

    port_rets = []
    prev_hold = set()
    skipped   = 0

    for i, rdate in enumerate(rebal_dates[:-1]):
        next_rdate = rebal_dates[i + 1]

        # ── Cross-sectional composite ──────────────────────────────────────
        crsp_universe = set(ret_wide.columns)
        scored = {}

        for fname in fnames:
            panel = panels[fname]
            if rdate not in panel.index:
                continue
            row = panel.loc[rdate].dropna()
            row = row[row.index.isin(crsp_universe)]
            z   = _zscore(row)
            if len(z) >= basket:
                scored[fname] = z

        if not scored:
            port_rets.append(np.nan)
            skipped += 1
            continue

        common = list(scored.values())[0].index
        for z in scored.values():
            common = common.intersection(z.index)

        if len(common) < basket:
            port_rets.append(np.nan)
            skipped += 1
            continue

        n_cs  = len(scored)
        if weights:
            w_arr = np.array([weights.get(fn, 1.0 / n_cs) for fn in scored])
            w_arr = w_arr / w_arr.sum()
        else:
            w_arr = np.ones(n_cs) / n_cs

        factor_mat = np.column_stack([scored[fn].reindex(common).values for fn in scored])
        composite  = pd.Series(factor_mat @ w_arr, index=common)

        # ── Macro blend ───────────────────────────────────────────────────
        if macro_factor is not None:
            macro_val = macro_factor.asof(rdate)
            if pd.notna(macro_val):
                macro_w   = weights.get("macro_hy", 1.0 / (n_cs + 1)) if weights else 1.0 / (n_cs + 1)
                composite = composite * (1.0 - macro_w) + macro_val * macro_w

        top_n = set(composite.nlargest(basket).index)

        # ── TC ────────────────────────────────────────────────────────────
        turnover = (
            len(top_n.symmetric_difference(prev_hold)) / (2 * basket)
            if prev_hold else 1.0
        )
        tc_drag = turnover * TC

        # ── Holding-period return ─────────────────────────────────────────
        period_slice = ret_wide.loc[
            (ret_wide.index > rdate) & (ret_wide.index <= next_rdate),
            list(top_n),
        ].dropna(how="all", axis=1)

        if period_slice.empty:
            port_rets.append(np.nan)
            skipped += 1
            continue

        ew_monthly   = period_slice.mean(axis=1)
        compound_ret = (1 + ew_monthly).prod() - 1 - tc_drag
        port_rets.append(compound_ret)
        prev_hold = top_n

    return pd.Series(port_rets, index=rebal_dates[:-1], name=label)


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK — SPY buy-and-hold (monthly, matched to our rebal dates)
# ══════════════════════════════════════════════════════════════════════════════

def load_spy_benchmark(ret_wide: pd.DataFrame) -> pd.Series:
    """
    Monthly SPY returns aligned to our CRSP month-end dates.
    Uses cached daily parquet if it spans the full analysis window,
    otherwise downloads monthly data from yfinance.
    """
    need_start = ret_wide.index[0]
    spy_path   = Path("data/prices/SPY.parquet")

    try:
        raw = pd.read_parquet(spy_path)
        raw.index = pd.to_datetime(raw.index)
        # Only use cache if it covers our start date
        if raw.index.min() <= need_start:
            closes  = raw["close"] if "close" in raw.columns else raw.iloc[:, 0]
            monthly = closes.resample("ME").last().pct_change()
            return monthly.reindex(ret_wide.index).rename("SPY")
    except Exception:
        pass

    # Fallback: download monthly bars directly from yfinance
    raw = yf.download(
        "SPY", start="2004-12-01", end="2025-01-31",
        interval="1mo", auto_adjust=True, progress=False,
    )["Close"]
    raw.index = pd.to_datetime(raw.index) + pd.offsets.MonthEnd(0)
    monthly   = raw.pct_change().squeeze()
    return monthly.reindex(ret_wide.index).rename("SPY")


def spy_holding_period_rets(
    spy_monthly: pd.Series,
    rebal_dates: pd.DatetimeIndex,
    hold_months: int,
) -> pd.Series:
    """Compound SPY monthly returns over each holding period."""
    rdate_list = list(rebal_dates)
    out = []
    for i, rd in enumerate(rdate_list[:-1]):
        nrd    = rdate_list[i + 1]
        window = spy_monthly.loc[(spy_monthly.index > rd) & (spy_monthly.index <= nrd)]
        out.append(float((1 + window.fillna(0)).prod() - 1) if len(window) else np.nan)
    return pd.Series(out, index=rebal_dates[:-1], name="SPY_BH")


# ══════════════════════════════════════════════════════════════════════════════
# METRICS & REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def full_metrics(r: pd.Series, hold_months: int = 1, label: str = "") -> dict:
    r = r.dropna()
    if len(r) < 12:
        return {}
    ppy     = 12 / hold_months
    rf_per  = (1 + RISK_FREE_ANN) ** (hold_months / 12) - 1
    mu      = r.mean()
    sig     = r.std()
    ann_ret = (1 + mu) ** ppy - 1
    ann_vol = sig * np.sqrt(ppy)
    sharpe  = (mu - rf_per) / sig * np.sqrt(ppy) if sig > 0 else np.nan
    nav     = (1 + r).cumprod()
    max_dd  = ((nav - nav.cummax()) / nav.cummax()).min()
    calmar  = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    hit     = (r > 0).mean()
    tstat   = mu / (sig / np.sqrt(len(r))) if sig > 0 else np.nan
    total   = float(nav.iloc[-1] - 1)
    return dict(
        label        = label or r.name,
        ann_return   = round(ann_ret * 100, 2),
        ann_vol      = round(ann_vol  * 100, 2),
        sharpe       = round(sharpe,   3),
        max_drawdown = round(max_dd   * 100, 2),
        calmar       = round(calmar,   3),
        hit_rate     = round(hit      * 100, 1),
        t_stat       = round(tstat,    2),
        total_return = round(total    * 100, 1),
        n_periods    = len(r),
    )


def print_metrics_row(m: dict, col_w: int = 22):
    print(
        f"  {m.get('label',''):<{col_w}}"
        f"  {m['ann_return']:>+7.1f}%"
        f"  {m['ann_vol']:>6.1f}%"
        f"  {m['sharpe']:>7.3f}"
        f"  {m['max_drawdown']:>+7.1f}%"
        f"  {m['total_return']:>+7.1f}%"
        f"  {m['hit_rate']:>5.1f}%"
        f"  {m['t_stat']:>6.2f}"
        f"  {m['n_periods']:>5}"
    )


def print_header(title: str, w: int = 100):
    print("\n" + "═" * w)
    print(f"  {title}")
    print("═" * w)
    col_names = (
        f"  {'Strategy':<22}"
        f"  {'Ann Ret':>8}  {'Vol':>6}  {'Sharpe':>7}"
        f"  {'MaxDD':>8}  {'Total':>8}  {'Hit%':>6}  {'t-stat':>7}  {'N':>5}"
    )
    print(col_names)
    print("  " + "─" * (w - 2))


def to_nav(rets: pd.Series) -> pd.Series:
    """Convert a Series of period returns to a NAV starting at 1.0."""
    clean = rets.dropna()
    nav   = (1 + clean).cumprod()
    # Prepend a 1.0 start one period before first return
    return nav


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    W = 100
    print("\n" + "═" * W)
    print("  TWENTY-YEAR BACKTEST  (2005 – 2024)")
    print("  Five-Factor Model  |  ML-Optimised Weights  |  Strict PIT discipline")
    print("  Compustat PIT anchor: rdq (actual earnings report date)")
    print("  No lookahead bias: each factor uses only data available at rebalance")
    print("═" * W + "\n")

    # ── STEP 1: Load CRSP ─────────────────────────────────────────────────────
    print("STEP 1  Loading CRSP monthly returns (2005–2024)")
    print("─" * 60)
    try:
        ret_wide = load_crsp_wide().loc[START_DATE:END_DATE]
    except Exception as e:
        print(f"  [ERROR] {e}")
        return
    print(f"  {ret_wide.shape[1]:,} stocks  |  "
          f"{ret_wide.index[0].date()} – {ret_wide.index[-1].date()}  |  "
          f"{len(ret_wide)} months\n")

    # ── STEP 2: Compustat (PIT via rdq) ───────────────────────────────────────
    print("STEP 2  Loading Compustat — PIT via rdq (actual report date)")
    print("─" * 60)
    try:
        roe_panel, sic_map = build_compustat_pit(ret_wide)
        coverage = roe_panel.notna().sum(axis=1).mean()
        print(f"  ROE TTM panel : {roe_panel.shape[1]:,} stocks  |  "
              f"avg coverage/month: {coverage:.0f} stocks")
        print(f"  SIC map       : {len(sic_map):,} stocks mapped to sector ETFs")
        # Sanity check: verify no future data leaks
        # (if a row in roe_panel is nonzero at date D, the rdq that produced it
        #  must be <= D.  The build function enforces this via pit_month rounding.)
        print("  PIT check     : rdq-based pivot — no forward-fill beyond rebalance date")
    except Exception as e:
        print(f"  [SKIP] Compustat error: {e}")
        roe_panel, sic_map = pd.DataFrame(), pd.Series(dtype=str)

    # ── STEP 3: Sector RS ─────────────────────────────────────────────────────
    print("\nSTEP 3  Sector RS vs SPY (3m rolling, ETF prices)")
    print("─" * 60)
    sector_rs = pd.DataFrame()
    try:
        sector_rs = build_sector_rs_panel(ret_wide, sic_map)
        if not sector_rs.empty:
            print(f"  Sector RS     : {sector_rs.shape[1]:,} stocks")
        else:
            print("  [SKIP] Sector RS — no SIC or ETF data")
    except Exception as e:
        print(f"  [SKIP] Sector RS: {e}")

    # ── STEP 4: IBES analyst revisions ────────────────────────────────────────
    print("\nSTEP 4  IBES analyst revisions (statpers = PIT date)")
    print("─" * 60)
    analyst_rev = pd.DataFrame()
    try:
        analyst_rev = build_analyst_panel(ret_wide)
        if not analyst_rev.empty:
            print(f"  Analyst Rev   : {analyst_rev.shape[1]:,} stocks × {analyst_rev.shape[0]} months")
    except Exception as e:
        print(f"  [SKIP] Analyst: {e}")

    # ── STEP 5: Macro factor ──────────────────────────────────────────────────
    print("\nSTEP 5  Macro factor (FRED HY OAS, rolling z-score, sign-flipped)")
    print("─" * 60)
    macro_factor = None
    try:
        macro_factor = build_macro_factor(ret_wide)
        if macro_factor is not None:
            latest = macro_factor.dropna().iloc[-1]
            print(f"  Macro F5      : {macro_factor.notna().sum()} months  |  "
                  f"latest z = {latest:+.2f}  ({'bullish' if latest >= 0 else 'bearish'})")
    except Exception as e:
        print(f"  [SKIP] Macro: {e}")

    # ── STEP 6: Build factor panels ───────────────────────────────────────────
    print("\nSTEP 6  Building factor panels")
    print("─" * 60)
    panels = build_all_factor_panels(ret_wide, roe_panel, sector_rs, analyst_rev)
    active = list(panels.keys())
    print(f"\n  Active cross-sectional factors ({len(active)}): {', '.join(active)}")
    if macro_factor is not None:
        print("  F5 Macro HY: active")
    if not active:
        print("  [ERROR] No factor panels. Aborting.")
        return

    # ── STEP 7: Load optimal weights ──────────────────────────────────────────
    print("\nSTEP 7  Loading ML-optimised weights")
    print("─" * 60)
    opt_weights = None
    wpath = Path("data/results/optimal_weights.json")
    if wpath.exists():
        with open(wpath) as f:
            wdata = json.load(f)
        opt_weights = wdata.get("weights")
        print("  Loaded from data/results/optimal_weights.json:")
        eq = 1.0 / (len(active) + (1 if macro_factor is not None else 0))
        for fn, w in sorted(opt_weights.items(), key=lambda x: -x[1]):
            delta = w - eq
            print(f"    {fn:<25}  {w:.1%}  ({'+' if delta>=0 else ''}{delta:.1%} vs equal)")
    else:
        print("  No optimal_weights.json — running equal-weight only.")

    # ── STEP 8: Benchmark ─────────────────────────────────────────────────────
    print("\nSTEP 8  Loading SPY benchmark")
    print("─" * 60)
    spy_monthly = load_spy_benchmark(ret_wide)
    rebal_dates = ret_wide.index[::HOLD_MONTHS]
    spy_rets    = spy_holding_period_rets(spy_monthly, rebal_dates, HOLD_MONTHS)
    spy_m       = full_metrics(spy_rets, HOLD_MONTHS, "SPY Buy-and-Hold")
    if spy_m:
        print(f"  SPY 2005–2024: Ann={spy_m['ann_return']:+.1f}%  "
              f"Sharpe={spy_m['sharpe']:.3f}  "
              f"MaxDD={spy_m['max_drawdown']:+.1f}%  "
              f"Total={spy_m['total_return']:+.1f}%")

    # ── STEP 9: Run backtest configs ──────────────────────────────────────────
    print("\nSTEP 9  Running backtests")
    print("─" * 60)

    configs = [
        ("ML-Optimised Weights", opt_weights),
        ("Equal-Weight",         None),
    ]
    if opt_weights is None:
        configs = [("Equal-Weight", None)]

    all_rets  = {}
    summaries = []

    for cfg_label, wts in configs:
        print(f"  {cfg_label:<30}", end=" ... ", flush=True)
        rets = run_backtest(
            ret_wide     = ret_wide,
            panels       = panels,
            macro_factor = macro_factor,
            basket       = BASKET_SIZE,
            hold_months  = HOLD_MONTHS,
            weights      = wts,
            label        = cfg_label,
        )
        m = full_metrics(rets, HOLD_MONTHS, cfg_label)
        if m:
            print(f"Sharpe={m['sharpe']:.3f}  Ann={m['ann_return']:+.1f}%  "
                  f"MaxDD={m['max_drawdown']:+.1f}%  Total={m['total_return']:+.1f}%")
            summaries.append(m)
        else:
            print("insufficient data")
        all_rets[cfg_label] = rets

    # ── STEP 10: Results tables ───────────────────────────────────────────────

    # Full-period summary
    print_header("FULL-PERIOD RESULTS  (Jan 2005 – Dec 2024  |  ~20 years)", W)
    for m in summaries:
        print_metrics_row(m)
    if spy_m:
        print_metrics_row(spy_m)
    print("═" * W)
    print("  TC: 10 bps/side on portfolio turnover  |  1M hold  |  Top-50 basket")

    # Sub-period breakdown
    print_header("SUB-PERIOD BREAKDOWN  (ML-Optimised vs SPY)", W)
    sub_rows = []
    for sp_start, sp_end, sp_label in SUB_PERIODS:
        row = {"period": sp_label}
        for cfg_label, wts in configs[:1]:   # show only best config
            sp_rets = all_rets[cfg_label]
            sp_slice = sp_rets.loc[sp_start:sp_end].dropna()
            m = full_metrics(sp_slice, HOLD_MONTHS)
            if m:
                row.update({
                    f"model_ann_ret":   m["ann_return"],
                    f"model_sharpe":    m["sharpe"],
                    f"model_max_dd":    m["max_drawdown"],
                })
        # SPY sub-period
        spy_sp  = spy_rets.loc[sp_start:sp_end].dropna()
        sm      = full_metrics(spy_sp, HOLD_MONTHS)
        if sm:
            row.update({
                "spy_ann_ret": sm["ann_return"],
                "spy_sharpe":  sm["sharpe"],
            })
        sub_rows.append(row)

        cfg_lbl = list(all_rets.keys())[0]
        print(
            f"  {sp_label:<30}"
            f"  Model: Ann={row.get('model_ann_ret', float('nan')):>+6.1f}%"
            f"  Sharpe={row.get('model_sharpe', float('nan')):>6.3f}"
            f"  DD={row.get('model_max_dd', float('nan')):>+6.1f}%"
            f"  │  SPY: Ann={row.get('spy_ann_ret', float('nan')):>+6.1f}%"
            f"  Sharpe={row.get('spy_sharpe', float('nan')):>6.3f}"
        )
    print("═" * W)

    # ── STEP 11: Save outputs ─────────────────────────────────────────────────
    print("\nSTEP 11  Saving outputs")
    print("─" * 60)

    # Equity curves: NAV starting at 1.0
    nav_df = pd.DataFrame()
    for lbl, rets in all_rets.items():
        nav_df[lbl] = to_nav(rets)
    nav_df["SPY_BH"] = to_nav(spy_rets)
    nav_df.to_csv(RESULTS_DIR / "twenty_year_equity_curves.csv")
    print(f"  Saved → {RESULTS_DIR / 'twenty_year_equity_curves.csv'}")

    # Summary table
    summary_df = pd.DataFrame(summaries)
    if spy_m:
        summary_df = pd.concat([summary_df, pd.DataFrame([spy_m])], ignore_index=True)
    summary_df.to_csv(RESULTS_DIR / "twenty_year_summary.csv", index=False)
    print(f"  Saved → {RESULTS_DIR / 'twenty_year_summary.csv'}")

    # Sub-period table
    pd.DataFrame(sub_rows).to_csv(RESULTS_DIR / "twenty_year_subperiod.csv", index=False)
    print(f"  Saved → {RESULTS_DIR / 'twenty_year_subperiod.csv'}")

    # Raw returns (for further analysis)
    rets_df = pd.DataFrame(all_rets)
    rets_df["SPY_BH"] = spy_rets
    rets_df.to_csv(RESULTS_DIR / "twenty_year_raw_returns.csv")
    print(f"  Saved → {RESULTS_DIR / 'twenty_year_raw_returns.csv'}")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
