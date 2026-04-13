"""
backtest/updated_model.py
-------------------------
Primary research harness for the SectorScope stock-selection model.

This script now supports two explicit modes:

  1. core
     Canonical three-factor workflow:
       - sue
       - mom_52wk_high
       - sector_rs_1m

  2. research
     Extended model with optional supporting signals:
       - earnings_yield
       - neg_dispersion
       - macro overlays

The canonical entry point for day-to-day work is `backtest/core_model.py`,
which calls this script in `core` mode. This file remains the broader
research surface for testing extensions and alternate weight schemes.
"""

import sys
import warnings
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sectorscope.metrics import compute_metrics
from sectorscope.utils   import zscore as _zscore, sic_to_etf as _sic_to_etf
from sectorscope.factors import (
    build_52wk_high,
    build_residual_52wk_high,
    build_earnings_yield,
    build_unemployment_macro,
    build_vix_macro,
    build_sector_rs_1m,
    build_neg_dispersion,
    build_sue,
    build_liquidity_screen,
    build_factor_panels,
)
from sectorscope.modeling import (
    FACTOR_WEIGHTS_OPT,
    parse_factor_list,
    get_selected_weights,
    get_default_excluded_factors,
    describe_model_mode,
)

# ── Config ─────────────────────────────────────────────────────────────────────
RESULTS_DIR   = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FULL_START    = "2005-01-01"
IS_END        = "2024-12-31"
OOS_START     = "2025-01-01"
OOS_END       = "2025-12-31"

HOLD_MONTHS   = 1
BASKET_PCT    = 0.25          # top 25% of liquid universe
UNIVERSE_SIZE = 1_000         # top-N by mkt cap liquidity screen
RISK_FREE_ANN = 0.0425
TC            = 0.001         # 10 bps per side

FACTOR_WEIGHTS = FACTOR_WEIGHTS_OPT

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_returns(oos_end: str = OOS_END) -> pd.DataFrame:
    from ingestion.wrds_returns import load_returns as _lr, returns_wide
    rw = returns_wide(_lr())
    rw.index = pd.to_datetime(rw.index) + pd.offsets.MonthEnd(0)
    return rw.loc[FULL_START:oos_end]


def load_compustat() -> pd.DataFrame:
    df = pd.read_parquet("data/fundamentals/compustat_quarterly.parquet")
    df["available_date"] = pd.to_datetime(df["available_date"])
    df["datadate"]       = pd.to_datetime(df["datadate"])
    df["permno"]         = pd.to_numeric(df["permno"], errors="coerce")
    return df.dropna(subset=["permno"]).assign(permno=lambda x: x["permno"].astype(int))


def load_ibes() -> pd.DataFrame:
    path = Path("data/analyst/ibes_signals.parquet")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["statpers"] = pd.to_datetime(df["statpers"]) + pd.offsets.MonthEnd(0)
    df["permno"]   = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    ret_wide:      pd.DataFrame,
    panels:        dict,
    is_liquid:     pd.DataFrame,
    macro:         pd.Series | None = None,
    vix_macro:     pd.Series | None = None,
    hold_months:   int   = HOLD_MONTHS,
    basket_pct:    float = BASKET_PCT,
    weights:       dict  = None,
    score_weighted: bool = False,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Returns (period_returns, holdings_log).

    basket_pct : fraction of scored universe to hold (e.g. 0.25 = top quartile).
    weights    : factor weights dict; defaults to FACTOR_WEIGHTS.
    """
    weights      = weights or FACTOR_WEIGHTS
    factor_names = list(panels.keys())
    dates        = ret_wide.index
    rebal_dates  = dates[::hold_months]

    port_rets   = []
    prev_hold   = set()
    log_rows    = []

    for i, rdate in enumerate(rebal_dates[:-1]):
        next_rdate = rebal_dates[i + 1]

        # Liquid universe at this rebalance date
        universe = set(ret_wide.columns)
        if not is_liquid.empty and rdate in is_liquid.index:
            liq_row  = is_liquid.loc[rdate].fillna(False)
            universe = universe.intersection(liq_row[liq_row].index)

        # Compute z-scored factor scores
        scored = {}
        for fn in factor_names:
            p = panels[fn]
            if rdate not in p.index:
                continue
            row = p.loc[rdate].dropna().replace([np.inf, -np.inf], np.nan).dropna()
            row = row[row.index.isin(universe)]
            z   = _zscore(row)
            if len(z) > 10:
                scored[fn] = z

        if not scored:
            port_rets.append(np.nan)
            continue

        # Intersection of stocks with all active factor scores
        common = list(scored.values())[0].index
        for z in scored.values():
            common = common.intersection(z.index)

        basket = max(10, int(len(common) * basket_pct))
        if len(common) < basket:
            port_rets.append(np.nan)
            continue

        # Weighted composite score
        cs_names = list(scored.keys())
        w_arr    = np.array([weights.get(fn, 1 / len(cs_names)) for fn in cs_names])
        w_arr    = w_arr / w_arr.sum()

        mat       = np.column_stack([scored[fn].reindex(common).values for fn in cs_names])
        composite = pd.Series(mat @ w_arr, index=common)

        top_n    = set(composite.nlargest(basket).index)
        turnover = len(top_n.symmetric_difference(prev_hold)) / (2 * basket) if prev_hold else 1.0
        tc_drag  = turnover * TC

        # ── Macro overlay ─────────────────────────────────────────────────────
        invested  = 1.0

        # Unemployment YOY
        u_val = macro.asof(rdate) if macro is not None else np.nan
        if not pd.isna(u_val):
            if u_val < -1.0:
                invested = min(invested, 0.70)
                if "earnings_yield" in cs_names and "mom_52wk_high" in cs_names:
                    ey_i  = cs_names.index("earnings_yield")
                    m52_i = cs_names.index("mom_52wk_high")
                    w_arr[ey_i]  = min(1.0, w_arr[ey_i]  + 0.10)
                    w_arr[m52_i] = max(0.0, w_arr[m52_i] - 0.10)
                    w_arr = w_arr / w_arr.sum()
            elif u_val < -0.5:
                invested = min(invested, 0.85)
                if "earnings_yield" in cs_names and "mom_52wk_high" in cs_names:
                    ey_i  = cs_names.index("earnings_yield")
                    m52_i = cs_names.index("mom_52wk_high")
                    w_arr[ey_i]  = min(1.0, w_arr[ey_i]  + 0.05)
                    w_arr[m52_i] = max(0.0, w_arr[m52_i] - 0.05)
                    w_arr = w_arr / w_arr.sum()

        # VIX Volatility (raw level)
        v_val = vix_macro.asof(rdate) if vix_macro is not None else np.nan
        if not pd.isna(v_val):
            if v_val > 30:   # Severe volatility (VIX > 30)
                invested = min(invested, 0.50)
            elif v_val > 25: # High volatility (VIX > 25)
                invested = min(invested, 0.80)

        period_slice = ret_wide.loc[
            (ret_wide.index > rdate) & (ret_wide.index <= next_rdate),
            list(top_n),
        ].dropna(how="all", axis=1)

        if period_slice.empty:
            port_rets.append(np.nan)
            continue

        if score_weighted:
            # Rank-proportional weights within basket: highest composite score → highest weight.
            # Rank rather than raw score avoids sensitivity to score magnitude/skew.
            live = [c for c in period_slice.columns if c in composite.index]
            if live:
                ranks = composite.reindex(live).rank()          # 1 = lowest, N = highest
                stock_w = (ranks / ranks.sum()).reindex(period_slice.columns).fillna(0)
                sw_monthly = period_slice.mul(stock_w, axis=1).sum(axis=1)
            else:
                sw_monthly = period_slice.mean(axis=1)
            raw_ret = (1 + sw_monthly).prod() - 1
        else:
            raw_ret = (1 + period_slice.mean(axis=1)).prod() - 1
        rf_period     = (1 + RISK_FREE_ANN) ** (hold_months / 12) - 1
        period_ret    = raw_ret * invested + rf_period * (1 - invested) - tc_drag
        port_rets.append(period_ret)
        prev_hold = top_n

        log_rows.append({
            "rebal_date": rdate,
            "n_stocks":   basket,
            "turnover":   round(turnover * 100, 1),
            "period_ret": round(period_ret * 100, 2),
            "raw_ret":    round(raw_ret * 100, 2),
            "tc_drag_bp": round(tc_drag * 10_000, 1),
        })

    holdings_log = pd.DataFrame(log_rows)
    return pd.Series(port_rets, index=rebal_dates[:-1]), holdings_log


def build_spy_benchmark(ret_wide: pd.DataFrame, hold_months: int) -> pd.Series:
    """3-month SPY total return benchmark."""
    raw = yf.download("SPY", start=str(ret_wide.index[0].date()),
                      end="2026-06-01", interval="1mo",
                      auto_adjust=True, progress=False)["Close"]
    # yfinance may return DataFrame with Ticker multi-index — squeeze to Series
    spy = raw.squeeze() if isinstance(raw, pd.DataFrame) else raw
    spy.index  = pd.to_datetime(spy.index) + pd.offsets.MonthEnd(0)
    spy_ret    = spy.pct_change().reindex(ret_wide.index)
    dates      = ret_wide.index
    rebal_dates = dates[::hold_months]
    bench_rets  = []
    for i, rd in enumerate(rebal_dates[:-1]):
        nrd = rebal_dates[i + 1]
        sl  = spy_ret.loc[(spy_ret.index > rd) & (spy_ret.index <= nrd)]
        bench_rets.append((1 + sl.fillna(0)).prod() - 1)
    return pd.Series(bench_rets, index=rebal_dates[:-1])


# ══════════════════════════════════════════════════════════════════════════════
# DECILE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_decile_backtest(
    ret_wide:    pd.DataFrame,
    panels:      dict,
    is_liquid:   pd.DataFrame,
    hold_months: int  = HOLD_MONTHS,
    weights:     dict = None,
    n_deciles:   int  = 10,
) -> dict[str, pd.Series]:
    """
    At each rebalance date rank all scored stocks into n_deciles equal buckets
    by composite score and track equal-weight period returns per bucket.

    No macro overlay applied here — pure cross-sectional factor spread.
    Returns dict: {"D1": pd.Series, ..., "D10": pd.Series, "SPY": pd.Series}.
    D1 = top decile (highest composite score), D10 = bottom.
    """
    weights      = weights or FACTOR_WEIGHTS
    factor_names = list(panels.keys())
    dates        = ret_wide.index
    rebal_dates  = dates[::hold_months]

    # accumulate per-period returns for each decile
    bucket_rets = {f"D{d}": [] for d in range(1, n_deciles + 1)}
    index_out   = []

    for i, rdate in enumerate(rebal_dates[:-1]):
        next_rdate = rebal_dates[i + 1]

        universe = set(ret_wide.columns)
        if not is_liquid.empty and rdate in is_liquid.index:
            liq_row  = is_liquid.loc[rdate].fillna(False)
            universe = universe.intersection(liq_row[liq_row].index)

        scored = {}
        for fn in factor_names:
            p = panels[fn]
            if rdate not in p.index:
                continue
            row = p.loc[rdate].dropna().replace([np.inf, -np.inf], np.nan).dropna()
            row = row[row.index.isin(universe)]
            z   = _zscore(row)
            if len(z) > n_deciles * 3:   # need enough stocks for meaningful deciles
                scored[fn] = z

        if not scored:
            for d in range(1, n_deciles + 1):
                bucket_rets[f"D{d}"].append(np.nan)
            index_out.append(rdate)
            continue

        common = list(scored.values())[0].index
        for z in scored.values():
            common = common.intersection(z.index)

        if len(common) < n_deciles * 3:
            for d in range(1, n_deciles + 1):
                bucket_rets[f"D{d}"].append(np.nan)
            index_out.append(rdate)
            continue

        cs_names = list(scored.keys())
        w_arr    = np.array([weights.get(fn, 1 / len(cs_names)) for fn in cs_names])
        w_arr    = w_arr / w_arr.sum()
        mat      = np.column_stack([scored[fn].reindex(common).values for fn in cs_names])
        composite = pd.Series(mat @ w_arr, index=common).sort_values(ascending=False)

        # Split into equal-sized decile buckets
        n        = len(composite)
        cut_size = n / n_deciles
        period_slice = ret_wide.loc[
            (ret_wide.index > rdate) & (ret_wide.index <= next_rdate)
        ].dropna(how="all", axis=1)

        for d in range(1, n_deciles + 1):
            lo = int(round((d - 1) * cut_size))
            hi = int(round(d * cut_size))
            bucket_stocks = set(composite.iloc[lo:hi].index)
            sl = period_slice.reindex(columns=list(bucket_stocks)).dropna(how="all", axis=1)
            if sl.empty:
                bucket_rets[f"D{d}"].append(np.nan)
            else:
                ret = (1 + sl.mean(axis=1)).prod() - 1
                bucket_rets[f"D{d}"].append(ret)

        index_out.append(rdate)

    return {k: pd.Series(v, index=index_out) for k, v in bucket_rets.items()}


def print_decile_table(
    label:       str,
    decile_rets: dict,
    spy_curve:   pd.Series,
    hold_months: int,
    min_p:       int = 4,
):
    W = 100
    print("\n" + "═" * W)
    print(f"  DECILE ANALYSIS — {label}")
    print(f"  D1 = top 10% of composite score  |  D10 = bottom 10%  |  no macro overlay")
    print("═" * W)

    hdr = (f"  {'':8}  {'Ann Ret':>8}  {'Ann Vol':>8}  {'Sharpe':>7}  "
           f"{'Max DD':>8}  {'Calmar':>7}  {'Hit%':>6}  {'t-stat':>7}  {'N':>5}")
    sep = "  " + "─" * (W - 2)
    print(hdr)
    print(sep)

    metrics_list = []
    n_deciles = len(decile_rets)

    for d in range(1, n_deciles + 1):
        key = f"D{d}"
        curve = decile_rets[key]
        m = compute_metrics(curve, hold_months=hold_months, min_periods=min_p)
        metrics_list.append(m)
        if not m:
            print(f"  {key:<8}  {'— insufficient data —':>70}")
            continue
        label_str = f"D{d} {'(top)' if d == 1 else '(bot)' if d == n_deciles else ''}"
        print(f"  {label_str:<8}"
              f"  {m['ann_return']:>+7.1f}%"
              f"  {m['ann_vol']:>7.1f}%"
              f"  {m['sharpe']:>7.3f}"
              f"  {m['max_drawdown']:>+7.1f}%"
              f"  {m['calmar']:>7.3f}"
              f"  {m['hit_rate']:>5.1f}%"
              f"  {m['t_stat']:>7.2f}"
              f"  {m['n_periods']:>5}")

    print(sep)

    # Spread: D1 minus D10
    m1  = metrics_list[0]   if metrics_list else {}
    m10 = metrics_list[-1]  if len(metrics_list) >= n_deciles else {}
    if m1 and m10:
        spread_ret = m1["ann_return"] - m10["ann_return"]
        spread_sh  = m1["sharpe"]     - m10["sharpe"]
        print(f"  {'D1–D10':8}  {spread_ret:>+7.1f}%"
              f"  {'':>8}  {spread_sh:>7.3f}"
              f"  {'':>8}  {'':>7}  {'':>6}  {'':>7}")

    print(sep)

    # SPY row
    m_spy = compute_metrics(spy_curve, hold_months=hold_months, min_periods=min_p)
    if m_spy:
        print(f"  {'SPY':<8}"
              f"  {m_spy['ann_return']:>+7.1f}%"
              f"  {m_spy['ann_vol']:>7.1f}%"
              f"  {m_spy['sharpe']:>7.3f}"
              f"  {m_spy['max_drawdown']:>+7.1f}%"
              f"  {m_spy['calmar']:>7.3f}"
              f"  {m_spy['hit_rate']:>5.1f}%"
              f"  {m_spy['t_stat']:>7.2f}"
              f"  {m_spy['n_periods']:>5}")
    print("═" * W)


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def print_results(label: str, curve: pd.Series, bench: pd.Series,
                  hold_months: int, log: pd.DataFrame, factor_names: list[str],
                  min_p: int = 4):
    m     = compute_metrics(curve, hold_months=hold_months, min_periods=min_p)
    m_spy = compute_metrics(bench, hold_months=hold_months, min_periods=min_p)
    if not m:
        print(f"\n  [{label}] — insufficient periods ({curve.dropna().__len__()})")
        return

    W = 80
    print("\n" + "═" * W)
    print(f"  {label}")
    dates = curve.dropna().index
    if len(dates):
        print(f"  Period : {dates[0].date()} – {dates[-1].date()}"
              f"  ({m['n_periods']} rebalances, {hold_months}-month hold)")
    print(f"  Basket : top {BASKET_PCT:.0%} of top-{UNIVERSE_SIZE} liquid universe")
    print(f"  Factors: {', '.join(factor_names)}")
    print("═" * W)

    hdr = (f"  {'':25}  {'Ann Ret':>8}  {'Ann Vol':>8}  {'Sharpe':>7}  "
           f"{'Sortino':>7}  {'Calmar':>7}  {'Max DD':>8}  {'Hit%':>6}  {'t-stat':>7}")
    sep = "  " + "─" * (W + 18 - 2)
    print(hdr)
    print(sep)

    def _row(name, mm):
        if not mm:
            return f"  {name:25}  {'— no data —':>70}"
        sortino_val = mm.get('sortino', float('nan'))
        calmar_val  = mm.get('calmar',  float('nan'))
        sortino_str = f"{sortino_val:>7.3f}" if not (sortino_val != sortino_val) else "    n/a"
        calmar_str  = f"{calmar_val:>7.3f}"  if not (calmar_val  != calmar_val)  else "    n/a"
        return (f"  {name:25}"
                f"  {mm['ann_return']:>+7.1f}%"
                f"  {mm['ann_vol']:>7.1f}%"
                f"  {mm['sharpe']:>7.3f}"
                f"  {sortino_str}"
                f"  {calmar_str}"
                f"  {mm['max_drawdown']:>+7.1f}%"
                f"  {mm['hit_rate']:>5.1f}%"
                f"  {mm['t_stat']:>7.2f}")

    print(_row("Updated model (top 25%)", m))
    print(_row("SPY benchmark", m_spy))

    if not log.empty:
        print(sep)
        print(f"\n  Rebalance log:")
        hdr2 = f"  {'Date':12}  {'N stocks':>9}  {'Turnover':>9}  {'Period ret':>11}  {'TC drag (bp)':>13}"
        print(hdr2)
        for _, r in log.iterrows():
            print(f"  {str(r['rebal_date'].date()):12}  {int(r['n_stocks']):>9}"
                  f"  {r['turnover']:>8.1f}%"
                  f"  {r['period_ret']:>+10.2f}%"
                  f"  {r['tc_drag_bp']:>13.1f}")

    print("═" * W)


def print_monthly_calendar(label: str, curve: pd.Series, bench: pd.Series, hold_months: int):
    """Year × month return calendar with annual totals and SPY comparison."""
    if hold_months != 1:
        return  # calendar only meaningful for monthly rebalance
    W = 120
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    print("\n" + "═" * W)
    print(f"  MONTHLY RETURNS — {label}")
    print("═" * W)
    hdr = f"  {'Year':>4}  " + "  ".join(f"{m:>5}" for m in months) + f"  {'Annual':>7}"
    print(hdr)
    print("  " + "─" * (W - 2))

    def _calendar(series, name):
        years = sorted(series.index.year.unique())
        for yr in years:
            yr_rets = series[series.index.year == yr]
            cells = []
            annual = (1 + yr_rets).prod() - 1
            for mo in range(1, 13):
                mo_data = yr_rets[yr_rets.index.month == mo]
                if mo_data.empty:
                    cells.append(f"{'---':>5}")
                else:
                    v = mo_data.iloc[0] * 100
                    cells.append(f"{v:>+5.1f}")
            ann_str = f"{annual*100:>+6.1f}%"
            print(f"  {yr:>4}  " + "  ".join(cells) + f"  {ann_str}")
        print()

    print(f"  Model:")
    _calendar(curve.dropna(), "Model")
    print(f"  SPY:")
    _calendar(bench.dropna(), "SPY")
    print("═" * W)


def print_annual_decile_breakdown(label: str, decile_rets: dict):
    """Annual return per decile — shows how the spread evolves year by year."""
    n_deciles = len(decile_rets)
    d1 = decile_rets.get("D1", pd.Series(dtype=float)).dropna()
    if d1.empty:
        return
    years = sorted(d1.index.year.unique())
    W = 130

    print("\n" + "═" * W)
    print(f"  ANNUAL DECILE RETURNS — {label}")
    print(f"  D1 = highest composite score  |  D{n_deciles} = lowest")
    print("═" * W)

    col_w = 7
    hdr = f"  {'Year':>4}  " + "  ".join(f"{'D'+str(d):>{col_w}}" for d in range(1, n_deciles+1)) + f"  {'D1-D10':>{col_w}}"
    print(hdr)
    print("  " + "─" * (W - 2))

    for yr in years:
        cells = []
        ann_rets = {}
        for d in range(1, n_deciles + 1):
            key = f"D{d}"
            s = decile_rets.get(key, pd.Series(dtype=float))
            yr_s = s[s.index.year == yr].dropna()
            if yr_s.empty:
                cells.append(f"{'---':>{col_w}}")
                ann_rets[d] = None
            else:
                ann = (1 + yr_s).prod() - 1
                cells.append(f"{ann*100:>+{col_w}.1f}%")
                ann_rets[d] = ann
        if ann_rets.get(1) is not None and ann_rets.get(n_deciles) is not None:
            spread = (ann_rets[1] - ann_rets[n_deciles]) * 100
            spread_str = f"{spread:>+{col_w}.1f}%"
        else:
            spread_str = f"{'---':>{col_w}}"
        print(f"  {yr:>4}  " + "  ".join(cells) + f"  {spread_str}")

    print("  " + "─" * (W - 2))
    # Average row
    avg_cells = []
    avg_rets = {}
    for d in range(1, n_deciles + 1):
        s = decile_rets.get(f"D{d}", pd.Series(dtype=float)).dropna()
        if s.empty:
            avg_cells.append(f"{'---':>{col_w}}")
        else:
            avg = (1 + s).prod() ** (12 / len(s)) - 1
            avg_cells.append(f"{avg*100:>+{col_w}.1f}%")
            avg_rets[d] = avg
    if avg_rets.get(1) is not None and avg_rets.get(n_deciles) is not None:
        avg_spread = (avg_rets[1] - avg_rets[n_deciles]) * 100
        avg_spread_str = f"{avg_spread:>+{col_w}.1f}%"
    else:
        avg_spread_str = f"{'---':>{col_w}}"
    print(f"  {'Avg':>4}  " + "  ".join(avg_cells) + f"  {avg_spread_str}")
    print("═" * W)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-mode", default="core",
                        choices=["core", "research"],
                        help="`core` = canonical three-factor model, `research` = extended stack.")
    parser.add_argument("--is-only",  action="store_true", help="Run IS period only")
    parser.add_argument("--oos-only", action="store_true", help="Run OOS period only")
    parser.add_argument("--no-tech",  action="store_true",
                        help="Exclude technical factors (mom_52wk_high, sector_rs_1m); "
                             "weights renormalise automatically over neg_dispersion + earnings_yield")
    parser.add_argument("--oos-start", default=None,
                        help="Override OOS start date (YYYY-MM-DD). Useful for post-event windows.")
    parser.add_argument("--exclude-months", default=None,
                        help="Comma-separated YYYY-MM months to drop from OOS curve before "
                             "re-annualizing (e.g. '2025-01,2025-02,2025-03').")
    parser.add_argument("--oos-end", default=None,
                        help="Override OOS end date (YYYY-MM-DD). Extends window beyond 2025-12-31.")
    parser.add_argument("--score-weighted", action="store_true",
                        help="Weight basket by composite score rank instead of equal-weighting.")
    parser.add_argument("--weight-scheme", default=None,
                        choices=["icir", "capped", "equal", "opt"],
                        help="Weight scheme. Core mode defaults to `opt`; research mode defaults to `icir`.")
    parser.add_argument("--exclude-factors", default=None,
                        help="Comma-separated factor names to drop (e.g. 'earnings_yield,sue').")
    parser.add_argument("--no-sector-neutral", action="store_true",
                        help="Use raw 52wk-high proximity instead of sector-demeaned residual.")
    parser.add_argument("--grid-search", action="store_true",
                        help="Grid search over factor weights (5%% steps) and rank by IS decile spread.")
    args = parser.parse_args()

    run_is  = not args.oos_only
    run_oos = not args.is_only

    if args.weight_scheme is None:
        args.weight_scheme = "opt" if args.model_mode == "core" else "icir"

    selected_weights = get_selected_weights(args.model_mode, args.weight_scheme)
    auto_excluded = get_default_excluded_factors(args.model_mode)
    user_excluded = parse_factor_list(args.exclude_factors)
    excluded_factors = sorted(set(auto_excluded + user_excluded))

    print("\n" + "═" * 70)
    print("  UPDATED MODEL BACKTEST")
    print(f"  Mode          : {args.model_mode.upper()}")
    print(f"  Weight scheme : {args.weight_scheme.upper()}")
    print(f"  Profile       : {describe_model_mode(args.model_mode)}")
    print("═" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n  Loading returns...")
    oos_end_date = args.oos_end or OOS_END
    ret_wide = load_returns(oos_end=oos_end_date)
    print(f"  {ret_wide.shape[1]:,} stocks | {ret_wide.index[0].date()} – {ret_wide.index[-1].date()}")

    print("  Loading Compustat...")
    fund    = load_compustat()
    sic_map = pd.Series({int(p): _sic_to_etf(s)
                         for p, s in fund.sort_values("datadate")
                                         .groupby("permno")
                                         [("siccd" if "siccd" in fund.columns else "sic")]
                                         .last().items()}).dropna()
    print(f"  {fund['permno'].nunique():,} permnos | {sic_map.notna().sum():,} SIC-mapped")

    print("  Loading IBES...")
    ibes = load_ibes()
    print(f"  {ibes['permno'].nunique() if not ibes.empty else 0:,} permnos")

    # ── Build factor panels (full history, no IS/OOS split — rolling windows warm up naturally) ──
    print("\n  Building factor panels...")

    panels = build_factor_panels(
        ret_wide=ret_wide,
        fund=fund,
        ibes=ibes,
        sic_map=sic_map,
        include_technical=not args.no_tech,
        sector_neutral_momentum=not args.no_sector_neutral,
    )
    factor_descriptions = [
        (
            "mom_52wk_high",
            "F1  Raw 52-week high proximity (no sector demeaning)..."
            if args.no_sector_neutral else
            "F1  Residual 52-week high (sector-neutral momentum)...",
        ),
        ("earnings_yield", "F2  Earnings yield (ibq_ttm / market_cap, sector-neutral)..."),
        ("sector_rs_1m", "F3  Sector RS 1-month (yfinance)..."),
        ("neg_dispersion", "F4  Analyst dispersion (neg)..."),
        ("sue", "F6  SUE — earnings surprise (seasonal random walk, rdq PIT)..."),
    ]
    skipped = {"mom_52wk_high", "sector_rs_1m"} if args.no_tech else set()
    for factor_name, label in factor_descriptions:
        if factor_name in skipped:
            continue
        panel = panels.get(factor_name, pd.DataFrame())
        print(f"  {label}")
        print(f"       {panel.shape[1]:,} stocks" if not panel.empty else "       [empty]")

    print("  Liquidity screen...")
    is_liq = build_liquidity_screen(fund, ret_wide, UNIVERSE_SIZE)

    for fn in excluded_factors:
        if fn in panels:
            del panels[fn]
            print(f"  [exclude-factors] Dropped: {fn}")

    print(f"\n  Active factors ({len(panels)}): {list(panels.keys())}")
    if not panels:
        print("  [ERROR] No factor panels built."); return

    print("  F5  Unemployment YOY macro overlay...")
    macro = build_unemployment_macro(ret_wide)
    if macro is not None:
        pct_def = (macro.dropna() < -1.0).mean() * 100
        print(f"       {pct_def:.1f}% of months in defensive mode (z < -1.0)")
    else:
        print("       [unavailable — fred_raw.parquet missing]")

    print("  F6  VIX volatility macro overlay...")
    vix_m = build_vix_macro(ret_wide)
    if vix_m is not None:
        pct_def = (vix_m.dropna() > 25).mean() * 100
        print(f"       {pct_def:.1f}% of months in defensive mode (VIX > 25)")
    else:
        print("       [unavailable]")

    # Recompute weights for active factors only (re-normalise if any missing)
    active_weights = {k: selected_weights[k] for k in panels if k in selected_weights}
    total_w = sum(active_weights.values())
    if total_w == 0:
        print("  [ERROR] Selected weight scheme has no overlap with active factors.")
        return
    active_weights = {k: v / total_w for k, v in active_weights.items()}
    print("  Weights used:")
    for fn, w in sorted(active_weights.items(), key=lambda x: -x[1]):
        print(f"    {fn:<25} {w:.1%}")

    # ── Grid search ───────────────────────────────────────────────────────────
    if args.grid_search:
        factor_names = list(panels.keys())
        if len(factor_names) != 3:
            print(f"\n  [--grid-search] Requires exactly 3 active factors (got {len(factor_names)}); skipping.")
        else:
            print(f"\n  Running weight grid search over {factor_names} (5% steps, IS only)...")
            is_idx   = ret_wide.index[ret_wide.index <= IS_END]
            panels_gs = {k: v.reindex(is_idx) for k, v in panels.items()}
            is_liq_gs = is_liq.reindex(is_idx) if not is_liq.empty else pd.DataFrame()

            results = []
            f0, f1, f2 = factor_names
            for a in range(5, 91, 5):
                for b in range(5, 96 - a, 5):
                    c = 100 - a - b
                    if c < 5:
                        continue
                    w = {f0: a / 100, f1: b / 100, f2: c / 100}
                    dr = run_decile_backtest(ret_wide.reindex(is_idx), panels_gs, is_liq_gs,
                                            hold_months=HOLD_MONTHS, weights=w)
                    m1  = compute_metrics(dr["D1"],  hold_months=HOLD_MONTHS, min_periods=4) or {}
                    m10 = compute_metrics(dr["D10"], hold_months=HOLD_MONTHS, min_periods=4) or {}
                    if not m1 or not m10:
                        continue
                    ret_spread = m1["ann_return"]  - m10["ann_return"]
                    sh_spread  = m1["sharpe"]      - m10["sharpe"]
                    results.append({
                        f0: a, f1: b, f2: c,
                        "ret_spread": round(ret_spread, 4),
                        "sh_spread":  round(sh_spread,  4),
                        "d1_sharpe":  round(m1["sharpe"], 4),
                        "d1_ret":     round(m1["ann_return"], 4),
                        "d10_ret":    round(m10["ann_return"], 4),
                    })

            results.sort(key=lambda x: x["ret_spread"] + x["sh_spread"], reverse=True)

            W = 100
            print("\n" + "═" * W)
            print(f"  WEIGHT GRID SEARCH — IS 2005–2024  |  ranked by ret_spread + sharpe_spread")
            print(f"  Factors: {f0}  |  {f1}  |  {f2}")
            print("═" * W)
            hdr = (f"  {'':>3}  {f0[:12]:>12}  {f1[:10]:>10}  {f2[:10]:>10}"
                   f"  {'RetSpread':>10}  {'ShSpread':>9}  {'D1 Shr':>7}  {'D1 Ret':>7}  {'D10 Ret':>8}")
            print(hdr)
            print("  " + "─" * (W - 2))
            for i, r in enumerate(results[:25], 1):
                print(f"  {i:>3}  {r[f0]:>11}%  {r[f1]:>9}%  {r[f2]:>9}%"
                      f"  {r['ret_spread']*100:>+9.2f}%"
                      f"  {r['sh_spread']:>9.3f}"
                      f"  {r['d1_sharpe']:>7.3f}"
                      f"  {r['d1_ret']*100:>+6.1f}%"
                      f"  {r['d10_ret']*100:>+7.1f}%")
            print("═" * W)

            # Auto-select best and update active_weights for the IS/OOS run below
            best = results[0]
            active_weights = {f0: best[f0] / 100, f1: best[f1] / 100, f2: best[f2] / 100}
            print(f"\n  Best weights selected: "
                  + ", ".join(f"{k}={v:.0%}" for k, v in active_weights.items()))

    # ── IS backtest ───────────────────────────────────────────────────────────
    if run_is:
        print(f"\n  Running IS backtest ({FULL_START} – {IS_END})...")
        is_idx = ret_wide.index[ret_wide.index <= IS_END]
        panels_is = {k: v.reindex(is_idx) for k, v in panels.items()}
        is_liq_is = is_liq.reindex(is_idx) if not is_liq.empty else pd.DataFrame()
        curve_is, log_is = run_backtest(
            ret_wide.reindex(is_idx), panels_is, is_liq_is,
            macro=macro, vix_macro=vix_m, hold_months=HOLD_MONTHS,
            basket_pct=BASKET_PCT, weights=active_weights,
            score_weighted=False,
        )


        spy_is = build_spy_benchmark(ret_wide.reindex(is_idx), HOLD_MONTHS)
        print_results("IN-SAMPLE  2005 – 2024", curve_is, spy_is, HOLD_MONTHS,
                      log_is, factor_names=list(active_weights.keys()))
        print_monthly_calendar("IN-SAMPLE 2005–2024", curve_is, spy_is, HOLD_MONTHS)

        decile_rets_is = run_decile_backtest(
            ret_wide.reindex(is_idx), panels_is, is_liq_is,
            hold_months=HOLD_MONTHS, weights=active_weights)
        print_decile_table("IN-SAMPLE 2005–2024", decile_rets_is, spy_is, HOLD_MONTHS)
        print_annual_decile_breakdown("IN-SAMPLE 2005–2024", decile_rets_is)

        # Save
        curve_is.to_csv(RESULTS_DIR / "updated_model_is_curve.csv", header=["period_ret"])

    # ── OOS backtest ──────────────────────────────────────────────────────────
    if run_oos:
        oos_from = args.oos_start or OOS_START
        oos_idx = ret_wide.index[(ret_wide.index >= oos_from) & (ret_wide.index <= oos_end_date)]
        if len(oos_idx) == 0:
            print("\n  [SKIP] No OOS data — run ingestion/refresh_2025.py first.")
        else:
            print(f"\n  Running OOS backtest ({oos_from} – {OOS_END})...")
            # Use full history panels so rolling windows are properly warmed
            panels_oos = {k: v.reindex(oos_idx) for k, v in panels.items()}
            is_liq_oos = is_liq.reindex(oos_idx) if not is_liq.empty else pd.DataFrame()
            curve_oos, log_oos = run_backtest(
                ret_wide.reindex(oos_idx), panels_oos, is_liq_oos,
                macro=macro, vix_macro=vix_m, hold_months=HOLD_MONTHS,
                basket_pct=BASKET_PCT, weights=active_weights,
                score_weighted=False,
            )


            spy_oos = build_spy_benchmark(ret_wide.reindex(oos_idx), HOLD_MONTHS)
            print_results("OUT-OF-SAMPLE  2025", curve_oos, spy_oos,
                          HOLD_MONTHS, log_oos, factor_names=list(active_weights.keys()),
                          min_p=2)
            print_monthly_calendar("OUT-OF-SAMPLE 2025", curve_oos, spy_oos, HOLD_MONTHS)

            decile_rets_oos = run_decile_backtest(
                ret_wide.reindex(oos_idx), panels_oos, is_liq_oos,
                hold_months=HOLD_MONTHS, weights=active_weights)
            print_decile_table("OUT-OF-SAMPLE 2025", decile_rets_oos, spy_oos,
                               HOLD_MONTHS, min_p=2)
            print_annual_decile_breakdown("OUT-OF-SAMPLE 2025", decile_rets_oos)

            # ── Adjusted OOS: exclude shock months + re-annualize to 12 months ──
            if args.exclude_months:
                excl = [pd.Period(m.strip(), "M") for m in args.exclude_months.split(",")]
                mask = pd.Series(
                    [pd.Period(d, "M") not in excl for d in curve_oos.index],
                    index=curve_oos.index)
                curve_adj = curve_oos[mask].dropna()
                spy_adj   = spy_oos[mask].dropna()
                n_kept    = len(curve_adj)
                if n_kept >= 2:
                    # Annualize compound return to 12-month equivalent
                    def _adj_metrics(c, label):
                        cum   = (1 + c).prod() - 1
                        ann_r = (1 + cum) ** (12 / n_kept) - 1
                        mu    = c.mean(); std = c.std()
                        rf_m  = (1 + RISK_FREE_ANN) ** (1 / 12) - 1
                        sh    = (mu - rf_m) / std * np.sqrt(12) if std > 0 else np.nan
                        nav   = (1 + c).cumprod()
                        mdd   = ((nav - nav.cummax()) / nav.cummax()).min()
                        hit   = (c > 0).mean()
                        print(f"  {label:<25}  {ann_r*100:>+7.1f}%"
                              f"  {std*np.sqrt(12)*100:>7.1f}%"
                              f"  {sh:>7.3f}"
                              f"  {mdd*100:>+7.1f}%"
                              f"  {hit*100:>5.1f}%")

                    W = 80
                    excl_str = ", ".join(args.exclude_months.split(","))
                    print("\n" + "═" * W)
                    print(f"  OOS ADJUSTED — excluding {excl_str} | {n_kept} months re-annualized to 12")
                    print("═" * W)
                    print(f"  {'':25}  {'Ann Ret':>8}  {'Ann Vol':>8}  {'Sharpe':>7}  {'Max DD':>8}  {'Hit%':>6}")
                    print("  " + "─" * (W - 2))
                    _adj_metrics(curve_adj, "Updated model (adj)")
                    _adj_metrics(spy_adj,   "SPY benchmark (adj)")
                    print("═" * W)

            # Side-by-side IS vs OOS summary
            m_is  = compute_metrics(curve_is,  HOLD_MONTHS) if run_is else {}
            m_oos = compute_metrics(curve_oos, HOLD_MONTHS, min_periods=2)
            if m_is and m_oos:
                W = 75
                print("\n" + "═" * W)
                print("  IS vs OOS SUMMARY")
                print("═" * W)
                print(f"  {'Metric':<20}  {'IS 2005-2024':>14}  {'OOS 2025':>12}")
                print("  " + "─" * (W - 2))
                for k, label in [("ann_return","Ann Return"),("ann_vol","Ann Vol"),
                                  ("sharpe","Sharpe"),("max_drawdown","Max DD"),
                                  ("hit_rate","Hit Rate"),("calmar","Calmar")]:
                    iv = m_is.get(k, np.nan)
                    ov = m_oos.get(k, np.nan)
                    if k in ("ann_return","ann_vol","max_drawdown","hit_rate"):
                        print(f"  {label:<20}  {iv:>+13.1f}%  {ov:>+11.1f}%")
                    else:
                        print(f"  {label:<20}  {iv:>14.3f}  {ov:>12.3f}")
                print("═" * W)

            curve_oos.to_csv(RESULTS_DIR / "updated_model_oos_curve.csv",
                             header=["period_ret"])

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
