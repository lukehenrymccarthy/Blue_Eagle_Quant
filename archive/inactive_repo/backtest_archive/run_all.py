"""
backtest/run_all.py
-------------------
Unified backtest runner — executes all three factor suites and writes
every result to a single summary file.

Factor suites
─────────────
  1. Technical     — momentum + MA crossover parameter sweep (price data only)
                     Data: CRSP monthly returns
                     Fast: ~1-2 min for 32 combos.

  2. Analyst       — 8 IBES consensus signals × long-only (top 25/50/100) + long-short
                     Data: IBES recommendation summary, CRSP returns
                     Skipped gracefully if data files are missing.

  3. Sector        — 8 sector signals using SPDR ETFs, applied to CRSP stock universe
                     Data: yfinance (auto-downloaded), CRSP returns
                     Always available (requires only an internet connection).

Output
──────
  data/results/all_factors_summary.csv          ← unified leaderboard (all suites)
  data/results/equity_curves_technical.parquet
  data/results/equity_curves_analyst.parquet
  data/results/equity_curves_sector.parquet

Usage
─────
  python backtest/run_all.py                     # run everything
  python backtest/run_all.py --skip technical    # skip technical suite
  python backtest/run_all.py --skip technical analyst
  python backtest/run_all.py --top 10 --plot
"""

import sys
import itertools
import argparse
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── Paths & shared constants ───────────────────────────────────────────────────
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RISK_FREE = 0.05 / 12   # monthly
TC        = 0.001        # 10 bps per side


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(r: pd.Series) -> dict:
    """Standard performance metrics from a monthly return series."""
    r = r.dropna()
    if len(r) < 12:
        return {}
    ann_ret = (1 + r.mean()) ** 12 - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe  = (r.mean() - RISK_FREE) / r.std() * np.sqrt(12) if r.std() > 0 else np.nan
    nav     = (1 + r).cumprod()
    max_dd  = ((nav - nav.cummax()) / nav.cummax()).min()
    calmar  = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    hit     = (r > 0).mean()
    t_stat  = r.mean() / (r.std() / np.sqrt(len(r))) if r.std() > 0 else np.nan
    return {
        "ann_return":   round(ann_ret * 100, 2),
        "ann_vol":      round(ann_vol  * 100, 2),
        "sharpe":       round(sharpe,   3),
        "max_drawdown": round(max_dd   * 100, 2),
        "calmar":       round(calmar,   3),
        "hit_rate":     round(hit      * 100, 1),
        "t_stat":       round(t_stat,   2),
        "n_months":     len(r),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SUITE 1 — TECHNICAL  (momentum + MA crossover, price data only)
# ═══════════════════════════════════════════════════════════════════════════════

# Full parameter grid — 4 × 2 × 4 × 3 = 96 combos, runs in ~2 min.
TECHNICAL_GRID = {
    "momentum_lookback": [3, 6, 12, 18],          # months of cumulative return
    "momentum_skip":     [0, 1],                   # skip last N months (reversal buffer)
    "ma_windows":        [(1, 6), (1, 12), (3, 12), (6, 12)],  # (short, long) MA pair
    "w_momentum":        [1.0, 0.7, 0.5],          # weight on momentum; 1 - w on MA
}

TECHNICAL_START = "2010-01-01"
TECHNICAL_END   = "2024-12-31"
PORTFOLIO_SIZE  = 25


def _zscore(s: pd.Series) -> pd.Series:
    """Winsorise at 1%/99%, then z-score cross-sectionally."""
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    s = s.clip(lo, hi)
    std = s.std()
    return (s - s.mean()) / std if std > 0 else pd.Series(0.0, index=s.index)


def run_technical_suite() -> tuple[pd.DataFrame, dict]:
    """
    Load CRSP monthly returns, then run a momentum + MA crossover parameter
    sweep.  All signals are price-only — no fundamentals, macro, or analyst
    data required.  Signals are computed vectorially before the main loop so
    each per-date step is just an index lookup + top-N selection.
    """
    print("\n" + "═" * 70)
    print("  SUITE 1 — TECHNICAL  (momentum + MA crossover)")
    print("═" * 70)

    try:
        from ingestion.wrds_returns import load_returns, returns_wide as _returns_wide
    except ImportError as e:
        print(f"  [SKIP] Missing dependency: {e}")
        return pd.DataFrame(), {}

    # ── Load CRSP returns ──────────────────────────────────────────────────────
    try:
        ret_df      = load_returns()
        ret_wide    = _returns_wide(ret_df)
        ret_wide.index = pd.to_datetime(ret_wide.index) + pd.offsets.MonthEnd(0)
        ret_wide    = ret_wide.loc[TECHNICAL_START:TECHNICAL_END]
        print(f"  CRSP: {ret_wide.shape[1]:,} stocks, "
              f"{ret_wide.index[0].date()} – {ret_wide.index[-1].date()}")
    except FileNotFoundError:
        print("  [SKIP] CRSP returns not found — run ingestion/wrds_returns.py first.")
        return pd.DataFrame(), {}

    # ── Pre-build synthetic price index (cumulative return, base=1) ───────────
    # Filling NaN with 0 means missing months don't break the chain.
    price_idx = (1 + ret_wide.fillna(0)).cumprod()

    # ── Pre-compute all momentum signals we'll need ────────────────────────────
    # mom_signals[(lb, skip)][date] → cross-sectional Series of cum returns
    # We compute them as rolling products on the return matrix (fast, vectorised).
    needed_lbs   = set(TECHNICAL_GRID["momentum_lookback"])
    needed_skips = set(TECHNICAL_GRID["momentum_skip"])
    mom_cache: dict[tuple, pd.DataFrame] = {}

    for lb in needed_lbs:
        # Full compound return over `lb` months ending at each date
        roll_prod = (1 + ret_wide.fillna(0)).rolling(lb).apply(np.prod, raw=True) - 1
        for skip in needed_skips:
            # shift(skip) pushes the window back: skip=1 gives 12-1 momentum
            mom_cache[(lb, skip)] = roll_prod.shift(skip)

    # ── Pre-compute all MA signals we'll need ─────────────────────────────────
    # ma_signals[(short, long)][date] → cross-sectional Series of price/ma - 1
    needed_ma = set(TECHNICAL_GRID["ma_windows"])
    ma_cache:  dict[tuple, pd.DataFrame] = {}

    for short_w, long_w in needed_ma:
        short_ma = price_idx.rolling(short_w, min_periods=1).mean()
        long_ma  = price_idx.rolling(long_w,  min_periods=max(long_w // 2, 1)).mean()
        ma_cache[(short_w, long_w)] = (short_ma / long_ma.replace(0, np.nan)) - 1

    # ── Parameter sweep ────────────────────────────────────────────────────────
    keys   = list(TECHNICAL_GRID.keys())
    combos = list(itertools.product(*TECHNICAL_GRID.values()))
    dates  = ret_wide.index

    print(f"  Running {len(combos)} combinations × {len(dates)-1} months ...\n")

    results, curves = [], {}

    for i, vals in enumerate(combos):
        params  = dict(zip(keys, vals))
        lb      = params["momentum_lookback"]
        skip    = params["momentum_skip"]
        ma_pair = params["ma_windows"]
        w_mom   = params["w_momentum"]
        w_ma    = 1.0 - w_mom

        label = (
            f"mom{lb}_skip{skip}"
            f"_ma{ma_pair[0]}-{ma_pair[1]}"
            f"_wm{int(w_mom * 10)}"
        )
        print(f"  [{i+1:>3}/{len(combos)}] {label}", end=" ... ", flush=True)

        mom_df = mom_cache[(lb, skip)]
        ma_df  = ma_cache[ma_pair]

        monthly_rets = []
        prev_hold    = set()

        for j in range(len(dates) - 1):
            rdate     = dates[j]
            next_date = dates[j + 1]

            mom_row = mom_df.loc[rdate].dropna()
            ma_row  = ma_df.loc[rdate].dropna()

            # Align to common stocks present on this date
            common = mom_row.index.intersection(ma_row.index)
            if len(common) < PORTFOLIO_SIZE:
                monthly_rets.append(np.nan)
                continue

            if w_mom == 1.0:
                score = _zscore(mom_row.loc[common])
            elif w_ma == 1.0:
                score = _zscore(ma_row.loc[common])
            else:
                score = (
                    _zscore(mom_row.loc[common]) * w_mom
                    + _zscore(ma_row.loc[common]) * w_ma
                )

            top_n    = set(score.nlargest(PORTFOLIO_SIZE).index)
            turnover = (
                len(top_n.symmetric_difference(prev_hold)) / (2 * PORTFOLIO_SIZE)
                if prev_hold else 1.0
            )
            port_ret = (
                ret_wide.loc[next_date, list(top_n)].dropna().mean()
                - turnover * TC
            )
            monthly_rets.append(port_ret)
            prev_hold = top_n

        r = pd.Series(monthly_rets, index=dates[:-1])
        m = compute_metrics(r)
        if not m:
            print("skipped")
            continue

        results.append({
            "category":          "Technical",
            "style":             "long-only",
            "run_id":            label,
            "momentum_lookback": lb,
            "momentum_skip":     skip,
            "ma_short":          ma_pair[0],
            "ma_long":           ma_pair[1],
            "w_momentum":        w_mom,
            **m,
        })
        curves[label] = r
        print(f"Sharpe={m['sharpe']:.3f}  Ann={m['ann_return']:+.1f}%  DD={m['max_drawdown']:.1f}%")

    summary = (
        pd.DataFrame(results)
        .sort_values("sharpe", ascending=False)
        .reset_index(drop=True)
    )

    pd.DataFrame(curves).to_parquet(RESULTS_DIR / "equity_curves_technical.parquet")
    print(f"\n  Saved equity curves → data/results/equity_curves_technical.parquet")

    return summary, curves


# ═══════════════════════════════════════════════════════════════════════════════
# SUITE 2 — FUNDAMENTAL SIGNALS  (Compustat)
# ═══════════════════════════════════════════════════════════════════════════════

FUNDAMENTAL_SIGNALS = {
    "roe_ttm":        "ROE (TTM)",
    "fcf_yield":      "FCF Yield",
    "earnings_yield": "Earnings Yield",
    "neg_pb":         "Value (neg P/B)",
    "gross_margin":   "Gross Margin",
    "asset_turnover": "Asset Turnover",
    "accruals":       "Accruals (neg)",
    "leverage":       "Leverage (neg)",
}
FUND_PORT_SIZES = [25, 50, 100]
FUND_START      = "2010-01-01"
FUND_END        = "2024-12-31"


def _engineer_fundamentals(f: pd.DataFrame) -> pd.DataFrame:
    """Add derived fundamental signals from raw Compustat columns."""
    f = f.copy()
    # Gross margin proxy: ibq_ttm / annualised quarterly sales
    f["gross_margin"] = np.where(
        f["saleq"] > 0, f["ibq_ttm"] / (f["saleq"] * 4), np.nan
    )
    # Asset turnover: efficiency of asset use
    f["asset_turnover"] = np.where(
        f["atq"] > 0, (f["saleq"] * 4) / f["atq"], np.nan
    )
    # Accruals: earnings quality (lower = higher cash-based earnings)
    f["accruals"] = np.where(
        f["atq"] > 0,
        -(f["ibq_ttm"] - f["oancfq_ttm"]) / f["atq"],   # negated: lower accruals = better
        np.nan,
    )
    # Value factor: negate P/B so high score = cheap
    if "pb_ratio" in f.columns:
        f["neg_pb"] = -f["pb_ratio"]
    elif "ceqq" in f.columns:
        mktcap = (f["cshoq"] * f["prccq"]).replace(0, np.nan)
        f["neg_pb"] = -mktcap / f["ceqq"].replace(0, np.nan)
    # Leverage: lower debt = higher quality (already negated)
    f["leverage"] = np.where(
        f["atq"] > 0, -(f["ltq"] / f["atq"]), np.nan
    )
    return f


def _fund_longonly(fund: pd.DataFrame, ret_wide: pd.DataFrame,
                   sig_col: str, port_size: int) -> pd.Series:
    dates = ret_wide.loc[FUND_START:FUND_END].index
    fund_sorted = fund.sort_values(["permno", "datadate"])
    avail = fund_sorted["available_date"].values
    rets, prev = [], set()

    for i, rdate in enumerate(dates[:-1]):
        next_date = dates[i + 1]
        snap = (fund_sorted[avail <= np.datetime64(rdate)]
                .groupby("permno")[sig_col]
                .last()
                .dropna())
        if len(snap) < port_size:
            rets.append(np.nan)
            continue
        top_n    = set(snap.nlargest(port_size).index)
        turnover = len(top_n.symmetric_difference(prev)) / (2 * port_size) if prev else 1.0
        if next_date not in ret_wide.index:
            rets.append(np.nan)
            continue
        valid    = ret_wide.loc[next_date].reindex(list(top_n)).dropna()
        rets.append(valid.mean() - turnover * TC if len(valid) > 0 else np.nan)
        prev = top_n
    return pd.Series(rets, index=dates[:-1])


def _fund_longshort(fund: pd.DataFrame, ret_wide: pd.DataFrame,
                    sig_col: str) -> pd.Series:
    dates = ret_wide.loc[FUND_START:FUND_END].index
    fund_sorted = fund.sort_values(["permno", "datadate"])
    avail = fund_sorted["available_date"].values
    rets, prev_l, prev_s = [], set(), set()

    for i, rdate in enumerate(dates[:-1]):
        next_date = dates[i + 1]
        snap = (fund_sorted[avail <= np.datetime64(rdate)]
                .groupby("permno")[sig_col]
                .last()
                .dropna())
        if len(snap) < 50:
            rets.append(np.nan)
            continue
        q     = max(1, len(snap) // 5)
        ranked = snap.sort_values(ascending=False)
        long_s  = set(ranked.head(q).index)
        short_s = set(ranked.tail(q).index)
        if next_date not in ret_wide.index:
            rets.append(np.nan)
            continue
        row      = ret_wide.loc[next_date]
        lr       = row.reindex(list(long_s)).dropna().mean()
        sr       = row.reindex(list(short_s)).dropna().mean()
        lt       = len(long_s.symmetric_difference(prev_l))  / (2 * max(len(long_s),  1))
        st_      = len(short_s.symmetric_difference(prev_s)) / (2 * max(len(short_s), 1))
        lr      -= lt  * TC
        sr      += st_ * TC
        rets.append(lr - sr if not (np.isnan(lr) or np.isnan(sr)) else np.nan)
        prev_l, prev_s = long_s, short_s
    return pd.Series(rets, index=dates[:-1])


def run_fundamental_suite() -> tuple[pd.DataFrame, dict]:
    print("\n" + "═" * 70)
    print("  SUITE 2 — FUNDAMENTAL SIGNALS  (Compustat)")
    print("═" * 70)

    fund_path    = Path("data/fundamentals/compustat_quarterly.parquet")
    returns_path = Path("data/fundamentals/crsp_monthly_returns.parquet")

    if not fund_path.exists():
        print("  [SKIP] Compustat data not found — run ingestion/wrds_fundamentals.py first.")
        return pd.DataFrame(), {}
    if not returns_path.exists():
        print("  [SKIP] CRSP returns not found — run ingestion/wrds_returns.py first.")
        return pd.DataFrame(), {}

    fund = pd.read_parquet(fund_path)
    fund["available_date"] = pd.to_datetime(fund["available_date"])
    fund["datadate"]       = pd.to_datetime(fund["datadate"])
    fund = _engineer_fundamentals(fund)

    returns  = pd.read_parquet(returns_path, columns=["permno", "date", "ret"])
    returns["date"] = pd.to_datetime(returns["date"]) + pd.offsets.MonthEnd(0)
    ret_wide = returns.dropna(subset=["ret"]).pivot(
        index="date", columns="permno", values="ret"
    )

    print(f"  Compustat : {fund['permno'].nunique():,} firms")
    print(f"  Returns   : {ret_wide.shape[1]:,} stocks\n")

    total  = len(FUNDAMENTAL_SIGNALS) * len(FUND_PORT_SIZES) + len(FUNDAMENTAL_SIGNALS)
    done   = 0
    results, curves = [], {}

    # Long-only
    for sig, label in FUNDAMENTAL_SIGNALS.items():
        if sig not in fund.columns:
            done += len(FUND_PORT_SIZES)
            continue
        for n in FUND_PORT_SIZES:
            run_id = f"{label} | top-{n}"
            print(f"  [{done+1:>3}/{total}] {run_id}", end=" ... ", flush=True)
            r = _fund_longonly(fund, ret_wide, sig, n)
            m = compute_metrics(r)
            if m:
                results.append({"category": "Fundamental", "style": "long-only",
                                 "run_id": run_id, "signal": sig, **m})
                curves[run_id] = r
                print(f"Sharpe={m['sharpe']:.3f}  Ann={m['ann_return']:+.1f}%  DD={m['max_drawdown']:.1f}%")
            else:
                print("skipped")
            done += 1

    # Long-short
    for sig, label in FUNDAMENTAL_SIGNALS.items():
        if sig not in fund.columns:
            done += 1
            continue
        run_id = f"{label} | L/S quintile"
        print(f"  [{done+1:>3}/{total}] {run_id}", end=" ... ", flush=True)
        r = _fund_longshort(fund, ret_wide, sig)
        m = compute_metrics(r)
        if m:
            results.append({"category": "Fundamental", "style": "long-short",
                             "run_id": run_id, "signal": sig, **m})
            curves[run_id] = r
            print(f"Sharpe={m['sharpe']:.3f}  Ann={m['ann_return']:+.1f}%  DD={m['max_drawdown']:.1f}%")
        else:
            print("skipped")
        done += 1

    summary = (pd.DataFrame(results)
               .sort_values("sharpe", ascending=False)
               .reset_index(drop=True))
    pd.DataFrame(curves).to_parquet(RESULTS_DIR / "equity_curves_fundamental.parquet")
    print(f"\n  Saved equity curves → data/results/equity_curves_fundamental.parquet")
    return summary, curves


# ═══════════════════════════════════════════════════════════════════════════════
# SUITE 3 — MACRO SIGNALS  (FRED market timing)
# ═══════════════════════════════════════════════════════════════════════════════

# (signal_col, label, sign)
# sign=+1: go long when signal is positive (rising = risk-on)
# sign=-1: go long when signal is negative (rising = risk-off, so we negate)
MACRO_TIMING_SIGNALS = [
    ("T10Y3M_zscore",      "10Y-3M Yield Curve",     +1),
    ("T10Y2Y_zscore",      "10Y-2Y Yield Curve",     +1),
    ("CPIAUCSL_zscore",    "CPI Z-Score",            -1),
    ("PAYEMS_zscore",      "Payrolls Z-Score",       -1),
    ("hy_spread_widening", "HY Spread Widening",     -1),
]
MACRO_START = "2005-01-01"
MACRO_END   = "2024-12-31"


def run_macro_suite() -> tuple[pd.DataFrame, dict]:
    """
    Market-timing backtest: each month, the FRED signal determines whether
    to be invested in the equal-weight CRSP universe (long-only = invest vs cash;
    long-short = long market vs short market).
    """
    print("\n" + "═" * 70)
    print("  SUITE 3 — MACRO SIGNALS  (FRED market timing)")
    print("═" * 70)

    macro_path   = Path("data/macro/fred_signals.parquet")
    returns_path = Path("data/fundamentals/crsp_monthly_returns.parquet")

    if not macro_path.exists():
        print("  [SKIP] FRED signals not found — run ingestion/fred_macro.py first.")
        return pd.DataFrame(), {}
    if not returns_path.exists():
        print("  [SKIP] CRSP returns not found — run ingestion/wrds_returns.py first.")
        return pd.DataFrame(), {}

    macro = pd.read_parquet(macro_path)
    macro.index = pd.to_datetime(macro.index)

    returns = pd.read_parquet(returns_path, columns=["permno", "date", "ret"])
    returns["date"] = pd.to_datetime(returns["date"]) + pd.offsets.MonthEnd(0)
    returns = returns.dropna(subset=["ret"])

    # Pre-compute equal-weight market return for each month
    ew_market = (returns.groupby("date")["ret"].mean()
                 .loc[MACRO_START:MACRO_END])
    dates = ew_market.index

    print(f"  FRED signals: {len(macro.columns)} series")
    print(f"  EW market   : {len(dates)} months  "
          f"({dates[0].date()} – {dates[-1].date()})\n")

    available_sigs = [s for s in MACRO_TIMING_SIGNALS if s[0] in macro.columns]
    total = len(available_sigs) * 2
    done  = 0
    results, curves = [], {}

    for sig_col, label, sign in available_sigs:
        for style, invested_when_pos in [("long-only", True), ("long-short", True)]:
            run_id = f"{label} | {'timing' if style == 'long-only' else 'L/S'}"
            print(f"  [{done+1:>2}/{total}] {run_id}", end=" ... ", flush=True)

            # Pre-compute the signed signal series aligned to rebalance dates
            sig_series = macro[sig_col].dropna() * sign
            sig_aligned = sig_series.reindex(dates, method="ffill")

            # Use expanding median as threshold so ~50% of months are invested
            # regardless of the signal's absolute level (avoids all-cash regimes)
            threshold = sig_aligned.expanding(min_periods=12).median()

            monthly_rets = []
            for i, rdate in enumerate(dates[:-1]):
                next_date = dates[i + 1]

                val  = sig_aligned.get(rdate)
                thr  = threshold.get(rdate)
                mkt_ret = ew_market.get(next_date, np.nan)

                if pd.isna(val) or pd.isna(thr) or np.isnan(mkt_ret):
                    monthly_rets.append(np.nan)
                    continue

                risk_on = val > thr  # above own historical median = directionally bullish

                if style == "long-only":
                    monthly_rets.append(mkt_ret if risk_on else 0.0)
                else:
                    monthly_rets.append(mkt_ret if risk_on else -mkt_ret)

            r = pd.Series(monthly_rets, index=dates[:-1])
            m = compute_metrics(r)
            if m:
                results.append({"category": "Macro", "style": style,
                                 "run_id": run_id, "signal": sig_col, **m})
                curves[run_id] = r
                print(f"Sharpe={m['sharpe']:.3f}  Ann={m['ann_return']:+.1f}%  DD={m['max_drawdown']:.1f}%")
            else:
                print("skipped")
            done += 1

    summary = (pd.DataFrame(results)
               .sort_values("sharpe", ascending=False)
               .reset_index(drop=True))
    pd.DataFrame(curves).to_parquet(RESULTS_DIR / "equity_curves_macro.parquet")
    print(f"\n  Saved equity curves → data/results/equity_curves_macro.parquet")
    return summary, curves


# ═══════════════════════════════════════════════════════════════════════════════
# SUITE 4 — ANALYST SIGNALS  (IBES)
# ═══════════════════════════════════════════════════════════════════════════════

ANALYST_SIGNALS = {
    "neg_meanrec":      "Consensus Bullishness",
    "buy_pct":          "Buy %",
    "net_upgrades":     "Net Upgrades",
    "rev_3m":           "Rec Revision 3m",
    "buy_pct_rev_3m":   "Buy % Change 3m",
    "coverage":         "Analyst Coverage",
    "coverage_chg_3m":  "Coverage Change 3m",
    "neg_dispersion":   "Analyst Conviction",
}
ANALYST_PORT_SIZES = [25, 50, 100]
ANALYST_START      = "2005-01-01"
ANALYST_END        = "2024-12-31"
QUINTILE_N         = 5


def _analyst_longonly(signals, ret_wide, sig_col, port_size) -> pd.Series:
    dates = ret_wide.loc[ANALYST_START:ANALYST_END].index
    rets, prev = [], set()
    for i, rdate in enumerate(dates[:-1]):
        next_date = dates[i + 1]
        snap = (
            signals[signals["statpers"] <= rdate]
            .sort_values("statpers")
            .groupby("permno")[sig_col]
            .last()
            .dropna()
        )
        if len(snap) < port_size:
            rets.append(np.nan)
            continue
        top_n    = set(snap.nlargest(port_size).index)
        turnover = len(top_n.symmetric_difference(prev)) / (2 * port_size) if prev else 1.0
        tc_drag  = turnover * TC
        if next_date not in ret_wide.index:
            rets.append(np.nan)
            continue
        valid    = ret_wide.loc[next_date].reindex(list(top_n)).dropna()
        rets.append(valid.mean() - tc_drag if len(valid) > 0 else np.nan)
        prev = top_n
    return pd.Series(rets, index=dates[:-1])


def _analyst_longshort(signals, ret_wide, sig_col) -> pd.Series:
    dates = ret_wide.loc[ANALYST_START:ANALYST_END].index
    rets, prev_l, prev_s = [], set(), set()
    for i, rdate in enumerate(dates[:-1]):
        next_date = dates[i + 1]
        snap = (
            signals[signals["statpers"] <= rdate]
            .sort_values("statpers")
            .groupby("permno")[sig_col]
            .last()
            .dropna()
        )
        if len(snap) < 50:
            rets.append(np.nan)
            continue
        q_size  = max(1, len(snap) // QUINTILE_N)
        ranked  = snap.sort_values(ascending=False)
        long_s  = set(ranked.head(q_size).index)
        short_s = set(ranked.tail(q_size).index)
        if next_date not in ret_wide.index:
            rets.append(np.nan)
            continue
        row       = ret_wide.loc[next_date]
        long_ret  = row.reindex(list(long_s)).dropna().mean()
        short_ret = row.reindex(list(short_s)).dropna().mean()
        long_to   = len(long_s.symmetric_difference(prev_l))  / (2 * max(len(long_s),  1))
        short_to  = len(short_s.symmetric_difference(prev_s)) / (2 * max(len(short_s), 1))
        long_ret  -= long_to  * TC
        short_ret += short_to * TC
        rets.append(
            long_ret - short_ret
            if not (np.isnan(long_ret) or np.isnan(short_ret))
            else np.nan
        )
        prev_l, prev_s = long_s, short_s
    return pd.Series(rets, index=dates[:-1])


def run_analyst_suite() -> tuple[pd.DataFrame, dict]:
    print("\n" + "═" * 70)
    print("  SUITE 2 — ANALYST SIGNALS  (IBES)")
    print("═" * 70)

    analyst_path = Path("data/analyst/ibes_signals.parquet")
    returns_path = Path("data/fundamentals/crsp_monthly_returns.parquet")

    if not analyst_path.exists():
        print("  [SKIP] IBES signals not found — run ingestion/ibes_analyst.py first.")
        return pd.DataFrame(), {}
    if not returns_path.exists():
        print("  [SKIP] CRSP returns not found — run ingestion/wrds_returns.py first.")
        return pd.DataFrame(), {}

    signals = pd.read_parquet(analyst_path)
    signals["statpers"] = pd.to_datetime(signals["statpers"]) + pd.offsets.MonthEnd(0)
    signals["permno"]   = signals["permno"].astype(int)

    returns = pd.read_parquet(returns_path, columns=["permno", "date", "ret"])
    returns["date"] = pd.to_datetime(returns["date"]) + pd.offsets.MonthEnd(0)
    ret_wide = returns.dropna(subset=["ret"]).pivot(
        index="date", columns="permno", values="ret"
    )

    print(f"  Signals : {signals['permno'].nunique():,} permnos")
    print(f"  Returns : {ret_wide.shape[1]:,} stocks\n")

    total  = len(ANALYST_SIGNALS) * len(ANALYST_PORT_SIZES) + len(ANALYST_SIGNALS)
    done   = 0
    results, curves = [], {}

    # Long-only
    for sig, label in ANALYST_SIGNALS.items():
        for n in ANALYST_PORT_SIZES:
            run_id = f"{label} | top-{n}"
            print(f"  [{done+1:>3}/{total}] {run_id}", end=" ... ", flush=True)
            r = _analyst_longonly(signals, ret_wide, sig, n)
            m = compute_metrics(r)
            if m:
                results.append({
                    "category": "Analyst",
                    "style":    "long-only",
                    "run_id":   run_id,
                    "signal":   sig,
                    **m,
                })
                curves[run_id] = r
                print(f"Sharpe={m['sharpe']:.3f}  Ann={m['ann_return']:+.1f}%  DD={m['max_drawdown']:.1f}%")
            else:
                print("skipped")
            done += 1

    # Long-short
    for sig, label in ANALYST_SIGNALS.items():
        run_id = f"{label} | L/S quintile"
        print(f"  [{done+1:>3}/{total}] {run_id}", end=" ... ", flush=True)
        r = _analyst_longshort(signals, ret_wide, sig)
        m = compute_metrics(r)
        if m:
            results.append({
                "category": "Analyst",
                "style":    "long-short",
                "run_id":   run_id,
                "signal":   sig,
                **m,
            })
            curves[run_id] = r
            print(f"Sharpe={m['sharpe']:.3f}  Ann={m['ann_return']:+.1f}%  DD={m['max_drawdown']:.1f}%")
        else:
            print("skipped")
        done += 1

    summary = (
        pd.DataFrame(results)
        .sort_values("sharpe", ascending=False)
        .reset_index(drop=True)
    )

    pd.DataFrame(curves).to_parquet(RESULTS_DIR / "equity_curves_analyst.parquet")
    print(f"\n  Saved equity curves → data/results/equity_curves_analyst.parquet")

    return summary, curves


# ═══════════════════════════════════════════════════════════════════════════════
# SUITE 3 — SECTOR ROTATION  (SPDR ETFs)
# ═══════════════════════════════════════════════════════════════════════════════

SECTOR_ETFS    = ["XLC", "XLY", "XLP", "XLE", "XLF",
                  "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]
SECTOR_SIGNALS = {
    "mom_1m":        "Momentum 1m",
    "mom_3m":        "Momentum 3m",
    "mom_6m":        "Momentum 6m",
    "mom_12m_skip1": "Momentum 12m (skip 1m)",
    "rel_vs_spy":    "Rel. Strength vs SPY (3m)",
    "ma_signal":     "Trend / 12m MA",
    "mean_rev_1m":   "Mean Reversion 1m",
    "vol_adj_mom":   "Vol-Adj. Momentum (3m)",
}
SECTOR_START = "2018-07-01"
SECTOR_END   = "2024-12-31"
LONG_N       = 3
SHORT_N      = 3


def _sic_to_etf(sic):
    if pd.isna(sic):
        return None
    sic = int(sic)
    if 1300 <= sic <= 1399 or 2900 <= sic <= 2999:                        return "XLE"
    if (1000 <= sic <= 1099 or 1200 <= sic <= 1299 or
            2600 <= sic <= 2699 or 2810 <= sic <= 2819 or
            2860 <= sic <= 2879 or 3300 <= sic <= 3399):                  return "XLB"
    if (3400 <= sic <= 3569 or 3720 <= sic <= 3743 or
            sic == 3812 or 4011 <= sic <= 4013 or
            4210 <= sic <= 4215 or 4510 <= sic <= 4512 or
            7510 <= sic <= 7515):                                          return "XLI"
    if (3570 <= sic <= 3579 or 3670 <= sic <= 3679 or
            7370 <= sic <= 7379):                                          return "XLK"
    if (4810 <= sic <= 4813 or 4820 <= sic <= 4822 or
            4830 <= sic <= 4841 or 4890 <= sic <= 4899 or
            7810 <= sic <= 7819 or 7990 <= sic <= 7999):                  return "XLC"
    if (2830 <= sic <= 2836 or 3841 <= sic <= 3851 or
            8000 <= sic <= 8099):                                          return "XLV"
    if 6500 <= sic <= 6552 or sic == 6798:                                return "XLRE"
    if 6000 <= sic <= 6499 or 6700 <= sic <= 6799:                       return "XLF"
    if (2000 <= sic <= 2199 or 5140 <= sic <= 5149 or
            5400 <= sic <= 5499 or sic == 5912):                          return "XLP"
    if 4900 <= sic <= 4991:                                               return "XLU"
    if (2300 <= sic <= 2399 or 2510 <= sic <= 2519 or
            5200 <= sic <= 5399 or 5500 <= sic <= 5699 or
            5700 <= sic <= 5899 or 5940 <= sic <= 5999 or
            7000 <= sic <= 7099 or 7200 <= sic <= 7299 or
            7900 <= sic <= 7989):                                          return "XLY"
    return None


def _pull_etf_data() -> tuple[pd.DataFrame, pd.Series, dict]:
    """Returns (etf_returns, spy_returns, signal_dict)."""
    print("  Downloading ETF prices from Yahoo Finance...")
    raw = yf.download(
        SECTOR_ETFS + ["SPY"],
        start="2017-01-01", end="2025-01-01",
        interval="1mo", auto_adjust=True, progress=False,
    )["Close"]
    raw.index = pd.to_datetime(raw.index) + pd.offsets.MonthEnd(0)
    raw = raw.sort_index()

    etf_ret = raw[SECTOR_ETFS].pct_change().dropna(how="any")
    spy_ret = raw["SPY"].pct_change().dropna()

    print(f"  ETF data : {len(etf_ret)} months "
          f"({etf_ret.index.min().date()} – {etf_ret.index.max().date()})")

    prices  = (1 + etf_ret.fillna(0)).cumprod()
    spy_3m  = (1 + spy_ret).rolling(3).apply(np.prod, raw=True) - 1
    mom_3m  = (1 + etf_ret).rolling(3).apply(np.prod, raw=True) - 1
    ma12    = prices.rolling(12, min_periods=10).mean()

    signals = {
        "mom_1m":        etf_ret,
        "mom_3m":        mom_3m,
        "mom_6m":        (1 + etf_ret).rolling(6).apply(np.prod, raw=True) - 1,
        "mom_12m_skip1": ((1 + etf_ret).rolling(12).apply(np.prod, raw=True) - 1).shift(1),
        "rel_vs_spy":    mom_3m.subtract(spy_3m, axis=0),
        "ma_signal":     (prices / ma12) - 1,
        "mean_rev_1m":   -etf_ret,
        "vol_adj_mom":   mom_3m / etf_ret.rolling(3).std().replace(0, np.nan),
    }
    return etf_ret, spy_ret, signals


def _sector_longonly(signal_df, stock_data, dates) -> pd.Series:
    rets, prev = [], set()
    for i, rdate in enumerate(dates[:-1]):
        next_date = dates[i + 1]
        if rdate not in signal_df.index:
            rets.append(np.nan)
            continue
        scores = signal_df.loc[rdate].dropna()
        if len(scores) < LONG_N:
            rets.append(np.nan)
            continue
        top_sec  = set(scores.nlargest(LONG_N).index)
        turnover = len(top_sec.symmetric_difference(prev)) / (2 * LONG_N) if prev else 1.0
        mask     = (stock_data["date"] == next_date) & (stock_data["sector_etf"].isin(top_sec))
        ret_val  = stock_data.loc[mask, "ret"].dropna().mean()
        rets.append(ret_val - turnover * TC if not np.isnan(ret_val) else np.nan)
        prev = top_sec
    return pd.Series(rets, index=dates[:-1])


def _sector_longshort(signal_df, stock_data, dates) -> pd.Series:
    rets, prev_l, prev_s = [], set(), set()
    for i, rdate in enumerate(dates[:-1]):
        next_date = dates[i + 1]
        if rdate not in signal_df.index:
            rets.append(np.nan)
            continue
        scores = signal_df.loc[rdate].dropna()
        if len(scores) < LONG_N + SHORT_N:
            rets.append(np.nan)
            continue
        long_s  = set(scores.nlargest(LONG_N).index)
        short_s = set(scores.nsmallest(SHORT_N).index)
        mask_l  = (stock_data["date"] == next_date) & (stock_data["sector_etf"].isin(long_s))
        mask_s  = (stock_data["date"] == next_date) & (stock_data["sector_etf"].isin(short_s))
        lr = stock_data.loc[mask_l, "ret"].dropna().mean()
        sr = stock_data.loc[mask_s, "ret"].dropna().mean()
        lt = len(long_s.symmetric_difference(prev_l))  / (2 * LONG_N)
        st = len(short_s.symmetric_difference(prev_s)) / (2 * SHORT_N)
        tc = (lt + st) / 2 * TC
        rets.append(lr - sr - tc if not (np.isnan(lr) or np.isnan(sr)) else np.nan)
        prev_l, prev_s = long_s, short_s
    return pd.Series(rets, index=dates[:-1])


def run_sector_suite() -> tuple[pd.DataFrame, dict]:
    print("\n" + "═" * 70)
    print("  SUITE 3 — SECTOR ROTATION  (SPDR ETFs)")
    print("═" * 70)

    returns_path = Path("data/fundamentals/crsp_monthly_returns.parquet")
    if not returns_path.exists():
        print("  [SKIP] CRSP returns not found — run ingestion/wrds_returns.py first.")
        return pd.DataFrame(), {}

    etf_ret, spy_ret, all_signals = _pull_etf_data()

    print("  Building stock–sector map from CRSP SIC codes...")
    stock_raw = pd.read_parquet(returns_path, columns=["permno", "date", "ret", "siccd"])
    stock_raw["date"]       = pd.to_datetime(stock_raw["date"]) + pd.offsets.MonthEnd(0)
    stock_raw["sector_etf"] = stock_raw["siccd"].apply(_sic_to_etf)
    stock_data = stock_raw.dropna(subset=["ret", "sector_etf"])
    print(f"  Mapped {stock_data['permno'].nunique():,} permnos to 11 sectors")

    etf_dates   = etf_ret.loc[SECTOR_START:SECTOR_END].index
    stock_dates = set(stock_data["date"].unique())
    rebal_dates = pd.DatetimeIndex(
        sorted(d for d in etf_dates if d in stock_dates)
    )
    print(f"  Rebalance dates: {rebal_dates[0].date()} – {rebal_dates[-1].date()} "
          f"({len(rebal_dates)} months)\n")

    total = len(SECTOR_SIGNALS) * 2
    done  = 0
    results, curves = [], {}

    for sig, label in SECTOR_SIGNALS.items():
        signal_df = all_signals[sig]

        for style, fn, style_label in [
            ("long-only",  _sector_longonly,  "long-only"),
            ("long-short", _sector_longshort, "L/S"),
        ]:
            run_id = f"{label} | {style_label}"
            print(f"  [{done+1:>2}/{total}] {run_id}", end=" ... ", flush=True)
            r = fn(signal_df, stock_data, rebal_dates)
            m = compute_metrics(r)
            if m:
                results.append({
                    "category": "Sector-Rotation",
                    "style":    style,
                    "run_id":   run_id,
                    "signal":   sig,
                    **m,
                })
                curves[run_id] = r
                print(f"Sharpe={m['sharpe']:.3f}  Ann={m['ann_return']:+.1f}%  DD={m['max_drawdown']:.1f}%")
            else:
                print("skipped")
            done += 1

    summary = (
        pd.DataFrame(results)
        .sort_values("sharpe", ascending=False)
        .reset_index(drop=True)
    )

    pd.DataFrame(curves).to_parquet(RESULTS_DIR / "equity_curves_sector.parquet")
    print(f"\n  Saved equity curves → data/results/equity_curves_sector.parquet")

    return summary, curves


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def print_leaderboard(df: pd.DataFrame, title: str, n: int = 10):
    top = df.head(n).copy()
    top.index = range(1, len(top) + 1)
    top.index.name = "rank"
    top["ann_return"]   = top["ann_return"].map("{:+.1f}%".format)
    top["ann_vol"]      = top["ann_vol"].map("{:.1f}%".format)
    top["sharpe"]       = top["sharpe"].map("{:.3f}".format)
    top["max_drawdown"] = top["max_drawdown"].map("{:.1f}%".format)
    top["calmar"]       = top["calmar"].map("{:.3f}".format)
    top["hit_rate"]     = top["hit_rate"].map("{:.1f}%".format)
    top["t_stat"]       = top["t_stat"].map("{:+.2f}".format)

    print("\n" + "─" * 110)
    print(f"  {title}")
    print("─" * 110)
    print(top[["category", "style", "run_id",
               "ann_return", "ann_vol", "sharpe",
               "max_drawdown", "calmar", "hit_rate", "t_stat", "n_months"]].to_string())
    print("─" * 110)


def print_by_category(all_results: pd.DataFrame, n: int):
    for cat in ["Technical", "Fundamental", "Macro", "Analyst", "Sector-Rotation"]:
        sub = all_results[all_results["category"] == cat]
        if sub.empty:
            continue
        print_leaderboard(
            sub.sort_values("sharpe", ascending=False).reset_index(drop=True),
            title=f"{cat.upper()} — Top {n} by Sharpe",
            n=n,
        )


def plot_top_curves(all_results: pd.DataFrame, equity_curves: dict, n: int = 3):
    """
    One subplot per category — top-N equity curves from each suite.
    Saved to data/results/equity_curves_all.png
    """
    categories = [c for c in ["Multi-Factor", "Analyst", "Sector-Rotation"]
                  if c in all_results["category"].values]
    if not categories:
        return

    fig, axes = plt.subplots(len(categories), 1,
                             figsize=(14, 5 * len(categories)))
    if len(categories) == 1:
        axes = [axes]

    bg     = "#0f1117"
    colors = ["#4fc3f7", "#81c784", "#ffb74d", "#f06292", "#ce93d8"]
    fig.patch.set_facecolor(bg)

    for ax, cat in zip(axes, categories):
        ax.set_facecolor(bg)
        ax.tick_params(colors="#aaaaaa", labelsize=9)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["bottom", "left"]:
            ax.spines[sp].set_color("#333")

        sub     = (all_results[all_results["category"] == cat]
                   .sort_values("sharpe", ascending=False))
        top_ids = sub.head(n)["run_id"].tolist()

        for i, run_id in enumerate(top_ids):
            if run_id not in equity_curves:
                continue
            r      = equity_curves[run_id].dropna()
            nav    = (1 + r).cumprod()
            sharpe = sub.loc[sub["run_id"] == run_id, "sharpe"].values[0]
            ax.plot(nav.index, nav.values,
                    color=colors[i % len(colors)], linewidth=1.6, alpha=0.9,
                    label=f"{run_id[:55]}  (SR={sharpe:.2f})")

        ax.axhline(1, color="#555", linewidth=0.6, linestyle="--")
        ax.set_title(f"{cat} — Top {n}", color="white", fontsize=11, pad=8)
        ax.set_ylabel("NAV", color="#aaa", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f×"))
        ax.legend(fontsize=7.5, loc="upper left",
                  facecolor="#1a1d26", edgecolor="#333", labelcolor="white")
        ax.grid(axis="y", color="#222", linewidth=0.5)

    plt.tight_layout()
    out = RESULTS_DIR / "equity_curves_all.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=bg)
    print(f"\n  Chart saved → {out.resolve()}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run all backtest suites")
    parser.add_argument("--skip", nargs="+", default=[],
                        choices=["technical", "fundamental", "macro", "analyst", "sector"],
                        help="Suites to skip (e.g. --skip technical analyst)")
    parser.add_argument("--top",  type=int, default=5,
                        help="Rows to show per leaderboard (default: 5)")
    parser.add_argument("--plot", action="store_true",
                        help="Save equity curve chart to data/results/")
    args = parser.parse_args()

    print("=" * 70)
    print("  SectorScope — All-Factor Backtest Runner")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    frames         = []
    equity_curves  = {}   # merged dict for plotting

    if "technical" not in args.skip:
        df, curves = run_technical_suite()
        if not df.empty:
            frames.append(df)
            equity_curves.update(curves)

    if "fundamental" not in args.skip:
        df, curves = run_fundamental_suite()
        if not df.empty:
            frames.append(df)
            equity_curves.update(curves)

    if "macro" not in args.skip:
        df, curves = run_macro_suite()
        if not df.empty:
            frames.append(df)
            equity_curves.update(curves)

    if "analyst" not in args.skip:
        df, curves = run_analyst_suite()
        if not df.empty:
            frames.append(df)
            equity_curves.update(curves)

    if "sector" not in args.skip:
        df, curves = run_sector_suite()
        if not df.empty:
            frames.append(df)
            equity_curves.update(curves)

    if not frames:
        print("\nNo results to report. Check that data files exist.")
        return

    all_results = pd.concat(frames, ignore_index=True)

    # ── Save combined summary ──────────────────────────────────────────────────
    out_csv = RESULTS_DIR / "all_factors_summary.csv"
    all_results.to_csv(out_csv, index=False)
    print(f"\n\n{'═' * 70}")
    print(f"  ALL RESULTS SAVED → {out_csv.resolve()}")
    print(f"  {len(all_results)} total configurations across "
          f"{all_results['category'].nunique()} suites")
    print("═" * 70)

    # ── Print leaderboards ─────────────────────────────────────────────────────
    print_leaderboard(
        all_results.sort_values("sharpe", ascending=False).reset_index(drop=True),
        title=f"OVERALL TOP {args.top} — All Suites Combined (ranked by Sharpe)",
        n=args.top,
    )
    print_by_category(all_results, n=args.top)

    # ── Best per category summary ──────────────────────────────────────────────
    print("\n  BEST CONFIG PER CATEGORY\n")
    for cat in ["Technical", "Fundamental", "Macro", "Analyst", "Sector-Rotation"]:
        sub = all_results[all_results["category"] == cat]
        if sub.empty:
            continue
        best = sub.sort_values("sharpe", ascending=False).iloc[0]
        print(f"  {cat:<18} │ {best['run_id'][:55]:<55}"
              f" │ Sharpe={best['sharpe']:+.3f}"
              f"  Ann={best['ann_return']:+.1f}%"
              f"  DD={best['max_drawdown']:.1f}%")
    print()

    if args.plot:
        plot_top_curves(all_results, equity_curves, n=3)


if __name__ == "__main__":
    main()
