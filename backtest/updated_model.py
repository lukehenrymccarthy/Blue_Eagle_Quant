"""
backtest/updated_model.py
--------------------------
Five-factor model rebuilt using IC-validated signals from factor_evaluation.py.

Factor changes vs original model
──────────────────────────────────
  F1  Momentum    : mom_12m_skip1   → mom_52wk_high
                    (52-week high proximity; George & Hwang 2004)
  F2  Quality     : ROE+Accruals+GP+FCF composite
                    → earnings_yield   (ibq_ttm / market_cap; sector-neutral)
  F3  Sector RS   : 3-month RS vs SPY → 1-month RS vs SPY
  F4  Analyst     : rev_3m (mean-rec revision) → neg_dispersion
                    (low analyst disagreement; consensus = quality signal)
  F5  Macro overlay: unemployment YOY regime
                    (reduces equity exposure when unemployment is rising)

Factor weights
──────────────
  ICIR-proportional cross-sectional factors (sum to 1):
    neg_dispersion   74.9%  (IS ICIR 1.949)
    earnings_yield   13.0%  (IS ICIR 0.339)
    mom_52wk_high     6.6%  (IS ICIR 0.172)
    sector_rs_1m      5.4%  (IS ICIR 0.141)

  Macro overlay (time-series, unemployment YOY):
    z < -1.0  → 70% invested + tilt quality weight +10%
    z < -0.5  → 85% invested + tilt quality weight +5%

Portfolio construction
──────────────────────
  Universe   : top 1 000 stocks by market cap (liquidity screen)
  Basket     : top 25% of scored universe at each rebalance
  Hold       : 3-month rebalance
  Cost       : 10 bps per side on turnover (equally weighted)
  Benchmark  : SPY (reported alongside)

Output
──────
  IS  period : 2005-01-01 – 2024-12-31
  OOS period : 2025-01-01 – 2025-12-31

Usage:
    python backtest/updated_model.py
    python backtest/updated_model.py --is-only
    python backtest/updated_model.py --oos-only
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
RISK_FREE_ANN = 0.05
TC            = 0.001         # 10 bps per side

SECTOR_ETFS   = ["XLC","XLY","XLP","XLE","XLF","XLV","XLI","XLB","XLRE","XLK","XLU"]

# ICIR-proportional weights (sum to 1)
# earnings_yield replaces inv_debt_equity per user preference
FACTOR_WEIGHTS = {
    "neg_dispersion":  0.749,
    "earnings_yield":  0.130,
    "mom_52wk_high":   0.066,
    "sector_rs_1m":    0.054,
}

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_returns() -> pd.DataFrame:
    from ingestion.wrds_returns import load_returns as _lr, returns_wide
    rw = returns_wide(_lr())
    rw.index = pd.to_datetime(rw.index) + pd.offsets.MonthEnd(0)
    return rw.loc[FULL_START:OOS_END]


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
# FACTOR BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_52wk_high(ret_wide: pd.DataFrame) -> pd.DataFrame:
    """
    52-week high proximity signal (George & Hwang 2004).

    Computed as:  exp( log_cumret[t] − max(log_cumret[t-11..t]) )

    Interpretation: 1.0 = stock AT its 52-week high (maximum bullish).
    < 1.0 = stock below its recent high → lower score.

    Lagged 1 month to avoid look-ahead.
    """
    log_r    = np.log1p(ret_wide.fillna(0))
    cum_log  = log_r.cumsum()
    roll_max = cum_log.rolling(12, min_periods=6).max()
    proximity = np.exp(cum_log - roll_max)
    return proximity.shift(1)   # signal known at start of month


def build_earnings_yield(fund: pd.DataFrame, ret_wide: pd.DataFrame,
                          sic_map: pd.Series) -> pd.DataFrame:
    """
    Earnings yield: TTM net income / market cap (inverse P/E).
    Higher = cheaper relative to earnings = value-quality signal.
    Point-in-time via available_date. Sector-neutral z-score.
    """
    fund = fund.copy()
    if "earnings_yield" not in fund.columns:
        fund["earnings_yield"] = fund["ibq_ttm"] / fund["market_cap"].replace(0, np.nan)

    piv = (fund.dropna(subset=["earnings_yield"])
               .sort_values("available_date")
               .pivot_table(index="available_date", columns="permno",
                            values="earnings_yield", aggfunc="last"))
    piv.index = pd.to_datetime(piv.index)
    panel = piv.resample("ME").last().ffill().reindex(ret_wide.index, method="ffill")

    def _row_z(row):
        s = row.dropna().replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) < 20:
            return pd.Series(dtype=float)
        df = s.to_frame("v")
        df["sec"] = sic_map.reindex(df.index).fillna("Unknown")
        lo, hi = df["v"].quantile(0.01), df["v"].quantile(0.99)
        df["v"] = df["v"].clip(lo, hi)
        def _sz(g):
            return (g - g.mean()) / g.std() if len(g) >= 3 and g.std() > 0 else g * 0
        return df.groupby("sec")["v"].transform(_sz)

    return panel.apply(_row_z, axis=1)


def build_unemployment_macro(ret_wide: pd.DataFrame) -> pd.Series | None:
    """
    Unemployment YOY macro overlay signal.
    UNRATE.diff(12), rolling z-score, negated + 1-month pub lag.
    Negative z → unemployment rising → defensive tilt.
    """
    raw_path = Path("data/macro/fred_raw.parquet")
    if not raw_path.exists():
        return None
    raw    = pd.read_parquet(raw_path)
    raw.index = pd.to_datetime(raw.index)
    if "UNRATE" not in raw.columns:
        return None
    unrate   = raw["UNRATE"].resample("ME").last().dropna()
    yoy      = unrate.diff(12)
    rm       = yoy.rolling(36, min_periods=12).mean()
    rs       = yoy.rolling(36, min_periods=12).std().replace(0, np.nan)
    macro_z  = -((yoy - rm) / rs).shift(1)   # negate + pub lag
    return macro_z.reindex(ret_wide.index, method="ffill")


def build_sector_rs_1m(ret_wide: pd.DataFrame, sic_map: pd.Series) -> pd.DataFrame:
    """
    1-month sector ETF return vs SPY (relative strength).
    Stocks inherit their sector's score via SIC → ETF mapping.
    """
    if sic_map.empty:
        return pd.DataFrame()

    raw = yf.download(SECTOR_ETFS + ["SPY"], start="2004-01-01", end="2026-06-01",
                      interval="1mo", auto_adjust=True, progress=False)["Close"]
    raw.index = pd.to_datetime(raw.index) + pd.offsets.MonthEnd(0)
    mr  = raw.pct_change().sort_index()
    spy = mr["SPY"].fillna(0)

    etf_rs = {etf: mr[etf].fillna(0) - spy
              for etf in SECTOR_ETFS if etf in mr.columns}
    etf_rs_df = pd.DataFrame(etf_rs).reindex(ret_wide.index, method="ffill")

    # Broadcast ETF signal to stocks via sic_map
    stock_to_etf = sic_map.reindex(ret_wide.columns).dropna()
    etf_to_idx   = {e: i for i, e in enumerate(etf_rs_df.columns)}
    matched      = [(s, e) for s, e in stock_to_etf.items() if e in etf_to_idx]
    if not matched:
        return pd.DataFrame()

    stocks = [s for s, _ in matched]
    idxs   = [etf_to_idx[e] for _, e in matched]
    panel  = pd.DataFrame(etf_rs_df.values[:, idxs],
                          index=etf_rs_df.index, columns=stocks)
    panel.columns = panel.columns.astype(ret_wide.columns.dtype)
    return panel


def build_neg_dispersion(ibes: pd.DataFrame, ret_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Negative analyst recommendation dispersion.
    Low dispersion (analysts agree) = consensus quality signal → positive score.

    Source: ibes.stdev (std of individual recommendations).
    neg_dispersion = -stdev, already computed and stored in IBES signals file.
    """
    if ibes.empty or "neg_dispersion" not in ibes.columns:
        return pd.DataFrame()

    piv = (ibes.dropna(subset=["neg_dispersion"])
               .pivot_table(index="statpers", columns="permno",
                            values="neg_dispersion", aggfunc="last"))
    piv.index    = pd.to_datetime(piv.index)
    piv.columns  = piv.columns.astype(ret_wide.columns.dtype)
    return piv.resample("ME").last().ffill().reindex(ret_wide.index, method="ffill")


def build_liquidity_screen(fund: pd.DataFrame, ret_wide: pd.DataFrame) -> pd.DataFrame:
    """Boolean panel: True if stock is in top UNIVERSE_SIZE by market cap."""
    if "market_cap" not in fund.columns:
        return pd.DataFrame(True, index=ret_wide.index, columns=ret_wide.columns)

    mkt = (fund.dropna(subset=["market_cap"])
               .sort_values("available_date")
               .pivot_table(index="available_date", columns="permno",
                            values="market_cap", aggfunc="last"))
    mkt.index = pd.to_datetime(mkt.index)
    mkt = mkt.resample("ME").last().ffill().reindex(ret_wide.index, method="ffill")

    def _top_k(row):
        s = row.dropna()
        if s.empty:
            return pd.Series(False, index=row.index)
        res = pd.Series(False, index=row.index)
        res.loc[s.nlargest(UNIVERSE_SIZE).index] = True
        return res

    return mkt.apply(_top_k, axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    ret_wide:    pd.DataFrame,
    panels:      dict,
    is_liquid:   pd.DataFrame,
    macro:       pd.Series | None = None,
    hold_months: int   = HOLD_MONTHS,
    basket_pct:  float = BASKET_PCT,
    weights:     dict  = None,
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

        # ── Unemployment YOY macro overlay ────────────────────────────────────
        macro_val = macro.asof(rdate) if macro is not None else np.nan
        invested  = 1.0
        if not pd.isna(macro_val):
            if macro_val < -1.0:
                invested = 0.70
                if "earnings_yield" in w_arr and "mom_52wk_high" in cs_names:
                    ey_i  = cs_names.index("earnings_yield")
                    m52_i = cs_names.index("mom_52wk_high")
                    w_arr[ey_i]  = min(1.0, w_arr[ey_i]  + 0.10)
                    w_arr[m52_i] = max(0.0, w_arr[m52_i] - 0.10)
                    w_arr = w_arr / w_arr.sum()
            elif macro_val < -0.5:
                invested = 0.85
                if "earnings_yield" in cs_names and "mom_52wk_high" in cs_names:
                    ey_i  = cs_names.index("earnings_yield")
                    m52_i = cs_names.index("mom_52wk_high")
                    w_arr[ey_i]  = min(1.0, w_arr[ey_i]  + 0.05)
                    w_arr[m52_i] = max(0.0, w_arr[m52_i] - 0.05)
                    w_arr = w_arr / w_arr.sum()

        period_slice = ret_wide.loc[
            (ret_wide.index > rdate) & (ret_wide.index <= next_rdate),
            list(top_n),
        ].dropna(how="all", axis=1)

        if period_slice.empty:
            port_rets.append(np.nan)
            continue

        ew_monthly    = period_slice.mean(axis=1)
        raw_ret       = (1 + ew_monthly).prod() - 1
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
                  hold_months: int, log: pd.DataFrame, min_p: int = 4):
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
    print(f"  Factors: {', '.join(FACTOR_WEIGHTS.keys())}")
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


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is-only",  action="store_true", help="Run IS period only")
    parser.add_argument("--oos-only", action="store_true", help="Run OOS period only")
    args = parser.parse_args()

    run_is  = not args.oos_only
    run_oos = not args.is_only

    print("\n" + "═" * 70)
    print("  UPDATED MODEL BACKTEST")
    print("  Factors: 52wk-high | inv-D/E | sector-RS-1m | analyst-dispersion")
    print("═" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n  Loading returns...")
    ret_wide = load_returns()
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

    print("  F1  52-week high proximity...")
    p_52wk = build_52wk_high(ret_wide)
    print(f"       {p_52wk.shape[1]:,} stocks")

    print("  F2  Earnings yield (ibq_ttm / market_cap, sector-neutral)...")
    p_inv_de = build_earnings_yield(fund, ret_wide, sic_map)
    print(f"       {p_inv_de.shape[1]:,} stocks" if not p_inv_de.empty else "       [empty]")

    print("  F3  Sector RS 1-month (yfinance)...")
    p_sector = build_sector_rs_1m(ret_wide, sic_map)
    print(f"       {p_sector.shape[1]:,} stocks" if not p_sector.empty else "       [empty]")

    print("  F4  Analyst dispersion (neg)...")
    p_disp = build_neg_dispersion(ibes, ret_wide)
    print(f"       {p_disp.shape[1]:,} stocks" if not p_disp.empty else "       [empty]")

    print("  Liquidity screen...")
    is_liq = build_liquidity_screen(fund, ret_wide)

    panels = {}
    if not p_52wk.empty:     panels["mom_52wk_high"]   = p_52wk
    if not p_inv_de.empty:   panels["earnings_yield"]  = p_inv_de
    if not p_sector.empty:   panels["sector_rs_1m"]    = p_sector
    if not p_disp.empty:     panels["neg_dispersion"]  = p_disp
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

    # Recompute weights for active factors only (re-normalise if any missing)
    active_weights = {k: FACTOR_WEIGHTS[k] for k in panels if k in FACTOR_WEIGHTS}
    total_w = sum(active_weights.values())
    active_weights = {k: v / total_w for k, v in active_weights.items()}
    print("  Weights used:")
    for fn, w in sorted(active_weights.items(), key=lambda x: -x[1]):
        print(f"    {fn:<25} {w:.1%}")

    # ── IS backtest ───────────────────────────────────────────────────────────
    if run_is:
        print(f"\n  Running IS backtest ({FULL_START} – {IS_END})...")
        is_idx = ret_wide.index[ret_wide.index <= IS_END]
        panels_is = {k: v.reindex(is_idx) for k, v in panels.items()}
        is_liq_is = is_liq.reindex(is_idx) if not is_liq.empty else pd.DataFrame()

        curve_is, log_is = run_backtest(
            ret_wide.reindex(is_idx), panels_is, is_liq_is,
            macro=macro, hold_months=HOLD_MONTHS,
            basket_pct=BASKET_PCT, weights=active_weights)

        spy_is = build_spy_benchmark(ret_wide.reindex(is_idx), HOLD_MONTHS)
        print_results("IN-SAMPLE  2005 – 2024", curve_is, spy_is, HOLD_MONTHS, log_is)

        decile_rets_is = run_decile_backtest(
            ret_wide.reindex(is_idx), panels_is, is_liq_is,
            hold_months=HOLD_MONTHS, weights=active_weights)
        print_decile_table("IN-SAMPLE 2005–2024", decile_rets_is, spy_is, HOLD_MONTHS)

        # Save
        curve_is.to_csv(RESULTS_DIR / "updated_model_is_curve.csv", header=["period_ret"])

    # ── OOS backtest ──────────────────────────────────────────────────────────
    if run_oos:
        oos_idx = ret_wide.index[ret_wide.index >= OOS_START]
        if len(oos_idx) == 0:
            print("\n  [SKIP] No OOS data — run ingestion/refresh_2025.py first.")
        else:
            print(f"\n  Running OOS backtest ({OOS_START} – {OOS_END})...")
            # Use full history panels so rolling windows are properly warmed
            panels_oos = {k: v.reindex(oos_idx) for k, v in panels.items()}
            is_liq_oos = is_liq.reindex(oos_idx) if not is_liq.empty else pd.DataFrame()

            curve_oos, log_oos = run_backtest(
                ret_wide.reindex(oos_idx), panels_oos, is_liq_oos,
                macro=macro, hold_months=HOLD_MONTHS,
                basket_pct=BASKET_PCT, weights=active_weights)

            spy_oos = build_spy_benchmark(ret_wide.reindex(oos_idx), HOLD_MONTHS)
            print_results("OUT-OF-SAMPLE  2025", curve_oos, spy_oos,
                          HOLD_MONTHS, log_oos, min_p=2)

            decile_rets_oos = run_decile_backtest(
                ret_wide.reindex(oos_idx), panels_oos, is_liq_oos,
                hold_months=HOLD_MONTHS, weights=active_weights)
            print_decile_table("OUT-OF-SAMPLE 2025", decile_rets_oos, spy_oos,
                               HOLD_MONTHS, min_p=2)

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
