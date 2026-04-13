"""
backtest/macro_comparison.py
----------------------------
Compares three macro regime filters on the five-factor model:
  A) No macro overlay (pure cross-sectional)
  B) HY credit spread widening (existing F5)
  C) YOY unemployment change (new candidate)

Also runs per-factor IC diagnostics to support factor reevaluation.

Usage:
    python backtest/macro_comparison.py
    python backtest/macro_comparison.py --basket 10 --hold 1
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sectorscope.metrics import compute_metrics
from sectorscope.utils import zscore as _zscore, sic_to_etf as _sic_to_etf

# ── Config ─────────────────────────────────────────────────────────────────────
RESULTS_DIR   = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RISK_FREE_ANN = 0.05
TC            = 0.001
START_DATE    = "2005-01-01"   # full history for factor construction
TRAIN_END     = "2024-12-31"   # last in-sample month
TEST_START    = "2025-01-01"   # out-of-sample window
END_DATE      = "2025-12-31"

SECTOR_ETFS   = ["XLC", "XLY", "XLP", "XLE", "XLF",
                 "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]
CREDIT_BETA_WINDOW = 36
UNIVERSE_SIZE = 1000

FACTOR_WEIGHTS = {
    "momentum_12m":       0.10,
    "fund_quality":       0.25,
    "sector_rs_3m":       0.25,
    "analyst_rev_3m":     0.25,
    "credit_spread_tilt": 0.15,
}

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS  (thin wrappers around the existing ingestion layer)
# ══════════════════════════════════════════════════════════════════════════════

def load_crsp_wide() -> pd.DataFrame:
    from ingestion.wrds_returns import load_returns, returns_wide
    ret_df = load_returns()
    rw = returns_wide(ret_df)
    rw.index = pd.to_datetime(rw.index) + pd.offsets.MonthEnd(0)
    return rw


def build_compustat_panels(ret_wide):
    import pyarrow.parquet as pq
    fund_path   = Path("data/fundamentals/compustat_quarterly.parquet")
    schema_names = pq.read_schema(fund_path).names
    sic_col   = "siccd" if "siccd" in schema_names else ("sic" if "sic" in schema_names else None)
    load_cols = ["permno", "datadate", "available_date",
                 "roe_ttm", "ibq_ttm", "oancfq_ttm", "atq", "saleq", "fcf_yield", "market_cap"]
    if sic_col: load_cols.append(sic_col)

    fund = pd.read_parquet(fund_path, columns=[c for c in load_cols if c in schema_names])
    fund["available_date"] = pd.to_datetime(fund["available_date"])
    fund["datadate"]       = pd.to_datetime(fund["datadate"])

    def _build_signal_panel(col):
        piv = (fund.dropna(subset=[col])
               .sort_values("available_date")
               .pivot_table(index="available_date", columns="permno",
                            values=col, aggfunc="last"))
        piv.index = pd.to_datetime(piv.index)
        return piv.resample("ME").last().ffill().reindex(ret_wide.index, method="ffill")

    if sic_col and sic_col in fund.columns:
        sic_latest = fund.sort_values("datadate").groupby("permno")[sic_col].last()
        sic_map    = sic_latest.apply(_sic_to_etf).dropna()
    else:
        sic_map = pd.Series(dtype=str)

    def _cs_z_sector_neutral(panel):
        def _row_z(row):
            s = row.dropna().replace([np.inf, -np.inf], np.nan)
            if len(s) < 10: return row * 0
            df = s.to_frame(name="val")
            df["sector"] = sic_map
            df["sector"] = df["sector"].fillna("Unknown")
            lo, hi = df["val"].quantile(0.01), df["val"].quantile(0.99)
            df["val"] = df["val"].clip(lo, hi)
            def _sec_z(grp):
                if len(grp) < 3 or grp.std() == 0: return grp * 0
                return (grp - grp.mean()) / grp.std()
            return df.groupby("sector")["val"].transform(_sec_z)
        return panel.apply(_row_z, axis=1)

    roe_z    = _cs_z_sector_neutral(_build_signal_panel("roe_ttm"))
    fcf_z    = _cs_z_sector_neutral(_build_signal_panel("fcf_yield")) if "fcf_yield" in fund.columns else pd.DataFrame()

    neg_acc_z = pd.DataFrame()
    if all(c in fund.columns for c in ["ibq_ttm", "oancfq_ttm", "atq"]):
        fund["neg_accruals"] = -(fund["ibq_ttm"] - fund["oancfq_ttm"]) / fund["atq"].replace(0, np.nan)
        neg_acc_z = _cs_z_sector_neutral(_build_signal_panel("neg_accruals"))

    gp_z = pd.DataFrame()
    if all(c in fund.columns for c in ["saleq", "atq"]):
        fund["gross_prof"] = fund["saleq"] / fund["atq"].replace(0, np.nan)
        gp_z = _cs_z_sector_neutral(_build_signal_panel("gross_prof"))

    components = [df for df in [roe_z, fcf_z, neg_acc_z, gp_z] if not df.empty]
    if components:
        fq = pd.concat(components, axis=0).groupby(level=0).mean()
        fq = fq.reindex(ret_wide.index, method="ffill")
    else:
        fq = pd.DataFrame()

    if "market_cap" in fund.columns:
        mkt = _build_signal_panel("market_cap")
        def _top_k(row):
            s = row.dropna()
            if len(s) == 0: return pd.Series(False, index=row.index)
            top_k = s.nlargest(UNIVERSE_SIZE).index
            res = pd.Series(False, index=row.index)
            res.loc[top_k] = True
            return res
        is_liq = mkt.apply(_top_k, axis=1)
    else:
        is_liq = pd.DataFrame(True, index=ret_wide.index, columns=ret_wide.columns)

    return fq, sic_map, is_liq


def build_sector_rs_panel(ret_wide, sic_map):
    raw = yf.download(SECTOR_ETFS + ["SPY"], start="2009-01-01", end="2025-06-01",
                      interval="1mo", auto_adjust=True, progress=False)["Close"]
    raw.index = pd.to_datetime(raw.index) + pd.offsets.MonthEnd(0)
    mr = raw.pct_change()
    spy_3m = (1 + mr["SPY"].fillna(0)).rolling(3).apply(np.prod, raw=True) - 1
    etf_rs = {}
    for etf in SECTOR_ETFS:
        if etf not in mr.columns: continue
        etf_3m = (1 + mr[etf].fillna(0)).rolling(3).apply(np.prod, raw=True) - 1
        etf_rs[etf] = etf_3m - spy_3m
    etf_rs_df = pd.DataFrame(etf_rs).reindex(ret_wide.index, method="ffill")
    if sic_map.empty: return pd.DataFrame()

    stock_to_etf = sic_map.reindex(ret_wide.columns).dropna()
    etf_to_idx   = {e: i for i, e in enumerate(etf_rs_df.columns)}
    matched      = [(s, e) for s, e in stock_to_etf.items() if e in etf_to_idx]
    if not matched: return pd.DataFrame()
    stocks_out = [s for s, _ in matched]
    etf_idxs   = [etf_to_idx[e] for _, e in matched]
    sp = pd.DataFrame(etf_rs_df.values[:, etf_idxs], index=etf_rs_df.index, columns=stocks_out)
    sp.columns = sp.columns.astype(ret_wide.columns.dtype)
    return sp


def build_analyst_panel(ret_wide):
    path = Path("data/analyst/ibes_signals.parquet")
    if not path.exists(): return pd.DataFrame()
    ibes = pd.read_parquet(path, columns=["permno", "statpers", "rev_3m"])
    ibes["statpers"] = pd.to_datetime(ibes["statpers"]) + pd.offsets.MonthEnd(0)
    ibes["permno"]   = ibes["permno"].astype(int)
    piv = ibes.dropna(subset=["rev_3m"]).pivot_table(
        index="statpers", columns="permno", values="rev_3m", aggfunc="last")
    return piv.resample("ME").last().ffill().reindex(ret_wide.index, method="ffill")


def build_credit_spread_panel(ret_wide, sic_map):
    macro_path = Path("data/macro/fred_signals.parquet")
    if not macro_path.exists() or sic_map.empty: return pd.DataFrame()

    signals    = pd.read_parquet(macro_path)
    signals.index = pd.to_datetime(signals.index) + pd.offsets.MonthEnd(0)
    if "hy_spread_widening" not in signals.columns: return pd.DataFrame()

    hy_spread    = signals["hy_spread_widening"].dropna().sort_index()
    roll_mean    = hy_spread.rolling(36, min_periods=12).mean()
    roll_std     = hy_spread.rolling(36, min_periods=12).std().replace(0, np.nan)
    credit_regime = ((hy_spread - roll_mean) / roll_std).reindex(ret_wide.index, method="ffill")
    delta_spread  = hy_spread.diff().dropna()

    raw = yf.download(SECTOR_ETFS, start="2007-01-01", end="2025-06-01",
                      interval="1mo", auto_adjust=True, progress=False)["Close"]
    raw.index   = pd.to_datetime(raw.index) + pd.offsets.MonthEnd(0)
    monthly_ret = raw.pct_change()

    etf_betas = {}
    for etf in SECTOR_ETFS:
        if etf not in monthly_ret.columns: continue
        er = monthly_ret[etf]
        common = delta_spread.index.intersection(er.index)
        roll_cov = er.reindex(common).rolling(CREDIT_BETA_WINDOW, min_periods=12).cov(delta_spread.reindex(common))
        roll_var = delta_spread.reindex(common).rolling(CREDIT_BETA_WINDOW, min_periods=12).var().replace(0, np.nan)
        etf_betas[etf] = roll_cov / roll_var

    if not etf_betas: return pd.DataFrame()
    beta_df          = pd.DataFrame(etf_betas).reindex(ret_wide.index, method="ffill")
    credit_tilt_etf  = beta_df.multiply(credit_regime, axis=0)

    stock_to_etf = sic_map.reindex(ret_wide.columns).dropna()
    etf_to_idx   = {e: i for i, e in enumerate(credit_tilt_etf.columns)}
    matched      = [(s, e) for s, e in stock_to_etf.items() if e in etf_to_idx]
    if not matched: return pd.DataFrame()

    stocks_out  = [s for s, _ in matched]
    etf_idxs    = [etf_to_idx[e] for _, e in matched]
    cp = pd.DataFrame(credit_tilt_etf.values[:, etf_idxs],
                      index=credit_tilt_etf.index, columns=stocks_out)
    cp.columns = cp.columns.astype(ret_wide.columns.dtype)
    return cp


# ══════════════════════════════════════════════════════════════════════════════
# MACRO SIGNALS  — existing + new candidate
# ══════════════════════════════════════════════════════════════════════════════

def build_macro_hy_spread(ret_wide) -> pd.Series | None:
    """Existing: 3-month HY spread change, rolling z-score, negated."""
    macro_path = Path("data/macro/fred_signals.parquet")
    if not macro_path.exists(): return None
    signals = pd.read_parquet(macro_path)
    signals.index = pd.to_datetime(signals.index) + pd.offsets.MonthEnd(0)
    if "hy_spread_widening" not in signals.columns: return None

    s         = signals["hy_spread_widening"].dropna()
    roll_mean = s.rolling(36, min_periods=12).mean()
    roll_std  = s.rolling(36, min_periods=12).std().replace(0, np.nan)
    macro_z   = -((s - roll_mean) / roll_std)          # negative = risk-off
    return macro_z.reindex(ret_wide.index, method="ffill")


def build_macro_unemployment_yoy(ret_wide) -> pd.Series | None:
    """
    New candidate: YOY change in unemployment rate.

    UNRATE_YOY = UNRATE.diff(12)  (percentage-point change vs 12 months ago)

    Rising unemployment (positive YOY) → risk-off → macro_z should be negative.
    Convention: macro_z < -1.0 triggers defensive tilt (same thresholds as HY spread).

    We compute a rolling z-score of the UNRATE_YOY series, then negate so that
    a spike in unemployment maps to a negative z — consistent with the existing
    macro overlay logic in run_one_config().

    Publication lag: UNRATE for month T is released early in month T+1.
    We shift the signal by 1 month to avoid look-ahead.
    """
    raw_path = Path("data/macro/fred_raw.parquet")
    if not raw_path.exists():
        print("  [WARN] fred_raw.parquet not found — unemployment macro unavailable")
        return None

    raw = pd.read_parquet(raw_path)
    raw.index = pd.to_datetime(raw.index)

    if "UNRATE" not in raw.columns:
        print("  [WARN] UNRATE not in fred_raw.parquet")
        return None

    unrate   = raw["UNRATE"].resample("ME").last().dropna()
    yoy      = unrate.diff(12)                            # pp change YOY

    roll_mean = yoy.rolling(36, min_periods=12).mean()
    roll_std  = yoy.rolling(36, min_periods=12).std().replace(0, np.nan)
    z         = (yoy - roll_mean) / roll_std              # positive = rising unemp
    macro_z   = -z                                        # negate → negative = risk-off
    macro_z   = macro_z.shift(1)                          # 1-month publication lag
    return macro_z.reindex(ret_wide.index, method="ffill")


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def build_factor_panels(ret_wide, fund_quality, sector_rs, analyst_rev, credit_spread):
    panels = {}

    roll = (1 + ret_wide.fillna(0)).rolling(12).apply(np.prod, raw=True) - 1
    panels["momentum_12m"] = roll.shift(1)

    if not fund_quality.empty:  panels["fund_quality"]       = fund_quality
    if not sector_rs.empty:     panels["sector_rs_3m"]       = sector_rs
    if not analyst_rev.empty:   panels["analyst_rev_3m"]     = analyst_rev
    if not credit_spread.empty: panels["credit_spread_tilt"] = credit_spread

    return panels


def run_backtest(
    ret_wide, panels, macro_factor, is_liquid_panel,
    basket, hold_months, weights=None,
):
    dates       = ret_wide.index
    rebal_dates = dates[::hold_months]
    factor_names = list(panels.keys())
    port_rets   = []
    prev_hold   = set()

    for i, rdate in enumerate(rebal_dates[:-1]):
        next_rdate = rebal_dates[i + 1]

        crsp_universe = set(ret_wide.columns)
        if rdate in is_liquid_panel.index:
            liquid_row    = is_liquid_panel.loc[rdate].fillna(False)
            crsp_universe = crsp_universe.intersection(liquid_row[liquid_row].index)

        scored = {}
        for fname in factor_names:
            panel = panels[fname]
            if rdate not in panel.index: continue
            row = panel.loc[rdate].dropna()
            row = row[row.index.isin(crsp_universe)]
            z   = _zscore(row)
            if len(z) >= basket: scored[fname] = z

        if not scored:
            port_rets.append(np.nan); continue

        common = list(scored.values())[0].index
        for z in scored.values():
            common = common.intersection(z.index)

        if len(common) < basket:
            port_rets.append(np.nan); continue

        effective_weights = (weights or FACTOR_WEIGHTS).copy()
        macro_val = macro_factor.asof(rdate) if macro_factor is not None else np.nan
        invested  = 1.0

        if not pd.isna(macro_val):
            if macro_val < -1.0:
                invested = 0.70
                if "fund_quality" in effective_weights and "momentum_12m" in effective_weights:
                    effective_weights["fund_quality"] += 0.10
                    effective_weights["momentum_12m"] = max(0, effective_weights["momentum_12m"] - 0.10)
            elif macro_val < -0.5:
                invested = 0.85
                if "fund_quality" in effective_weights and "momentum_12m" in effective_weights:
                    effective_weights["fund_quality"] += 0.05
                    effective_weights["momentum_12m"] = max(0, effective_weights["momentum_12m"] - 0.05)

        cs_names = list(scored.keys())
        w_arr    = np.array([effective_weights.get(fn, 1 / len(cs_names)) for fn in cs_names])
        w_arr    = w_arr / w_arr.sum() if w_arr.sum() > 0 else np.ones(len(cs_names)) / len(cs_names)

        factor_mat = np.column_stack([scored[fn].reindex(common).values for fn in cs_names])
        composite  = pd.Series(factor_mat @ w_arr, index=common)

        top_n    = set(composite.nlargest(basket).index)
        turnover = len(top_n.symmetric_difference(prev_hold)) / (2 * basket) if prev_hold else 1.0
        tc_drag  = turnover * TC

        period_slice = ret_wide.loc[
            (ret_wide.index > rdate) & (ret_wide.index <= next_rdate),
            list(top_n),
        ].dropna(how="all", axis=1)

        if period_slice.empty:
            port_rets.append(np.nan); continue

        ew_monthly    = period_slice.mean(axis=1)
        raw_ret       = (1 + ew_monthly).prod() - 1
        rf_per_period = (1 + RISK_FREE_ANN) ** (hold_months / 12) - 1
        compound_ret  = raw_ret * invested + rf_per_period * (1 - invested) - tc_drag
        port_rets.append(compound_ret)
        prev_hold = top_n

    return pd.Series(port_rets, index=rebal_dates[:-1])


# ══════════════════════════════════════════════════════════════════════════════
# FACTOR IC DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_factor_ic(panels: dict, ret_wide: pd.DataFrame, fwd_months: int = 1) -> pd.DataFrame:
    """
    Per-factor monthly IC (rank correlation with fwd_months-ahead return).

    Returns a DataFrame with columns:
        factor, mean_ic, ic_std, icir, t_stat, pct_positive, n_months
    """
    fwd_ret = ret_wide.rolling(fwd_months).apply(
        lambda x: (1 + x).prod() - 1, raw=True
    ).shift(-fwd_months)

    rows = []
    for fname, panel in panels.items():
        ics = []
        common_dates = panel.index.intersection(fwd_ret.index)
        for dt in common_dates:
            scores = panel.loc[dt].dropna()
            fwd    = fwd_ret.loc[dt].dropna()
            common = scores.index.intersection(fwd.index)
            if len(common) < 30: continue
            ic, _ = sp_stats.spearmanr(scores.loc[common], fwd.loc[common])
            if not np.isnan(ic): ics.append(ic)

        if not ics:
            continue

        ics    = np.array(ics)
        mean_ic = ics.mean()
        ic_std  = ics.std()
        icir    = mean_ic / ic_std if ic_std > 0 else np.nan
        t_stat  = mean_ic / (ic_std / np.sqrt(len(ics))) if ic_std > 0 else np.nan

        rows.append({
            "factor":       fname,
            "mean_ic":      round(mean_ic, 4),
            "ic_std":       round(ic_std, 4),
            "icir":         round(icir, 3),
            "t_stat":       round(t_stat, 2),
            "pct_positive": round((ics > 0).mean() * 100, 1),
            "n_months":     len(ics),
        })

    return pd.DataFrame(rows)


def compute_factor_turnover(panels: dict) -> pd.DataFrame:
    """Mean monthly factor rank turnover — how much does each factor's top-decile change?"""
    rows = []
    for fname, panel in panels.items():
        turnovers = []
        prev_top = None
        for dt in panel.index.sort_values():
            row = panel.loc[dt].dropna()
            if len(row) < 50: continue
            k = max(int(len(row) * 0.10), 10)
            top = set(row.nlargest(k).index)
            if prev_top is not None:
                to = len(top.symmetric_difference(prev_top)) / (2 * k)
                turnovers.append(to)
            prev_top = top
        if turnovers:
            rows.append({"factor": fname,
                         "mean_turnover": round(np.mean(turnovers) * 100, 1),
                         "median_turnover": round(np.median(turnovers) * 100, 1)})
    return pd.DataFrame(rows)


def compute_factor_correlation(panels: dict) -> pd.DataFrame:
    """Average cross-sectional rank correlation between pairs of factors."""
    fnames = list(panels.keys())
    common_dates = None
    for p in panels.values():
        common_dates = p.index if common_dates is None else common_dates.intersection(p.index)

    corr_data = {f: [] for f in fnames}
    for dt in common_dates:
        rows_at_t = {}
        for fn in fnames:
            row = panels[fn].loc[dt].dropna()
            rows_at_t[fn] = row

        common_stocks = None
        for row in rows_at_t.values():
            common_stocks = row.index if common_stocks is None else common_stocks.intersection(row.index)
        if len(common_stocks) < 30: continue

        for fn in fnames:
            corr_data[fn].append(rows_at_t[fn].reindex(common_stocks).rank())

    if not any(corr_data[f] for f in fnames):
        return pd.DataFrame()

    # Build mean cross-sectional correlation matrix
    corr_matrix = pd.DataFrame(index=fnames, columns=fnames, dtype=float)
    for f1 in fnames:
        for f2 in fnames:
            if f1 == f2:
                corr_matrix.loc[f1, f2] = 1.0
                continue
            if len(corr_data[f1]) == 0 or len(corr_data[f2]) == 0:
                corr_matrix.loc[f1, f2] = np.nan
                continue
            pairs = list(zip(corr_data[f1], corr_data[f2]))
            ics = [sp_stats.spearmanr(r1, r2)[0] for r1, r2 in pairs
                   if len(r1) >= 30 and len(r2) >= 30]
            corr_matrix.loc[f1, f2] = round(np.mean(ics), 3) if ics else np.nan

    return corr_matrix


# ══════════════════════════════════════════════════════════════════════════════
# MACRO SIGNAL DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def macro_signal_stats(name: str, signal: pd.Series, ret_wide: pd.DataFrame) -> dict:
    """Summary stats for a macro signal and its correlation with market returns."""
    if signal is None or signal.dropna().empty:
        return {"name": name, "available": False}

    s = signal.dropna()
    # SPY returns as market proxy
    try:
        spy = yf.download("SPY", start=str(s.index[0].date()), end="2025-01-01",
                          interval="1mo", auto_adjust=True, progress=False)["Close"].pct_change()
        spy.index = pd.to_datetime(spy.index) + pd.offsets.MonthEnd(0)
        common = s.index.intersection(spy.index)
        fwd_spy = spy.shift(-1).reindex(common)
        ic_spy, p_spy = sp_stats.spearmanr(s.reindex(common).dropna(),
                                           fwd_spy.reindex(s.reindex(common).dropna().index))
    except Exception:
        ic_spy, p_spy = np.nan, np.nan

    # How often does the signal trigger defensive mode?
    pct_defensive = (s < -1.0).mean() * 100
    pct_caution   = ((s >= -1.0) & (s < -0.5)).mean() * 100

    return {
        "name":           name,
        "available":      True,
        "mean":           round(float(s.mean()), 3),
        "std":            round(float(s.std()), 3),
        "min":            round(float(s.min()), 3),
        "max":            round(float(s.max()), 3),
        "pct_defensive":  round(pct_defensive, 1),   # z < -1.0 → 70% invested
        "pct_caution":    round(pct_caution, 1),      # -1 ≤ z < -0.5 → 85% invested
        "ic_fwd_spy":     round(float(ic_spy), 4) if not np.isnan(ic_spy) else np.nan,
        "p_fwd_spy":      round(float(p_spy), 4) if not np.isnan(p_spy) else np.nan,
    }


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def _pct(v): return f"{v:>+7.1f}%" if not pd.isna(v) else "    n/a"
def _f3(v):  return f"{v:>7.3f}"   if not pd.isna(v) else "    n/a"


def print_macro_comparison(results: dict, basket: int, hold: int):
    W = 100
    print("\n" + "═" * W)
    print(f"  MACRO OVERLAY COMPARISON  —  Top {basket} | {hold}-month hold")
    print("═" * W)
    hdr = (f"  {'Variant':<28}  {'Ann Ret':>7}  {'Ann Vol':>7}  "
           f"{'Sharpe':>7}  {'Max DD':>7}  {'Calmar':>7}  {'Hit%':>6}  {'t-stat':>7}  {'N':>5}")
    sep = "  " + "─" * 28 + "──" + "──────────".join(["─" * 7] * 6) + "──" + "─" * 7
    print(hdr)
    print(sep)
    for label, m in results.items():
        if not m:
            print(f"  {label:<28}  {'— no data —':>60}")
            continue
        print(f"  {label:<28} "
              f" {_pct(m['ann_return'])}"
              f"  {m['ann_vol']:>6.1f}%"
              f"  {_f3(m['sharpe'])}"
              f"  {_pct(m['max_drawdown'])}"
              f"  {_f3(m['calmar'])}"
              f"  {m['hit_rate']:>5.1f}%"
              f"  {_f3(m['t_stat'])}"
              f"  {int(m['n_periods']):>5}")
    print("═" * W)


def print_ic_table(ic_df: pd.DataFrame):
    if ic_df.empty: return
    W = 90
    print("\n" + "═" * W)
    print("  FACTOR IC DIAGNOSTICS  —  Rank IC vs 1-month fwd return")
    print("═" * W)
    hdr = f"  {'Factor':<25}  {'Mean IC':>8}  {'IC Std':>7}  {'ICIR':>7}  {'t-stat':>7}  {'%Pos':>6}  {'N':>5}"
    sep = "  " + "─" * 25 + "──" + "─────────" + "──────────".join(["─" * 7] * 4) + "──" + "─" * 5
    print(hdr)
    print(sep)
    for _, row in ic_df.sort_values("icir", ascending=False).iterrows():
        print(f"  {row['factor']:<25} "
              f"  {row['mean_ic']:>+8.4f}"
              f"  {row['ic_std']:>7.4f}"
              f"  {row['icir']:>7.3f}"
              f"  {row['t_stat']:>7.2f}"
              f"  {row['pct_positive']:>5.1f}%"
              f"  {int(row['n_months']):>5}")
    print("═" * W)


def print_turnover_table(to_df: pd.DataFrame):
    if to_df.empty: return
    W = 60
    print("\n" + "═" * W)
    print("  FACTOR TURNOVER  —  Monthly top-decile turnover")
    print("═" * W)
    hdr = f"  {'Factor':<25}  {'Mean %':>8}  {'Median %':>9}"
    sep = "  " + "─" * 25 + "──" + "─" * 8 + "──" + "─" * 9
    print(hdr)
    print(sep)
    for _, row in to_df.sort_values("mean_turnover").iterrows():
        print(f"  {row['factor']:<25}  {row['mean_turnover']:>7.1f}%  {row['median_turnover']:>8.1f}%")
    print("═" * W)


def print_corr_table(corr_df: pd.DataFrame):
    if corr_df.empty: return
    W = max(80, 20 + 10 * len(corr_df.columns))
    print("\n" + "═" * W)
    print("  FACTOR PAIR CORRELATIONS  —  Mean rank correlation across months")
    print("═" * W)
    cols = list(corr_df.columns)
    col_width = max(len(c) for c in cols) + 2
    header = "  " + f"{'Factor':<25}" + "".join(f"{c:>{col_width}}" for c in cols)
    print(header)
    print("  " + "─" * (25 + col_width * len(cols)))
    for idx in corr_df.index:
        row_str = "  " + f"{idx:<25}"
        for col in cols:
            v = corr_df.loc[idx, col]
            if pd.isna(v):
                row_str += f"{'n/a':>{col_width}}"
            elif idx == col:
                row_str += f"{'1.000':>{col_width}}"
            else:
                row_str += f"{v:>{col_width}.3f}"
        print(row_str)
    print("═" * W)


def print_macro_stats(stats_list: list):
    W = 90
    print("\n" + "═" * W)
    print("  MACRO SIGNAL DIAGNOSTICS")
    print("═" * W)
    hdr = (f"  {'Signal':<30}  {'Mean':>6}  {'Std':>6}  "
           f"{'%Def':>6}  {'%Caut':>6}  {'IC(SPY)':>8}  {'p-val':>7}")
    sep = "  " + "─" * 30 + "──" + "──────────".join(["─" * 6] * 4) + "──" + "─" * 8 + "──" + "─" * 7
    print(hdr)
    print(sep)
    for s in stats_list:
        if not s.get("available", False):
            print(f"  {s['name']:<30}  — unavailable")
            continue
        ic_str = f"{s['ic_fwd_spy']:>+8.4f}" if not pd.isna(s.get("ic_fwd_spy", np.nan)) else "     n/a"
        p_str  = f"{s['p_fwd_spy']:>7.4f}"  if not pd.isna(s.get("p_fwd_spy", np.nan)) else "    n/a"
        print(f"  {s['name']:<30}  {s['mean']:>6.3f}  {s['std']:>6.3f}  "
              f"  {s['pct_defensive']:>5.1f}%  {s['pct_caution']:>5.1f}%"
              f"  {ic_str}  {p_str}")
    print("═" * W)
    print("  %Def  = % of months signal is in defensive mode (z < -1.0, invested 70%)")
    print("  %Caut = % of months in caution mode (-1.0 ≤ z < -0.5, invested 85%)")
    print("  IC(SPY) = rank IC of macro signal vs SPY 1-month forward return")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def print_oos_comparison(in_sample: dict, out_of_sample: dict, basket: int, hold: int):
    """Side-by-side in-sample vs 2025 OOS comparison."""
    W = 110
    print("\n" + "═" * W)
    print(f"  IN-SAMPLE vs OUT-OF-SAMPLE 2025  —  Top {basket} | {hold}-month hold")
    print(f"  In-sample  : {START_DATE} – {TRAIN_END}")
    print(f"  OOS (2025) : {TEST_START} – {END_DATE}")
    print("═" * W)
    hdr = (f"  {'Variant':<28}  {'── In-Sample ───────────────────────':<38}"
           f"  {'── OOS 2025 ────────────────'}")
    sub = (f"  {'':28}  {'Ann Ret':>7}  {'Sharpe':>7}  {'Max DD':>7}  {'Hit%':>6}  {'N':>4}"
           f"     {'Ann Ret':>7}  {'Sharpe':>7}  {'Hit%':>6}  {'N':>4}")
    sep = "  " + "─" * (W - 2)
    print(hdr)
    print(sub)
    print(sep)
    for label in in_sample:
        m_in  = in_sample.get(label, {})
        m_out = out_of_sample.get(label, {})
        def _r(v): return f"{v:>+7.1f}%" if not pd.isna(v) else "    n/a"
        def _s(v): return f"{v:>7.3f}"   if not pd.isna(v) else "    n/a"
        in_str = (f" {_r(m_in.get('ann_return', np.nan))}"
                  f"  {_s(m_in.get('sharpe', np.nan))}"
                  f"  {_r(m_in.get('max_drawdown', np.nan))}"
                  f"  {m_in.get('hit_rate', np.nan):>5.1f}%"
                  f"  {int(m_in.get('n_periods', 0)):>4}") if m_in else "  — no data —"
        out_str = (f"   {_r(m_out.get('ann_return', np.nan))}"
                   f"  {_s(m_out.get('sharpe', np.nan))}"
                   f"  {m_out.get('hit_rate', np.nan):>5.1f}%"
                   f"  {int(m_out.get('n_periods', 0)):>4}") if m_out else "   — no data —"
        print(f"  {label:<28} {in_str}  {out_str}")
    print("═" * W)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basket",    type=int, default=10)
    parser.add_argument("--hold",      type=int, default=1)
    parser.add_argument("--train-end", default=TRAIN_END,
                        help="Last in-sample month (default: 2024-12-31)")
    parser.add_argument("--test-start", default=TEST_START,
                        help="First OOS month (default: 2025-01-01)")
    parser.add_argument("--test-end",   default=END_DATE)
    parser.add_argument("--skip-ic", action="store_true")
    args = parser.parse_args()

    print("\n" + "═" * 70)
    print("  MACRO COMPARISON — train on history, test OOS 2025")
    print("═" * 70)

    # ── Load full history (needed for rolling factor construction) ────────────
    try:
        print(f"\n  Loading CRSP: {START_DATE} → {args.test_end}...")
        ret_wide_full = load_crsp_wide().loc[START_DATE:args.test_end]
        print(f"  CRSP: {ret_wide_full.shape[1]:,} stocks | "
              f"{ret_wide_full.index[0].date()} – {ret_wide_full.index[-1].date()}")
    except FileNotFoundError as e:
        print(f"\n  [ERROR] {e}"); return

    # ── Split into in-sample and OOS index ranges ─────────────────────────────
    is_idx  = ret_wide_full.index[ret_wide_full.index <= args.train_end]
    oos_idx = ret_wide_full.index[ret_wide_full.index >= args.test_start]

    if len(oos_idx) == 0:
        print(f"\n  [ERROR] No data found for OOS period {args.test_start}+")
        print("  Run:  python ingestion/refresh_2025.py   to pull 2025 CRSP data first.")
        return

    print(f"\n  In-sample  : {is_idx[0].date()} – {is_idx[-1].date()}  ({len(is_idx)} months)")
    print(f"  OOS (2025) : {oos_idx[0].date()} – {oos_idx[-1].date()}  ({len(oos_idx)} months)")

    # ── Load supporting panels (full history so rolling windows are warm) ─────
    fund_panel = pd.DataFrame()
    sic_map    = pd.Series(dtype=str)
    is_liq     = pd.DataFrame()
    try:
        print("\n  Loading Compustat...")
        fund_panel, sic_map, is_liq = build_compustat_panels(ret_wide_full)
        print(f"  Fund Quality: {fund_panel.shape[1]:,} stocks")
    except Exception as e:
        print(f"  [SKIP] Compustat: {e}")

    sector_rs = pd.DataFrame()
    try:
        print("  Building sector RS panel (yfinance)...")
        sector_rs = build_sector_rs_panel(ret_wide_full, sic_map)
    except Exception as e:
        print(f"  [SKIP] Sector RS: {e}")

    analyst_rev = pd.DataFrame()
    try:
        print("  Loading IBES analyst revisions...")
        analyst_rev = build_analyst_panel(ret_wide_full)
    except Exception as e:
        print(f"  [SKIP] Analyst: {e}")

    credit_panel = pd.DataFrame()
    try:
        print("  Building credit spread tilt panel...")
        credit_panel = build_credit_spread_panel(ret_wide_full, sic_map)
        if not credit_panel.empty:
            print(f"  Credit spread: {credit_panel.shape[1]:,} stocks")
    except Exception as e:
        print(f"  [SKIP] Credit spread: {e}")

    # ── Build factor panels on full history ───────────────────────────────────
    print("\n  Building factor panels (full history)...")
    panels_full = build_factor_panels(ret_wide_full, fund_panel, sector_rs,
                                      analyst_rev, credit_panel)
    print(f"  Active factors: {list(panels_full.keys())}")

    # ── Restrict panels to each window ───────────────────────────────────────
    def _slice_panels(panels, idx):
        return {k: v.reindex(idx).dropna(how="all") for k, v in panels.items()}

    panels_is  = _slice_panels(panels_full, is_idx)
    panels_oos = _slice_panels(panels_full, oos_idx)
    is_liq_is  = is_liq.reindex(is_idx)   if not is_liq.empty else pd.DataFrame()
    is_liq_oos = is_liq.reindex(oos_idx)  if not is_liq.empty else pd.DataFrame()

    # ── Build macro signals ───────────────────────────────────────────────────
    print("\n  Building macro signals...")
    macro_hy    = build_macro_hy_spread(ret_wide_full)
    macro_unemp = build_macro_unemployment_yoy(ret_wide_full)

    macro_variants = {
        "No Macro (baseline)":           None,
        "HY Spread Widening (existing)": macro_hy,
        "Unemployment YOY (candidate)":  macro_unemp,
    }

    # ── Macro signal diagnostics ──────────────────────────────────────────────
    macro_stats = [
        macro_signal_stats("HY Spread Widening", macro_hy,    ret_wide_full),
        macro_signal_stats("Unemployment YOY",   macro_unemp, ret_wide_full),
    ]
    print_macro_stats(macro_stats)

    # ── Run in-sample and OOS backtests ───────────────────────────────────────
    print(f"\n  Running backtests: Top {args.basket} | {args.hold}-month hold")
    print(f"  {'Variant':<30}  {'IS Sharpe':>10}  {'OOS Sharpe':>11}")
    print("  " + "─" * 55)

    results_is  = {}
    results_oos = {}

    for label, macro in macro_variants.items():
        if macro is not None and macro.dropna().empty:
            results_is[label]  = {}
            results_oos[label] = {}
            continue
        # In-sample
        try:
            curve_is = run_backtest(ret_wide_full.reindex(is_idx), panels_is,
                                    macro, is_liq_is,
                                    basket=args.basket, hold_months=args.hold)
            m_is = compute_metrics(curve_is, hold_months=args.hold)
            results_is[label] = m_is or {}
        except Exception as e:
            results_is[label] = {}
            print(f"  [{label}] IS error: {e}")

        # OOS
        try:
            curve_oos = run_backtest(ret_wide_full.reindex(oos_idx), panels_oos,
                                     macro, is_liq_oos,
                                     basket=args.basket, hold_months=args.hold)
            m_oos = compute_metrics(curve_oos, hold_months=args.hold, min_periods=3)
            results_oos[label] = m_oos or {}
        except Exception as e:
            results_oos[label] = {}
            print(f"  [{label}] OOS error: {e}")

        is_sh  = results_is[label].get("sharpe", np.nan)
        oos_sh = results_oos[label].get("sharpe", np.nan)
        is_str  = f"{is_sh:>+9.3f}" if not pd.isna(is_sh)  else "       n/a"
        oos_str = f"{oos_sh:>+9.3f}" if not pd.isna(oos_sh) else "       n/a"
        print(f"  {label:<30}  IS={is_str}  OOS={oos_str}")

    print_oos_comparison(results_is, results_oos, args.basket, args.hold)

    # ── Full grid: basket × hold × macro for OOS ─────────────────────────────
    print("\n  Running OOS grid (baskets × holds)...")
    BASKETS = [10, 25, 50]
    HOLDS   = [1, 2, 3]
    grid_rows = []

    for hold in HOLDS:
        for basket in BASKETS:
            for mlabel, macro in [("no_macro", None), ("hy_spread", macro_hy),
                                   ("unemp_yoy", macro_unemp)]:
                for window, rw, pan, liq in [
                    ("is",  ret_wide_full.reindex(is_idx),  panels_is,  is_liq_is),
                    ("oos", ret_wide_full.reindex(oos_idx), panels_oos, is_liq_oos),
                ]:
                    try:
                        curve = run_backtest(rw, pan, macro, liq,
                                             basket=basket, hold_months=hold)
                        m = compute_metrics(curve, hold_months=hold,
                                            min_periods=3 if window == "oos" else None)
                        if m:
                            grid_rows.append({"window": window, "macro": mlabel,
                                              "basket": basket, "hold": hold, **m})
                    except Exception:
                        pass

    grid_df   = pd.DataFrame(grid_rows)
    grid_path = RESULTS_DIR / "macro_oos_grid.csv"
    grid_df.to_csv(grid_path, index=False)
    print(f"  Saved → {grid_path}")

    if not grid_df.empty:
        oos_grid = grid_df[grid_df["window"] == "oos"]
        is_grid  = grid_df[grid_df["window"] == "is"]
        print("\n  OOS 2025 grid (mean across basket sizes):")
        W = 80
        print("  " + "─" * W)
        print(f"  {'Macro Variant':<25}  {'Hold':>5}  {'Ann Ret':>8}  {'Sharpe':>7}  {'Max DD':>8}  {'Hit%':>6}")
        print("  " + "─" * W)
        for hold in HOLDS:
            sub = oos_grid[oos_grid["hold"] == hold]
            for ml in ["no_macro", "hy_spread", "unemp_yoy"]:
                row = sub[sub["macro"] == ml]
                if row.empty: continue
                print(f"  {ml:<25}  {hold:>5}m"
                      f"  {row['ann_return'].mean():>+7.1f}%"
                      f"  {row['sharpe'].mean():>7.3f}"
                      f"  {row['max_drawdown'].mean():>+7.1f}%"
                      f"  {row['hit_rate'].mean():>5.1f}%")
            print("  " + "─" * W)

    # ── Factor IC diagnostics (in-sample only — use history to assess factors) ─
    if not args.skip_ic:
        print("\n  Computing factor IC on in-sample data...")
        ret_is = ret_wide_full.reindex(is_idx)
        ic_df  = compute_factor_ic(panels_is, ret_is, fwd_months=1)
        print_ic_table(ic_df)

        to_df   = compute_factor_turnover(panels_is)
        print_turnover_table(to_df)

        corr_df = compute_factor_correlation(panels_is)
        print_corr_table(corr_df)

        ic_df.to_csv(RESULTS_DIR / "factor_ic_diagnostics.csv", index=False)
        to_df.to_csv(RESULTS_DIR / "factor_turnover.csv", index=False)

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
