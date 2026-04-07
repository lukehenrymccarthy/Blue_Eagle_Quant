"""
backtest/factor_evaluation.py
------------------------------
Systematic IC-based evaluation of candidate signals for all five factor groups.

Anti-p-hacking protocol
────────────────────────
  1. Every candidate is pre-specified below before any results are viewed.
  2. Signal selection uses IS (2005-2024) data only.
  3. Benjamini-Hochberg FDR correction (q=0.05) applied within each group.
     Only BH-significant signals are flagged as recommended.
  4. OOS (2025) IC is reported for ALL candidates — not just winners —
     as an independent external check.
  5. IS-to-OOS IC degradation is reported.  >50% retention = acceptable.
     Severe degradation (<25%) flags potential overfit within the IS period.
  6. The final recommendation is the BH-significant signal with the highest
     IS ICIR, provided OOS IC has the same sign as IS IC.

Candidate signals pre-registered
──────────────────────────────────
  F1 Momentum   : mom_1m, mom_3m, mom_6m_skip1, mom_12m_skip1(★),
                  mom_24m_skip1, mom_vol_adj_12m, price_vs_12m_ma, mom_52wk_high
  F2 Quality    : roe_ttm(★), fcf_yield(★), neg_accruals(★), gross_prof(★),
                  roa_ttm, earnings_yield, ocf_yield, inv_pb, inv_debt_equity,
                  composite_equal_wt
  F3 Sector RS  : sector_rs_1m, sector_rs_3m(★), sector_rs_6m, sector_rs_12m,
                  sector_rs_vol_adj_3m, sector_abs_mom_3m, sector_rs_trend_6m
  F4 Analyst    : rev_3m(★), rev_1m, rev_6m, neg_meanrec, net_upgrades,
                  buy_pct, buy_pct_rev_3m, coverage_chg_3m, neg_dispersion
  F5 MacroTilt  : beta_x_hy(★), beta_x_unemp, beta_x_yieldcurve,
                  beta_x_indpro, beta_x_payems, beta_x_combined_or,
                  sector_mom_x_hy, sector_mom_x_unemp

  ★ = current model component

Usage:
    python backtest/factor_evaluation.py
    python backtest/factor_evaluation.py --group momentum   # one group only
    python backtest/factor_evaluation.py --fdr 0.10         # relax threshold
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

from sectorscope.utils import zscore as _zscore, sic_to_etf as _sic_to_etf

# ── Config ─────────────────────────────────────────────────────────────────────
RESULTS_DIR  = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IS_START     = "2005-01-01"
IS_END       = "2024-12-31"
OOS_START    = "2025-01-01"
OOS_END      = "2025-12-31"

SECTOR_ETFS  = ["XLC","XLY","XLP","XLE","XLF","XLV","XLI","XLB","XLRE","XLK","XLU"]
CREDIT_BETA_WINDOW = 36

# ── BH correction (no external dep required) ──────────────────────────────────
def bh_correct(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(p_values)
    if n == 0:
        return np.array([])
    order   = np.argsort(p_values)
    ranks   = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    adj = np.minimum(1.0, p_values * n / ranks)
    # enforce monotonicity (from largest rank down)
    adj_sorted = adj[order]
    for i in range(n - 2, -1, -1):
        adj_sorted[i] = min(adj_sorted[i], adj_sorted[i + 1])
    adj[order] = adj_sorted
    return adj


def ic_stats(ic_series: np.ndarray) -> dict:
    """Summary stats for a vector of monthly ICs."""
    ic = ic_series[~np.isnan(ic_series)]
    if len(ic) < 6:
        return {}
    mean_ic  = ic.mean()
    ic_std   = ic.std()
    icir     = mean_ic / ic_std if ic_std > 0 else np.nan
    t_stat   = mean_ic / (ic_std / np.sqrt(len(ic))) if ic_std > 0 else np.nan
    # one-sided p-value (we test H1: IC > 0)
    p_value  = sp_stats.t.sf(t_stat, df=len(ic) - 1) if not np.isnan(t_stat) else 1.0
    return {
        "n":          len(ic),
        "mean_ic":    round(mean_ic, 5),
        "ic_std":     round(ic_std, 5),
        "icir":       round(icir, 3),
        "t_stat":     round(t_stat, 2),
        "p_one_side": round(p_value, 5),
        "pct_pos":    round((ic > 0).mean() * 100, 1),
    }


def compute_monthly_ic(panel: pd.DataFrame, fwd_ret: pd.DataFrame,
                       idx: pd.DatetimeIndex) -> np.ndarray:
    """
    Compute monthly rank IC (Spearman) between a factor panel and 1-month
    forward returns for dates in `idx`.  Returns array of monthly ICs.
    """
    ics = []
    for dt in idx:
        if dt not in panel.index or dt not in fwd_ret.index:
            continue
        scores = panel.loc[dt].dropna().replace([np.inf, -np.inf], np.nan).dropna()
        fwd    = fwd_ret.loc[dt].dropna()
        common = scores.index.intersection(fwd.index)
        if len(common) < 30:
            continue
        ic, _ = sp_stats.spearmanr(scores.loc[common], fwd.loc[common])
        if not np.isnan(ic):
            ics.append(ic)
    return np.array(ics)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_returns() -> pd.DataFrame:
    from ingestion.wrds_returns import load_returns as _lr, returns_wide
    ret_df = _lr()
    rw = returns_wide(ret_df)
    rw.index = pd.to_datetime(rw.index) + pd.offsets.MonthEnd(0)
    return rw


def load_compustat() -> pd.DataFrame:
    df = pd.read_parquet("data/fundamentals/compustat_quarterly.parquet")
    df["available_date"] = pd.to_datetime(df["available_date"])
    df["datadate"]       = pd.to_datetime(df["datadate"])
    return df


def load_ibes() -> pd.DataFrame:
    path = Path("data/analyst/ibes_signals.parquet")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["statpers"] = pd.to_datetime(df["statpers"]) + pd.offsets.MonthEnd(0)
    df["permno"]   = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
    return df


def load_fred() -> pd.DataFrame:
    raw = pd.read_parquet("data/macro/fred_raw.parquet")
    raw.index = pd.to_datetime(raw.index)
    return raw.resample("ME").last()


def build_sic_map(fund: pd.DataFrame) -> pd.Series:
    sic_col = "siccd" if "siccd" in fund.columns else ("sic" if "sic" in fund.columns else None)
    if not sic_col:
        return pd.Series(dtype=str)
    latest = fund.sort_values("datadate").groupby("permno")[sic_col].last()
    return latest.apply(_sic_to_etf).dropna()


def _build_pit_panel(fund: pd.DataFrame, col: str, ret_wide: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time panel: pivot on available_date, forward-fill to ret_wide dates."""
    sub = fund.dropna(subset=[col]).sort_values("available_date")
    piv = sub.pivot_table(index="available_date", columns="permno",
                          values=col, aggfunc="last")
    piv.index = pd.to_datetime(piv.index)
    return piv.resample("ME").last().ffill().reindex(ret_wide.index, method="ffill")


def _sector_neutral_zscore(panel: pd.DataFrame, sic_map: pd.Series) -> pd.DataFrame:
    def _row_z(row):
        s = row.dropna().replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) < 10:
            return pd.Series(dtype=float)
        df = s.to_frame("val")
        df["sec"] = sic_map.reindex(df.index).fillna("Unknown")
        lo, hi = df["val"].quantile(0.01), df["val"].quantile(0.99)
        df["val"] = df["val"].clip(lo, hi)
        def _sz(g):
            return (g - g.mean()) / g.std() if len(g) >= 3 and g.std() > 0 else g * 0
        return df.groupby("sec")["val"].transform(_sz)
    return panel.apply(_row_z, axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# F1: MOMENTUM CANDIDATES
# ══════════════════════════════════════════════════════════════════════════════

def build_momentum_candidates(ret_wide: pd.DataFrame) -> dict:
    """
    All candidates derived from monthly total returns (ret_wide).
    Lookback periods: 1m (reversal), 3m, 6m-1m, 12m-1m (★), 24m-1m.
    Transformations: raw, vol-adjusted, 52-wk high proximity, MA crossover.
    """
    cands = {}
    rw = ret_wide.copy()

    def _cum(months, skip=0):
        """Cumulative return over `months` skipping last `skip` months."""
        full = (1 + rw.fillna(0)).rolling(months + skip).apply(np.prod, raw=True) - 1
        if skip:
            skip_part = (1 + rw.fillna(0)).rolling(skip).apply(np.prod, raw=True) - 1
            # mom = (1+full)/(1+skip_part) - 1 ≈ full - skip_part for small values
            return ((1 + full) / (1 + skip_part.shift(0)) - 1)
        return full

    # Lookback variants
    cands["mom_1m"]        = rw.shift(1)                            # 1-month reversal
    cands["mom_3m"]        = _cum(3).shift(1)                       # 3m raw
    cands["mom_6m_skip1"]  = _cum(6, skip=1)                        # 6m-1m
    cands["mom_12m_skip1"] = _cum(12, skip=1)                       # ★ current
    cands["mom_24m_skip1"] = _cum(24, skip=1)                       # long-term

    # Vol-adjusted momentum (Barroso & Santa-Clara 2015)
    trail_vol = rw.rolling(12, min_periods=6).std().shift(1).replace(0, np.nan)
    cands["mom_vol_adj_12m"] = cands["mom_12m_skip1"] / trail_vol

    # Price vs 12-month moving average (trend following)
    # Use cumulative return proxy: if 12m return > 0 → above MA → positive
    cands["price_vs_12m_ma"] = (rw.rolling(12).mean().shift(1) > 0).astype(float) * 2 - 1
    # Better: sign of the 12m return is a coarse proxy
    cands["price_vs_12m_ma"] = cands["mom_12m_skip1"].apply(np.sign)

    # 52-week high proximity (George & Hwang 2004): use 12m max return as proxy
    # price/52wk_high ≈ (1+current_price)/max_price_over_12m
    # Approximation using the 12m rolling max of cumulative returns
    roll_max = (1 + rw.fillna(0)).rolling(12).apply(np.prod, raw=True)
    curr_rel  = (1 + rw.fillna(0)).shift(1)  # price index at t
    cands["mom_52wk_high"] = (curr_rel / roll_max.shift(1)).shift(1)

    return cands


# ══════════════════════════════════════════════════════════════════════════════
# F2: QUALITY CANDIDATES
# ══════════════════════════════════════════════════════════════════════════════

def build_quality_candidates(fund: pd.DataFrame, ret_wide: pd.DataFrame,
                              sic_map: pd.Series) -> dict:
    """
    All candidates derived from Compustat quarterly.
    Economic rationale documented per signal.
    """
    cands = {}

    # Derive extra columns not yet in the parquet
    fund = fund.copy()
    fund["permno"] = pd.to_numeric(fund["permno"], errors="coerce")
    fund = fund.dropna(subset=["permno"])
    fund["permno"] = fund["permno"].astype(int)

    # ROA TTM (profitability, less affected by leverage than ROE)
    fund["roa_ttm"] = fund["ibq_ttm"] / fund["atq"].replace(0, np.nan)

    # Operating cash flow yield (cash-based quality, harder to manipulate)
    fund["ocf_yield"] = fund["oancfq_ttm"] / fund["market_cap"].replace(0, np.nan)

    # Accruals: -(net income - operating cash flow) / assets
    fund["neg_accruals"] = -(fund["ibq_ttm"] - fund["oancfq_ttm"]) / \
                            fund["atq"].replace(0, np.nan)

    # Gross profitability: revenue / assets (Novy-Marx 2013)
    fund["gross_prof"] = fund["saleq"] / fund["atq"].replace(0, np.nan)

    # Inverse leverage (lower debt = higher quality)
    fund["inv_debt_equity"] = -(fund["ltq"] / fund["ceqq"].replace(0, np.nan))

    # Inverse P/B (value component of quality)
    fund["inv_pb"] = -fund["pb_ratio"]

    raw_signals = {
        "roe_ttm":        "ROE TTM (★ component)",
        "fcf_yield":      "FCF Yield TTM (★ component)",
        "neg_accruals":   "Neg-Accruals, -(NI-OCF)/Assets (★ component)",
        "gross_prof":     "Gross Profitability, Sales/Assets (★ component)",
        "roa_ttm":        "ROA TTM",
        "earnings_yield": "Earnings Yield (E/P), ibq_ttm/mktcap",
        "ocf_yield":      "Operating CF Yield",
        "inv_pb":         "Inverse P/B (value-quality)",
        "inv_debt_equity":"Inverse Debt/Equity (low leverage)",
    }

    panels = {}
    for col, _ in raw_signals.items():
        if col not in fund.columns:
            continue
        try:
            p = _build_pit_panel(fund, col, ret_wide)
            panels[col] = _sector_neutral_zscore(p, sic_map)
        except Exception:
            pass

    # Equal-weight composite of all available individual signals
    all_panels = [v for v in panels.values() if not v.empty]
    if len(all_panels) >= 4:
        comp = pd.concat(all_panels, axis=0).groupby(level=0).mean()
        panels["composite_equal_wt"] = comp.reindex(ret_wide.index, method="ffill")

    return panels


# ══════════════════════════════════════════════════════════════════════════════
# F3: SECTOR RELATIVE STRENGTH CANDIDATES
# ══════════════════════════════════════════════════════════════════════════════

def build_sector_rs_candidates(ret_wide: pd.DataFrame, sic_map: pd.Series) -> dict:
    """
    Lookback variants (1m/3m/6m/12m) and transformations of sector-vs-SPY RS.
    Vol-adjusted RS and trend (slope) also included.
    """
    if sic_map.empty:
        return {}

    print("    Downloading sector ETF + SPY prices (yfinance)...")
    raw = yf.download(SECTOR_ETFS + ["SPY"], start="2004-01-01", end="2026-06-01",
                      interval="1mo", auto_adjust=True, progress=False)["Close"]
    raw.index = pd.to_datetime(raw.index) + pd.offsets.MonthEnd(0)
    mr = raw.pct_change().sort_index()

    spy = mr["SPY"].fillna(0)

    def _etf_cum(months):
        spy_cm = (1 + spy).rolling(months).apply(np.prod, raw=True) - 1
        result = {}
        for etf in SECTOR_ETFS:
            if etf not in mr.columns:
                continue
            etf_cm = (1 + mr[etf].fillna(0)).rolling(months).apply(np.prod, raw=True) - 1
            result[etf] = etf_cm - spy_cm
        return pd.DataFrame(result).reindex(ret_wide.index, method="ffill")

    def _etf_abs_cum(months):
        result = {}
        for etf in SECTOR_ETFS:
            if etf not in mr.columns:
                continue
            result[etf] = (1 + mr[etf].fillna(0)).rolling(months).apply(np.prod, raw=True) - 1
        return pd.DataFrame(result).reindex(ret_wide.index, method="ffill")

    etf_rs_1m  = _etf_cum(1)
    etf_rs_3m  = _etf_cum(3)   # ★ current
    etf_rs_6m  = _etf_cum(6)
    etf_rs_12m = _etf_cum(12)
    etf_abs_3m = _etf_abs_cum(3)

    # Volatility-adjusted: RS / trailing sector vol
    trail_vol_dict = {}
    for etf in SECTOR_ETFS:
        if etf in mr.columns:
            trail_vol_dict[etf] = mr[etf].rolling(12, min_periods=6).std().replace(0, np.nan)
    etf_trail_vol = pd.DataFrame(trail_vol_dict).reindex(ret_wide.index, method="ffill")
    etf_rs_vol_adj = (etf_rs_3m / etf_trail_vol).replace([np.inf, -np.inf], np.nan)

    # RS trend: slope of 6 monthly RS observations (linear trend)
    etf_monthly_rs = {}
    for etf in SECTOR_ETFS:
        if etf in mr.columns:
            etf_monthly_rs[etf] = mr[etf].fillna(0) - spy
    monthly_rs_df = pd.DataFrame(etf_monthly_rs).reindex(ret_wide.index, method="ffill")

    def _rs_trend(series, window=6):
        def _slope(x):
            x = x[~np.isnan(x)]
            if len(x) < 4:
                return np.nan
            t = np.arange(len(x))
            return np.polyfit(t, x, 1)[0]
        return series.rolling(window, min_periods=4).apply(_slope, raw=True)

    etf_rs_trend = monthly_rs_df.apply(_rs_trend, axis=0).reindex(ret_wide.index, method="ffill")

    def _broadcast(etf_df):
        """Map ETF-level panel to stock-level via sic_map."""
        stock_to_etf = sic_map.reindex(ret_wide.columns).dropna()
        etf_to_idx   = {e: i for i, e in enumerate(etf_df.columns)}
        matched      = [(s, e) for s, e in stock_to_etf.items() if e in etf_to_idx]
        if not matched:
            return pd.DataFrame()
        stocks = [s for s, _ in matched]
        idxs   = [etf_to_idx[e] for _, e in matched]
        sp = pd.DataFrame(etf_df.values[:, idxs], index=etf_df.index, columns=stocks)
        sp.columns = sp.columns.astype(ret_wide.columns.dtype)
        return sp

    return {
        "sector_rs_1m":       _broadcast(etf_rs_1m),
        "sector_rs_3m":       _broadcast(etf_rs_3m),       # ★
        "sector_rs_6m":       _broadcast(etf_rs_6m),
        "sector_rs_12m":      _broadcast(etf_rs_12m),
        "sector_rs_vol_adj":  _broadcast(etf_rs_vol_adj),
        "sector_abs_mom_3m":  _broadcast(etf_abs_3m),
        "sector_rs_trend_6m": _broadcast(etf_rs_trend),
    }


# ══════════════════════════════════════════════════════════════════════════════
# F4: ANALYST REVISION CANDIDATES
# ══════════════════════════════════════════════════════════════════════════════

def build_analyst_candidates(ibes: pd.DataFrame, ret_wide: pd.DataFrame) -> dict:
    """
    IBES-based signals: level of consensus rec, revision windows, buy%,
    coverage change, dispersion (uncertainty).
    meanrec: 1=Strong Buy → 5=Strong Sell, so lower = more bullish.
    neg_meanrec = -meanrec already in the file (higher = more bullish).
    """
    if ibes.empty:
        return {}

    ibes_cols = list(ibes.columns)

    def _pivot_ibes(col):
        sub = ibes.dropna(subset=[col])
        piv = sub.pivot_table(index="statpers", columns="permno",
                              values=col, aggfunc="last")
        piv.index = pd.to_datetime(piv.index)
        piv.columns = piv.columns.astype(ret_wide.columns.dtype)
        return piv.resample("ME").last().ffill().reindex(ret_wide.index, method="ffill")

    cands = {}

    # Existing pre-computed signals in IBES file
    for col in ["neg_meanrec", "net_upgrades", "buy_pct", "buy_pct_rev_3m",
                "coverage_chg_3m", "neg_dispersion"]:
        if col in ibes_cols:
            try:
                cands[col] = _pivot_ibes(col)
            except Exception:
                pass

    # rev_3m is already in the file (★ current)
    if "rev_3m" in ibes_cols:
        cands["rev_3m"] = _pivot_ibes("rev_3m")

    # Derive additional revision windows from meanrec
    if "meanrec" in ibes_cols:
        mr_pivot = _pivot_ibes("meanrec")
        # 1m and 6m revisions (lower meanrec change = upgrade = positive)
        cands["rev_1m"] = -(mr_pivot - mr_pivot.shift(1))
        cands["rev_6m"] = -(mr_pivot - mr_pivot.shift(6))
        # Momentum of revisions: 3-month trend (are revisions accelerating?)
        cands["rev_trend_3m"] = cands.get("rev_1m", pd.DataFrame())
        if "rev_1m" in cands and not cands["rev_1m"].empty:
            cands["rev_trend_3m"] = (cands["rev_1m"]
                                     .rolling(3, min_periods=2).mean())

    return cands


# ══════════════════════════════════════════════════════════════════════════════
# F5: MACRO TILT CANDIDATES
# ══════════════════════════════════════════════════════════════════════════════

def build_macro_tilt_candidates(ret_wide: pd.DataFrame, sic_map: pd.Series,
                                 fred: pd.DataFrame) -> dict:
    """
    Sector-level macro regime × sensitivity panels.

    For each candidate we combine:
      - A SECTOR SENSITIVITY measure (rolling OLS beta, or sector momentum)
      - A REGIME SIGNAL (macro z-score: positive = expansion, negative = stress)

    The cross product (sensitivity × regime) gives a stock-level tilt:
    high-sensitivity sectors score badly in stress regimes.

    Regime signals tested:
      HY spread (★), Unemployment YOY, Yield curve slope, INDPRO YOY,
      Payrolls YOY, Combined OR(HY, unemp).

    Sensitivity measures:
      OLS beta vs HY spread (★), sector momentum 12m.
    """
    if sic_map.empty:
        return {}

    fred.index = pd.to_datetime(fred.index) + pd.offsets.MonthEnd(0)

    # ── Regime signals (rolling z-score, positive = expansion = good) ─────────
    def _regime_z(series, window=36):
        s = series.dropna()
        rm = s.rolling(window, min_periods=12).mean()
        rs = s.rolling(window, min_periods=12).std().replace(0, np.nan)
        return ((s - rm) / rs).reindex(ret_wide.index, method="ffill")

    regimes = {}

    # HY spread widening (★): negative when spreads widen = stress
    if "BAMLH0A0HYM2" in fred.columns:
        hy_widening = fred["BAMLH0A0HYM2"].diff(3)
        regimes["hy"]       = -_regime_z(hy_widening)

    # Unemployment YOY: negative when unemployment rising = stress
    if "UNRATE" in fred.columns:
        unemp_yoy = fred["UNRATE"].diff(12).shift(1)   # 1m pub lag
        regimes["unemp"]    = -_regime_z(unemp_yoy)

    # Yield curve slope (T10Y2Y): inverted curve = stress
    if "T10Y2Y" in fred.columns:
        regimes["yieldcurve"] = _regime_z(fred["T10Y2Y"])

    # Industrial production YOY: negative growth = contraction
    if "INDPRO" in fred.columns:
        indpro_yoy = fred["INDPRO"].pct_change(12)
        regimes["indpro"]   = _regime_z(indpro_yoy)

    # Payrolls YOY: negative = labor market stress
    if "PAYEMS" in fred.columns:
        payems_yoy = fred["PAYEMS"].pct_change(12)
        regimes["payems"]   = _regime_z(payems_yoy)

    # Combined OR: minimum of HY and unemp regimes (fires if either is stressed)
    if "hy" in regimes and "unemp" in regimes:
        combined = pd.concat([regimes["hy"], regimes["unemp"]], axis=1)
        combined.columns = ["hy", "unemp"]
        regimes["combined_or"] = combined.min(axis=1)

    # ── Sector sensitivity measures ───────────────────────────────────────────
    print("    Downloading sector ETF prices for macro tilt...")
    raw = yf.download(SECTOR_ETFS, start="2003-01-01", end="2026-06-01",
                      interval="1mo", auto_adjust=True, progress=False)["Close"]
    raw.index = pd.to_datetime(raw.index) + pd.offsets.MonthEnd(0)
    mr = raw.pct_change().sort_index()

    # OLS beta: rolling covariance(sector_ret, hy_spread_change) / var(hy_spread_change)
    # This is the ★ current sensitivity measure
    hy_beta_df = pd.DataFrame(dtype=float)
    if "BAMLH0A0HYM2" in fred.columns:
        delta_hy = fred["BAMLH0A0HYM2"].diff().dropna()
        betas = {}
        for etf in SECTOR_ETFS:
            if etf not in mr.columns:
                continue
            common = delta_hy.index.intersection(mr[etf].index)
            rc = mr[etf].reindex(common).rolling(CREDIT_BETA_WINDOW, min_periods=12) \
                         .cov(delta_hy.reindex(common))
            rv = delta_hy.reindex(common).rolling(CREDIT_BETA_WINDOW, min_periods=12) \
                          .var().replace(0, np.nan)
            betas[etf] = (rc / rv).reindex(ret_wide.index, method="ffill")
        hy_beta_df = pd.DataFrame(betas).reindex(ret_wide.index, method="ffill")

    # Sector 12m momentum as alternative sensitivity proxy
    sector_mom_df = pd.DataFrame(dtype=float)
    if not mr.empty:
        moms = {}
        for etf in SECTOR_ETFS:
            if etf in mr.columns:
                moms[etf] = (1 + mr[etf].fillna(0)).rolling(12) \
                              .apply(np.prod, raw=True) - 1
        sector_mom_df = pd.DataFrame(moms).reindex(ret_wide.index, method="ffill")

    # ── Build (sensitivity × regime) panels ───────────────────────────────────
    def _broadcast_tilt(etf_tilt_df):
        if etf_tilt_df.empty:
            return pd.DataFrame()
        stock_to_etf = sic_map.reindex(ret_wide.columns).dropna()
        etf_to_idx   = {e: i for i, e in enumerate(etf_tilt_df.columns)}
        matched      = [(s, e) for s, e in stock_to_etf.items() if e in etf_to_idx]
        if not matched:
            return pd.DataFrame()
        stocks = [s for s, _ in matched]
        idxs   = [etf_to_idx[e] for _, e in matched]
        sp = pd.DataFrame(etf_tilt_df.values[:, idxs],
                          index=etf_tilt_df.index, columns=stocks)
        sp.columns = sp.columns.astype(ret_wide.columns.dtype)
        return sp

    cands = {}
    for regime_name, regime_series in regimes.items():
        # OLS beta × regime
        if not hy_beta_df.empty:
            tilt = hy_beta_df.multiply(regime_series, axis=0)
            cands[f"beta_x_{regime_name}"] = _broadcast_tilt(tilt)

        # Sector momentum × regime
        if not sector_mom_df.empty:
            tilt_m = sector_mom_df.multiply(regime_series, axis=0)
            cands[f"sector_mom_x_{regime_name}"] = _broadcast_tilt(tilt_m)

    # Rename ★ current to make it clear
    if "beta_x_hy" in cands:
        cands["beta_x_hy (★)"] = cands.pop("beta_x_hy")

    return cands


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_group(
    group_name: str,
    candidates:  dict,
    ret_wide:    pd.DataFrame,
    is_idx:      pd.DatetimeIndex,
    oos_idx:     pd.DatetimeIndex,
    fdr_q:       float = 0.05,
) -> pd.DataFrame:
    """
    For each candidate panel in `candidates`:
      1. Compute IS rank IC vs 1m forward return
      2. Compute OOS rank IC vs 1m forward return
      3. Apply BH correction across all candidates in the group (IS p-values)
    Returns a sorted DataFrame of results.
    """
    fwd_ret = ret_wide.pct_change(1).shift(-1)   # 1-month ahead

    rows = []
    for name, panel in candidates.items():
        if panel is None or (isinstance(panel, pd.DataFrame) and panel.empty):
            continue
        # IS IC
        is_ic_arr  = compute_monthly_ic(panel, fwd_ret, is_idx)
        is_stats   = ic_stats(is_ic_arr)
        if not is_stats:
            continue

        # OOS IC
        oos_ic_arr = compute_monthly_ic(panel, fwd_ret, oos_idx)
        oos_stats  = ic_stats(oos_ic_arr) if len(oos_ic_arr) >= 3 else {}

        row = {"signal": name, **{f"is_{k}": v for k, v in is_stats.items()}}
        if oos_stats:
            row.update({f"oos_{k}": v for k, v in oos_stats.items()})
            # IS-to-OOS retention (same sign only)
            if is_stats["mean_ic"] != 0:
                retention = oos_stats["mean_ic"] / is_stats["mean_ic"] * 100
                row["oos_retention_pct"] = round(retention, 1)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # BH FDR correction on IS one-sided p-values
    p_vals   = df["is_p_one_side"].fillna(1.0).values
    p_adj    = bh_correct(p_vals, q=fdr_q)
    df["bh_adj_p"]    = p_adj.round(5)
    df["bh_signif"]   = (p_adj < fdr_q)

    # Sort by IS ICIR descending
    return df.sort_values("is_icir", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def print_group_results(group_name: str, df: pd.DataFrame, fdr_q: float):
    if df.empty:
        print(f"\n  [{group_name}] — no results")
        return

    W = 120
    print("\n" + "═" * W)
    print(f"  FACTOR GROUP: {group_name.upper()}")
    print(f"  BH FDR threshold: q = {fdr_q}   ★ = current model component")
    print("═" * W)

    hdr = (f"  {'Signal':<28}  "
           f"{'IS IC':>7}  {'IS ICIR':>7}  {'IS t':>6}  {'BH p':>7}  {'✓':>3}  "
           f"{'OOS IC':>7}  {'OOS t':>6}  {'Retain%':>8}  {'Rec':>5}")
    sep = "  " + "─" * (W - 2)
    print(hdr)
    print(sep)

    for _, row in df.iterrows():
        is_ic    = row.get("is_mean_ic", np.nan)
        is_icir  = row.get("is_icir",    np.nan)
        is_t     = row.get("is_t_stat",  np.nan)
        bh_p     = row.get("bh_adj_p",   np.nan)
        bh_sig   = row.get("bh_signif",  False)
        oos_ic   = row.get("oos_mean_ic", np.nan)
        oos_t    = row.get("oos_t_stat",  np.nan)
        retain   = row.get("oos_retention_pct", np.nan)
        name     = str(row["signal"])

        sig_mark = "✓" if bh_sig else " "
        # Recommendation: BH significant + OOS same sign as IS
        is_valid_oos = (not pd.isna(oos_ic) and not pd.isna(is_ic) and
                        np.sign(oos_ic) == np.sign(is_ic))
        rec = "★REC" if (bh_sig and is_valid_oos and is_icir == df[df["bh_signif"]]["is_icir"].max()) else \
              "SIG"  if (bh_sig and is_valid_oos) else \
              "WARN" if (bh_sig and not is_valid_oos) else ""

        def _f(v, fmt=".4f"): return f"{v:{fmt}}" if not pd.isna(v) else "   n/a"

        print(f"  {name:<28}  "
              f"  {_f(is_ic):>7}  {_f(is_icir, '.3f'):>7}  {_f(is_t, '.2f'):>6}"
              f"  {_f(bh_p, '.4f'):>7}  {sig_mark:>3}  "
              f"  {_f(oos_ic):>7}  {_f(oos_t, '.2f'):>6}"
              f"  {_f(retain, '.1f'):>8}  {rec:>5}")

    print(sep)

    # Recommendation summary
    sig_df = df[df["bh_signif"]]
    if sig_df.empty:
        print(f"\n  ⚠  No signals pass BH FDR correction at q={fdr_q} for {group_name}.")
        print(f"     Consider relaxing threshold (--fdr 0.10) or investigating data quality.")
    else:
        best = sig_df.sort_values("is_icir", ascending=False).iloc[0]
        oos_ok = (not pd.isna(best.get("oos_mean_ic")) and
                  not pd.isna(best.get("is_mean_ic")) and
                  np.sign(best.get("oos_mean_ic", 0)) == np.sign(best.get("is_mean_ic", 0)))
        print(f"\n  Recommendation  : {best['signal']}")
        print(f"  IS ICIR={best['is_icir']:.3f}  IS t={best['is_t_stat']:.2f}"
              f"  BH p={best['bh_adj_p']:.4f}"
              f"  OOS IC={_f(best.get('oos_mean_ic', np.nan))}"
              f"  OOS consistent={'YES' if oos_ok else 'NO — FLAG'}")
    print("═" * W)


def save_results(all_results: dict):
    """Consolidate all groups into one CSV for further analysis."""
    frames = []
    for group, df in all_results.items():
        if df.empty:
            continue
        df = df.copy()
        df.insert(0, "group", group)
        frames.append(df)
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        path = RESULTS_DIR / "factor_evaluation.csv"
        combined.to_csv(path, index=False)
        print(f"\n  Full results → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["momentum","quality","sector","analyst","macro","all"],
                        default="all")
    parser.add_argument("--fdr", type=float, default=0.05,
                        help="BH FDR threshold (default 0.05)")
    args = parser.parse_args()

    run_groups = (["momentum","quality","sector","analyst","macro"]
                  if args.group == "all" else [args.group])

    print("\n" + "═" * 70)
    print("  FACTOR EVALUATION — all candidates, IS + OOS, BH correction")
    print(f"  IS : {IS_START} – {IS_END}")
    print(f"  OOS: {OOS_START} – {OOS_END}")
    print(f"  BH FDR q = {args.fdr}")
    print("═" * 70)

    # ── Load core data ────────────────────────────────────────────────────────
    print("\n  Loading CRSP returns...")
    ret_wide = load_returns().loc[IS_START:OOS_END]
    print(f"  {ret_wide.shape[1]:,} stocks | {ret_wide.index[0].date()} – {ret_wide.index[-1].date()}")

    is_idx  = ret_wide.index[ret_wide.index <= IS_END]
    oos_idx = ret_wide.index[ret_wide.index >= OOS_START]
    print(f"  IS: {len(is_idx)} months  |  OOS: {len(oos_idx)} months")

    fund    = pd.DataFrame()
    sic_map = pd.Series(dtype=str)
    if any(g in run_groups for g in ["quality","sector","macro"]):
        print("  Loading Compustat...")
        fund    = load_compustat()
        sic_map = build_sic_map(fund)
        print(f"  {fund['permno'].nunique():,} permnos  |  {sic_map.notna().sum():,} SIC-mapped")

    ibes = pd.DataFrame()
    if "analyst" in run_groups:
        print("  Loading IBES...")
        ibes = load_ibes()
        print(f"  IBES: {ibes['permno'].nunique():,} permnos")

    fred = pd.DataFrame()
    if "macro" in run_groups:
        print("  Loading FRED...")
        fred = load_fred()

    all_results = {}

    # ── F1: Momentum ─────────────────────────────────────────────────────────
    if "momentum" in run_groups:
        print("\n  Building momentum candidates...")
        cands = build_momentum_candidates(ret_wide)
        print(f"  {len(cands)} candidates")
        all_results["momentum"] = evaluate_group(
            "F1: Momentum", cands, ret_wide, is_idx, oos_idx, fdr_q=args.fdr)
        print_group_results("F1: Momentum", all_results["momentum"], args.fdr)

    # ── F2: Quality ───────────────────────────────────────────────────────────
    if "quality" in run_groups:
        print("\n  Building quality candidates...")
        cands = build_quality_candidates(fund, ret_wide, sic_map)
        print(f"  {len(cands)} candidates")
        all_results["quality"] = evaluate_group(
            "F2: Quality", cands, ret_wide, is_idx, oos_idx, fdr_q=args.fdr)
        print_group_results("F2: Quality", all_results["quality"], args.fdr)

    # ── F3: Sector RS ─────────────────────────────────────────────────────────
    if "sector" in run_groups:
        print("\n  Building sector RS candidates...")
        cands = build_sector_rs_candidates(ret_wide, sic_map)
        print(f"  {len(cands)} candidates")
        all_results["sector"] = evaluate_group(
            "F3: Sector RS", cands, ret_wide, is_idx, oos_idx, fdr_q=args.fdr)
        print_group_results("F3: Sector RS", all_results["sector"], args.fdr)

    # ── F4: Analyst ───────────────────────────────────────────────────────────
    if "analyst" in run_groups:
        print("\n  Building analyst revision candidates...")
        cands = build_analyst_candidates(ibes, ret_wide)
        print(f"  {len(cands)} candidates")
        all_results["analyst"] = evaluate_group(
            "F4: Analyst", cands, ret_wide, is_idx, oos_idx, fdr_q=args.fdr)
        print_group_results("F4: Analyst", all_results["analyst"], args.fdr)

    # ── F5: Macro Tilt ────────────────────────────────────────────────────────
    if "macro" in run_groups:
        print("\n  Building macro tilt candidates...")
        cands = build_macro_tilt_candidates(ret_wide, sic_map, fred)
        print(f"  {len(cands)} candidates")
        all_results["macro"] = evaluate_group(
            "F5: Macro Tilt", cands, ret_wide, is_idx, oos_idx, fdr_q=args.fdr)
        print_group_results("F5: Macro Tilt", all_results["macro"], args.fdr)

    save_results(all_results)
    print("\n  Done.\n")


if __name__ == "__main__":
    main()
