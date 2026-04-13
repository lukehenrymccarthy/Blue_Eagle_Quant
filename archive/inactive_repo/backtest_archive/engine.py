"""
backtest/engine.py
------------------
Five-factor model library — factor construction, composite scoring, and
single-run backtest logic.

Imported by backtest/run_all.py.  For the full parameter sweep, run:
    python backtest/run_all.py
"""

import sys
import warnings
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PORTFOLIO_SIZE   = 25          # top N stocks held each month
TRANSACTION_COST = 0.001       # 10bps per side (round trip = 20bps)
RISK_FREE_RATE   = 0.05 / 12  # monthly risk-free (adjust to current rate)
START_DATE       = "2010-01-01"
END_DATE         = "2024-12-31"


# ═══════════════════════════════════════════════════════════════════════════════
# FACTOR CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_momentum_factor(
    returns_wide: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    lookback: int,
    skip: int,
) -> pd.Series:
    """
    Classic momentum: cumulative return over [lookback+skip, skip+1] months ago.
    E.g. lookback=12, skip=1 → 12-month return ending 1 month ago (12-1 momentum).
    Returns a cross-sectional Series indexed by permno.
    """
    end_idx   = rebalance_date - pd.DateOffset(months=skip)
    start_idx = rebalance_date - pd.DateOffset(months=lookback + skip)

    window = returns_wide.loc[
        (returns_wide.index >= start_idx) &
        (returns_wide.index <= end_idx)
    ]

    if len(window) < max(lookback - 2, 1):
        return pd.Series(dtype=float)

    # Compound returns over the window
    cum_ret = (1 + window.fillna(0)).prod() - 1
    return cum_ret


def compute_ma_crossover_factor(
    prices_monthly: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    short_window: int,
    long_window: int,
) -> pd.Series:
    """
    Price-above-MA signal:
      - Short MA / Long MA - 1  (positive = short MA above long MA = bullish)
    Higher values = stronger uptrend.
    """
    history = prices_monthly.loc[prices_monthly.index <= rebalance_date]

    if len(history) < long_window:
        return pd.Series(dtype=float)

    short_ma = history.tail(short_window).mean()
    long_ma  = history.tail(long_window).mean()

    signal = (short_ma / long_ma.replace(0, np.nan)) - 1
    return signal


def compute_technical_factor(
    returns_wide: pd.DataFrame,
    prices_monthly: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    momentum_lookback: int,
    momentum_skip: int,
    ma_windows: tuple,
) -> pd.Series:
    """Combine momentum and MA crossover into a single technical factor score."""
    mom = compute_momentum_factor(
        returns_wide, rebalance_date, momentum_lookback, momentum_skip
    )
    ma_short, ma_long = ma_windows
    ma  = compute_ma_crossover_factor(
        prices_monthly, rebalance_date, ma_short, ma_long
    )

    combined = pd.concat([mom, ma], axis=1).mean(axis=1)
    return combined


def compute_fundamental_factor(
    fundamentals: pd.DataFrame,
    rebalance_date: pd.Timestamp,
) -> pd.Series:
    """
    Composite fundamental score from:
      - ROE (TTM)          : profitability
      - FCF yield          : cash generation efficiency
      - Earnings yield     : value (inverse P/E)

    Each is cross-sectionally z-scored and averaged.
    Returns a Series indexed by permno.
    """
    pit = fundamentals[fundamentals["available_date"] <= rebalance_date].copy()
    pit = (
        pit.sort_values("datadate")
        .groupby("permno")
        .last()
        .reset_index()
        .set_index("permno")
    )

    scores = pd.DataFrame(index=pit.index)
    for col in ["roe_ttm", "fcf_yield", "earnings_yield"]:
        if col not in pit.columns:
            continue
        s = pit[col].replace([np.inf, -np.inf], np.nan)
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        s = s.clip(lo, hi)
        scores[col] = (s - s.mean()) / s.std()

    # Accruals: low accruals = high earnings quality (Sloan 1996).
    # accruals = (net income - operating cash flow) / total assets — negate so high = good.
    if all(c in pit.columns for c in ["ibq_ttm", "oancfq_ttm", "atq"]):
        acc = -(pit["ibq_ttm"] - pit["oancfq_ttm"]) / pit["atq"].replace(0, np.nan)
        s   = acc.replace([np.inf, -np.inf], np.nan)
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        s = s.clip(lo, hi)
        scores["neg_accruals"] = (s - s.mean()) / s.std()

    # Gross profitability proxy (Novy-Marx 2013): quarterly revenue / total assets.
    # Higher = more revenue per asset dollar = higher quality business.
    if all(c in pit.columns for c in ["saleq", "atq"]):
        gp = pit["saleq"] / pit["atq"].replace(0, np.nan)
        s  = gp.replace([np.inf, -np.inf], np.nan)
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        s = s.clip(lo, hi)
        scores["gross_profitability"] = (s - s.mean()) / s.std()

    return scores.mean(axis=1).dropna()


def compute_macro_scalar(
    macro_signals: pd.DataFrame,
    rebalance_date: pd.Timestamp,
) -> float:
    """
    Return a single macro regime score for position-sizing decisions.
    Positive = risk-on, negative = risk-off.

    Signals (IC-tested in analysis/macro_ic.py):
      - CPIAUCSL_zscore  (t=−3.20)  high inflation = headwind (negate)
      - PAYEMS_zscore    (t=−2.66)  hot labor market = Fed tightening (negate)
      - hy_spread_widening (t=−2.36) widening spreads = risk-off (negate)
    """
    available = macro_signals[macro_signals.index <= rebalance_date]
    if available.empty:
        return 0.0
    row = available.iloc[-1]
    components = []
    for col, sign in [
        ("CPIAUCSL_zscore",    -1),
        ("PAYEMS_zscore",      -1),
        ("hy_spread_widening", -1),
    ]:
        if col in row.index and pd.notna(row[col]):
            components.append(sign * row[col])
    return float(np.mean(components)) if components else 0.0


def compute_macro_factor(
    macro_signals: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    equity_index: pd.Index,
) -> pd.Series:
    """Broadcast macro regime score to all stocks (kept for compatibility)."""
    macro_score = compute_macro_scalar(macro_signals, rebalance_date)
    return pd.Series(macro_score, index=equity_index)


def compute_sector_valuation_factor(
    fundamentals: pd.DataFrame,
    rebalance_date: pd.Timestamp,
) -> pd.Series:
    """
    Sector valuation: z-score of each sector's current median P/B
    vs its own trailing 36-month history. Cheap sectors score high.

    Returns a Series indexed by permno (each stock gets its sector's score).
    """
    pit = fundamentals[fundamentals["available_date"] <= rebalance_date].copy()
    pit = (
        pit.sort_values("datadate")
        .groupby("permno")
        .last()
        .reset_index()
    )

    if "pb_ratio" not in pit.columns or "sic" not in pit.columns:
        return pd.Series(dtype=float)

    # Map SIC to broad GICS-like sector (simplified 11-sector mapping)
    def sic_to_sector(sic):
        if pd.isna(sic): return "Unknown"
        sic = int(sic)
        if   sic < 1000:  return "Agriculture"
        elif sic < 1500:  return "Mining"
        elif sic < 2000:  return "Construction"
        elif sic < 4000:  return "Manufacturing"
        elif sic < 5000:  return "Transport_Utilities"
        elif sic < 5200:  return "Wholesale"
        elif sic < 6000:  return "Retail"
        elif sic < 6500:  return "Finance"
        elif sic < 7000:  return "RealEstate"
        elif sic < 8000:  return "Services"
        else:             return "PublicAdmin"

    pit["sector"] = pit["sic"].apply(sic_to_sector)

    # Historical sector medians for z-scoring
    hist_start = rebalance_date - pd.DateOffset(months=36)
    hist = fundamentals[
        (fundamentals["available_date"] >= hist_start) &
        (fundamentals["available_date"] <= rebalance_date)
    ].copy()
    hist["sector"] = hist["sic"].apply(sic_to_sector)

    hist_median = (
        hist.groupby(["sector", "available_date"])["pb_ratio"]
        .median()
        .reset_index()
        .groupby("sector")["pb_ratio"]
        .agg(["mean", "std"])
    )

    # Current sector median P/B
    current_pb = pit.groupby("sector")["pb_ratio"].median()

    # Z-score: negative because cheap (low P/B) = high score
    sector_score = pd.Series(dtype=float)
    for sector in current_pb.index:
        if sector not in hist_median.index:
            continue
        mean = hist_median.loc[sector, "mean"]
        std  = hist_median.loc[sector, "std"]
        if std > 0:
            z = -(current_pb[sector] - mean) / std   # negate: cheap = good
            sector_score[sector] = z

    # Map back to permno level
    pit_indexed = pit.set_index("permno")
    return pit_indexed["sector"].map(sector_score).dropna()


def compute_analyst_factor(
    ibes_panel: pd.DataFrame | None,
    rebalance_date: pd.Timestamp,
) -> pd.Series:
    """
    Analyst factor: 3-month IBES consensus recommendation revision (rev_3m).
    Positive values = analysts becoming more bullish over the past 3 months.
    Returns empty Series if IBES panel not provided or no data before rebalance_date.
    """
    if ibes_panel is None or ibes_panel.empty:
        return pd.Series(dtype=float)
    available = ibes_panel.loc[ibes_panel.index <= rebalance_date]
    if available.empty:
        return pd.Series(dtype=float)
    return available.iloc[-1].dropna()


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-SECTIONAL NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def cross_section_zscore(s: pd.Series, winsor_pct: float = 0.01) -> pd.Series:
    """
    Winsorize then z-score a cross-sectional factor.
    Winsorizing at 1%/99% prevents outliers from dominating the score.
    """
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    lo, hi = s.quantile(winsor_pct), s.quantile(1 - winsor_pct)
    s = s.clip(lo, hi)
    std = s.std()
    if std == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def build_composite_score(
    factors: dict,
    weights: tuple,
) -> pd.Series:
    """
    Combine normalized factor scores into a single composite.
    factors: dict with keys technical, fundamental, sector_val, analyst
    weights: tuple (w_tech, w_fund, w_sector, w_analyst)
    Macro is intentionally excluded — used as a regime overlay at execution time.
    """
    keys = ["technical", "fundamental", "sector_val", "analyst"]
    w    = dict(zip(keys, weights))

    # Align all factors to a common index
    all_permnos = set()
    for s in factors.values():
        if isinstance(s, pd.Series):
            all_permnos |= set(s.index)

    composite = pd.Series(0.0, index=list(all_permnos))
    total_w   = 0.0

    for key in keys:
        f = factors.get(key)
        if f is None or (isinstance(f, pd.Series) and f.empty):
            continue
        f_norm = cross_section_zscore(f)
        composite = composite.add(f_norm * w[key], fill_value=0)
        total_w += w[key]

    if total_w > 0:
        composite /= total_w

    return composite.dropna().sort_values(ascending=False)


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(monthly_returns: pd.Series, rf: float = RISK_FREE_RATE) -> dict:
    """
    Compute performance metrics from a monthly return series.
    """
    r = monthly_returns.dropna()
    if len(r) < 12:
        return {}

    ann_return = (1 + r.mean()) ** 12 - 1
    ann_vol    = r.std() * np.sqrt(12)
    sharpe     = (r.mean() - rf) / r.std() * np.sqrt(12) if r.std() > 0 else np.nan

    # Max drawdown
    nav       = (1 + r).cumprod()
    rolling_max = nav.cummax()
    drawdowns   = (nav - rolling_max) / rolling_max
    max_dd      = drawdowns.min()

    calmar = ann_return / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "ann_return":   round(ann_return * 100, 2),   # percent
        "ann_vol":      round(ann_vol    * 100, 2),
        "sharpe":       round(sharpe, 3),
        "max_drawdown": round(max_dd * 100, 2),        # percent (negative)
        "calmar":       round(calmar, 3),
        "n_months":     len(r),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_single_backtest(
    ret_wide:       pd.DataFrame,
    prices_monthly: pd.DataFrame,
    fundamentals:   pd.DataFrame,
    macro_signals:  pd.DataFrame,
    momentum_lookback: int,
    momentum_skip:     int,
    ma_windows:        tuple,
    factor_weights:    tuple,
    ibes_panel:        pd.DataFrame | None = None,
    start: str = START_DATE,
    end:   str = END_DATE,
) -> tuple[pd.Series, list]:
    """
    Run one walk-forward backtest for a given parameter combination.
    Returns (monthly_portfolio_returns, list_of_monthly_holdings).
    """
    rebalance_dates = pd.date_range(start=start, end=end, freq="ME")
    monthly_returns = []
    holdings_log    = []
    prev_holdings   = set()

    for i, rdate in enumerate(rebalance_dates[:-1]):
        next_date = rebalance_dates[i + 1]

        # ── Build factor scores ───────────────────────────────────────────────
        f_tech = compute_technical_factor(
            ret_wide, prices_monthly, rdate,
            momentum_lookback, momentum_skip, ma_windows
        )
        f_fund = compute_fundamental_factor(fundamentals, rdate)
        f_sect = compute_sector_valuation_factor(fundamentals, rdate)
        f_anal = compute_analyst_factor(ibes_panel, rdate)

        # Macro excluded from composite — used as regime overlay below
        factors = {
            "technical":   f_tech,
            "fundamental": f_fund,
            "sector_val":  f_sect,
            "analyst":     f_anal,
        }

        # ── Composite score & portfolio selection ─────────────────────────────
        composite = build_composite_score(factors, factor_weights)

        if composite.empty or len(composite) < PORTFOLIO_SIZE:
            monthly_returns.append(np.nan)
            continue

        top_n = set(composite.head(PORTFOLIO_SIZE).index)

        # ── Transaction cost ──────────────────────────────────────────────────
        if prev_holdings:
            turnover = len(top_n.symmetric_difference(prev_holdings)) / (2 * PORTFOLIO_SIZE)
        else:
            turnover = 1.0
        tc = turnover * TRANSACTION_COST

        # ── Equal-weight portfolio return next month ──────────────────────────
        next_ret_row = ret_wide.loc[
            ret_wide.index.to_period("M") == next_date.to_period("M")
        ]

        if next_ret_row.empty:
            monthly_returns.append(np.nan)
            continue

        stock_rets = next_ret_row.iloc[0][list(top_n)].dropna()
        raw_ret    = stock_rets.mean()

        # ── Macro regime overlay: scale invested fraction by stress level ─────
        macro_score = compute_macro_scalar(macro_signals, rdate)
        if   macro_score < -1.0:
            invested = 0.70   # 30% to cash — stressed regime
        elif macro_score < -0.5:
            invested = 0.85   # 15% to cash — cautious regime
        else:
            invested = 1.00   # fully invested

        port_ret = raw_ret * invested + RISK_FREE_RATE * (1 - invested) - tc
        monthly_returns.append(port_ret)

        holdings_log.append({
            "date":     rdate,
            "holdings": list(top_n),
            "turnover": round(turnover, 3),
            "tc_drag":  round(tc * 100, 4),
        })
        prev_holdings = top_n

    return pd.Series(monthly_returns, index=rebalance_dates[:-1]), holdings_log


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETER SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

def run_parameter_sweep(
    ret_wide:       pd.DataFrame,
    prices_monthly: pd.DataFrame,
    fundamentals:   pd.DataFrame,
    macro_signals:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Test every combination in PARAM_GRID and return a summary DataFrame
    ranked by Sharpe ratio.
    """
    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))

    print(f"\nRunning {len(combos)} parameter combinations...\n")

    results      = []
    equity_curves = {}

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        label  = (
            f"mom{params['momentum_lookback']}_"
            f"skip{params['momentum_skip']}_"
            f"ma{params['ma_windows'][0]}-{params['ma_windows'][1]}_"
            f"w{'-'.join(str(int(w*100)) for w in params['factor_weights'])}"
        )

        print(f"  [{i+1:>3}/{len(combos)}] {label}", end=" ... ", flush=True)

        try:
            port_rets, _ = run_single_backtest(
                ret_wide       = ret_wide,
                prices_monthly = prices_monthly,
                fundamentals   = fundamentals,
                macro_signals  = macro_signals,
                momentum_lookback = params["momentum_lookback"],
                momentum_skip     = params["momentum_skip"],
                ma_windows        = params["ma_windows"],
                factor_weights    = params["factor_weights"],
            )

            metrics = compute_metrics(port_rets)
            if not metrics:
                print("skipped (insufficient data)")
                continue

            row = {
                "run_id":              label,
                "momentum_lookback":   params["momentum_lookback"],
                "momentum_skip":       params["momentum_skip"],
                "ma_short":            params["ma_windows"][0],
                "ma_long":             params["ma_windows"][1],
                "w_technical":         params["factor_weights"][0],
                "w_fundamental":       params["factor_weights"][1],
                "w_macro":             params["factor_weights"][2],
                "w_sector_val":        params["factor_weights"][3],
                "w_analyst":           params["factor_weights"][4],
                **metrics,
            }
            results.append(row)
            equity_curves[label] = port_rets

            print(f"Sharpe={metrics['sharpe']:.2f}  "
                  f"Ann.Ret={metrics['ann_return']:.1f}%  "
                  f"MaxDD={metrics['max_drawdown']:.1f}%")

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    summary = (
        pd.DataFrame(results)
        .sort_values("sharpe", ascending=False)
        .reset_index(drop=True)
    )
    summary.index.name = "rank"

    # Save results
    summary.to_parquet(RESULTS_DIR / "backtest_summary.parquet", index=True)
    summary.to_csv(RESULTS_DIR / "backtest_summary.csv", index=True)

    eq_df = pd.DataFrame(equity_curves)
    eq_df.to_parquet(RESULTS_DIR / "equity_curves.parquet")

    return summary, equity_curves


