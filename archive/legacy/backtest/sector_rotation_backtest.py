"""
backtest/sector_rotation_backtest.py
--------------------------------------
Sector rotation strategy: buy outperforming sectors, avoid laggards.

Signal
------
At each month-end, compute each sector ETF's 3-month cumulative return
relative to SPY.  Sectors with positive relative strength are classified
as BUY; those with negative or zero are SELL / avoid.

An optional credit-stress filter (HY spread z-score from FRED) tightens
selection when credit is under pressure: in stress, only the top-N
positive-RS sectors are held rather than all of them.

Portfolio
---------
Equal-weight the BUY sectors each month.
If no sector qualifies as BUY, hold cash at the risk-free rate.

Benchmarks
----------
  1. SPY              (S&P 500)
  2. EW-All           (equal-weight all 11 sector ETFs, rebalanced monthly)

Period
------
  Warm-up  : 2024-01-01  (3 months needed for first RS signal)
  Backtest : 2025-01-01 – present

Outputs
-------
  data/results/sector_rotation_monthly.csv   — per-month returns + holdings
  data/results/sector_rotation_summary.csv   — annualised performance table
  Console                                    — current signals + results table
"""

import sys
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import date

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sectorscope.metrics import compute_metrics

# ── Config ─────────────────────────────────────────────────────────────────────
RESULTS_DIR  = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SECTOR_ETFS = {
    "XLC":  "Communication",
    "XLY":  "Consumer Disc",
    "XLP":  "Consumer Stap",
    "XLE":  "Energy",
    "XLF":  "Financials",
    "XLV":  "Health Care",
    "XLI":  "Industrials",
    "XLB":  "Materials",
    "XLRE": "Real Estate",
    "XLK":  "Technology",
    "XLU":  "Utilities",
}

WARMUP_START   = "2024-10-01"        # need RS_WINDOW months before first signal
BACKTEST_START = "2025-04-01"        # first period counted in results (Q2 2025)
BACKTEST_END   = date.today().strftime("%Y-%m-%d")

RS_WINDOW      = 3     # months of relative strength for buy/sell signal
REBAL_MONTHS   = 3     # hold each portfolio for 3 months before rebalancing
RISK_FREE_ANN  = 0.05
TC             = 0.001  # 10 bps per side on turnover

STRESS_THRESHOLD  = 1.0   # used for display only (regime label)


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_prices() -> pd.DataFrame:
    tickers = list(SECTOR_ETFS.keys()) + ["SPY"]
    print(f"  Downloading monthly prices for {tickers} ...")
    raw = yf.download(
        tickers,
        start=WARMUP_START,
        end=BACKTEST_END,
        interval="1mo",
        auto_adjust=True,
        progress=False,
    )["Close"]
    raw.index = pd.to_datetime(raw.index) + pd.offsets.MonthEnd(0)
    raw = raw.sort_index()
    print(f"  Prices: {raw.shape[1]} tickers | "
          f"{raw.index[0].date()} – {raw.index[-1].date()} "
          f"({len(raw)} months)")
    return raw


def load_credit_regime() -> pd.Series:
    """
    Rolling z-score of HY spread level from FRED signals.
    Returns empty Series if file not available.
    """
    macro_path = Path("data/macro/fred_signals.parquet")
    if not macro_path.exists():
        return pd.Series(dtype=float)
    try:
        signals = pd.read_parquet(macro_path)
        signals.index = pd.to_datetime(signals.index) + pd.offsets.MonthEnd(0)
        if "hy_spread_widening" not in signals.columns:
            return pd.Series(dtype=float)
        s = signals["hy_spread_widening"].dropna()
        roll_mean = s.rolling(36, min_periods=12).mean()
        roll_std  = s.rolling(36, min_periods=12).std().replace(0, np.nan)
        return ((s - roll_mean) / roll_std).rename("credit_z")
    except Exception:
        return pd.Series(dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL
# ══════════════════════════════════════════════════════════════════════════════

def build_rs_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """
    3-month cumulative return for each sector ETF minus SPY.
    Positive = outperforming → BUY signal.
    """
    monthly_ret = prices.pct_change()
    cum3 = (1 + monthly_ret.fillna(0)).rolling(RS_WINDOW).apply(np.prod, raw=True) - 1
    spy_3m = cum3["SPY"]
    sector_cols = [t for t in SECTOR_ETFS if t in cum3.columns]
    rs = cum3[sector_cols].subtract(spy_3m, axis=0)
    return rs  # (date × sector ETF) — positive = outperforming SPY


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    prices:        pd.DataFrame,
    rs_panel:      pd.DataFrame,
    credit_regime: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Monthly walk-forward: signal at end of month t → portfolio return in month t+1.

    Returns
    -------
    monthly_df   : per-month DataFrame (strategy ret, SPY ret, EW ret, holdings, n_buy)
    signals_df   : per-month buy/sell classification per sector
    """
    monthly_ret  = prices.pct_change()
    sector_cols  = [t for t in SECTOR_ETFS if t in prices.columns]

    # Rebalance dates: every REBAL_MONTHS steps within the backtest window
    all_dates    = rs_panel.loc[BACKTEST_START:].index
    rebal_dates  = all_dates[::REBAL_MONTHS]

    records      = []
    signal_rows  = []
    prev_held    = set()

    for i, rdate in enumerate(rebal_dates[:-1]):
        next_rdate = rebal_dates[i + 1]

        # ── Relative strength at rdate ────────────────────────────────────────
        rs_row = rs_panel.loc[rdate, sector_cols].dropna()

        # ── Credit regime (display only) ──────────────────────────────────────
        credit_z = float(credit_regime.asof(rdate)) if not credit_regime.empty else np.nan
        stressed = (not np.isnan(credit_z)) and (credit_z > STRESS_THRESHOLD)

        # ── Classify sectors ──────────────────────────────────────────────────
        buy_sectors  = sorted(rs_row[rs_row > 0].index.tolist(),
                              key=lambda t: rs_row[t], reverse=True)
        sell_sectors = sorted(rs_row[rs_row <= 0].index.tolist())

        # ── Transaction cost ──────────────────────────────────────────────────
        held_set = set(buy_sectors)
        if prev_held:
            turnover = len(held_set.symmetric_difference(prev_held)) / max(2 * len(held_set), 1)
        else:
            turnover = 1.0
        tc_drag = turnover * TC

        # ── Compound return over the holding period ───────────────────────────
        period_slice = monthly_ret.loc[
            (monthly_ret.index > rdate) & (monthly_ret.index <= next_rdate)
        ]

        if buy_sectors:
            avail = [s for s in buy_sectors if s in period_slice.columns]
            if avail and not period_slice.empty:
                ew_monthly = period_slice[avail].mean(axis=1)
                port_ret   = float((1 + ew_monthly).prod() - 1) - tc_drag
            else:
                port_ret = np.nan
        else:
            rf_period = (1 + RISK_FREE_ANN) ** (REBAL_MONTHS / 12) - 1
            port_ret  = rf_period

        spy_period = float((1 + period_slice["SPY"].fillna(0)).prod() - 1) if "SPY" in period_slice.columns else np.nan
        ew_period  = float((1 + period_slice[sector_cols].fillna(0).mean(axis=1)).prod() - 1)

        records.append({
            "period_end":  next_rdate,
            "strategy":    port_ret,
            "spy":         spy_period,
            "ew_sectors":  ew_period,
            "n_buy":       len(buy_sectors),
            "n_sell":      len(sell_sectors),
            "holdings":    ", ".join(buy_sectors) if buy_sectors else "CASH",
            "stressed":    stressed,
            "credit_z":    round(credit_z, 3) if not np.isnan(credit_z) else "",
            "tc_drag":     round(tc_drag * 100, 3),
        })

        sig_row = {"date": rdate}
        for t in sector_cols:
            if t in rs_row.index:
                sig_row[t] = "BUY" if t in buy_sectors else "SELL"
            else:
                sig_row[t] = "N/A"
        signal_rows.append(sig_row)

        prev_held = held_set

    monthly_df  = pd.DataFrame(records).set_index("period_end")
    signals_df  = pd.DataFrame(signal_rows).set_index("date")
    return monthly_df, signals_df


# ══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def print_current_signals(rs_panel: pd.DataFrame, credit_regime: pd.Series):
    latest_date = rs_panel.index[-1]
    sector_cols = [t for t in SECTOR_ETFS if t in rs_panel.columns]
    rs_row      = rs_panel.loc[latest_date, sector_cols].dropna().sort_values(ascending=False)

    credit_z  = float(credit_regime.asof(latest_date)) if not credit_regime.empty else np.nan
    stressed  = (not np.isnan(credit_z)) and (credit_z > STRESS_THRESHOLD)

    W = 72
    print("\n" + "═" * W)
    print(f"  CURRENT SECTOR SIGNALS  —  as of {latest_date.date()}")
    if not np.isnan(credit_z):
        regime_str = f"STRESSED (z={credit_z:+.2f})" if stressed else f"Normal (z={credit_z:+.2f})"
        print(f"  Credit Regime : {regime_str}")
    print("═" * W)
    print(f"  {'ETF':<6}  {'Sector':<16}  {'3M RS vs SPY':>13}  {'Signal':>7}  Bar")
    print("  " + "─" * (W - 2))

    for etf, rs_val in rs_row.items():
        name   = SECTOR_ETFS.get(etf, etf)
        signal = "  BUY " if rs_val > 0 else " SELL "
        bar_len = int(abs(rs_val) * 100)
        bar     = ("█" * min(bar_len, 30)) if rs_val > 0 else ("░" * min(bar_len, 30))
        sign    = "+" if rs_val > 0 else ""
        print(f"  {etf:<6}  {name:<16}  {sign}{rs_val*100:>+8.2f}%  {signal}  {bar}")

    buy_list  = [e for e, v in rs_row.items() if v > 0]
    sell_list = [e for e, v in rs_row.items() if v <= 0]
    print("  " + "─" * (W - 2))
    print(f"  BUY  ({len(buy_list)}) : {', '.join(buy_list) if buy_list else 'none'}")
    print(f"  SELL ({len(sell_list)}): {', '.join(sell_list) if sell_list else 'none'}")
    print("═" * W)


def calc_metrics(r: pd.Series) -> dict:
    """Metrics with small-sample fallback (bypasses minimum period guard)."""
    m = compute_metrics(r, hold_months=REBAL_MONTHS)
    if m:
        return m
    r = r.dropna()
    if len(r) < 2:
        return {}
    ppy      = 12 / REBAL_MONTHS
    rf       = (1 + RISK_FREE_ANN) ** (REBAL_MONTHS / 12) - 1
    mu, std  = r.mean(), r.std()
    ann_ret  = (1 + mu) ** ppy - 1
    ann_vol  = std * np.sqrt(ppy)
    sharpe   = (mu - rf) / std * np.sqrt(ppy) if std > 0 else np.nan
    downside = r[r < rf] - rf
    down_std = float(np.sqrt((downside ** 2).mean())) if len(downside) > 0 else np.nan
    sortino  = (mu - rf) / down_std * np.sqrt(ppy) if down_std and down_std > 0 else np.nan
    nav      = (1 + r).cumprod()
    max_dd   = float(((nav - nav.cummax()) / nav.cummax()).min())
    calmar   = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    t_stat   = mu / (std / np.sqrt(len(r))) if std > 0 else np.nan
    return {
        "ann_return":   round(ann_ret * 100, 2),
        "ann_vol":      round(ann_vol  * 100, 2),
        "sharpe":       round(sharpe,   3),
        "sortino":      round(sortino,  3) if not np.isnan(sortino) else np.nan,
        "max_drawdown": round(max_dd   * 100, 2),
        "calmar":       round(calmar,   3),
        "hit_rate":     round((r > 0).mean() * 100, 1),
        "t_stat":       round(t_stat,   2),
        "n_months":     len(r) * REBAL_MONTHS,
    }


def print_performance_table(monthly_df: pd.DataFrame):
    strat = monthly_df["strategy"].dropna()
    spy   = monthly_df["spy"].dropna()
    ew    = monthly_df["ew_sectors"].dropna()

    m_strat = calc_metrics(strat)
    m_spy   = calc_metrics(spy)
    m_ew    = calc_metrics(ew)

    W = 80
    start = monthly_df.index[0].date()
    end   = monthly_df.index[-1].date()
    print("\n" + "═" * W)
    print(f"  SECTOR ROTATION RESULTS  —  {start} to {end}")
    print(f"  Signal : {RS_WINDOW}-month RS vs SPY  |  Rebalance every {REBAL_MONTHS} months  |  Equal-weight BUY sectors")
    print(f"  Cost   : 10 bps/side on turnover")
    print("═" * W)

    rows = [
        ("Ann. Return",  "ann_return",   True),
        ("Ann. Vol",     "ann_vol",      False),
        ("Sharpe",       "sharpe",       False),
        ("Sortino",      "sortino",      False),
        ("Max Drawdown", "max_drawdown", True),
        ("Calmar",       "calmar",       False),
        ("Hit Rate %",   "hit_rate",     False),
        ("t-stat",       "t_stat",       False),
        ("N Months",     "n_months",     False),
    ]

    print(f"\n  {'Metric':<20}  {'Sector Rotation':>17}  {'EW All Sectors':>15}  {'SPY':>10}")
    print("  " + "─" * (W - 2))
    for label, key, signed in rows:
        vs  = m_strat.get(key, float("nan"))
        ve  = m_ew.get(key, float("nan"))
        vsp = m_spy.get(key, float("nan"))
        fmt = lambda v: (f"{v:>+8.2f}" if signed else f"{v:>8.3f}") if not np.isnan(v) else "     n/a"
        print(f"  {label:<20}  {fmt(vs):>17}  {fmt(ve):>15}  {fmt(vsp):>10}")

    alpha = m_strat.get("ann_return", 0) - m_spy.get("ann_return", 0)
    print("  " + "─" * (W - 2))
    print(f"  {'Alpha vs SPY':<20}  {alpha:>+17.2f}%")
    print("═" * W)


def print_monthly_log(monthly_df: pd.DataFrame):
    n = len(monthly_df)
    print(f"\n  Period detail ({n} periods × {REBAL_MONTHS} months each):")
    print(f"  {'Period End':<12}  {'Strategy':>9}  {'SPY':>7}  {'EW':>7}  "
          f"{'#Buy':>5}  {'Holdings'}")
    print("  " + "─" * 76)
    for dt, row in monthly_df.iterrows():
        strat_s = f"{row['strategy']*100:>+7.2f}%" if not np.isnan(row["strategy"]) else "    n/a"
        spy_s   = f"{row['spy']*100:>+6.2f}%"      if not np.isnan(row["spy"])      else "   n/a"
        ew_s    = f"{row['ew_sectors']*100:>+6.2f}%" if not np.isnan(row["ew_sectors"]) else "   n/a"
        print(f"  {str(dt.date()):<12}  {strat_s}  {spy_s}  {ew_s}  "
              f"{int(row['n_buy']):>5}  {row['holdings']}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 72)
    print("  SECTOR ROTATION BACKTEST")
    print(f"  Period : {BACKTEST_START} – {BACKTEST_END}")
    print(f"  Signal : {RS_WINDOW}-month sector RS vs SPY  (positive = BUY)  |  Rebalance every {REBAL_MONTHS}M")
    print(f"  ETFs   : {', '.join(SECTOR_ETFS.keys())}")
    print("═" * 72 + "\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    prices        = load_prices()
    credit_regime = load_credit_regime()
    if credit_regime.empty:
        print("  [INFO] No FRED credit data — credit stress filter disabled")
    else:
        print(f"  Credit regime loaded ({len(credit_regime)} months)")

    # ── Build signals ──────────────────────────────────────────────────────────
    rs_panel = build_rs_panel(prices)

    # ── Current signals (latest month) ─────────────────────────────────────────
    print_current_signals(rs_panel, credit_regime)

    # ── Run backtest ───────────────────────────────────────────────────────────
    print("\n  Running backtest...")
    monthly_df, signals_df = run_backtest(prices, rs_panel, credit_regime)

    if monthly_df.empty:
        print("  [ERROR] No backtest periods — check date range.")
        return

    # ── Results ────────────────────────────────────────────────────────────────
    print_performance_table(monthly_df)
    print_monthly_log(monthly_df)

    # ── Save ───────────────────────────────────────────────────────────────────
    monthly_path  = RESULTS_DIR / "sector_rotation_monthly.csv"
    summary_path  = RESULTS_DIR / "sector_rotation_summary.csv"
    signals_path  = RESULTS_DIR / "sector_rotation_signals.csv"

    monthly_df.to_csv(monthly_path)

    strat = monthly_df["strategy"].dropna()
    spy   = monthly_df["spy"].dropna()
    ew    = monthly_df["ew_sectors"].dropna()
    summary_rows = [
        {"label": "Sector Rotation", **calc_metrics(strat)},
        {"label": "EW All Sectors",  **calc_metrics(ew)},
        {"label": "SPY",             **calc_metrics(spy)},
    ]
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    signals_df.to_csv(signals_path)

    print(f"\n  Saved → {monthly_path}")
    print(f"  Saved → {summary_path}")
    print(f"  Saved → {signals_path}")


if __name__ == "__main__":
    main()
