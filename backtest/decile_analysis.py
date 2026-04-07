"""
backtest/decile_analysis.py
---------------------------
Decile return analysis for the five-factor composite score.

Splits the liquid universe into 10 equal buckets by composite score each month,
computes equal-weighted returns per decile, and reports:
  - Mean monthly return per decile
  - Annualised return and Sharpe per decile
  - Spread (D10 - D1)
  - Monotonicity check
  - IS (2010–2024) and OOS (2025) side-by-side

Usage:
    python backtest/decile_analysis.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.five_factor_model import (
    load_crsp_wide,
    build_compustat_panels,
    build_sector_rs_panel,
    build_analyst_panel,
    build_macro_factor,
    build_unemployment_tilt_panel,
    build_all_factor_panels,
    FACTOR_WEIGHTS,
)
from sectorscope.utils import zscore as _zscore

SECTOR_ETFS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]

RESULTS_DIR = Path("data/results")
LOAD_START  = "2007-01-01"
IS_START    = "2010-01-01"
IS_END      = "2024-12-31"
OOS_START   = "2025-01-01"
OOS_END     = "2025-12-31"
RISK_FREE_ANN = 0.05


def load_optimal_weights() -> dict:
    import json
    path = Path("data/results/optimal_weights.json")
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return data.get("weights", FACTOR_WEIGHTS)
    return FACTOR_WEIGHTS


def build_composite(panels: dict, weights: dict, date: pd.Timestamp,
                    liquid_stocks: set) -> pd.Series | None:
    """Compute weighted composite z-score for all liquid stocks at a given date."""
    scored = {}
    for fname, panel in panels.items():
        if date not in panel.index:
            continue
        row = panel.loc[date].dropna()
        row = row[row.index.isin(liquid_stocks)]
        z = _zscore(row)
        if len(z) >= 50:
            scored[fname] = z

    if not scored:
        return None

    common = list(scored.values())[0].index
    for z in scored.values():
        common = common.intersection(z.index)
    if len(common) < 50:
        return None

    w_arr = np.array([weights.get(fn, 1 / len(scored)) for fn in scored])
    w_arr = w_arr / w_arr.sum()
    factor_mat = np.column_stack([scored[fn].reindex(common).values for fn in scored])
    composite = pd.Series(factor_mat @ w_arr, index=common)
    return composite


def run_decile_backtest(ret_wide: pd.DataFrame, panels: dict,
                        liq_panel: pd.DataFrame, weights: dict,
                        label: str) -> pd.DataFrame:
    """
    For each monthly rebalance date, rank stocks into 10 deciles by composite score,
    compute equal-weighted next-month return per decile.
    Returns a DataFrame (date × decile) of monthly returns.
    """
    dates = ret_wide.index
    decile_rets = {d: [] for d in range(1, 11)}
    index_dates = []

    for i in range(len(dates) - 1):
        rdate     = dates[i]
        next_date = dates[i + 1]

        liquid_stocks = set(ret_wide.columns)
        if not liq_panel.empty and rdate in liq_panel.index:
            liquid_row = liq_panel.loc[rdate].fillna(False)
            liquid_stocks = liquid_stocks.intersection(liquid_row[liquid_row].index)

        composite = build_composite(panels, weights, rdate, liquid_stocks)
        if composite is None or len(composite) < 100:
            continue

        composite_sorted = composite.sort_values()
        n = len(composite_sorted)
        decile_bounds = np.array_split(composite_sorted.index, 10)

        fwd_rets = ret_wide.loc[next_date]
        valid = True
        period_rets = {}
        for d, stocks in enumerate(decile_bounds, 1):
            r = fwd_rets.reindex(stocks).dropna()
            if len(r) == 0:
                valid = False
                break
            period_rets[d] = float(r.mean())

        if not valid:
            continue

        for d in range(1, 11):
            decile_rets[d].append(period_rets[d])
        index_dates.append(next_date)

    if not index_dates:
        return pd.DataFrame()

    df = pd.DataFrame(decile_rets, index=index_dates)
    df.columns = [f"D{d}" for d in range(1, 11)]
    print(f"  {label}: {len(df)} monthly periods, {len(composite_sorted):,} stocks in universe")
    return df


def compute_decile_stats(monthly_rets: pd.DataFrame) -> pd.DataFrame:
    """Annualised stats per decile from monthly return series."""
    rows = []
    rf_monthly = (1 + RISK_FREE_ANN) ** (1 / 12) - 1

    for col in monthly_rets.columns:
        r = monthly_rets[col].dropna()
        mu  = r.mean()
        std = r.std()
        ann_ret = (1 + mu) ** 12 - 1
        ann_vol = std * np.sqrt(12)
        sharpe  = (mu - rf_monthly) / std * np.sqrt(12) if std > 0 else np.nan
        max_dd  = float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min())
        hit     = (r > 0).mean()
        rows.append({
            "decile":     col,
            "mean_monthly": round(mu * 100, 3),
            "ann_return":   round(ann_ret * 100, 2),
            "ann_vol":      round(ann_vol * 100, 2),
            "sharpe":       round(sharpe, 3),
            "max_drawdown": round(max_dd * 100, 2),
            "hit_rate":     round(hit * 100, 1),
            "n":            len(r),
        })
    return pd.DataFrame(rows).set_index("decile")


def annual_decile_table(monthly_rets: pd.DataFrame, label: str, lines: list) -> None:
    """Year × decile annual return table."""
    df = monthly_rets.copy()
    df.index = pd.to_datetime(df.index)
    df["year"] = df.index.year
    deciles = [f"D{d}" for d in range(1, 11)]

    lines.append(f"\n  {label} — ANNUAL RETURNS BY DECILE")
    cols = deciles + ["D10–D1"]
    col_w = 8
    sep = "  " + "─" * (6 + 2 + (col_w + 2) * len(cols))
    hdr = f"  {'Year':>6}  " + "  ".join(f"{'D'+str(d):>{col_w}}" for d in range(1, 11)) + f"  {'D10-D1':>{col_w}}"
    lines.append(sep)
    lines.append(hdr)
    lines.append(sep)

    for yr, grp in df.groupby("year"):
        ann_rets = {}
        for col in deciles:
            r = grp[col].dropna()
            ann_rets[col] = (1 + r).prod() - 1 if len(r) > 0 else np.nan
        spread = ann_rets["D10"] - ann_rets["D1"]
        row = f"  {yr:>6}  "
        row += "  ".join(
            f"{ann_rets[d]*100:>+{col_w-1}.1f}%" if not np.isnan(ann_rets[d]) else f"{'---':>{col_w}}"
            for d in deciles
        )
        row += f"  {spread*100:>+{col_w-1}.1f}%" if not np.isnan(spread) else f"  {'---':>{col_w}}"
        lines.append(row)
    lines.append(sep)


def monthly_decile_table(monthly_rets: pd.DataFrame, label: str, lines: list) -> None:
    """Full monthly return history per decile."""
    df = monthly_rets.copy()
    df.index = pd.to_datetime(df.index)
    deciles = [f"D{d}" for d in range(1, 11)]
    months  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    col_w = 7

    lines.append(f"\n  {label} — MONTHLY RETURNS")
    hdr_base = f"  {'':>10}  " + "  ".join(f"{m:>{col_w}}" for m in months)
    sep = "  " + "─" * (len(hdr_base) - 2)

    for dec in deciles:
        r = df[dec].dropna()
        r.index = pd.to_datetime(r.index)
        pivot = r.to_frame("ret")
        pivot["year"]  = pivot.index.year
        pivot["month"] = pivot.index.month
        years = sorted(pivot["year"].unique())

        lines.append(f"\n  {dec}")
        lines.append(sep)
        lines.append(hdr_base)
        lines.append(sep)
        for yr in years:
            row = f"  {yr:>10}  "
            for mo in range(1, 13):
                val = pivot[(pivot["year"]==yr) & (pivot["month"]==mo)]["ret"]
                if len(val) > 0:
                    row += f"  {val.iloc[0]*100:>+{col_w-1}.1f}%"
                else:
                    row += f"  {'---':>{col_w}}"
            lines.append(row)
        lines.append(sep)


def print_decile_table(is_stats: pd.DataFrame, oos_stats: pd.DataFrame,
                       lines: list) -> None:
    W = 108

    def section(stats: pd.DataFrame, period_label: str):
        lines.append(f"\n  {period_label}")
        sep = "  " + "─" * 92
        hdr = (f"  {'Decile':>8}  │  {'Mean Mo%':>8}  {'Ann Ret':>8}  {'Ann Vol':>8}  "
               f"{'Sharpe':>8}  {'Max DD':>8}  {'Hit%':>7}  {'N':>5}")
        lines.append(sep)
        lines.append(hdr)
        lines.append(sep)
        for dec, row in stats.iterrows():
            marker = "  ◀ top"  if dec == "D10" else ("  ◀ bot" if dec == "D1" else "")
            lines.append(
                f"  {dec:>8}  │"
                f"  {row['mean_monthly']:>+7.3f}%"
                f"  {row['ann_return']:>+7.2f}%"
                f"  {row['ann_vol']:>7.2f}%"
                f"  {row['sharpe']:>8.3f}"
                f"  {row['max_drawdown']:>+7.2f}%"
                f"  {row['hit_rate']:>6.1f}%"
                f"  {int(row['n']):>5}"
                f"{marker}"
            )
        lines.append(sep)

        # Spread row
        d10 = stats.loc["D10"]
        d1  = stats.loc["D1"]
        lines.append(
            f"  {'D10–D1':>8}  │"
            f"  {d10['mean_monthly']-d1['mean_monthly']:>+7.3f}%"
            f"  {d10['ann_return']-d1['ann_return']:>+7.2f}%"
            f"  {'':>8}"
            f"  {'':>8}"
            f"  {'':>8}"
            f"  {'':>6}"
        )
        lines.append(sep)

        # Monotonicity
        ann_rets = stats["ann_return"].values
        mono_up  = sum(ann_rets[i] < ann_rets[i+1] for i in range(len(ann_rets)-1))
        lines.append(f"  Monotonicity: {mono_up}/9 deciles increasing D1→D10")

    section(is_stats,  "IN-SAMPLE  (2010–2024)")
    if not oos_stats.empty:
        section(oos_stats, "OUT-OF-SAMPLE  (2025)")

    # Side-by-side spread comparison
    lines.append("\n  SPREAD SUMMARY  (D10 Ann Return − D1 Ann Return)")
    lines.append("  " + "─" * 50)
    lines.append(f"  {'':20}  {'IS':>10}  {'OOS':>10}")
    lines.append("  " + "─" * 50)
    is_spread  = is_stats.loc["D10", "ann_return"]  - is_stats.loc["D1", "ann_return"]
    oos_spread = oos_stats.loc["D10", "ann_return"] - oos_stats.loc["D1", "ann_return"] if not oos_stats.empty else float("nan")
    is_sharpe_spread  = is_stats.loc["D10", "sharpe"]  - is_stats.loc["D1", "sharpe"]
    oos_sharpe_spread = oos_stats.loc["D10", "sharpe"] - oos_stats.loc["D1", "sharpe"] if not oos_stats.empty else float("nan")
    lines.append(f"  {'Ann Return spread':20}  {is_spread:>+9.2f}%  {oos_spread:>+9.2f}%")
    lines.append(f"  {'Sharpe spread':20}  {is_sharpe_spread:>+10.3f}  {oos_sharpe_spread:>+10.3f}")
    lines.append("  " + "─" * 50)


def fetch_benchmark_monthly(start: str, end: str) -> pd.DataFrame:
    """
    Download SPY and equal-weight sector ETF monthly total returns.

    Returns DataFrame indexed by month-end date with columns:
        SPY, EW-Sector
    """
    tickers = ["SPY"] + SECTOR_ETFS
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    # Resample to month-end
    monthly = raw.resample("ME").last()
    rets = monthly.pct_change().dropna(how="all")

    bm = pd.DataFrame(index=rets.index)
    bm["SPY"] = rets["SPY"]
    # Equal-weight across available sector ETFs each month
    sector_cols = [c for c in SECTOR_ETFS if c in rets.columns]
    bm["EW-Sector"] = rets[sector_cols].mean(axis=1)
    return bm.dropna(how="all")


def compute_benchmark_stats(monthly_bm: pd.DataFrame) -> pd.DataFrame:
    """Annualised stats for SPY and EW-Sector benchmarks."""
    rows = []
    rf_monthly = (1 + RISK_FREE_ANN) ** (1 / 12) - 1
    for col in monthly_bm.columns:
        r = monthly_bm[col].dropna()
        mu  = r.mean()
        std = r.std()
        ann_ret = (1 + mu) ** 12 - 1
        ann_vol = std * np.sqrt(12)
        sharpe  = (mu - rf_monthly) / std * np.sqrt(12) if std > 0 else np.nan
        max_dd  = float(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min())
        hit     = (r > 0).mean()
        rows.append({
            "label":        col,
            "ann_return":   round(ann_ret * 100, 2),
            "ann_vol":      round(ann_vol * 100, 2),
            "sharpe":       round(sharpe, 3),
            "max_drawdown": round(max_dd * 100, 2),
            "hit_rate":     round(hit * 100, 1),
            "n":            len(r),
        })
    return pd.DataFrame(rows).set_index("label")


def main():
    print("\n" + "═" * 70)
    print("  DECILE ANALYSIS  —  Loading data")
    print("═" * 70)

    ret_full = load_crsp_wide().loc[LOAD_START:]
    print(f"  CRSP : {ret_full.shape[1]:,} stocks | "
          f"{ret_full.index[0].date()} – {ret_full.index[-1].date()}")

    fund_panel = pd.DataFrame()
    sic_map    = pd.Series(dtype=str)
    liq_panel  = pd.DataFrame()
    try:
        fund_panel, sic_map, liq_panel = build_compustat_panels(ret_full)
        print(f"  Fund Quality: {fund_panel.shape[1]:,} stocks")
    except Exception as e:
        print(f"  [SKIP] Compustat: {e}")

    sector_rs = pd.DataFrame()
    try:
        sector_rs = build_sector_rs_panel(ret_full, sic_map)
    except Exception as e:
        print(f"  [SKIP] Sector RS: {e}")

    analyst_rev = pd.DataFrame()
    try:
        analyst_rev = build_analyst_panel(ret_full)
    except Exception as e:
        print(f"  [SKIP] Analyst: {e}")

    unemp_tilt = pd.DataFrame()
    try:
        unemp_tilt = build_unemployment_tilt_panel(ret_full, sic_map)
    except Exception as e:
        print(f"  [SKIP] Unemp tilt: {e}")

    print("\n  Building factor panels...")
    panels = build_all_factor_panels(ret_full, fund_panel, sector_rs, analyst_rev, unemp_tilt)
    weights = load_optimal_weights()
    print(f"  Active factors: {list(panels.keys())}")

    # ── Slice IS and OOS ──────────────────────────────────────────────────────
    ret_is  = ret_full.loc[IS_START:IS_END]
    ret_oos = ret_full.loc[OOS_START:OOS_END]

    def slice_panels(p, idx):
        return {k: v.reindex(idx, method="ffill") for k, v in p.items()}

    panels_is  = slice_panels(panels, ret_is.index)
    panels_oos = slice_panels(panels, ret_oos.index)
    liq_is  = liq_panel.reindex(ret_is.index,  method="ffill") if not liq_panel.empty else pd.DataFrame()
    liq_oos = liq_panel.reindex(ret_oos.index, method="ffill") if not liq_panel.empty else pd.DataFrame()

    print("\n  Running IS decile backtest...")
    is_monthly  = run_decile_backtest(ret_is,  panels_is,  liq_is,  weights, "IS")
    print("  Running OOS decile backtest...")
    oos_monthly = run_decile_backtest(ret_oos, panels_oos, liq_oos, weights, "OOS")

    is_stats  = compute_decile_stats(is_monthly)
    oos_stats = compute_decile_stats(oos_monthly) if not oos_monthly.empty else pd.DataFrame()

    # ── Build report ──────────────────────────────────────────────────────────
    lines = []
    W = 108
    lines.append("═" * W)
    lines.append("  SECTORSCOPE — DECILE RETURN ANALYSIS")
    lines.append("  Five-factor composite score ranked into 10 equal buckets each month")
    lines.append("  D1 = lowest composite score  |  D10 = highest composite score")
    lines.append(f"  IS : {IS_START} → {IS_END}   |   OOS : {OOS_START} → {OOS_END}")
    lines.append("  Universe: Top 1,000 stocks by market cap  |  Equal-weight within decile")
    lines.append("  No transaction costs (gross returns)")
    lines.append("═" * W)

    print_decile_table(is_stats, oos_stats, lines)

    annual_decile_table(is_monthly, "IN-SAMPLE (2010–2024)", lines)
    monthly_decile_table(is_monthly, "IN-SAMPLE (2010–2024)", lines)

    lines.append("")
    lines.append("═" * W)

    report = "\n".join(lines)
    print("\n" + report)

    # ── Benchmarks (SPY + EW-Sector) ──────────────────────────────────────────
    print("\n  Fetching benchmark returns (SPY + sector ETFs)...")
    try:
        bm_is  = fetch_benchmark_monthly(IS_START,  IS_END)
        bm_oos = fetch_benchmark_monthly(OOS_START, OOS_END)
        # Align to backtest dates
        bm_is  = bm_is.reindex(is_monthly.index)
        bm_oos = bm_oos.reindex(oos_monthly.index) if not oos_monthly.empty else bm_oos.iloc[0:0]
        bm_stats_is  = compute_benchmark_stats(bm_is.dropna(how="all"))
        bm_stats_oos = compute_benchmark_stats(bm_oos.dropna(how="all")) if not bm_oos.empty else pd.DataFrame()
        print(f"  Benchmarks IS: {len(bm_is)} periods  |  OOS: {len(bm_oos)} periods")
    except Exception as e:
        print(f"  [WARN] Benchmark fetch failed: {e}")
        bm_stats_is  = pd.DataFrame()
        bm_stats_oos = pd.DataFrame()

    # ── Save ──────────────────────────────────────────────────────────────────
    report_path = RESULTS_DIR / "decile_report.txt"
    report_path.write_text(report)

    is_monthly.to_csv(RESULTS_DIR / "decile_monthly_is.csv")
    oos_monthly.to_csv(RESULTS_DIR / "decile_monthly_oos.csv")
    is_stats.to_csv(RESULTS_DIR  / "decile_stats_is.csv")
    oos_stats.to_csv(RESULTS_DIR / "decile_stats_oos.csv")
    if not bm_stats_is.empty:
        bm_stats_is.to_csv(RESULTS_DIR  / "benchmark_stats_is.csv")
    if not bm_stats_oos.empty:
        bm_stats_oos.to_csv(RESULTS_DIR / "benchmark_stats_oos.csv")

    print(f"\n  Saved → {report_path}")
    print(f"  Saved → data/results/decile_monthly_is.csv")
    print(f"  Saved → data/results/decile_monthly_oos.csv")
    print(f"  Saved → data/results/decile_stats_is.csv / oos.csv")
    if not bm_stats_is.empty:
        print(f"  Saved → data/results/benchmark_stats_is.csv / oos.csv")


if __name__ == "__main__":
    main()
