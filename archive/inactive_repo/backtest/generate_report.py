"""
backtest/generate_report.py
---------------------------
Generates a clean, formatted backtest proof document covering:
  1. Top 5 factor candidates tested per factor category (by IS ICIR)
  2. Top-10 / 1-month hold performance — IS (2010–2024) and OOS (2025)

Usage:
    python backtest/generate_report.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sectorscope.metrics import compute_metrics

RESULTS_DIR = Path("data/results")
REPORT_PATH = RESULTS_DIR / "backtest_report.txt"

FACTOR_DISPLAY = {
    # Momentum
    "mom_52wk_high":     "52-Wk High Proximity ★",   # current model factor
    "mom_vol_adj_12m":   "Vol-Adj 12m Momentum",
    "mom_1m":            "1-Month Momentum",
    "mom_3m":            "3-Month Momentum",
    "mom_6m_skip1":      "6m-1m Momentum",
    "price_vs_12m_ma":   "Price vs 12m MA",
    "mom_12m_skip1":     "12m-1m Momentum",
    "mom_24m_skip1":     "24m-1m Momentum",
    # Quality
    "inv_debt_equity":   "Inv Debt/Equity     ★",
    "neg_accruals":      "Neg Accruals       ★",
    "earnings_yield":    "Earnings Yield      ★",
    "gross_prof":        "Gross Profitability",
    "roe_ttm":           "ROE TTM",
    "roa_ttm":           "ROA TTM",
    "fcf_yield":         "FCF Yield",
    "composite_equal_wt":"Quality Composite",
    "inv_pb":            "Inv P/B",
    "ocf_yield":         "OCF Yield",
    # Sector
    "sector_rs_1m":      "Sector RS 1m        ★",   # current model factor
    "sector_rs_3m":      "Sector RS 3m",
    "sector_rs_6m":      "Sector RS 6m",
    "sector_rs_12m":     "Sector RS 12m",
    "sector_rs_vol_adj": "Sector RS Vol-Adj 3m",
    "sector_abs_mom_3m": "Sector Abs Mom 3m",
    "sector_rs_trend_6m":"Sector RS Trend 6m",
    # Analyst
    "neg_dispersion":    "Neg Analyst Dispersion ★",
    "rev_1m":            "Rec Revision 1m",
    "rev_trend_3m":      "Rec Rev Trend 3m",
    "rev_6m":            "Rec Revision 6m",
    "rev_3m":            "Rec Revision 3m     ★",
    "net_upgrades":      "Net Upgrades",
    "buy_pct_rev_3m":    "Buy% Rev 3m",
    "coverage_chg_3m":   "Coverage Change 3m",
    "buy_pct":           "Buy %",
    "neg_meanrec":       "Neg Mean Rec",
    # Macro — traditional named indicators only
    "beta_x_unemp":      "Unemployment YOY      ★",
    "beta_x_indpro":     "Industrial Production YOY",
    "beta_x_payems":     "Nonfarm Payrolls YOY",
    "beta_x_yieldcurve": "Yield Curve (10Y-2Y)",
    "beta_x_hy (★)":     "HY Credit Spread",
    "beta_x_hy":         "HY Credit Spread",
}

GROUP_TITLES = {
    "momentum": "F1  MOMENTUM",
    "quality":  "F2  QUALITY",
    "sector":   "F3  SECTOR RELATIVE STRENGTH",
    "analyst":  "F4  ANALYST SIGNALS",
    "macro":    "F5  MACRO — TRADITIONAL ECONOMIC INDICATORS",
}

# Traditional macro indicators to show in F5 (in display order)
MACRO_TRADITIONAL = [
    "beta_x_unemp",
    "beta_x_indpro",
    "beta_x_payems",
    "beta_x_yieldcurve",
    "beta_x_hy",
    "beta_x_hy (★)",
]


# Signals selected in the live model that must always appear in their group's table
MODEL_SELECTED = {
    "momentum": {"mom_52wk_high"},
    "quality":  {"inv_debt_equity", "earnings_yield", "neg_accruals"},
    "sector":   {"sector_rs_1m"},
    "analyst":  {"neg_dispersion"},
}


def top5_per_group(fe: pd.DataFrame) -> dict:
    """Return top 5 rows per group sorted by IS ICIR descending.
    Always includes model-selected signals even if they fall outside the top 5.
    For macro, use fixed traditional indicator list instead of top-5 by ICIR."""
    out = {}
    for grp in ["momentum", "quality", "sector", "analyst"]:
        sub = fe[fe["group"] == grp].sort_values("is_icir", ascending=False)
        top5 = sub.head(5)
        pinned = MODEL_SELECTED.get(grp, set())
        missing = pinned - set(top5["signal"])
        if missing:
            extra = sub[sub["signal"].isin(missing)]
            top5 = pd.concat([top5, extra]).drop_duplicates("signal")
            top5 = top5.sort_values("is_icir", ascending=False)
        out[grp] = top5.reset_index(drop=True)

    # Macro: fixed set of traditional indicators, sorted by IS ICIR
    macro_rows = fe[fe["signal"].isin(MACRO_TRADITIONAL)].copy()
    macro_rows = macro_rows.sort_values("is_icir", ascending=False).reset_index(drop=True)
    out["macro"] = macro_rows
    return out


def format_factor_table(rows: pd.DataFrame) -> list[str]:
    lines = []
    hdr = (f"  {'Rank':>4}  {'Signal':<30}  {'IS ICIR':>8}  {'IS IC%+':>7}  "
           f"{'OOS ICIR':>9}  {'OOS IC%+':>8}  {'BH Sig':>7}")
    sep = "  " + "─" * 84
    lines.append(sep)
    lines.append(hdr)
    lines.append(sep)
    for i, row in rows.iterrows():
        sig = row["signal"]
        display = FACTOR_DISPLAY.get(sig, sig)
        bh = "Yes" if row["bh_signif"] else "No"
        oos_icir = f"{row['oos_icir']:>+9.3f}" if not pd.isna(row["oos_icir"]) else "      N/A"
        oos_pct  = f"{row['oos_pct_pos']:>7.1f}%" if not pd.isna(row["oos_pct_pos"]) else "     N/A"
        lines.append(
            f"  {i+1:>4}  {display:<30}  {row['is_icir']:>+8.3f}  "
            f"{row['is_pct_pos']:>6.1f}%  {oos_icir}  {oos_pct}  {bh:>7}"
        )
    lines.append(sep)
    return lines


def monthly_table(curve: pd.Series, label: str) -> list[str]:
    """Format monthly returns as a calendar-style table."""
    lines = []
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    df = curve.dropna().to_frame("ret")
    df["year"]  = df.index.year
    df["month"] = df.index.month

    years = sorted(df["year"].unique())
    hdr = f"  {'Year':>5}  " + "  ".join(f"{m:>6}" for m in months) + f"  {'Annual':>8}"
    sep = "  " + "─" * (len(hdr) - 2)
    lines.append(f"\n  {label}")
    lines.append(sep)
    lines.append(hdr)
    lines.append(sep)

    for yr in years:
        yr_data = df[df["year"] == yr]
        row_str = f"  {yr:>5}  "
        ann = 1.0
        for mo in range(1, 13):
            val = yr_data[yr_data["month"] == mo]["ret"]
            if len(val) > 0:
                r = float(val.iloc[0])
                ann *= (1 + r)
                row_str += f"  {r*100:>+5.1f}%"
            else:
                row_str += f"  {'---':>6}"
        ann_ret = (ann - 1) * 100
        row_str += f"  {ann_ret:>+7.1f}%"
        lines.append(row_str)

    lines.append(sep)
    return lines


def format_perf_block(m: dict, label: str) -> list[str]:
    lines = [
        f"  {label}",
        f"  {'─'*55}",
        f"  Ann Return  : {m['ann_return']:>+7.2f}%",
        f"  Ann Vol     : {m['ann_vol']:>7.2f}%",
        f"  Sharpe      : {m['sharpe']:>7.3f}",
        f"  Sortino     : {m.get('sortino', float('nan')):>7.3f}" if not pd.isna(m.get('sortino', float('nan'))) else f"  Sortino     :     N/A",
        f"  Max Drawdown: {m['max_drawdown']:>+7.2f}%",
        f"  Calmar      : {m['calmar']:>7.3f}",
        f"  Hit Rate    : {m['hit_rate']:>7.1f}%",
        f"  t-stat      : {m['t_stat']:>7.2f}",
        f"  N Periods   : {m['n_periods']:>7}",
    ]
    return lines


def main():
    lines = []

    W = 90
    lines.append("═" * W)
    lines.append("  SECTORSCOPE FIVE-FACTOR MODEL — BACKTEST PROOF")
    lines.append(f"  Generated: {pd.Timestamp.today().strftime('%Y-%m-%d')}")
    lines.append("  In-Sample (IS) : 2010-01-01 → 2024-12-31   |   15 years")
    lines.append("  Out-of-Sample  : 2025-01-01 → 2025-12-31   |   1 year")
    lines.append("  Universe       : Top 1,000 stocks by market cap (CRSP NYSE/AMEX/NASDAQ)")
    lines.append("  Costs          : 10 bps/side on turnover")
    lines.append("  Risk-Free Rate : 5.0% annual")
    lines.append("═" * W)

    # ── MODEL FACTORS ────────────────────────────────────────────────────────
    lines.append("")
    lines.append("─" * W)
    lines.append("  MODEL FACTORS & WEIGHTS")
    lines.append("─" * W)

    factors = [
        ("F1", "Momentum",                 "52-week high proximity (price / rolling 12m high)",        "10%"),
        ("F2", "Fundamental Quality",      "Inv D/E + Earnings Yield + Neg Accruals (avg z-score)",    "25%"),
        ("F3", "Sector Relative Strength", "1-month sector ETF return vs SPY",                        "25%"),
        ("F4", "Analyst Signal",           "Neg analyst dispersion",                                  "25%"),
        ("F5", "Unemployment YOY Tilt",    "Sector beta × unemployment YOY regime z-score",           "15%"),
    ]

    sep_f = "  " + "─" * 84
    hdr_f = f"  {'':3}  {'Factor':<26}  {'Description':<48}  {'Weight':>6}"
    lines.append(sep_f)
    lines.append(hdr_f)
    lines.append(sep_f)
    for code, name, desc, weight in factors:
        lines.append(f"  {code}   {name:<26}  {desc:<48}  {weight:>6}")
    lines.append(sep_f)
    lines.append("")
    lines.append("  Composite score = weighted sum of cross-sectional z-scores per factor")
    lines.append("  Macro overlay   = unemployment YOY z-score adjusts invested fraction")
    lines.append("                    (z < −1.0 → 70% invested, z < −0.5 → 85% invested)")

    # ── SECTION 1: Factor Candidates ─────────────────────────────────────────
    lines.append("")
    lines.append("─" * W)
    lines.append("  SECTION 1 — FACTOR EVALUATION")
    lines.append("  Top 5 candidates per factor group, ranked by IS ICIR (★ selected factors always shown)")
    lines.append("  IC = Spearman rank correlation vs next-month return")
    lines.append("  BH Sig = passed Benjamini-Hochberg FDR correction (q=0.05) within group")
    lines.append("  ★ = selected in live five-factor model")
    lines.append("─" * W)

    fe = pd.read_csv(RESULTS_DIR / "factor_evaluation.csv")
    top5 = top5_per_group(fe)

    for grp, title in GROUP_TITLES.items():
        lines.append(f"\n  {title}")
        lines.extend(format_factor_table(top5[grp]))

    # ── SECTION 2: Top-10 / 1-Month Hold Performance ─────────────────────────
    lines.append("")
    lines.append("─" * W)
    lines.append("  SECTION 2 — TOP 100 STOCKS / 1-MONTH HOLD")
    lines.append("  Equal-weight, rebalanced monthly, net of transaction costs")
    lines.append("─" * W)

    # IS curve (from five_factor_curves which covers 2010-2025)
    curves_full = pd.read_parquet(RESULTS_DIR / "five_factor_curves.parquet")
    is_curve  = curves_full["top100_hold1m"].loc[:"2024-12-31"].dropna()
    oos_curve = pd.read_parquet(RESULTS_DIR / "oos_2025_curves.parquet")["top100_hold1m"].dropna()

    is_m  = compute_metrics(is_curve,  hold_months=1)
    oos_m = compute_metrics(oos_curve, hold_months=1, min_periods=1)

    lines.append("")
    lines.append("  PERFORMANCE SUMMARY")
    lines.append("")

    # Side-by-side summary
    hdr = f"  {'Metric':<20}  {'IS (2010–2024)':>16}  {'OOS (2025)':>14}"
    sep = "  " + "─" * 56
    lines.append(sep)
    lines.append(hdr)
    lines.append(sep)

    metrics = [
        ("Ann Return",   "ann_return",   "%"),
        ("Ann Vol",      "ann_vol",      "%"),
        ("Sharpe",       "sharpe",       ""),
        ("Sortino",      "sortino",      ""),
        ("Max Drawdown", "max_drawdown", "%"),
        ("Calmar",       "calmar",       ""),
        ("Hit Rate",     "hit_rate",     "%"),
        ("t-stat",       "t_stat",       ""),
        ("N Periods",    "n_periods",    ""),
    ]

    for label, key, unit in metrics:
        iv = is_m.get(key, float("nan"))
        ov = oos_m.get(key, float("nan"))
        if key == "n_periods":
            lines.append(f"  {label:<20}  {int(iv):>16}  {int(ov):>14}")
        elif unit == "%":
            lines.append(f"  {label:<20}  {iv:>+15.2f}%  {ov:>+13.2f}%")
        else:
            iv_s = f"{iv:>16.3f}" if not pd.isna(iv) else f"{'N/A':>16}"
            ov_s = f"{ov:>14.3f}" if not pd.isna(ov) else f"{'N/A':>14}"
            lines.append(f"  {label:<20}  {iv_s}  {ov_s}")
    lines.append(sep)

    # Monthly return tables
    lines.extend(monthly_table(is_curve, "IN-SAMPLE MONTHLY RETURNS — Top 100 / 1-Month Hold (2010–2024)"))
    lines.extend(monthly_table(oos_curve, "OUT-OF-SAMPLE MONTHLY RETURNS — Top 100 / 1-Month Hold (2025)"))

    # NAV growth
    lines.append("")
    lines.append("  CUMULATIVE NAV (starts at 1.00)")
    lines.append("")
    nav_is  = (1 + is_curve).cumprod()
    nav_oos = (1 + oos_curve).cumprod()
    lines.append(f"  IS  start NAV  : 1.000   |   IS  end NAV : {nav_is.iloc[-1]:.3f}  "
                 f"(×{nav_is.iloc[-1]:.1f} over 15 years)")
    lines.append(f"  OOS start NAV  : 1.000   |   OOS end NAV : {nav_oos.iloc[-1]:.3f}  "
                 f"(2025 standalone year)")

    lines.append("")
    lines.append("═" * W)

    # ── SECTION 3: Basket Size × Hold Period Grid ────────────────────────────
    lines.append("")
    lines.append("─" * W)
    lines.append("  SECTION 3 — BASKET SIZE × HOLD PERIOD GRID")
    lines.append("  All configurations: 4 basket sizes × 4 hold periods = 16 combinations")
    lines.append("  Note: OOS Sharpe/Calmar unreliable for 3m (N=3) and 6m (N=1) — too few periods")
    lines.append("─" * W)

    is_df  = pd.read_csv(RESULTS_DIR / "oos_is_summary.csv")
    oos_df = pd.read_csv(RESULTS_DIR / "oos_2025_summary.csv")

    basket_labels = {10: "Top 10 ", 25: "Top 25 ", 50: "Top 50 ", 100: "Top 100"}
    hold_labels   = {1: "1-Month Hold", 2: "2-Month Hold", 3: "3-Month Hold", 6: "6-Month Hold"}

    for hold in [1, 2, 3, 6]:
        lines.append(f"\n  {hold_labels[hold]}  (OOS N = {oos_df[oos_df['hold_months']==hold]['n_periods'].iloc[0]} periods)")

        col_w = 13
        hdr = (f"  {'Basket':<9}  │"
               f"  {'IS Ret':>{col_w}}  {'IS Sharpe':>{col_w}}  {'IS MaxDD':>{col_w}}  {'IS Hit%':>{col_w}}  {'IS Calmar':>{col_w}}"
               f"  ║"
               f"  {'OOS Ret':>{col_w}}  {'OOS Sharpe':>{col_w}}  {'OOS MaxDD':>{col_w}}  {'OOS Hit%':>{col_w}}  {'OOS Calmar':>{col_w}}")
        sep = "  " + "─" * 9 + "──┼" + "─" * 73 + "──╫" + "─" * 74

        lines.append(sep)
        lines.append(hdr)
        lines.append(sep)

        for basket in [10, 25, 50, 100]:
            ir = is_df[(is_df["basket_size"] == basket) & (is_df["hold_months"] == hold)]
            or_ = oos_df[(oos_df["basket_size"] == basket) & (oos_df["hold_months"] == hold)]

            def fmt_pct(v, w=col_w):
                return f"{v:>+{w-1}.2f}%" if not pd.isna(v) else f"{'N/A':>{w}}"
            def fmt_num(v, w=col_w):
                return f"{v:>{w}.3f}" if not pd.isna(v) else f"{'N/A':>{w}}"

            if ir.empty:
                is_cols = ["N/A"] * 5
            else:
                r = ir.iloc[0]
                is_cols = [fmt_pct(r["ann_return"]), fmt_num(r["sharpe"]),
                           fmt_pct(r["max_drawdown"]), fmt_pct(r["hit_rate"]),
                           fmt_num(r["calmar"])]

            if or_.empty:
                oos_cols = ["N/A"] * 5
            else:
                r = or_.iloc[0]
                oos_cols = [fmt_pct(r["ann_return"]), fmt_num(r["sharpe"]),
                            fmt_pct(r["max_drawdown"]), fmt_pct(r["hit_rate"]),
                            fmt_num(r["calmar"])]

            lines.append(
                f"  {basket_labels[basket]:<9}  │"
                f"  {'  '.join(is_cols)}"
                f"  ║"
                f"  {'  '.join(oos_cols)}"
            )
        lines.append(sep)

    # ── IS vs OOS Sharpe Δ summary ────────────────────────────────────────────
    lines.append("")
    lines.append("  IS → OOS SHARPE CHANGE  (positive = improved OOS)")
    sep2 = "  " + "─" * 62
    lines.append(sep2)
    hdr2 = f"  {'':9}  {'1m Hold':>10}  {'2m Hold':>10}  {'3m Hold':>10}  {'6m Hold':>10}"
    lines.append(hdr2)
    lines.append(sep2)
    for basket in [10, 25, 50, 100]:
        row = f"  {basket_labels[basket]:<9}"
        for hold in [1, 2, 3, 6]:
            ir  = is_df[(is_df["basket_size"]  == basket) & (is_df["hold_months"]  == hold)]
            or_ = oos_df[(oos_df["basket_size"] == basket) & (oos_df["hold_months"] == hold)]
            if ir.empty or or_.empty:
                row += f"  {'N/A':>10}"
                continue
            delta = or_.iloc[0]["sharpe"] - ir.iloc[0]["sharpe"]
            if pd.isna(delta):
                row += f"  {'N/A':>10}"
            else:
                arrow = "▲" if delta >= 0 else "▼"
                row += f"  {arrow}{delta:>+8.3f}"
        lines.append(row)
    lines.append(sep2)

    lines.append("")
    lines.append("═" * W)

    # ── SECTION 4: Decile Return Summary ─────────────────────────────────────
    lines.append("")
    lines.append("─" * W)
    lines.append("  SECTION 4 — DECILE RETURN SUMMARY")
    lines.append("  Universe ranked monthly into 10 equal buckets by composite score")
    lines.append("  D1 = lowest score (short candidates)  |  D10 = highest score (long candidates)")
    lines.append("  Gross returns, equal-weight within each decile, no transaction costs")
    lines.append("─" * W)

    is_dec  = pd.read_csv(RESULTS_DIR / "decile_stats_is.csv",  index_col="decile")
    oos_dec = pd.read_csv(RESULTS_DIR / "decile_stats_oos.csv", index_col="decile")

    # Summary stats table
    col_w = 10
    sep_d  = "  " + "─" * (8 + 2 + (col_w + 2) * 8)
    hdr_d  = (f"  {'Decile':>8}  "
              f"{'IS Ret':>{col_w}}  {'IS Sharpe':>{col_w}}  {'IS MaxDD':>{col_w}}  {'IS Hit%':>{col_w}}  "
              f"{'OOS Ret':>{col_w}}  {'OOS Sharpe':>{col_w}}  {'OOS MaxDD':>{col_w}}  {'OOS Hit%':>{col_w}}")

    lines.append("")
    lines.append(sep_d)
    lines.append(hdr_d)
    lines.append(sep_d)

    for dec in [f"D{d}" for d in range(1, 11)]:
        ir  = is_dec.loc[dec]  if dec in is_dec.index  else None
        or_ = oos_dec.loc[dec] if dec in oos_dec.index else None
        marker = "  ◀ top" if dec == "D10" else ("  ◀ bot" if dec == "D1" else "")

        def fmtp(v):
            return f"{v:>+{col_w-1}.1f}%" if not pd.isna(v) else f"{'N/A':>{col_w}}"
        def fmtn(v):
            return f"{v:>{col_w}.3f}" if not pd.isna(v) else f"{'N/A':>{col_w}}"

        is_cols  = (fmtp(ir["ann_return"]),  fmtn(ir["sharpe"]),  fmtp(ir["max_drawdown"]),  fmtp(ir["hit_rate"]))  if ir  is not None else ("N/A",)*4
        oos_cols = (fmtp(or_["ann_return"]), fmtn(or_["sharpe"]), fmtp(or_["max_drawdown"]), fmtp(or_["hit_rate"])) if or_ is not None else ("N/A",)*4

        lines.append(f"  {dec:>8}  {'  '.join(is_cols)}  {'  '.join(oos_cols)}{marker}")

    lines.append(sep_d)

    # Spread rows
    def spread_row(label, col, fmt_fn):
        d10_is  = is_dec.loc["D10",  col] if "D10" in is_dec.index  else float("nan")
        d1_is   = is_dec.loc["D1",   col] if "D1"  in is_dec.index  else float("nan")
        d10_oos = oos_dec.loc["D10", col] if "D10" in oos_dec.index else float("nan")
        d1_oos  = oos_dec.loc["D1",  col] if "D1"  in oos_dec.index else float("nan")
        s_is  = fmt_fn(d10_is  - d1_is)  if not (pd.isna(d10_is)  or pd.isna(d1_is))  else f"{'N/A':>{col_w}}"
        s_oos = fmt_fn(d10_oos - d1_oos) if not (pd.isna(d10_oos) or pd.isna(d1_oos)) else f"{'N/A':>{col_w}}"
        lines.append(f"  {'D10–D1':>8}  {s_is:<{col_w*4+6}}  {s_oos}")

    spread_row("Ann Return spread", "ann_return", lambda v: f"{v:>+{col_w-1}.1f}%")
    spread_row("Sharpe spread",     "sharpe",     lambda v: f"{v:>{col_w}.3f}")
    lines.append(sep_d)

    # Annual returns by decile (IS only)
    lines.append("")
    lines.append("  IN-SAMPLE ANNUAL RETURNS BY DECILE (2010–2024)")
    is_monthly = pd.read_csv(RESULTS_DIR / "decile_monthly_is.csv", index_col=0, parse_dates=True)
    is_monthly.index = pd.to_datetime(is_monthly.index)
    decile_cols = [f"D{d}" for d in range(1, 11)]
    ann_col_w = 8

    sep_a = "  " + "─" * (6 + 2 + (ann_col_w + 2) * 11)
    hdr_a = (f"  {'Year':>6}  " +
             "  ".join(f"{c:>{ann_col_w}}" for c in decile_cols) +
             f"  {'D10-D1':>{ann_col_w}}")
    lines.append(sep_a)
    lines.append(hdr_a)
    lines.append(sep_a)

    is_monthly["year"] = is_monthly.index.year
    for yr, grp in is_monthly.groupby("year"):
        ann = {}
        for col in decile_cols:
            r = grp[col].dropna()
            ann[col] = (1 + r).prod() - 1 if len(r) > 0 else float("nan")
        spread = ann["D10"] - ann["D1"] if not (pd.isna(ann["D10"]) or pd.isna(ann["D1"])) else float("nan")
        row = f"  {yr:>6}  "
        row += "  ".join(
            f"{ann[c]*100:>+{ann_col_w-1}.1f}%" if not pd.isna(ann[c]) else f"{'---':>{ann_col_w}}"
            for c in decile_cols
        )
        row += f"  {spread*100:>+{ann_col_w-1}.1f}%" if not pd.isna(spread) else f"  {'---':>{ann_col_w}}"
        lines.append(row)

    # Average row
    lines.append(sep_a)
    avg_row = f"  {'Avg':>6}  "
    avg_ann = {}
    for col in decile_cols:
        mu = is_monthly[col].mean()
        avg_ann[col] = (1 + mu) ** 12 - 1
    avg_spread = avg_ann["D10"] - avg_ann["D1"]
    avg_row += "  ".join(f"{avg_ann[c]*100:>+{ann_col_w-1}.1f}%" for c in decile_cols)
    avg_row += f"  {avg_spread*100:>+{ann_col_w-1}.1f}%"
    lines.append(avg_row)
    lines.append(sep_a)

    # Monotonicity summary
    avg_returns = [avg_ann[f"D{d}"] for d in range(1, 11)]
    mono_up = sum(avg_returns[i] < avg_returns[i+1] for i in range(9))
    lines.append(f"\n  IS monotonicity : {mono_up}/9 deciles increasing D1 → D10")
    lines.append(f"  IS D10–D1 spread: {(avg_ann['D10'] - avg_ann['D1'])*100:>+.2f}% annualised (avg)")

    if not oos_dec.empty:
        oos_monthly = pd.read_csv(RESULTS_DIR / "decile_monthly_oos.csv", index_col=0, parse_dates=True)
        oos_d10 = (1 + oos_monthly["D10"].dropna().mean()) ** 12 - 1
        oos_d1  = (1 + oos_monthly["D1"].dropna().mean())  ** 12 - 1
        lines.append(f"  OOS D10–D1 spread: {(oos_d10 - oos_d1)*100:>+.2f}% annualised (2025, N=11 months)")

    lines.append("")
    lines.append("═" * W)

    # ── Print & Save ─────────────────────────────────────────────────────────
    report = "\n".join(lines)
    print(report)
    REPORT_PATH.write_text(report)
    print(f"\n  Saved → {REPORT_PATH}")


if __name__ == "__main__":
    main()
