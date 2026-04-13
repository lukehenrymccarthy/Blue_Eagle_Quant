"""
show_portfolio.py
-----------------
Shows the current model portfolio and latest backtest results.
Run from the project root:

    python show_portfolio.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ── helpers ───────────────────────────────────────────────────────────────────

def _separator(char="─", width=58):
    print(char * width)


def _header(title):
    _separator("═")
    print(f"  {title}")
    _separator("═")


def _section(title):
    print()
    _separator()
    print(f"  {title}")
    _separator()


# ── current portfolio ─────────────────────────────────────────────────────────

def show_current_portfolio():
    """Score today's signals and print the current 10-stock basket."""
    import numpy as np
    from backtest.core_model import ACTIVE_FACTORS, BASKET_SIZE, build_core_inputs
    from sectorscope.core_strategy import active_oos_end
    from sectorscope.modeling import FACTOR_WEIGHTS_OPT
    from sectorscope.utils import zscore as _zscore
    from sectorscope.factors import build_liquidity_screen

    print("  Loading data and scoring factors (this takes ~30 seconds)...")
    oos_end = active_oos_end()
    ret_wide, panels, is_liq = build_core_inputs(oos_end=oos_end)

    # Most recent month-end available
    target = ret_wide.index.max()

    scored = {}
    for fn in ACTIVE_FACTORS:
        panel = panels[fn]
        if target not in panel.index:
            continue
        row = (
            panel.loc[target]
            .dropna()
            .replace([float("inf"), float("-inf")], float("nan"))
            .dropna()
        )
        if target in is_liq.index:
            liq_row = is_liq.loc[target].fillna(False)
            row = row[row.index.isin(liq_row[liq_row].index)]
        scored[fn] = _zscore(row)

    if len(scored) != len(ACTIVE_FACTORS):
        print("  Warning: not all factors available at this date.")

    common = list(scored.values())[0].index
    for z in scored.values():
        common = common.intersection(z.index)

    factor_names = list(scored.keys())
    w_arr = __import__("numpy").array(
        [FACTOR_WEIGHTS_OPT.get(fn, 0.0) for fn in factor_names]
    )
    w_arr = w_arr / w_arr.sum()
    mat = __import__("numpy").column_stack(
        [scored[fn].reindex(common).values for fn in factor_names]
    )
    composite = pd.Series(mat @ w_arr, index=common).sort_values(ascending=False)
    top10 = composite.head(BASKET_SIZE)

    crsp = pd.read_parquet(
        "data/fundamentals/crsp_monthly_returns.parquet",
        columns=["permno", "date", "ticker"],
    )
    crsp["date"] = pd.to_datetime(crsp["date"])
    ticker_map = crsp.sort_values("date").groupby("permno")["ticker"].last().str.upper()

    _section(f"CURRENT PORTFOLIO  (scored at {target.strftime('%b %d, %Y')})")
    print(f"  {'Rank':<5} {'Ticker':<8} {'Composite Score':>16}")
    print(f"  {'----':<5} {'------':<8} {'---------------':>16}")
    for rank, (permno, score) in enumerate(top10.items(), 1):
        ticker = ticker_map.get(permno, f"permno:{permno}")
        print(f"  {rank:<5} {ticker:<8} {score:>16.3f}")

    # Compare to previous portfolio (last logged OOS rebalance)
    detail_path = Path("data/results/core_model_backtest_detail.csv")
    if detail_path.exists():
        detail = pd.read_csv(detail_path)
        oos = detail[detail["window"] == "OOS"].sort_values("rebal_date")
        if not oos.empty:
            prev_tickers = set(
                t.strip() for t in str(oos.iloc[-1]["tickers"]).split(",") if t.strip()
            )
            current_tickers = set(ticker_map.get(p, "") for p in top10.index)
            current_tickers.discard("")
            buys  = sorted(current_tickers - prev_tickers)
            sells = sorted(prev_tickers - current_tickers)
            holds = sorted(current_tickers & prev_tickers)
            prev_date = oos.iloc[-1]["rebal_date"]
            print()
            print(f"  vs. previous basket ({prev_date}):")
            if buys:
                print(f"  BUY  (+) : {', '.join(buys)}")
            if sells:
                print(f"  SELL (-) : {', '.join(sells)}")
            if holds:
                print(f"  HOLD (=) : {', '.join(holds)}")

    print()
    print(f"  Factor weights:  SUE {FACTOR_WEIGHTS_OPT.get('sue', 0)*100:.0f}%  |  "
          f"Sector RS {FACTOR_WEIGHTS_OPT.get('sector_rs_1m', 0)*100:.0f}%  |  "
          f"Momentum {FACTOR_WEIGHTS_OPT.get('mom_52wk_high', 0)*100:.0f}%")
    print(f"  Next rebalance:  every 2 months from {target.strftime('%b %Y')}")


# ── backtest summary ──────────────────────────────────────────────────────────

def show_backtest_summary():
    summary_path = Path("data/results/core_model_portfolio_summary.csv")
    annual_path  = Path("data/results/oos_annual_summary.csv")

    if not summary_path.exists():
        print("  No backtest results found. Run: python backtest/core_model_report.py")
        return

    df = pd.read_csv(summary_path)

    _section("IN-SAMPLE BACKTEST  (2005 – 2024, 19 years)")
    is_row = df[df["window"] == "IS"].iloc[0]
    rows = [
        ("Ann. Return",    f"{is_row['model_ann_return']:.2f}%",    f"{is_row['spy_ann_return']:.2f}%"),
        ("Ann. Vol",       f"{is_row['model_ann_vol']:.2f}%",       f"{is_row['spy_ann_vol']:.2f}%"),
        ("Sharpe",         f"{is_row['model_sharpe']:.3f}",         f"{is_row['spy_sharpe']:.3f}"),
        ("Sortino",        f"{is_row['model_sortino']:.3f}",        f"{is_row['spy_sortino']:.3f}"),
        ("Calmar",         f"{is_row['model_calmar']:.3f}",         f"{is_row['spy_calmar']:.3f}"),
        ("Max Drawdown",   f"{is_row['model_max_drawdown']:.2f}%",  f"{is_row['spy_max_drawdown']:.2f}%"),
        ("Hit Rate",       f"{is_row['model_hit_rate']:.1f}%",      f"{is_row['spy_hit_rate']:.1f}%"),
    ]
    print(f"  {'Metric':<16} {'Model':>10} {'SPY':>10}")
    print(f"  {'------':<16} {'-----':>10} {'---':>10}")
    for label, model_val, spy_val in rows:
        print(f"  {label:<16} {model_val:>10} {spy_val:>10}")

    if annual_path.exists():
        ann = pd.read_csv(annual_path).set_index("metric")
        _section("OUT-OF-SAMPLE  (Apr 2025 – Mar 2026, live)")
        oos_rows = [
            ("Ann. Return",   f"{ann.loc['ann_return_pct','model']}%",    f"{ann.loc['ann_return_pct','spy']}%"),
            ("Ann. Vol",      f"{ann.loc['ann_vol_pct','model']}%",       f"{ann.loc['ann_vol_pct','spy']}%"),
            ("Sharpe",        f"{ann.loc['sharpe','model']}",             f"{ann.loc['sharpe','spy']}"),
            ("Sortino",       f"{ann.loc['sortino','model']}",            f"{ann.loc['sortino','spy']}"),
            ("Calmar",        f"{ann.loc['calmar','model']}",             f"{ann.loc['calmar','spy']}"),
            ("Max Drawdown",  f"{ann.loc['max_drawdown_pct','model']}%",  f"{ann.loc['max_drawdown_pct','spy']}%"),
            ("Best Month",    f"{ann.loc['best_month_pct','model']}%",    f"{ann.loc['best_month_pct','spy']}%"),
            ("Worst Month",   f"{ann.loc['worst_month_pct','model']}%",   f"{ann.loc['worst_month_pct','spy']}%"),
        ]
        print(f"  {'Metric':<16} {'Model':>10} {'SPY':>10}")
        print(f"  {'------':<16} {'-----':>10} {'---':>10}")
        for label, model_val, spy_val in oos_rows:
            print(f"  {label:<16} {model_val:>10} {spy_val:>10}")
        print()
        print("  Note: OOS Sortino/Calmar are based on 11 monthly observations.")
        print("  Use the IS ratios as the primary long-run risk benchmarks.")


# ── period-by-period OOS ──────────────────────────────────────────────────────

def show_oos_periods():
    path = Path("data/results/core_model_backtest_detail.csv")
    if not path.exists():
        return
    df = pd.read_csv(path)
    oos = df[df["window"] == "OOS"].copy()
    if oos.empty:
        return

    _section("OOS PERIOD RETURNS  (each 2-month holding window)")
    print(f"  {'Rebal Date':<13} {'Hold Until':<13} {'Model':>8} {'SPY':>8} {'Excess':>8}  Holdings")
    print(f"  {'----------':<13} {'----------':<13} {'-----':>8} {'---':>8} {'------':>8}  --------")
    for _, row in oos.iterrows():
        end = row["period_end_date"] if str(row["period_end_date"]) != "nan" else "open"
        tickers = row["tickers"][:40] + "…" if len(str(row["tickers"])) > 40 else row["tickers"]
        print(
            f"  {row['rebal_date']:<13} {end:<13} "
            f"{row['net_ret_pct']:>7.2f}% {row['spy_ret_pct']:>7.2f}% "
            f"{row['excess_ret_pct']:>7.2f}%  {tickers}"
        )


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _header("SECTORSCOPE  |  Three-Factor Equity Model")
    print(f"  Strategy : SUE (80%)  +  Sector RS (15%)  +  Momentum (5%)")
    print(f"  Universe : Top 1000 US stocks by market cap")
    print(f"  Hold     : 10 stocks, rebalanced every 2 months")

    show_backtest_summary()
    show_oos_periods()
    show_current_portfolio()

    print()
    _separator("═")
    print("  To regenerate backtest results:")
    print("    python backtest/core_model_report.py")
    print("    python backtest/oos_annual_summary.py")
    _separator("═")
    print()
