"""
backtest/signal_portfolio_search.py
-----------------------------------
Search for the best stock-selection portfolio built directly from individual
signals rather than a fixed pre-defined factor model.

Workflow:
  1. Build individual stock-level candidate signals from existing research code
  2. Orient each signal in the empirically favorable direction using IS IC
  3. Shortlist the strongest stock-selection signals
  4. Backtest single-signal, pair, and triple-signal portfolios
  5. Rank portfolios by IS performance and report OOS behavior

Usage:
    python backtest/signal_portfolio_search.py
    python backtest/signal_portfolio_search.py --top-signals 12 --max-combo 3
"""

import sys
import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.factor_evaluation import (
    IS_END,
    OOS_END,
    OOS_START,
    build_analyst_candidates,
    build_momentum_candidates,
    build_quality_candidates,
    build_sector_rs_candidates,
    build_sic_map,
    compute_monthly_ic,
    ic_stats,
    load_compustat,
    load_ibes,
    load_returns,
)
from backtest.updated_model import run_backtest, run_decile_backtest, build_spy_benchmark
from sectorscope.factors import build_liquidity_screen
from sectorscope.metrics import compute_metrics

RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STOCK_SIGNAL_GROUPS = ("momentum", "quality", "sector", "analyst")


def build_stock_signal_panels(
    ret_wide: pd.DataFrame,
    fund: pd.DataFrame,
    ibes: pd.DataFrame,
    sic_map: pd.Series,
) -> dict[str, pd.DataFrame]:
    panels: dict[str, pd.DataFrame] = {}
    grouped = {
        "momentum": build_momentum_candidates(ret_wide),
        "quality": build_quality_candidates(fund, ret_wide, sic_map),
        "sector": build_sector_rs_candidates(ret_wide, sic_map),
        "analyst": build_analyst_candidates(ibes, ret_wide),
    }
    for group, group_panels in grouped.items():
        for signal_name, panel in group_panels.items():
            if panel is None or panel.empty:
                continue
            panels[f"{group}::{signal_name}"] = panel
    return panels


def orient_signal_panels(
    panels: dict[str, pd.DataFrame],
    ret_wide: pd.DataFrame,
    is_idx: pd.DatetimeIndex,
    min_abs_icir: float = 0.05,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Flip signals with negative IS IC so all retained panels point in the
    empirically favorable direction for portfolio ranking.
    """
    fwd_ret = ret_wide.pct_change(1).shift(-1)
    oriented: dict[str, pd.DataFrame] = {}
    rows = []

    for name, panel in panels.items():
        is_ic_arr = compute_monthly_ic(panel, fwd_ret, is_idx)
        stats = ic_stats(is_ic_arr)
        if not stats:
            continue
        sign = 1.0 if stats["mean_ic"] >= 0 else -1.0
        oriented[name] = panel * sign
        rows.append({
            "signal": name,
            "direction": "as_is" if sign > 0 else "flipped",
            "is_mean_ic": stats["mean_ic"],
            "is_icir": stats["icir"],
            "is_t_stat": stats["t_stat"],
            "retain": abs(stats["icir"]) >= min_abs_icir,
        })

    meta = pd.DataFrame(rows).sort_values("is_icir", ascending=False)
    keep = set(meta.loc[meta["retain"], "signal"])
    oriented = {k: v for k, v in oriented.items() if k in keep}
    meta = meta[meta["signal"].isin(keep)].reset_index(drop=True)
    return oriented, meta


def shortlist_signals(meta: pd.DataFrame, top_signals: int) -> list[str]:
    """
    Prefer signals with stronger IS ICIR after orientation.
    """
    if meta.empty:
        return []
    return meta.sort_values(["is_icir", "is_t_stat"], ascending=False)["signal"].head(top_signals).tolist()


def evaluate_combo(
    combo: tuple[str, ...],
    panels: dict[str, pd.DataFrame],
    ret_wide: pd.DataFrame,
    is_liq: pd.DataFrame,
    is_idx: pd.DatetimeIndex,
    oos_idx: pd.DatetimeIndex,
    basket_pct: float,
    hold_months: int,
) -> dict | None:
    combo_panels = {name: panels[name] for name in combo}
    combo_weights = {name: 1 / len(combo) for name in combo}

    curve_is, _ = run_backtest(
        ret_wide.reindex(is_idx),
        {k: v.reindex(is_idx) for k, v in combo_panels.items()},
        is_liq.reindex(is_idx),
        macro=None,
        vix_macro=None,
        hold_months=hold_months,
        basket_pct=basket_pct,
        weights=combo_weights,
        score_weighted=False,
    )
    curve_oos, _ = run_backtest(
        ret_wide.reindex(oos_idx),
        {k: v.reindex(oos_idx) for k, v in combo_panels.items()},
        is_liq.reindex(oos_idx),
        macro=None,
        vix_macro=None,
        hold_months=hold_months,
        basket_pct=basket_pct,
        weights=combo_weights,
        score_weighted=False,
    )

    m_is = compute_metrics(curve_is, hold_months=hold_months, min_periods=6)
    m_oos = compute_metrics(curve_oos, hold_months=hold_months, min_periods=2)
    if not m_is:
        return None

    dec_is = run_decile_backtest(
        ret_wide.reindex(is_idx),
        {k: v.reindex(is_idx) for k, v in combo_panels.items()},
        is_liq.reindex(is_idx),
        hold_months=hold_months,
        weights=combo_weights,
    )
    d1_is = compute_metrics(dec_is["D1"], hold_months=hold_months, min_periods=4) or {}
    d10_is = compute_metrics(dec_is["D10"], hold_months=hold_months, min_periods=4) or {}
    spread_is = (d1_is.get("ann_return", np.nan) - d10_is.get("ann_return", np.nan)
                 if d1_is and d10_is else np.nan)

    dec_oos = run_decile_backtest(
        ret_wide.reindex(oos_idx),
        {k: v.reindex(oos_idx) for k, v in combo_panels.items()},
        is_liq.reindex(oos_idx),
        hold_months=hold_months,
        weights=combo_weights,
    )
    d1_oos = compute_metrics(dec_oos["D1"], hold_months=hold_months, min_periods=2) or {}
    d10_oos = compute_metrics(dec_oos["D10"], hold_months=hold_months, min_periods=2) or {}
    spread_oos = (d1_oos.get("ann_return", np.nan) - d10_oos.get("ann_return", np.nan)
                  if d1_oos and d10_oos else np.nan)

    return {
        "combo": " + ".join(combo),
        "n_signals": len(combo),
        "is_ann_return": m_is["ann_return"],
        "is_sharpe": m_is["sharpe"],
        "is_calmar": m_is["calmar"],
        "is_max_dd": m_is["max_drawdown"],
        "is_hit_rate": m_is["hit_rate"],
        "is_d1_d10_spread": round(spread_is, 2) if spread_is == spread_is else np.nan,
        "oos_ann_return": m_oos.get("ann_return", np.nan) if m_oos else np.nan,
        "oos_sharpe": m_oos.get("sharpe", np.nan) if m_oos else np.nan,
        "oos_calmar": m_oos.get("calmar", np.nan) if m_oos else np.nan,
        "oos_max_dd": m_oos.get("max_drawdown", np.nan) if m_oos else np.nan,
        "oos_hit_rate": m_oos.get("hit_rate", np.nan) if m_oos else np.nan,
        "oos_d1_d10_spread": round(spread_oos, 2) if spread_oos == spread_oos else np.nan,
    }


def print_top_results(df: pd.DataFrame, title: str, top_n: int = 15) -> None:
    print("\n" + "═" * 120)
    print(f"  {title}")
    print("═" * 120)
    if df.empty:
        print("  No portfolios evaluated.")
        return
    cols = [
        "combo", "n_signals", "is_ann_return", "is_sharpe", "is_calmar",
        "is_d1_d10_spread", "oos_ann_return", "oos_sharpe", "oos_d1_d10_spread",
    ]
    print(df[cols].head(top_n).to_string(index=False))
    print("═" * 120)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-signals", type=int, default=12,
                        help="Shortlist this many oriented signals before combo search.")
    parser.add_argument("--max-combo", type=int, default=3, choices=[1, 2, 3],
                        help="Maximum number of signals per portfolio.")
    parser.add_argument("--basket-pct", type=float, default=0.25,
                        help="Top fraction of liquid universe to hold.")
    parser.add_argument("--hold-months", type=int, default=1,
                        help="Holding period in months.")
    parser.add_argument("--universe-size", type=int, default=1000,
                        help="Universe size based on market cap.")
    args = parser.parse_args()

    print("\n" + "═" * 80)
    print("  SIGNAL PORTFOLIO SEARCH")
    print("  Search best portfolio directly from individual stock signals")
    print("═" * 80)

    print("\n  Loading data...")
    ret_wide = load_returns().loc[:OOS_END]
    fund = load_compustat()
    ibes = load_ibes()
    sic_map = build_sic_map(fund)
    is_liq = build_liquidity_screen(fund, ret_wide, args.universe_size)
    is_idx = ret_wide.index[ret_wide.index <= IS_END]
    oos_idx = ret_wide.index[(ret_wide.index >= OOS_START) & (ret_wide.index <= OOS_END)]
    print(f"  Returns: {ret_wide.shape[1]:,} stocks")
    print(f"  IS months: {len(is_idx)} | OOS months: {len(oos_idx)}")

    print("\n  Building candidate stock signals...")
    raw_panels = build_stock_signal_panels(ret_wide, fund, ibes, sic_map)
    print(f"  Candidate panels: {len(raw_panels)}")

    print("  Orienting signals by IS IC direction...")
    oriented_panels, meta = orient_signal_panels(raw_panels, ret_wide, is_idx)
    meta.to_csv(RESULTS_DIR / "signal_orientation.csv", index=False)
    print(f"  Retained oriented signals: {len(oriented_panels)}")

    shortlist = shortlist_signals(meta, args.top_signals)
    if not shortlist:
        print("  [ERROR] No signals survived orientation / filtering.")
        return

    print("\n  Shortlist:")
    for sig in shortlist:
        row = meta.loc[meta["signal"] == sig].iloc[0]
        print(f"    {sig:<35} ICIR={row['is_icir']:.3f}  t={row['is_t_stat']:.2f}  dir={row['direction']}")

    print("\n  Evaluating portfolio combinations...")
    results = []
    for k in range(1, args.max_combo + 1):
        for combo in itertools.combinations(shortlist, k):
            row = evaluate_combo(
                combo=combo,
                panels=oriented_panels,
                ret_wide=ret_wide,
                is_liq=is_liq,
                is_idx=is_idx,
                oos_idx=oos_idx,
                basket_pct=args.basket_pct,
                hold_months=args.hold_months,
            )
            if row:
                results.append(row)

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print("  [ERROR] No viable portfolios evaluated.")
        return

    results_df = results_df.sort_values(
        ["is_sharpe", "is_ann_return", "is_d1_d10_spread"],
        ascending=False,
    ).reset_index(drop=True)
    results_df.to_csv(RESULTS_DIR / "signal_portfolio_search.csv", index=False)

    print_top_results(results_df, "TOP PORTFOLIOS BY IN-SAMPLE PERFORMANCE")

    robust_df = results_df.sort_values(
        ["oos_sharpe", "oos_ann_return", "is_sharpe"],
        ascending=False,
    ).reset_index(drop=True)
    print_top_results(robust_df, "TOP PORTFOLIOS BY OOS FOLLOW-THROUGH")

    spy_is = build_spy_benchmark(ret_wide.reindex(is_idx), args.hold_months)
    spy_oos = build_spy_benchmark(ret_wide.reindex(oos_idx), args.hold_months)
    m_spy_is = compute_metrics(spy_is, hold_months=args.hold_months, min_periods=6)
    m_spy_oos = compute_metrics(spy_oos, hold_months=args.hold_months, min_periods=2)
    print("\n  SPY benchmark:")
    print(f"    IS  ann_ret={m_spy_is.get('ann_return', np.nan):.2f}%  sharpe={m_spy_is.get('sharpe', np.nan):.3f}")
    print(f"    OOS ann_ret={m_spy_oos.get('ann_return', np.nan):.2f}%  sharpe={m_spy_oos.get('sharpe', np.nan):.3f}")
    print(f"\n  Saved → {RESULTS_DIR / 'signal_portfolio_search.csv'}")


if __name__ == "__main__":
    main()
