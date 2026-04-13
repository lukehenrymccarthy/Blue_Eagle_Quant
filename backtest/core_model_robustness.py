"""
backtest/core_model_robustness.py
---------------------------------
Robustness workflow for the canonical core model.

Goals:
  - Use exact basket counts rather than basket percentages
  - Compare the canonical 3-factor model against a simpler 2-factor variant
  - Split 2025 into pre-shock and post-shock windows
  - Inspect sector exposure concentration in the live holdings

Usage:
    python backtest/core_model_robustness.py
    python backtest/core_model_robustness.py --basket-sizes 5,10,15,20,25,30,40,50,75,100 --hold-months-list 1,2,3,6
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sectorscope.core_strategy import (
    FULL_START,
    IS_END,
    OOS_END,
    OOS_START,
    build_spy_benchmark,
    load_compustat,
    load_ibes,
    load_returns,
    run_backtest_exact,
)
from sectorscope.factors import build_factor_panels, build_liquidity_screen
from sectorscope.metrics import compute_metrics
from sectorscope.utils import sic_to_etf as _sic_to_etf

RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

THREE_FACTOR_WEIGHTS = {"sue": 0.80, "mom_52wk_high": 0.05, "sector_rs_1m": 0.15}
TWO_FACTOR_WEIGHTS = {"sue": 0.90, "mom_52wk_high": 0.10}


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def build_core_inputs(oos_end: str = OOS_END):
    ret_wide = load_returns(oos_end=oos_end)
    fund = load_compustat()
    ibes = load_ibes()
    sic_col = "siccd" if "siccd" in fund.columns else "sic"
    sic_map = pd.Series({
        int(p): _sic_to_etf(s)
        for p, s in fund.sort_values("datadate").groupby("permno")[sic_col].last().items()
    }).dropna()
    panels = build_factor_panels(
        ret_wide=ret_wide,
        fund=fund,
        ibes=ibes,
        sic_map=sic_map,
        include_technical=True,
        sector_neutral_momentum=True,
    )
    panels = {k: v for k, v in panels.items() if k in {"sue", "mom_52wk_high", "sector_rs_1m"}}
    is_liq = build_liquidity_screen(fund, ret_wide, universe_size=1000)
    return ret_wide, panels, is_liq, sic_map

def summarize_window(series: pd.Series, hold_months: int, min_periods: int) -> dict:
    return compute_metrics(series, hold_months=hold_months, min_periods=min_periods) or {}


def summarize_sector_exposure(log_df: pd.DataFrame, sic_map: pd.Series) -> str:
    if log_df.empty or "holdings" not in log_df.columns:
        return ""
    counts: dict[str, int] = {}
    total = 0
    for holdings in log_df["holdings"]:
        if not holdings:
            continue
        permnos = [int(x) for x in holdings.split(",") if x]
        sectors = sic_map.reindex(permnos).fillna("Unknown")
        for sec in sectors:
            counts[str(sec)] = counts.get(str(sec), 0) + 1
            total += 1
    if total == 0:
        return ""
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
    return "; ".join(f"{sec}:{cnt/total:.1%}" for sec, cnt in ranked)


def run_robustness_grid(
    basket_sizes: list[int],
    hold_months_list: list[int],
    shock_split: str,
    oos_end: str,
) -> pd.DataFrame:
    ret_wide, panels, is_liq, sic_map = build_core_inputs(oos_end=oos_end)
    is_idx = ret_wide.index[ret_wide.index <= IS_END]
    oos_idx = ret_wide.index[(ret_wide.index >= OOS_START) & (ret_wide.index <= oos_end)]
    pre_idx = ret_wide.index[(ret_wide.index >= OOS_START) & (ret_wide.index < shock_split)]
    post_idx = ret_wide.index[(ret_wide.index >= shock_split) & (ret_wide.index <= oos_end)]

    model_variants = {
        "two_factor": {k: v for k, v in panels.items() if k in TWO_FACTOR_WEIGHTS},
        "three_factor": {k: v for k, v in panels.items() if k in THREE_FACTOR_WEIGHTS},
    }
    weights_map = {
        "two_factor": TWO_FACTOR_WEIGHTS,
        "three_factor": THREE_FACTOR_WEIGHTS,
    }

    rows = []
    for model_name, model_panels in model_variants.items():
        for basket_size in basket_sizes:
            for hold_months in hold_months_list:
                curve_is, log_is = run_backtest_exact(
                    ret_wide.reindex(is_idx),
                    {k: v.reindex(is_idx) for k, v in model_panels.items()},
                    is_liq.reindex(is_idx),
                    basket_size=basket_size,
                    hold_months=hold_months,
                    weights=weights_map[model_name],
                )
                curve_oos, log_oos = run_backtest_exact(
                    ret_wide.reindex(oos_idx),
                    {k: v.reindex(oos_idx) for k, v in model_panels.items()},
                    is_liq.reindex(oos_idx),
                    basket_size=basket_size,
                    hold_months=hold_months,
                    weights=weights_map[model_name],
                )
                curve_pre, _ = run_backtest_exact(
                    ret_wide.reindex(pre_idx),
                    {k: v.reindex(pre_idx) for k, v in model_panels.items()},
                    is_liq.reindex(pre_idx),
                    basket_size=basket_size,
                    hold_months=hold_months,
                    weights=weights_map[model_name],
                )
                curve_post, log_post = run_backtest_exact(
                    ret_wide.reindex(post_idx),
                    {k: v.reindex(post_idx) for k, v in model_panels.items()},
                    is_liq.reindex(post_idx),
                    basket_size=basket_size,
                    hold_months=hold_months,
                    weights=weights_map[model_name],
                )

                m_is = summarize_window(curve_is, hold_months, 4)
                m_oos = summarize_window(curve_oos, hold_months, 2)
                m_pre = summarize_window(curve_pre, hold_months, 1)
                m_post = summarize_window(curve_post, hold_months, 1)
                spy_is = summarize_window(build_spy_benchmark(ret_wide.reindex(is_idx), hold_months), hold_months, 4)
                spy_oos = summarize_window(build_spy_benchmark(ret_wide.reindex(oos_idx), hold_months), hold_months, 2)

                if not m_is:
                    continue

                rows.append({
                    "model": model_name,
                    "basket_size": basket_size,
                    "hold_months": hold_months,
                    "is_ann_return": m_is.get("ann_return"),
                    "is_sharpe": m_is.get("sharpe"),
                    "is_sortino": m_is.get("sortino"),
                    "is_calmar": m_is.get("calmar"),
                    "spy_is_sharpe": spy_is.get("sharpe"),
                    "oos_ann_return": m_oos.get("ann_return"),
                    "oos_sharpe": m_oos.get("sharpe"),
                    "oos_sortino": m_oos.get("sortino"),
                    "oos_calmar": m_oos.get("calmar"),
                    "spy_oos_sharpe": spy_oos.get("sharpe"),
                    "pre_shock_ann_return": m_pre.get("ann_return"),
                    "pre_shock_sharpe": m_pre.get("sharpe"),
                    "post_shock_ann_return": m_post.get("ann_return"),
                    "post_shock_sharpe": m_post.get("sharpe"),
                    "post_shock_sector_exposure": summarize_sector_exposure(log_post, sic_map),
                    "full_oos_sector_exposure": summarize_sector_exposure(log_oos, sic_map),
                })
    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No robustness results produced.")
        return
    print("\n" + "═" * 160)
    print("  CORE MODEL ROBUSTNESS")
    print("  Ranked by post-shock Sharpe, then OOS Sharpe, then IS Sharpe")
    print("═" * 160)
    ranked = df.sort_values(
        ["post_shock_sharpe", "oos_sharpe", "is_sharpe"],
        ascending=False,
    ).reset_index(drop=True)
    ranked.insert(0, "rank", ranked.index + 1)
    cols = [
        "rank", "model", "basket_size", "hold_months",
        "is_sharpe", "spy_is_sharpe",
        "oos_sharpe", "spy_oos_sharpe",
        "pre_shock_sharpe", "post_shock_sharpe",
        "post_shock_sector_exposure",
    ]
    print(ranked[cols].to_string(index=False))
    print("═" * 160)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basket-sizes", default="5,10,15,20,25,30,40,50,75,100",
                        help="Comma-separated exact basket sizes.")
    parser.add_argument("--hold-months-list", default="1,2,3,6",
                        help="Comma-separated hold lengths.")
    parser.add_argument("--shock-split", default="2025-04-01",
                        help="Date used to split 2025 into pre-shock and post-shock windows.")
    parser.add_argument("--oos-end", default=OOS_END,
                        help="OOS end date.")
    args = parser.parse_args()

    basket_sizes = parse_int_list(args.basket_sizes)
    hold_months_list = parse_int_list(args.hold_months_list)

    print("\n" + "═" * 90)
    print("  CORE MODEL TARIFF-SHOCK ROBUSTNESS")
    print("  Core model remains the three-factor spec: SUE | residual 52wk-high | sector RS")
    print(f"  IS         : {FULL_START} to {IS_END}")
    print(f"  OOS        : {OOS_START} to {args.oos_end}")
    print(f"  Shock split: {args.shock_split}")
    print(f"  Baskets    : {basket_sizes}")
    print(f"  Holds      : {hold_months_list}")
    print("═" * 90)

    df = run_robustness_grid(
        basket_sizes=basket_sizes,
        hold_months_list=hold_months_list,
        shock_split=args.shock_split,
        oos_end=args.oos_end,
    )
    out_path = RESULTS_DIR / "core_model_robustness.csv"
    df.to_csv(out_path, index=False)
    print_summary(df)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
