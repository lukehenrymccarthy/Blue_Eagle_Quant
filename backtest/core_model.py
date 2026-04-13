"""
Locked live research configuration for the core equity model.

Model:
  - SUE
  - residual 52-week-high momentum
  - sector relative strength

Portfolio construction:
  - 10 stocks
  - 2-month hold
  - top 1000 market-cap universe
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sectorscope.core_strategy import (
    FULL_START,
    IS_END,
    OOS_START,
    UNIVERSE_SIZE,
    active_oos_end,
    build_spy_benchmark,
    load_compustat,
    load_ibes,
    load_returns,
    run_backtest_exact,
)
from sectorscope.factors import build_factor_panels, build_liquidity_screen
from sectorscope.metrics import compute_metrics
from sectorscope.modeling import FACTOR_WEIGHTS_OPT
from sectorscope.utils import sic_to_etf as _sic_to_etf

RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASKET_SIZE = 10
HOLD_MONTHS = 2
ACTIVE_FACTORS = ("sue", "mom_52wk_high", "sector_rs_1m")


def build_core_inputs(oos_end: str | None = None):
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
    panels = {k: v for k, v in panels.items() if k in ACTIVE_FACTORS}
    is_liq = build_liquidity_screen(fund, ret_wide, universe_size=UNIVERSE_SIZE)
    return ret_wide, panels, is_liq


def print_block(title: str, model_metrics: dict, spy_metrics: dict) -> None:
    print("\n" + "═" * 90)
    print(f"  {title}")
    print("═" * 90)
    if not model_metrics:
        print("  No usable results.")
        return
    print(
        f"  Model : ann_ret={model_metrics.get('ann_return', float('nan')):+.2f}%  "
        f"sharpe={model_metrics.get('sharpe', float('nan')):.3f}  "
        f"sortino={model_metrics.get('sortino', float('nan')):.3f}  "
        f"calmar={model_metrics.get('calmar', float('nan')):.3f}  "
        f"max_dd={model_metrics.get('max_drawdown', float('nan')):+.2f}%  "
        f"hit={model_metrics.get('hit_rate', float('nan')):.1f}%  "
        f"n={model_metrics.get('n_periods', 0)}"
    )
    if spy_metrics:
        print(
            f"  SPY   : ann_ret={spy_metrics.get('ann_return', float('nan')):+.2f}%  "
            f"sharpe={spy_metrics.get('sharpe', float('nan')):.3f}  "
            f"sortino={spy_metrics.get('sortino', float('nan')):.3f}  "
            f"calmar={spy_metrics.get('calmar', float('nan')):.3f}  "
            f"max_dd={spy_metrics.get('max_drawdown', float('nan')):+.2f}%  "
            f"hit={spy_metrics.get('hit_rate', float('nan')):.1f}%  "
            f"n={spy_metrics.get('n_periods', 0)}"
        )
    print("═" * 90)


def format_block(title: str, model_metrics: dict, spy_metrics: dict) -> str:
    lines = []
    lines.append(title)
    if not model_metrics:
        lines.append("No usable results.")
        return "\n".join(lines)
    lines.append(
        f"Model : ann_ret={model_metrics.get('ann_return', float('nan')):+.2f}%  "
        f"sharpe={model_metrics.get('sharpe', float('nan')):.3f}  "
        f"sortino={model_metrics.get('sortino', float('nan')):.3f}  "
        f"calmar={model_metrics.get('calmar', float('nan')):.3f}  "
        f"max_dd={model_metrics.get('max_drawdown', float('nan')):+.2f}%  "
        f"hit={model_metrics.get('hit_rate', float('nan')):.1f}%  "
        f"n={model_metrics.get('n_periods', 0)}"
    )
    if spy_metrics:
        lines.append(
            f"SPY   : ann_ret={spy_metrics.get('ann_return', float('nan')):+.2f}%  "
            f"sharpe={spy_metrics.get('sharpe', float('nan')):.3f}  "
            f"sortino={spy_metrics.get('sortino', float('nan')):.3f}  "
            f"calmar={spy_metrics.get('calmar', float('nan')):.3f}  "
            f"max_dd={spy_metrics.get('max_drawdown', float('nan')):+.2f}%  "
            f"hit={spy_metrics.get('hit_rate', float('nan')):.1f}%  "
            f"n={spy_metrics.get('n_periods', 0)}"
        )
    return "\n".join(lines)


def overlap_pair(left: pd.Series, right: pd.Series) -> tuple[pd.Series, pd.Series]:
    pair = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    if pair.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    return pair["left"], pair["right"]


def main():
    oos_end = active_oos_end()
    ret_wide, panels, is_liq = build_core_inputs(oos_end=oos_end)
    is_idx = ret_wide.index[ret_wide.index <= IS_END]
    oos_idx = ret_wide.index[(ret_wide.index >= OOS_START) & (ret_wide.index <= oos_end)]

    curve_is, _ = run_backtest_exact(
        ret_wide.reindex(is_idx),
        {k: v.reindex(is_idx) for k, v in panels.items()},
        is_liq.reindex(is_idx),
        basket_size=BASKET_SIZE,
        hold_months=HOLD_MONTHS,
        weights=FACTOR_WEIGHTS_OPT,
        score_weighted=False,
    )
    curve_oos, _ = run_backtest_exact(
        ret_wide.reindex(oos_idx),
        {k: v.reindex(oos_idx) for k, v in panels.items()},
        is_liq.reindex(oos_idx),
        basket_size=BASKET_SIZE,
        hold_months=HOLD_MONTHS,
        weights=FACTOR_WEIGHTS_OPT,
        score_weighted=False,
    )

    spy_is = build_spy_benchmark(ret_wide.reindex(is_idx), HOLD_MONTHS)
    spy_oos = build_spy_benchmark(ret_wide.reindex(oos_idx), HOLD_MONTHS)
    curve_is_aligned, spy_is_aligned = overlap_pair(curve_is, spy_is)
    curve_oos_aligned, spy_oos_aligned = overlap_pair(curve_oos, spy_oos)

    m_is = compute_metrics(curve_is_aligned, hold_months=HOLD_MONTHS, min_periods=4) or {}
    m_oos = compute_metrics(curve_oos_aligned, hold_months=HOLD_MONTHS, min_periods=2) or {}
    m_spy_is = compute_metrics(spy_is_aligned, hold_months=HOLD_MONTHS, min_periods=4) or {}
    m_spy_oos = compute_metrics(spy_oos_aligned, hold_months=HOLD_MONTHS, min_periods=2) or {}

    print("\n" + "═" * 90)
    print("  CORE MODEL")
    print(f"  Factors     : {', '.join(ACTIVE_FACTORS)}")
    print(f"  Basket size : {BASKET_SIZE}")
    print(f"  Hold months : {HOLD_MONTHS}")
    print(f"  Universe    : top {UNIVERSE_SIZE} by market cap")
    print(f"  IS window   : {FULL_START} to {IS_END}")
    print(f"  OOS window  : {OOS_START} to {oos_end}")
    print("═" * 90)

    print_block("IN SAMPLE", m_is, m_spy_is)
    print_block("OUT OF SAMPLE", m_oos, m_spy_oos)

    curve_is.to_csv(RESULTS_DIR / "core_model_is_curve.csv", header=["period_ret"])
    curve_oos.to_csv(RESULTS_DIR / "core_model_oos_curve.csv", header=["period_ret"])
    report = "\n".join([
        "CORE MODEL",
        f"Factors     : {', '.join(ACTIVE_FACTORS)}",
        f"Basket size : {BASKET_SIZE}",
        f"Hold months : {HOLD_MONTHS}",
        f"Universe    : top {UNIVERSE_SIZE} by market cap",
        f"IS window   : {FULL_START} to {IS_END}",
        f"OOS window  : {OOS_START} to {oos_end}",
        "",
        format_block("IN SAMPLE", m_is, m_spy_is),
        "",
        format_block("OUT OF SAMPLE", m_oos, m_spy_oos),
    ])
    (RESULTS_DIR / "core_model_report.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
