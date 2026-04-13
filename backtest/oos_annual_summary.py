"""
backtest/oos_annual_summary.py
------------------------------
Produces data/results/oos_annual_summary.csv

Computes portfolio metrics for the full April 2025 – March 2026 OOS year.
The locked model is scored monthly (hold = 1 month) over that window so all
12 calendar months are covered.  This is a presentation layer only — the
production strategy holds 2 months; monthly scoring here is used purely to
get uninterrupted monthly return coverage for metric computation.

Run:
    python backtest/oos_annual_summary.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.core_model import ACTIVE_FACTORS, BASKET_SIZE, build_core_inputs
from sectorscope.core_strategy import IS_END, OOS_START, build_spy_benchmark, run_backtest_exact
from sectorscope.metrics import compute_metrics
from sectorscope.modeling import FACTOR_WEIGHTS_OPT

RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_START = "2025-04-01"
WINDOW_END   = "2026-03-31"
HOLD_MONTHS  = 1          # monthly scoring for full-year coverage


def _spy_monthly(ret_wide: pd.DataFrame) -> pd.Series:
    spy = pd.read_parquet("data/prices/SPY.parquet")
    dates = pd.to_datetime(spy["date"] if "date" in spy.columns else spy.index)
    dates = pd.DatetimeIndex(dates).tz_localize(None)
    s = pd.Series(spy["close"].values, index=dates).sort_index()
    s = s.resample("ME").last()
    s.index = pd.to_datetime(s.index) + pd.offsets.MonthEnd(0)
    return s.pct_change().reindex(ret_wide.index)


def _format(v, pct=False, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if pct:
        return round(float(v), decimals)
    return round(float(v), decimals)


def main():
    ret_wide, panels, is_liq = build_core_inputs(oos_end=WINDOW_END)

    oos_idx = ret_wide.index[
        (ret_wide.index >= WINDOW_START) & (ret_wide.index <= WINDOW_END)
    ]

    curve, _ = run_backtest_exact(
        ret_wide.reindex(oos_idx),
        {k: v.reindex(oos_idx) for k, v in panels.items()},
        is_liq.reindex(oos_idx),
        basket_size=BASKET_SIZE,
        hold_months=HOLD_MONTHS,
        weights=FACTOR_WEIGHTS_OPT,
        score_weighted=False,
    )

    spy_curve = _spy_monthly(ret_wide.reindex(oos_idx))
    spy_bench = build_spy_benchmark(ret_wide.reindex(oos_idx), HOLD_MONTHS)

    # Align model and SPY on overlapping non-NaN dates
    both = pd.concat([curve.rename("model"), spy_bench.rename("spy")], axis=1).dropna()
    model_r = both["model"]
    spy_r   = both["spy"]

    model_m = compute_metrics(model_r, hold_months=HOLD_MONTHS, min_periods=3) or {}
    spy_m   = compute_metrics(spy_r,   hold_months=HOLD_MONTHS, min_periods=3) or {}

    total_model = round(((1 + model_r).prod() - 1) * 100, 2)
    total_spy   = round(((1 + spy_r).prod()   - 1) * 100, 2)

    rows = [
        ("period",          "Apr 2025 – Mar 2026",  "Apr 2025 – Mar 2026"),
        ("n_months",        int(model_m.get("n_months", 0)), int(spy_m.get("n_months", 0))),
        ("total_return_pct",total_model,             total_spy),
        ("ann_return_pct",  _format(model_m.get("ann_return")),  _format(spy_m.get("ann_return"))),
        ("ann_vol_pct",     _format(model_m.get("ann_vol")),     _format(spy_m.get("ann_vol"))),
        ("sharpe",          _format(model_m.get("sharpe"), decimals=3), _format(spy_m.get("sharpe"), decimals=3)),
        ("sortino",         _format(model_m.get("sortino"), decimals=3), _format(spy_m.get("sortino"), decimals=3)),
        ("calmar",          _format(model_m.get("calmar"), decimals=3),  _format(spy_m.get("calmar"), decimals=3)),
        ("max_drawdown_pct",_format(model_m.get("max_drawdown")), _format(spy_m.get("max_drawdown"))),
        ("hit_rate_pct",    _format(model_m.get("hit_rate")),    _format(spy_m.get("hit_rate"))),
        ("best_month_pct",  round(float(model_r.max()) * 100, 2), round(float(spy_r.max()) * 100, 2)),
        ("worst_month_pct", round(float(model_r.min()) * 100, 2), round(float(spy_r.min()) * 100, 2)),
        ("t_stat",          _format(model_m.get("t_stat"), decimals=2), _format(spy_m.get("t_stat"), decimals=2)),
    ]

    out = pd.DataFrame(rows, columns=["metric", "model", "spy"])
    out.to_csv(RESULTS_DIR / "oos_annual_summary.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nSaved → {RESULTS_DIR / 'oos_annual_summary.csv'}")


if __name__ == "__main__":
    main()
