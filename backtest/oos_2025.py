"""
backtest/oos_2025.py
--------------------
Train / Test split evaluation:
  - In-sample  (IS)  : 2010-01-01 → 2024-12-31
  - Out-of-sample (OOS): 2025-01-01 → 2025-12-31

Factor panels and weights are fit/determined on IS data only.
OOS results reflect genuine out-of-sample performance.

Usage:
    python backtest/oos_2025.py
    python backtest/oos_2025.py --no-macro
"""

import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.five_factor_model import (
    load_crsp_wide,
    build_compustat_panels,
    build_sector_rs_panel,
    build_analyst_panel,
    build_macro_factor,
    build_hy_tilt_panel,
    build_all_factor_panels,
    load_optimal_weights,
    run_one_config,
    BASKET_SIZES,
    HOLD_MONTHS,
    FACTOR_WEIGHTS,
)
from sectorscope.metrics import compute_metrics

RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IS_START  = "2010-01-01"
IS_END    = "2024-12-31"
OOS_START = "2025-01-01"
OOS_END   = "2025-12-31"
# Warmup needed: 12m momentum + 36m credit beta → load from 2007
LOAD_START = "2007-01-01"


def slice_panels(panels: dict, ret_slice: pd.DataFrame) -> dict:
    """Reindex all factor panels to match ret_slice's index."""
    out = {}
    for name, panel in panels.items():
        out[name] = panel.reindex(ret_slice.index, method="ffill")
    return out


def print_comparison_table(is_df: pd.DataFrame, oos_df: pd.DataFrame,
                           active_factors: list[str]):
    W = 110
    print("\n" + "═" * W)
    print("  FIVE-FACTOR MODEL  —  IN-SAMPLE vs OUT-OF-SAMPLE")
    print(f"  IS  Period : {IS_START} – {IS_END}   |   "
          f"OOS Period : {OOS_START} – {OOS_END}")
    print(f"  Factors    : {', '.join(active_factors)}")
    print(f"  Cost       : 10 bps/side on turnover   |   Equal-weight portfolio")
    print("═" * W)

    for hold in HOLD_MONTHS:
        is_sub  = is_df[is_df["hold_months"]  == hold].sort_values("basket_size")
        oos_sub = oos_df[oos_df["hold_months"] == hold].sort_values("basket_size")
        if is_sub.empty: continue

        print(f"\n  Holding: {hold} Month{'s' if hold > 1 else ''}")
        hdr = (f"  {'Basket':>7}  │"
               f"  {'IS Ret':>7}  {'IS Sharpe':>9}  {'IS MaxDD':>8}  {'IS Hit%':>7}"
               f"  ║"
               f"  {'OOS Ret':>7}  {'OOS Sharpe':>10}  {'OOS MaxDD':>9}  {'OOS Hit%':>8}  {'OOS N':>6}")
        sep = "  " + "─" * 7 + "──┼" + "─" * 46 + "──╫" + "─" * 52
        print(sep)
        print(hdr)
        print(sep)

        for _, irow in is_sub.iterrows():
            basket = int(irow["basket_size"])
            orow_df = oos_sub[oos_sub["basket_size"] == basket]
            if orow_df.empty:
                oos_vals = ("N/A", "N/A", "N/A", "N/A", "N/A")
            else:
                orow = orow_df.iloc[0]
                oos_vals = (
                    f"{orow['ann_return']:>+7.1f}%",
                    f"{orow['sharpe']:>10.3f}",
                    f"{orow['max_drawdown']:>+9.1f}%",
                    f"{orow['hit_rate']:>8.1f}%",
                    f"{int(orow['n_periods']):>6}",
                )
            print(f"  {'Top ' + str(basket):>7}  │"
                  f"  {irow['ann_return']:>+6.1f}%"
                  f"  {irow['sharpe']:>9.3f}"
                  f"  {irow['max_drawdown']:>+8.1f}%"
                  f"  {irow['hit_rate']:>7.1f}%"
                  f"  ║"
                  f"  {oos_vals[0]:>7}  {oos_vals[1]:>10}  {oos_vals[2]:>9}"
                  f"  {oos_vals[3]:>8}  {oos_vals[4]:>6}")
    print("\n" + "═" * W)


def main():
    parser = argparse.ArgumentParser(description="IS vs OOS 2025 backtest")
    parser.add_argument("--no-macro", action="store_true")
    args = parser.parse_args()
    use_macro = not args.no_macro

    print("\n" + "═" * 70)
    print("  FIVE-FACTOR MODEL  —  Loading data (2007–2025 for warmup)")
    print("═" * 70)

    # ── 1. Load full history (warmup included) ────────────────────────────────
    ret_full = load_crsp_wide().loc[LOAD_START:]
    print(f"  CRSP : {ret_full.shape[1]:,} stocks | "
          f"{ret_full.index[0].date()} – {ret_full.index[-1].date()}")

    fund_panel = pd.DataFrame()
    sic_map    = pd.Series(dtype=str)
    liq_panel  = pd.DataFrame()
    try:
        print("\n  Loading Compustat...")
        fund_panel, sic_map, liq_panel = build_compustat_panels(ret_full)
        print(f"  Fund Quality : {fund_panel.shape[1]:,} stocks")
    except Exception as e:
        print(f"  [SKIP] Compustat: {e}")

    sector_rs = pd.DataFrame()
    try:
        sector_rs = build_sector_rs_panel(ret_full, sic_map)
    except Exception as e:
        print(f"  [SKIP] Sector RS: {e}")

    analyst_rev = pd.DataFrame()
    try:
        print("\n  Loading IBES analyst revisions...")
        analyst_rev = build_analyst_panel(ret_full)
    except Exception as e:
        print(f"  [SKIP] Analyst: {e}")

    macro_factor = None
    if use_macro:
        try:
            print("\n  Loading FRED macro factor...")
            macro_factor = build_macro_factor(ret_full)
        except Exception as e:
            print(f"  [SKIP] Macro: {e}")

    hy_tilt = pd.DataFrame()
    try:
        print("\n  Building HY spread tilt panel...")
        hy_tilt = build_hy_tilt_panel(ret_full, sic_map)
        if not hy_tilt.empty:
            print(f"  HY Tilt      : {hy_tilt.shape[1]:,} stocks × {hy_tilt.shape[0]} months")
    except Exception as e:
        print(f"  [SKIP] HY tilt: {e}")

    print("\n" + "═" * 70)
    print("  Building factor panels (full history)")
    print("═" * 70 + "\n")
    panels_full = build_all_factor_panels(
        ret_full, fund_panel, sector_rs, analyst_rev, hy_tilt)
    active_factors = list(panels_full.keys())

    opt_weights = load_optimal_weights()
    if opt_weights:
        print(f"\n  Weights (from IS optimisation):")
        for fn, w in opt_weights.items():
            print(f"    {fn:25s} {w:.1%}")
    else:
        print("\n  No optimal weights — using defaults.")

    # ── 2. Slice to IS and OOS windows ───────────────────────────────────────
    ret_is  = ret_full.loc[IS_START:IS_END]
    # Start OOS from the last IS date so the first rebalance (IS_END) captures
    # January 2025 returns — giving a full 12 periods across 2025.
    ret_oos = ret_full.loc[IS_END:OOS_END]

    panels_is  = slice_panels(panels_full, ret_is)
    panels_oos = slice_panels(panels_full, ret_oos)

    liq_is  = liq_panel.reindex(ret_is.index,  method="ffill") if not liq_panel.empty else pd.DataFrame()
    liq_oos = liq_panel.reindex(ret_oos.index, method="ffill") if not liq_panel.empty else pd.DataFrame()

    # ── 3. Run IS backtest ────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print(f"  IN-SAMPLE backtest  ({IS_START} – {IS_END})")
    print("═" * 70 + "\n")

    is_results, is_curves = [], {}
    for hold in HOLD_MONTHS:
        for basket in BASKET_SIZES:
            label = f"top{basket}_hold{hold}m"
            print(f"  IS  {label:<22}", end=" ... ", flush=True)
            try:
                curve, monthly = run_one_config(
                    ret_wide        = ret_is,
                    panels          = panels_is,
                    macro_factor    = macro_factor,
                    is_liquid_panel = liq_is,
                    basket          = basket,
                    hold_months     = hold,
                    weights         = opt_weights,
                )
                m = compute_metrics(curve, hold_months=hold, monthly_curve=monthly)
                if not m:
                    print("skipped (too few periods)")
                    continue
                print(f"Sharpe={m['sharpe']:>6.3f}  Ann={m['ann_return']:>+6.1f}%")
                is_results.append({"basket_size": basket, "hold_months": hold, **m})
                is_curves[label] = curve
            except Exception as e:
                print(f"ERROR — {e}")

    # ── 4. Run OOS backtest ───────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print(f"  OUT-OF-SAMPLE backtest  ({OOS_START} – {OOS_END})")
    print("═" * 70 + "\n")

    oos_results, oos_curves = [], {}
    for hold in HOLD_MONTHS:
        for basket in BASKET_SIZES:
            label = f"top{basket}_hold{hold}m"
            print(f"  OOS {label:<22}", end=" ... ", flush=True)
            try:
                curve, monthly = run_one_config(
                    ret_wide        = ret_oos,
                    panels          = panels_oos,
                    macro_factor    = macro_factor,
                    is_liquid_panel = liq_oos,
                    basket          = basket,
                    hold_months     = hold,
                    weights         = opt_weights,
                )
                m = compute_metrics(curve, hold_months=hold, min_periods=1, monthly_curve=monthly)
                if not m:
                    print("skipped (no data)")
                    continue
                print(f"Sharpe={m['sharpe']:>6.3f}  Ann={m['ann_return']:>+6.1f}%  N={m['n_periods']}")
                oos_results.append({"basket_size": basket, "hold_months": hold, **m})
                oos_curves[label] = curve
            except Exception as e:
                print(f"ERROR — {e}")

    if not is_results or not oos_results:
        print("\n  [ERROR] Insufficient results.")
        return

    is_df  = pd.DataFrame(is_results)
    oos_df = pd.DataFrame(oos_results)

    print_comparison_table(is_df, oos_df, active_factors)

    # ── 5. Save results ───────────────────────────────────────────────────────
    is_df.to_csv(RESULTS_DIR  / "oos_is_summary.csv",  index=False)
    oos_df.to_csv(RESULTS_DIR / "oos_2025_summary.csv", index=False)
    pd.DataFrame(oos_curves).to_parquet(RESULTS_DIR / "oos_2025_curves.parquet")

    print(f"\n  Saved → {RESULTS_DIR}/oos_is_summary.csv")
    print(f"  Saved → {RESULTS_DIR}/oos_2025_summary.csv")
    print(f"  Saved → {RESULTS_DIR}/oos_2025_curves.parquet")


if __name__ == "__main__":
    main()
