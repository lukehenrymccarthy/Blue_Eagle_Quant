"""
backtest/optimize_weights.py
-----------------------------
Find optimal factor weights for the five-factor model using constrained
optimisation over the historical backtest period.

Method
------
1. Pre-compute (once):
   - Cross-sectional z-scores for each factor at every rebalance date
   - Compound stock returns for each holding period
   After this, every objective evaluation is pure NumPy — ~1ms per trial.
2. scipy SLSQP with N random restarts maximising in-sample Sharpe.
   Constraints: each weight ∈ [W_MIN, W_MAX], weights sum to 1.0
3. Walk-forward validation:
   In-sample  : START  → SPLIT   (optimise)
   Out-of-sample: SPLIT → END    (honest evaluation)
4. Save optimal weights to data/results/optimal_weights.json

Usage
-----
    python backtest/optimize_weights.py
    python backtest/optimize_weights.py --restarts 100 --basket 50
"""

import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.five_factor_model import (
    load_crsp_wide, build_compustat_panels, build_sector_rs_panel,
    build_analyst_panel, build_macro_factor, build_all_factor_panels,
    _zscore, RISK_FREE_ANN, TC,
)

RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

START_DATE  = "2010-01-01"
END_DATE    = "2024-12-31"
SPLIT_DATE  = "2019-12-31"
BASKET      = 50
HOLD_MONTHS = 1
W_MIN       = 0.03
W_MAX       = 0.45
N_RESTARTS  = 50


# ── Pre-computation (runs once) ───────────────────────────────────────────────

def build_fast_records(
    ret_wide:     pd.DataFrame,
    panels:       dict[str, pd.DataFrame],
    macro_factor: pd.Series | None,
    basket:       int,
    hold_months:  int,
) -> tuple[list[dict], list[str]]:
    """
    Pre-compute everything needed for fast weight optimisation.

    Returns:
        records     — list of per-rebalance dicts (numpy arrays only)
        cs_names    — ordered list of cross-sectional factor names

    Each record:
        z_matrix    : (n_common, n_cs_factors)  cross-sectional z-scores
        period_rets : (n_common,)               compound return each stock
        macro_val   : float | None
        rdate       : pd.Timestamp
    """
    crsp_universe = set(ret_wide.columns)
    factor_names  = list(panels.keys())
    records       = []

    dates      = ret_wide.index
    n_dates    = len(dates)

    for i in range(0, n_dates - hold_months, hold_months):
        rdate      = dates[i]
        next_rdate = dates[i + hold_months]

        # ── Cross-sectional z-scores ───────────────────────────────────────
        scored = {}
        for fname in factor_names:
            panel = panels[fname]
            if rdate not in panel.index:
                continue
            row = panel.loc[rdate].dropna()
            row = row[row.index.isin(crsp_universe)]
            z   = _zscore(row)
            if len(z) >= basket:
                scored[fname] = z

        if len(scored) < 2:
            continue

        common = list(scored.values())[0].index
        for z in scored.values():
            common = common.intersection(z.index)
        if len(common) < basket:
            continue

        # ── Pre-compute compound returns for this holding period ───────────
        period_slice = ret_wide.loc[
            (ret_wide.index > rdate) & (ret_wide.index <= next_rdate),
            list(common),
        ]
        compound_rets = (1 + period_slice.fillna(0)).prod(axis=0) - 1
        period_rets   = compound_rets.reindex(common).fillna(0).values  # (n_common,)

        # ── Macro scalar ───────────────────────────────────────────────────
        macro_val = None
        if macro_factor is not None:
            v = macro_factor.asof(rdate)
            if not pd.isna(v):
                macro_val = float(v)

        records.append({
            "rdate":       rdate,
            "z_matrix":    np.column_stack([scored[fn].reindex(common).values
                                            for fn in scored]),
            "period_rets": period_rets,
            "macro_val":   macro_val,
            "cs_names":    list(scored.keys()),
        })

    # Canonical CS factor order = all factor_names that appear in ANY record
    seen = []
    for rec in records:
        for n in rec["cs_names"]:
            if n not in seen:
                seen.append(n)
    cs_names = seen
    return records, cs_names


# ── Fast backtest (pure NumPy) ────────────────────────────────────────────────

def fast_backtest(
    records:     list[dict],
    cs_weights:  np.ndarray,       # (n_cs,) — must match record["z_matrix"] cols
    macro_w:     float,
    cs_names:    list[str],
    basket:      int = BASKET,
) -> np.ndarray:
    """
    Run the backtest for given weights using pre-computed arrays.
    Returns array of holding-period returns (length = len(records)).
    """
    port_rets = np.full(len(records), np.nan)
    prev_top  = None

    for idx, rec in enumerate(records):
        z_mat = rec["z_matrix"]   # (n_common, n_cs)
        n_cs  = z_mat.shape[1]

        # Align cs_weights to this record's factor order
        rec_names = rec["cs_names"]
        w = np.array([cs_weights[cs_names.index(n)] if n in cs_names else 0.0
                      for n in rec_names])
        w = w / w.sum() if w.sum() > 0 else np.ones(n_cs) / n_cs

        composite = z_mat @ w   # (n_common,)

        # Blend macro
        mv = rec["macro_val"]
        if macro_w > 0 and mv is not None:
            composite = composite * (1 - macro_w) + mv * macro_w

        # Top-basket selection
        top_idx = np.argpartition(composite, -basket)[-basket:]

        # Transaction cost
        if prev_top is not None:
            curr_set = set(top_idx.tolist())
            prev_set = set(prev_top.tolist())
            turnover = len(curr_set.symmetric_difference(prev_set)) / (2 * basket)
        else:
            turnover = 1.0
        tc_drag = turnover * TC

        port_rets[idx] = rec["period_rets"][top_idx].mean() - tc_drag
        prev_top = top_idx

    return port_rets[~np.isnan(port_rets)]


def sharpe(rets: np.ndarray, hold_months: int = HOLD_MONTHS) -> float:
    if len(rets) < 6:
        return -99.0
    ppy = 12 / hold_months
    rf  = (1 + RISK_FREE_ANN) ** (hold_months / 12) - 1
    mu, sig = rets.mean(), rets.std()
    return float((mu - rf) / sig * np.sqrt(ppy)) if sig > 0 else -99.0


# ── Optimiser ─────────────────────────────────────────────────────────────────

def optimise(
    records_is: list[dict],
    cs_names:   list[str],
    has_macro:  bool,
    n_restarts: int,
    lam:        float = 0.5,   # L2 regularisation toward equal weights
) -> tuple[dict, float]:
    """
    Maximise in-sample Sharpe via SLSQP + random restarts.
    Returns (weights_dict, best_sharpe).
    """
    n_cs    = len(cs_names)
    n_total = n_cs + (1 if has_macro else 0)

    eq_w = 1.0 / n_total

    def neg_sharpe(x):
        cs_w   = x[:n_cs]
        m_w    = float(x[n_cs]) if has_macro else 0.0
        rets   = fast_backtest(records_is, cs_w, m_w, cs_names)
        sh     = sharpe(rets)
        # L2 penalty: shrinks toward equal weights to reduce overfitting
        penalty = lam * float(np.sum((x - eq_w) ** 2))
        return -sh + penalty

    bounds      = [(W_MIN, W_MAX)] * n_total
    constraints = {"type": "eq", "fun": lambda x: x.sum() - 1.0}

    best_val = np.inf
    best_x   = None
    rng      = np.random.default_rng(42)

    for t in range(n_restarts):
        x0 = rng.dirichlet(np.ones(n_total))
        x0 = np.clip(x0, W_MIN, W_MAX)
        x0 /= x0.sum()

        try:
            res = minimize(
                neg_sharpe, x0,
                method      = "SLSQP",
                bounds      = bounds,
                constraints = constraints,
                options     = {"ftol": 1e-9, "maxiter": 300},
            )
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_x   = res.x.copy()
        except Exception:
            pass

        if (t + 1) % 10 == 0:
            print(f"    restart {t+1:>3}/{n_restarts}  "
                  f"best in-sample Sharpe: {-best_val:.3f}")

    if best_x is None:
        best_x = np.full(n_total, 1.0 / n_total)

    weights_out = {cs_names[i]: round(float(best_x[i]), 6) for i in range(n_cs)}
    if has_macro:
        weights_out["macro_hy"] = round(float(best_x[n_cs]), 6)

    return weights_out, -best_val


# ── Reporting ─────────────────────────────────────────────────────────────────

def perf_stats(rets: np.ndarray, label: str, hold_months: int = HOLD_MONTHS) -> dict:
    ppy = 12 / hold_months
    rf  = (1 + RISK_FREE_ANN) ** (hold_months / 12) - 1
    ann = (1 + rets.mean()) ** ppy - 1
    vol = rets.std() * np.sqrt(ppy)
    sh  = (rets.mean() - rf) / rets.std() * np.sqrt(ppy) if rets.std() > 0 else np.nan
    nav = np.cumprod(1 + rets)
    dd  = np.min((nav - np.maximum.accumulate(nav)) / np.maximum.accumulate(nav))
    hit = np.mean(rets > 0)
    return {"label": label, "ann_return": round(ann*100,2), "ann_vol": round(vol*100,2),
            "sharpe": round(sh,3), "max_drawdown": round(dd*100,2),
            "hit_rate": round(hit*100,1), "n": len(rets)}


def print_table(rows: list[dict]):
    W = 92
    print("\n" + "═" * W)
    print("  RESULTS  —  Equal-weight vs Optimised")
    print("═" * W)
    print(f"  {'':30s}  {'Ann Ret':>8}  {'Ann Vol':>8}  {'Sharpe':>7}  "
          f"{'Max DD':>8}  {'Hit%':>6}  {'N':>5}")
    print("  " + "─" * (W - 2))
    for r in rows:
        print(f"  {r['label']:30s}  {r['ann_return']:>+7.1f}%  {r['ann_vol']:>7.1f}%  "
              f"{r['sharpe']:>7.3f}  {r['max_drawdown']:>+7.1f}%  "
              f"{r['hit_rate']:>5.1f}%  {r['n']:>5}")
    print("═" * W)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restarts", type=int, default=N_RESTARTS)
    parser.add_argument("--basket",   type=int, default=BASKET)
    parser.add_argument("--start",    default=START_DATE)
    parser.add_argument("--end",      default=END_DATE)
    parser.add_argument("--split",    default=SPLIT_DATE)
    args = parser.parse_args()

    print("\n" + "═" * 70)
    print("  FIVE-FACTOR WEIGHT OPTIMISATION")
    print(f"  In-sample : {args.start} – {args.split}")
    print(f"  OOS       : {args.split} – {args.end}")
    print(f"  Basket    : top-{args.basket}  |  Hold: {HOLD_MONTHS}M  "
          f"|  Restarts: {args.restarts}")
    print(f"  Bounds    : each weight ∈ [{W_MIN:.0%}, {W_MAX:.0%}],  sum = 100%")
    print("═" * 70)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n  Loading data...")
    ret_wide = load_crsp_wide().loc[args.start:args.end]

    roe_panel, sic_map, is_liquid_panel = pd.DataFrame(), pd.Series(dtype=str), pd.DataFrame()
    try:
        roe_panel, sic_map, is_liquid_panel = build_compustat_panels(ret_wide)
    except Exception as e:
        print(f"  [SKIP] Compustat: {e}")

    sector_rs = pd.DataFrame()
    try:
        sector_rs = build_sector_rs_panel(ret_wide, sic_map)
    except Exception as e:
        print(f"  [SKIP] Sector RS: {e}")

    analyst_rev = pd.DataFrame()
    try:
        analyst_rev = build_analyst_panel(ret_wide)
    except Exception as e:
        print(f"  [SKIP] Analyst: {e}")

    macro_factor = None
    try:
        macro_factor = build_macro_factor(ret_wide)
    except Exception as e:
        print(f"  [SKIP] Macro: {e}")

    panels       = build_all_factor_panels(ret_wide, roe_panel, sector_rs, analyst_rev)
    factor_names = list(panels.keys())
    has_macro    = macro_factor is not None
    n_factors    = len(factor_names) + (1 if has_macro else 0)

    print(f"  Active factors ({n_factors}): {', '.join(factor_names)}"
          + (" + macro_hy" if has_macro else ""))

    # ── Pre-compute z-scores + period returns ─────────────────────────────────
    print("\n  Pre-computing z-scores and period returns (runs once)...")
    all_records, cs_names = build_fast_records(
        ret_wide, panels, macro_factor, args.basket, HOLD_MONTHS
    )
    print(f"  {len(all_records)} rebalance periods with sufficient coverage")

    split_ts    = pd.Timestamp(args.split)
    records_is  = [r for r in all_records if r["rdate"] <= split_ts]
    records_oos = [r for r in all_records if r["rdate"] >  split_ts]
    print(f"  In-sample : {len(records_is)} periods  "
          f"| OOS: {len(records_oos)} periods")

    if not records_is:
        print("  [ERROR] No in-sample records.")
        return

    # ── Equal-weight baseline ─────────────────────────────────────────────────
    n_cs     = len(cs_names)
    eq_cs_w  = np.ones(n_cs) / n_factors
    eq_m_w   = (1.0 / n_factors) if has_macro else 0.0

    eq_rets_is  = fast_backtest(records_is,  eq_cs_w, eq_m_w, cs_names, args.basket)
    eq_rets_oos = fast_backtest(records_oos, eq_cs_w, eq_m_w, cs_names, args.basket)

    # ── Optimise ───────────────────────────────────────────────────────────────
    print(f"\n  Optimising weights ({args.restarts} random restarts)...")
    opt_weights, is_sharpe = optimise(records_is, cs_names, has_macro, args.restarts)

    opt_cs_w = np.array([opt_weights.get(n, 1.0/n_cs) for n in cs_names])
    opt_m_w  = opt_weights.get("macro_hy", 0.0)

    opt_rets_is  = fast_backtest(records_is,  opt_cs_w, opt_m_w, cs_names, args.basket)
    opt_rets_oos = fast_backtest(records_oos, opt_cs_w, opt_m_w, cs_names, args.basket)

    # ── Print optimal weights ─────────────────────────────────────────────────
    all_names = cs_names + (["macro_hy"] if has_macro else [])
    eq_w_val  = 1.0 / n_factors

    print("\n" + "═" * 70)
    print("  OPTIMAL WEIGHTS")
    print("═" * 70)
    print(f"  {'Factor':25s}  {'Equal':>8}  {'Optimal':>8}  {'Change':>8}  Bar")
    print("  " + "─" * 65)
    for fname in all_names:
        opt_w = opt_weights.get(fname, 0.0)
        delta = opt_w - eq_w_val
        bar   = "█" * int(opt_w * 50)
        print(f"  {fname:25s}  {eq_w_val:>7.1%}  {opt_w:>7.1%}  {delta:>+7.1%}  {bar}")
    print("═" * 70)

    # ── Performance comparison ────────────────────────────────────────────────
    rows = [
        perf_stats(eq_rets_is,   "Equal-weight   [IN-SAMPLE]    "),
        perf_stats(opt_rets_is,  "Optimised      [IN-SAMPLE]    "),
        perf_stats(eq_rets_oos,  "Equal-weight   [OUT-OF-SAMPLE]"),
        perf_stats(opt_rets_oos, "Optimised      [OUT-OF-SAMPLE]"),
    ]
    print_table(rows)

    oos_lift = rows[3]["sharpe"] - rows[2]["sharpe"]
    print(f"\n  OOS Sharpe lift: {oos_lift:+.3f}", end="  ")
    if oos_lift > 0.05:
        print("✓ Weights generalise well out-of-sample.")
    elif oos_lift > -0.05:
        print("~ Marginal — weights are close to equal.")
    else:
        print("✗ Possible overfitting — consider more restarts.")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "weights":           opt_weights,
        "in_sample_sharpe":  round(is_sharpe, 4),
        "oos_sharpe":        round(rows[3]["sharpe"], 4),
        "equal_oos_sharpe":  round(rows[2]["sharpe"], 4),
        "basket":            args.basket,
        "hold_months":       HOLD_MONTHS,
        "split_date":        args.split,
        "constraints":       {"min": W_MIN, "max": W_MAX, "sum": 1.0},
    }
    out_path = RESULTS_DIR / "optimal_weights.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
