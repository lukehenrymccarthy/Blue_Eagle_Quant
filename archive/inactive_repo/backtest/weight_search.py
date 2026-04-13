"""
backtest/weight_search.py
─────────────────────────
Fast grid search over factor weights for the 3-factor model
(SUE, 52wk-high, sector_rs_1m).

Pre-computes cross-sectional z-scores once, then sweeps all weight
combinations via matrix ops — no repeated panel iteration.

Usage:
    python backtest/weight_search.py
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.updated_model import (
    load_returns, load_compustat, load_ibes,
    build_52wk_high, build_sue, build_sector_rs_1m,
    build_liquidity_screen,
    FULL_START, IS_END, HOLD_MONTHS, UNIVERSE_SIZE,
)
from sectorscope.utils import zscore as _zscore, sic_to_etf as _sic_to_etf
from sectorscope.metrics import compute_metrics

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading returns...")
ret_wide = load_returns()
print(f"  {ret_wide.shape[1]:,} stocks")

print("Loading Compustat...")
fund = load_compustat()
sic_map = pd.Series({
    int(p): _sic_to_etf(s)
    for p, s in fund.sort_values("datadate")
                    .groupby("permno")
                    [("siccd" if "siccd" in fund.columns else "sic")]
                    .last().items()
}).dropna()

print("Loading IBES...")
ibes = load_ibes()

# ── Build panels ─────────────────────────────────────────────────────────────
print("Building factor panels (IS only)...")
is_idx = ret_wide.index[ret_wide.index <= IS_END]

p_sue    = build_sue(fund, ret_wide).reindex(is_idx)
p_mom    = build_52wk_high(ret_wide).reindex(is_idx)
p_sector = build_sector_rs_1m(ret_wide, sic_map).reindex(is_idx)
is_liq   = build_liquidity_screen(fund, ret_wide).reindex(is_idx)

panels = {"sue": p_sue, "mom_52wk_high": p_mom, "sector_rs_1m": p_sector}
factor_names = list(panels.keys())
print(f"  Factors: {factor_names}")

# ── Pre-compute per-date z-scores ─────────────────────────────────────────────
# For each rebalance date, compute z-score of each factor over the liquid universe
# and store as arrays.  All weight combos share the same z-scores.
print("Pre-computing z-scores across IS dates...")
dates       = is_idx
rebal_dates = dates[::HOLD_MONTHS]

records = []   # list of (date, stock_ids, z_matrix [N x 3], next_returns [N])

for i, rdate in enumerate(rebal_dates[:-1]):
    next_rdate = rebal_dates[i + 1]

    universe = set(ret_wide.columns)
    if not is_liq.empty and rdate in is_liq.index:
        liq_row  = is_liq.loc[rdate].fillna(False)
        universe = universe.intersection(liq_row[liq_row].index)

    scored = {}
    for fn in factor_names:
        p = panels[fn]
        if rdate not in p.index:
            continue
        row = p.loc[rdate].dropna().replace([np.inf, -np.inf], np.nan).dropna()
        row = row[row.index.isin(universe)]
        z   = _zscore(row)
        if len(z) > 30:
            scored[fn] = z

    if len(scored) < 3:
        continue

    common = scored[factor_names[0]].index
    for z in scored.values():
        common = common.intersection(z.index)
    if len(common) < 30:
        continue

    period_rets = ret_wide.loc[
        (ret_wide.index > rdate) & (ret_wide.index <= next_rdate),
        list(common),
    ].dropna(how="all", axis=1)
    if period_rets.empty:
        continue

    # Compound period return per stock
    stock_ret = (1 + period_rets.reindex(columns=list(common)).fillna(0)).prod() - 1

    z_mat = np.column_stack([
        scored[fn].reindex(common).values for fn in factor_names
    ])
    records.append((rdate, np.array(common), z_mat, stock_ret.reindex(common).values))

print(f"  {len(records)} rebalance periods with full data")

# ── Grid search ───────────────────────────────────────────────────────────────
N_DECILES = 10
print("Sweeping weight grid (5% steps)...")

results = []
combos  = []
for a in range(5, 91, 5):
    for b in range(5, 96 - a, 5):
        c = 100 - a - b
        if c < 5:
            continue
        combos.append((a / 100, b / 100, c / 100))

bucket_ret_store = {}  # combo_idx -> list of (d, period_ret)

for ci, (wa, wb, wc) in enumerate(combos):
    w = np.array([wa, wb, wc])
    bucket_rets = {f"D{d}": [] for d in range(1, N_DECILES + 1)}

    for rdate, stocks, z_mat, next_r in records:
        composite = pd.Series(z_mat @ w, index=stocks).sort_values(ascending=False)
        n = len(composite)
        cut = n / N_DECILES
        for d in range(1, N_DECILES + 1):
            lo = int(round((d - 1) * cut))
            hi = int(round(d * cut))
            idx = composite.iloc[lo:hi].index
            r   = next_r[np.isin(stocks, idx)]
            bucket_rets[f"D{d}"].append(r.mean() if len(r) else np.nan)

    m1  = compute_metrics(pd.Series(bucket_rets["D1"]),  hold_months=HOLD_MONTHS, min_periods=4) or {}
    m10 = compute_metrics(pd.Series(bucket_rets["D10"]), hold_months=HOLD_MONTHS, min_periods=4) or {}
    if not m1 or not m10:
        continue

    ret_spread = m1["ann_return"] - m10["ann_return"]
    sh_spread  = m1["sharpe"]     - m10["sharpe"]
    score      = ret_spread + sh_spread

    results.append({
        "sue": int(wa * 100), "mom": int(wb * 100), "sector": int(wc * 100),
        "ret_spread": ret_spread, "sh_spread": sh_spread, "score": score,
        "d1_ret": m1["ann_return"], "d1_sharpe": m1["sharpe"],
        "d10_ret": m10["ann_return"], "d10_sharpe": m10["sharpe"],
    })

results.sort(key=lambda x: x["score"], reverse=True)

# ── Print results ─────────────────────────────────────────────────────────────
W = 105
print("\n" + "═" * W)
print("  WEIGHT GRID SEARCH — IS 2005–2024")
print("  Ranked by: ret_spread + sharpe_spread   |   D1=top decile, D10=bottom")
print(f"  Factors : sue | mom_52wk_high | sector_rs_1m   ({len(results)} combos tested)")
print("═" * W)
print(f"  {'Rk':>3}  {'SUE':>5}  {'Mom':>5}  {'Sec':>5}"
      f"  {'RetSprd':>8}  {'ShSprd':>7}  {'Score':>7}"
      f"  {'D1 Ret':>7}  {'D1 Shr':>7}  {'D10 Ret':>8}  {'D10 Shr':>8}")
print("  " + "─" * (W - 2))
for i, r in enumerate(results[:25], 1):
    print(f"  {i:>3}  {r['sue']:>4}%  {r['mom']:>4}%  {r['sector']:>4}%"
          f"  {r['ret_spread']:>+7.2f}%"
          f"  {r['sh_spread']:>7.3f}"
          f"  {r['score']:>+6.3f}"
          f"  {r['d1_ret']:>+6.1f}%"
          f"  {r['d1_sharpe']:>7.3f}"
          f"  {r['d10_ret']:>+7.1f}%"
          f"  {r['d10_sharpe']:>8.3f}")
print("═" * W)

best = results[0]
print(f"\n  Top weights: sue={best['sue']}%  mom={best['mom']}%  sector={best['sector']}%")
print(f"  IS D1–D10 ret spread: {best['ret_spread']:+.2f}%   Sharpe spread: {best['sh_spread']:+.3f}")
print(f"\n  To run full IS+OOS with these weights, add to FACTOR_WEIGHTS_CAPPED:")
print(f'    "sue": {best["sue"]/100:.2f}, "mom_52wk_high": {best["mom"]/100:.2f}, "sector_rs_1m": {best["sector"]/100:.2f}')
