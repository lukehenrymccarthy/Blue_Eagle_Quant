import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sectorscope.metrics import compute_metrics
from sectorscope.utils import zscore as _zscore, sic_to_etf as _sic_to_etf

# ── Paths & global config ─────────────────────────────────────────────────────
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RISK_FREE_ANN = 0.05          # 5% annual risk-free rate
TC            = 0.001         # 10 bps per side
START_DATE    = "2010-01-01"
END_DATE      = "2024-12-31"

BASKET_SIZES  = [10, 25, 50, 100]
HOLD_MONTHS   = [1, 2, 3, 6]
UNIVERSE_SIZE = 1000

SECTOR_ETFS = ["XLC", "XLY", "XLP", "XLE", "XLF",
               "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]

FACTOR_WEIGHTS = {
    "momentum_12m":   0.10,
    "fund_quality":   0.30,
    "sector_rs_3m":   0.30,
    "analyst_rev_3m": 0.30,
}

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — each function returns a pre-built panel or Series
# ══════════════════════════════════════════════════════════════════════════════

def load_crsp_wide() -> pd.DataFrame:
    """Load CRSP monthly returns as (date × permno) wide DataFrame."""
    from ingestion.wrds_returns import load_returns, returns_wide
    ret_df = load_returns()
    rw = returns_wide(ret_df)
    rw.index = pd.to_datetime(rw.index) + pd.offsets.MonthEnd(0)
    return rw

def build_compustat_panels(ret_wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load Compustat and build:
      - fund_quality_panel : composite of ROE TTM, neg-accruals, gross prof, fcf_yield.
      - sic_map            : Series(permno → sector ETF ticker).
      - is_liquid_panel    : boolean panel for top market cap stocks.
    """
    fund_path = Path("data/fundamentals/compustat_quarterly.parquet")
    import pyarrow.parquet as pq
    schema_names = pq.read_schema(fund_path).names
    sic_col   = "siccd" if "siccd" in schema_names else ("sic" if "sic" in schema_names else None)
    load_cols = ["permno", "datadate", "available_date",
                 "roe_ttm", "ibq_ttm", "oancfq_ttm", "atq", "saleq", "fcf_yield", "market_cap"]
    if sic_col: load_cols.append(sic_col)

    fund = pd.read_parquet(fund_path, columns=[c for c in load_cols if c in schema_names])
    fund["available_date"] = pd.to_datetime(fund["available_date"])
    fund["datadate"]       = pd.to_datetime(fund["datadate"])

    def _build_signal_panel(col: str) -> pd.DataFrame:
        piv = (fund.dropna(subset=[col])
               .sort_values("available_date")
               .pivot_table(index="available_date", columns="permno",
                            values=col, aggfunc="last"))
        piv.index = pd.to_datetime(piv.index)
        return piv.resample("ME").last().ffill().reindex(ret_wide.index, method="ffill")

    if sic_col and sic_col in fund.columns:
        sic_latest = fund.sort_values("datadate").groupby("permno")[sic_col].last()
        sic_map    = sic_latest.apply(_sic_to_etf).dropna()
    else:
        sic_map = pd.Series(dtype=str)

    def _cs_zscore_panel_sector_neutral(panel: pd.DataFrame) -> pd.DataFrame:
        def _row_z(row):
            s = row.dropna().replace([np.inf, -np.inf], np.nan)
            if len(s) < 10: return row * 0
            df = s.to_frame(name="val")
            df["sector"] = sic_map
            df["sector"] = df["sector"].fillna("Unknown")
            lo, hi = df["val"].quantile(0.01), df["val"].quantile(0.99)
            df["val"] = df["val"].clip(lo, hi)
            def _sec_z(grp):
                if len(grp) < 3 or grp.std() == 0: return grp * 0
                return (grp - grp.mean()) / grp.std()
            return df.groupby("sector")["val"].transform(_sec_z)
        return panel.apply(_row_z, axis=1)

    roe_z = _cs_zscore_panel_sector_neutral(_build_signal_panel("roe_ttm"))

    fcf_z = pd.DataFrame(dtype=float)
    if "fcf_yield" in fund.columns:
        fcf_z = _cs_zscore_panel_sector_neutral(_build_signal_panel("fcf_yield"))

    neg_acc_z = pd.DataFrame(dtype=float)
    if all(c in fund.columns for c in ["ibq_ttm", "oancfq_ttm", "atq"]):
        fund["neg_accruals"] = -(fund["ibq_ttm"] - fund["oancfq_ttm"]) / fund["atq"].replace(0, np.nan)
        neg_acc_z = _cs_zscore_panel_sector_neutral(_build_signal_panel("neg_accruals"))

    gp_z = pd.DataFrame(dtype=float)
    if all(c in fund.columns for c in ["saleq", "atq"]):
        fund["gross_prof"] = fund["saleq"] / fund["atq"].replace(0, np.nan)
        gp_z = _cs_zscore_panel_sector_neutral(_build_signal_panel("gross_prof"))

    components = [df for df in [roe_z, fcf_z, neg_acc_z, gp_z] if not df.empty]
    if components:
        fund_quality_panel = pd.concat(components, axis=0).groupby(level=0).mean()
        fund_quality_panel = fund_quality_panel.reindex(ret_wide.index, method="ffill")
    else:
        fund_quality_panel = pd.DataFrame()

    if "market_cap" in fund.columns:
        mkt_cap_panel = _build_signal_panel("market_cap")
        def _top_k(row):
            s = row.dropna()
            if len(s) == 0: return pd.Series(False, index=row.index)
            top_k = s.nlargest(UNIVERSE_SIZE).index
            res = pd.Series(False, index=row.index)
            res.loc[top_k] = True
            return res
        is_liquid_panel = mkt_cap_panel.apply(_top_k, axis=1)
    else:
        is_liquid_panel = pd.DataFrame(True, index=ret_wide.index, columns=ret_wide.columns)

    return fund_quality_panel, sic_map, is_liquid_panel

def build_sector_rs_panel(ret_wide: pd.DataFrame, sic_map: pd.Series) -> pd.DataFrame:
    print("  Downloading sector ETF prices (yfinance)...")
    raw = yf.download(
        SECTOR_ETFS + ["SPY"],
        start="2009-01-01", end="2025-06-01",
        interval="1mo", auto_adjust=True, progress=False,
    )["Close"]
    raw.index = pd.to_datetime(raw.index) + pd.offsets.MonthEnd(0)
    raw = raw.sort_index()

    monthly_ret = raw.pct_change()
    spy_3m  = (1 + monthly_ret["SPY"].fillna(0)).rolling(3).apply(np.prod, raw=True) - 1
    etf_rs  = {}
    for etf in SECTOR_ETFS:
        if etf not in monthly_ret.columns: continue
        etf_3m = (1 + monthly_ret[etf].fillna(0)).rolling(3).apply(np.prod, raw=True) - 1
        etf_rs[etf] = etf_3m - spy_3m

    etf_rs_df = pd.DataFrame(etf_rs).reindex(ret_wide.index, method="ffill")

    if sic_map.empty: return pd.DataFrame()

    stock_universe = ret_wide.columns
    stock_to_etf   = sic_map.reindex(stock_universe).dropna()
    etf_list   = list(etf_rs_df.columns)
    etf_to_idx = {e: i for i, e in enumerate(etf_list)}

    matched = [(s, e) for s, e in stock_to_etf.items() if e in etf_to_idx]
    if not matched: return pd.DataFrame()

    stocks_out = [s for s, _ in matched]
    etf_idxs   = [etf_to_idx[e] for _, e in matched]

    rs_matrix    = etf_rs_df.values
    sector_panel = pd.DataFrame(
        rs_matrix[:, etf_idxs],
        index   = etf_rs_df.index,
        columns = stocks_out,
    )
    sector_panel.columns = sector_panel.columns.astype(ret_wide.columns.dtype)
    return sector_panel

def build_analyst_panel(ret_wide: pd.DataFrame) -> pd.DataFrame:
    analyst_path = Path("data/analyst/ibes_signals.parquet")
    if not analyst_path.exists(): return pd.DataFrame()

    ibes = pd.read_parquet(analyst_path, columns=["permno", "statpers", "rev_3m"])
    ibes["statpers"] = pd.to_datetime(ibes["statpers"]) + pd.offsets.MonthEnd(0)
    ibes["permno"]   = ibes["permno"].astype(int)

    piv = (ibes.dropna(subset=["rev_3m"])
           .pivot_table(index="statpers", columns="permno",
                        values="rev_3m", aggfunc="last"))
    piv.index = pd.to_datetime(piv.index)
    monthly = piv.resample("ME").last().ffill()
    return monthly.reindex(ret_wide.index, method="ffill")

def build_macro_factor(ret_wide: pd.DataFrame) -> pd.Series | None:
    macro_path = Path("data/macro/fred_signals.parquet")
    if not macro_path.exists(): return None

    signals = pd.read_parquet(macro_path)
    signals.index = pd.to_datetime(signals.index) + pd.offsets.MonthEnd(0)

    if "hy_spread_widening" not in signals.columns: return None

    s = signals["hy_spread_widening"].dropna()
    roll_mean = s.rolling(36, min_periods=12).mean()
    roll_std  = s.rolling(36, min_periods=12).std().replace(0, np.nan)
    macro_z   = -((s - roll_mean) / roll_std)
    return macro_z.reindex(ret_wide.index, method="ffill")

def build_all_factor_panels(
    ret_wide:         pd.DataFrame,
    fund_quality:     pd.DataFrame,
    sector_rs:        pd.DataFrame,
    analyst_rev:      pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    panels = {}

    print("  F1  Momentum 12m-1m          — vectorised rolling...")
    roll = (1 + ret_wide.fillna(0)).rolling(12).apply(np.prod, raw=True) - 1
    panels["momentum_12m"] = roll.shift(1)
    print(f"       {ret_wide.shape[1]:,} stocks | {ret_wide.shape[0]} months")

    if not fund_quality.empty:
        print(f"  F2  Fund Quality             — {fund_quality.shape[1]:,} stocks")
        panels["fund_quality"] = fund_quality
    else:
        print("  F2  Fund Quality             — [SKIP]")

    if not sector_rs.empty:
        print(f"  F3  Sector RS vs SPY 3m      — {sector_rs.shape[1]:,} stocks mapped")
        panels["sector_rs_3m"] = sector_rs
    else:
        print("  F3  Sector RS vs SPY 3m      — [SKIP]")

    if not analyst_rev.empty:
        print(f"  F4  Analyst Rec Revision 3m  — {analyst_rev.shape[1]:,} stocks")
        panels["analyst_rev_3m"] = analyst_rev
    else:
        print("  F4  Analyst Rec Revision 3m  — [SKIP]")

    return panels

def load_optimal_weights() -> dict | None:
    path = Path("data/results/optimal_weights.json")
    if not path.exists(): return None
    with open(path) as f: data = json.load(f)
    return data.get("weights")

def run_one_config(
    ret_wide:        pd.DataFrame,
    panels:          dict[str, pd.DataFrame],
    macro_factor:    pd.Series | None,
    is_liquid_panel: pd.DataFrame,
    basket:          int,
    hold_months:     int,
    weights:         dict | None = None,
) -> pd.Series:
    dates        = ret_wide.index
    rebal_dates  = dates[::hold_months]
    factor_names = list(panels.keys())

    port_rets = []
    prev_hold = set()

    for i, rdate in enumerate(rebal_dates[:-1]):
        next_rdate = rebal_dates[i + 1]

        crsp_universe = set(ret_wide.columns)
        if rdate in is_liquid_panel.index:
            liquid_row = is_liquid_panel.loc[rdate].fillna(False)
            crsp_universe = crsp_universe.intersection(liquid_row[liquid_row].index)

        scored = {}
        for fname in factor_names:
            panel = panels[fname]
            if rdate not in panel.index: continue
            row = panel.loc[rdate].dropna()
            row = row[row.index.isin(crsp_universe)]
            z   = _zscore(row)
            if len(z) >= basket: scored[fname] = z

        if not scored:
            port_rets.append(np.nan)
            continue

        common = list(scored.values())[0].index
        for z in scored.values():
            common = common.intersection(z.index)

        if len(common) < basket:
            port_rets.append(np.nan)
            continue

        effective_weights = weights.copy() if weights else FACTOR_WEIGHTS.copy()
        macro_val = macro_factor.asof(rdate) if macro_factor is not None else np.nan

        invested = 1.0
        if not pd.isna(macro_val):
            if macro_val < -1.0:
                invested = 0.70
                if "fund_quality" in effective_weights and "momentum_12m" in effective_weights:
                    effective_weights["fund_quality"] += 0.10
                    effective_weights["momentum_12m"] = max(0, effective_weights["momentum_12m"] - 0.10)
            elif macro_val < -0.5:
                invested = 0.85
                if "fund_quality" in effective_weights and "momentum_12m" in effective_weights:
                    effective_weights["fund_quality"] += 0.05
                    effective_weights["momentum_12m"] = max(0, effective_weights["momentum_12m"] - 0.05)

        cs_factor_names = list(scored.keys())
        w_arr = np.array([effective_weights.get(fn, 1 / len(cs_factor_names)) for fn in cs_factor_names])
        if w_arr.sum() > 0:
            w_arr = w_arr / w_arr.sum()
        else:
            w_arr = np.ones(len(cs_factor_names)) / len(cs_factor_names)

        factor_mat = np.column_stack([scored[fn].reindex(common).values for fn in cs_factor_names])
        composite  = pd.Series(factor_mat @ w_arr, index=common)

        top_n = set(composite.nlargest(basket).index)

        turnover = len(top_n.symmetric_difference(prev_hold)) / (2 * basket) if prev_hold else 1.0
        tc_drag = turnover * TC

        period_slice = ret_wide.loc[
            (ret_wide.index > rdate) & (ret_wide.index <= next_rdate),
            list(top_n),
        ].dropna(how="all", axis=1)

        if period_slice.empty:
            port_rets.append(np.nan)
            continue

        ew_monthly = period_slice.mean(axis=1)
        raw_ret    = (1 + ew_monthly).prod() - 1

        rf_per_period = (1 + RISK_FREE_ANN) ** (hold_months / 12) - 1
        compound_ret  = raw_ret * invested + rf_per_period * (1 - invested) - tc_drag
        port_rets.append(compound_ret)
        prev_hold = top_n

    return pd.Series(port_rets, index=rebal_dates[:-1])

def print_results_table(summary: pd.DataFrame, active_factors: list[str],
                        start: str = "", end: str = ""):
    W = 98
    period_str = f"{start} – {end}" if start and end else "see data"
    print("\n" + "═" * W)
    print("  FIVE-FACTOR MODEL  —  BACKTEST SUMMARY")
    print(f"  Period : {period_str}")
    print(f"  Factors: {', '.join(active_factors)}")
    print(f"  Cost   : 10 bps/side on turnover   |   Equal-weight portfolio")
    print("═" * W)
    print()

    hdr = (f"  {'Basket':>7}  │  {'Ann Ret':>7}  {'Ann Vol':>7}  "
           f"{'Sharpe':>7}  {'Max DD':>7}  {'Calmar':>7}  "
           f"{'Hit%':>6}  {'t-stat':>7}  {'N':>5}")
    sep = "  " + "─" * 7 + "──┼──" + "──────────────".join(["─" * 7] * 7) + "──" + "─" * 7

    for hold in HOLD_MONTHS:
        sub = summary[summary["hold_months"] == hold].sort_values("basket_size")
        if sub.empty: continue
        n_example = sub.iloc[0]["n_periods"] if len(sub) else "?"
        print(f"  Holding: {hold} Month{'s' if hold > 1 else '':1s}  (~{int(n_example)} periods)")
        print(sep)
        print(hdr)
        print(sep)
        for _, row in sub.iterrows():
            print(f"  {'Top ' + str(int(row['basket_size'])):>7}  │ "
                  f" {row['ann_return']:>+6.1f}%"
                  f"  {row['ann_vol']:>6.1f}%"
                  f"  {row['sharpe']:>7.3f}"
                  f"  {row['max_drawdown']:>+7.1f}%"
                  f"  {row['calmar']:>7.3f}"
                  f"  {row['hit_rate']:>5.1f}%"
                  f"  {row['t_stat']:>7.2f}"
                  f"  {int(row['n_periods']):>5}")
        print()
    print("═" * W)

def main():
    parser = argparse.ArgumentParser(description="Five-factor equal-weight backtest")
    parser.add_argument("--no-macro", action="store_true")
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end",   default=END_DATE)
    args = parser.parse_args()

    use_macro = not args.no_macro
    start     = args.start
    end       = args.end

    print("\n" + "═" * 70)
    print("  FIVE-FACTOR MODEL  —  Loading data")
    print("═" * 70)

    try:
        print("\n  Loading CRSP monthly returns...")
        ret_wide = load_crsp_wide().loc[start:end]
        print(f"  CRSP : {ret_wide.shape[1]:,} stocks | {ret_wide.index[0].date()} – {ret_wide.index[-1].date()}")
    except FileNotFoundError as e:
        print(f"\n  [ERROR] {e}")
        return

    fund_panel, sic_map, is_liquid_panel = pd.DataFrame(), pd.Series(dtype=str), pd.DataFrame()
    try:
        print("\n  Loading Compustat (ROE / Accruals / Gross Profit / FCF Yield + SIC)...")
        fund_panel, sic_map, is_liquid_panel = build_compustat_panels(ret_wide)
        print(f"  Fund Quality  : {fund_panel.shape[1]:,} stocks")
    except Exception as e:
        print(f"  [SKIP] Compustat error: {e}")

    sector_rs = pd.DataFrame()
    try:
        sector_rs = build_sector_rs_panel(ret_wide, sic_map)
    except Exception as e:
        print(f"  [SKIP] Sector RS error: {e}")

    analyst_rev = pd.DataFrame()
    try:
        print("\n  Loading IBES analyst revisions...")
        analyst_rev = build_analyst_panel(ret_wide)
    except Exception as e:
        print(f"  [SKIP] Analyst error: {e}")

    macro_factor = None
    if use_macro:
        try:
            print("\n  Loading FRED macro factor...")
            macro_factor = build_macro_factor(ret_wide)
        except Exception as e:
            print(f"  [SKIP] Macro error: {e}")

    print("\n" + "═" * 70)
    print("  Building factor panels")
    print("═" * 70 + "\n")
    panels = build_all_factor_panels(ret_wide, fund_panel, sector_rs, analyst_rev)
    active_factors = list(panels.keys())
    
    opt_weights = load_optimal_weights()
    if opt_weights:
        print(f"\n  Optimal weights loaded:")
        for fn, w in opt_weights.items(): print(f"    {fn:25s} {w:.1%}")
    else:
        print("\n  No optimal weights found — using equal weights.")

    if len(active_factors) == 0:
        print("  [ERROR] No factor panels built.")
        return

    print("═" * 70)
    print(f"  Running backtest grid")
    print("═" * 70 + "\n")

    results       = []
    equity_curves = {}

    for hold in HOLD_MONTHS:
        for basket in BASKET_SIZES:
            label = f"top{basket}_hold{hold}m"
            print(f"  {label:<22}", end=" ... ", flush=True)
            try:
                curve = run_one_config(
                    ret_wide        = ret_wide,
                    panels          = panels,
                    macro_factor    = macro_factor if use_macro else None,
                    is_liquid_panel = is_liquid_panel,
                    basket          = basket,
                    hold_months     = hold,
                    weights         = opt_weights,
                )
                m = compute_metrics(curve, hold_months=hold)
                if not m:
                    print("skipped")
                    continue
                print(f"Sharpe={m['sharpe']:>6.3f}  Ann={m['ann_return']:>+6.1f}%")
                results.append({"basket_size": basket, "hold_months": hold, "run_id": label, **m})
                equity_curves[label] = curve
            except Exception as e:
                print(f"ERROR — {e}")

    if not results:
        print("\n  [ERROR] No results produced.")
        return

    summary = pd.DataFrame(results)
    print_results_table(summary, active_factors, start=start, end=end)

    summary_path = RESULTS_DIR / "five_factor_summary.csv"
    curves_path  = RESULTS_DIR / "five_factor_curves.parquet"
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(equity_curves).to_parquet(curves_path)
    print(f"\n  Saved → {summary_path}")
    print(f"  Saved → {curves_path}")

if __name__ == "__main__":
    main()
