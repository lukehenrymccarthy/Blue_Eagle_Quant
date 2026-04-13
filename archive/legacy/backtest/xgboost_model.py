"""
backtest/xgboost_model.py
--------------------------
Walk-forward XGBoost stock-ranking model.

Architecture
------------
Each month, an XGBoost regressor is trained on a rolling 60-month window of
(stock × feature) rows. The target is the cross-sectional percentile rank of
each stock's next-month return — this makes the model focus purely on
*relative* ordering rather than predicting absolute return magnitudes, which
are too noisy to model directly.

Features (14 total, all PIT-safe):
  Returns   : ret_1m, ret_3m, ret_6m, mom_12m_1, vol_12m, beta_spy
  Fundamentals: roe_ttm, fcf_yield, earnings_yield, neg_accruals, gross_prof
  Analyst   : rev_3m
  Sector    : sector_rs_3m, credit_spread_tilt

Walk-forward protocol
---------------------
  Train window : 60 months
  Retrain      : every 3 months (fast; monthly available via --retrain-monthly)
  Universe     : top-1000 stocks by market cap each month
  Portfolio    : top-N by predicted rank score, equal-weight
  Costs        : 10 bps per side on turnover

Usage
-----
  python backtest/xgboost_model.py
  python backtest/xgboost_model.py --basket 25 --hold 1
  python backtest/xgboost_model.py --basket 50 --hold 3 --retrain-monthly
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sectorscope.metrics import compute_metrics
from sectorscope.utils import sic_to_etf as _sic_to_etf

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR   = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RISK_FREE_ANN  = 0.05
TC             = 0.001
START_DATE     = "2010-01-01"
END_DATE       = "2024-12-31"
TRAIN_WINDOW   = 60    # months of history used to train each model
UNIVERSE_SIZE  = 1000  # restrict to top-N stocks by market cap
SECTOR_ETFS    = ["XLC","XLY","XLP","XLE","XLF","XLV","XLI","XLB","XLRE","XLK","XLU"]
CREDIT_BETA_WIN = 36   # months for rolling credit spread beta

FEATURE_COLS = [
    "ret_1m", "ret_3m", "ret_6m", "mom_12m_1", "vol_12m", "beta_spy",
    "roe_ttm", "fcf_yield", "earnings_yield", "neg_accruals", "gross_prof",
    "rev_3m", "sector_rs_3m", "credit_spread_tilt",
]

XGB_PARAMS = dict(
    n_estimators     = 300,
    max_depth        = 4,
    learning_rate    = 0.03,
    subsample        = 0.8,
    colsample_bytree = 0.7,
    min_child_weight = 30,   # prevents overfitting on small groups
    reg_lambda       = 2.0,
    objective        = "reg:squarederror",
    random_state     = 42,
    n_jobs           = -1,
    verbosity        = 0,
)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_crsp_wide() -> pd.DataFrame:
    from ingestion.wrds_returns import load_returns, returns_wide
    ret_df = load_returns()
    rw = returns_wide(ret_df)
    rw.index = pd.to_datetime(rw.index) + pd.offsets.MonthEnd(0)
    return rw


def build_all_feature_panels(ret_wide: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Build every feature panel as a (date × permno) DataFrame.
    Returns (panels_dict, sic_map, is_liquid_panel).
    """
    import pyarrow.parquet as pq

    panels = {}

    # ── Returns-based features ────────────────────────────────────────────────
    print("  Return features (1M/3M/6M/12M mom, vol, beta)...")

    def _cum(w):
        return (1 + ret_wide.fillna(0)).rolling(w).apply(np.prod, raw=True) - 1

    panels["ret_1m"]    = ret_wide.copy()
    panels["ret_3m"]    = _cum(3)
    panels["ret_6m"]    = _cum(6)
    panels["mom_12m_1"] = _cum(12).shift(1)   # skip last month
    panels["vol_12m"]   = ret_wide.rolling(12, min_periods=6).std()

    # Beta to SPY
    spy_raw = yf.download("SPY", start="2007-01-01", end="2025-06-01",
                           interval="1mo", auto_adjust=True, progress=False)["Close"]
    spy_ret = spy_raw.squeeze().pct_change()
    spy_ret.index = pd.to_datetime(spy_ret.index) + pd.offsets.MonthEnd(0)
    spy_ret = spy_ret.reindex(ret_wide.index)
    spy_var = spy_ret.rolling(12, min_periods=6).var().replace(0, np.nan)
    # Vectorised: each column's rolling cov with SPY / SPY var
    panels["beta_spy"] = ret_wide.apply(
        lambda col: col.rolling(12, min_periods=6).cov(spy_ret) / spy_var
    )

    # ── Fundamentals ──────────────────────────────────────────────────────────
    print("  Fundamental features (Compustat)...")
    fund_path    = Path("data/fundamentals/compustat_quarterly.parquet")
    schema_names = pq.read_schema(fund_path).names
    sic_col      = "siccd" if "siccd" in schema_names else ("sic" if "sic" in schema_names else None)

    load_cols = ["permno", "datadate", "available_date",
                 "roe_ttm", "fcf_yield", "earnings_yield",
                 "ibq_ttm", "oancfq_ttm", "atq", "saleq", "market_cap"]
    if sic_col:
        load_cols.append(sic_col)

    fund = pd.read_parquet(fund_path,
                           columns=[c for c in load_cols if c in schema_names])
    fund["available_date"] = pd.to_datetime(fund["available_date"])
    fund["neg_accruals"]   = -(fund["ibq_ttm"] - fund["oancfq_ttm"]) / fund["atq"].replace(0, np.nan)
    fund["gross_prof"]     = fund["saleq"] / fund["atq"].replace(0, np.nan)

    def _pivot_ffill(col: str) -> pd.DataFrame:
        piv = (fund.dropna(subset=[col])
               .sort_values("available_date")
               .pivot_table(index="available_date", columns="permno",
                            values=col, aggfunc="last"))
        piv.index = pd.to_datetime(piv.index)
        return (piv.resample("ME").last()
                .ffill()
                .reindex(ret_wide.index, method="ffill"))

    for col in ["roe_ttm", "fcf_yield", "earnings_yield", "neg_accruals", "gross_prof"]:
        panels[col] = _pivot_ffill(col)

    # ── Liquidity filter: top-UNIVERSE_SIZE by market cap ────────────────────
    mkt_panel = _pivot_ffill("market_cap")

    def _top_k(row):
        s = row.dropna()
        if s.empty:
            return pd.Series(False, index=row.index)
        result = pd.Series(False, index=row.index)
        result.loc[s.nlargest(UNIVERSE_SIZE).index] = True
        return result

    is_liquid_panel = mkt_panel.apply(_top_k, axis=1)

    # ── SIC map ───────────────────────────────────────────────────────────────
    if sic_col and sic_col in fund.columns:
        sic_latest = fund.sort_values("datadate").groupby("permno")[sic_col].last()
        sic_map    = sic_latest.apply(_sic_to_etf).dropna()
    else:
        sic_map = pd.Series(dtype=str)

    # ── IBES analyst revisions ─────────────────────────────────────────────────
    print("  Analyst features (IBES)...")
    analyst_path = Path("data/analyst/ibes_signals.parquet")
    if analyst_path.exists():
        ibes = pd.read_parquet(analyst_path, columns=["permno", "statpers", "rev_3m"])
        ibes["statpers"] = pd.to_datetime(ibes["statpers"]) + pd.offsets.MonthEnd(0)
        ibes["permno"]   = ibes["permno"].astype(int)
        piv = (ibes.dropna(subset=["rev_3m"])
               .pivot_table(index="statpers", columns="permno",
                            values="rev_3m", aggfunc="last"))
        piv.index = pd.to_datetime(piv.index)
        panels["rev_3m"] = (piv.resample("ME").last()
                            .ffill()
                            .reindex(ret_wide.index, method="ffill"))

    # ── Sector RS vs SPY (3-month) ─────────────────────────────────────────────
    print("  Sector / credit spread features (yfinance + FRED)...")
    raw = yf.download(SECTOR_ETFS + ["SPY"],
                      start="2007-01-01", end="2025-06-01",
                      interval="1mo", auto_adjust=True, progress=False)["Close"]
    raw.index = pd.to_datetime(raw.index) + pd.offsets.MonthEnd(0)
    mret = raw.pct_change()

    spy_3m  = (1 + mret["SPY"].fillna(0)).rolling(3).apply(np.prod, raw=True) - 1
    etf_rs  = {}
    for etf in SECTOR_ETFS:
        if etf not in mret.columns:
            continue
        etf_3m     = (1 + mret[etf].fillna(0)).rolling(3).apply(np.prod, raw=True) - 1
        etf_rs[etf] = etf_3m - spy_3m
    etf_rs_df = pd.DataFrame(etf_rs).reindex(ret_wide.index, method="ffill")

    # Credit spread tilt (replaces oil beta tilt)
    macro_path = Path("data/macro/fred_signals.parquet")
    credit_tilt_etf = pd.DataFrame()
    if macro_path.exists():
        fred = pd.read_parquet(macro_path)
        fred.index = pd.to_datetime(fred.index) + pd.offsets.MonthEnd(0)
        if "hy_spread_widening" in fred.columns:
            hy_spread    = fred["hy_spread_widening"].dropna().sort_index()
            delta_spread = hy_spread.diff().dropna()
            roll_mean    = hy_spread.rolling(36, min_periods=12).mean()
            roll_std     = hy_spread.rolling(36, min_periods=12).std().replace(0, np.nan)
            credit_regime = ((hy_spread - roll_mean) / roll_std).reindex(ret_wide.index, method="ffill")

            etf_betas = {}
            for etf in SECTOR_ETFS:
                if etf not in mret.columns:
                    continue
                common   = delta_spread.index.intersection(mret[etf].index)
                roll_cov = mret[etf].reindex(common).rolling(CREDIT_BETA_WIN, min_periods=12).cov(delta_spread.reindex(common))
                roll_var = delta_spread.reindex(common).rolling(CREDIT_BETA_WIN, min_periods=12).var().replace(0, np.nan)
                etf_betas[etf] = roll_cov / roll_var

            beta_df         = pd.DataFrame(etf_betas).reindex(ret_wide.index, method="ffill")
            credit_tilt_etf = beta_df.multiply(credit_regime, axis=0)

    # Broadcast sector RS and credit spread tilt to stock level via sic_map
    if not sic_map.empty:
        stock_to_etf = sic_map.reindex(ret_wide.columns).dropna()
        etf_cols     = list(etf_rs_df.columns)
        etf_to_idx   = {e: i for i, e in enumerate(etf_cols)}
        matched      = [(s, e) for s, e in stock_to_etf.items() if e in etf_to_idx]

        if matched:
            stocks_out = [s for s, _ in matched]
            etf_idxs   = [etf_to_idx[e] for _, e in matched]

            rs_mat = etf_rs_df.reindex(columns=etf_cols).values
            rs_panel = pd.DataFrame(
                rs_mat[:, etf_idxs], index=etf_rs_df.index, columns=stocks_out)
            rs_panel.columns = rs_panel.columns.astype(ret_wide.columns.dtype)
            panels["sector_rs_3m"] = rs_panel.reindex(ret_wide.index, method="ffill")

            if not credit_tilt_etf.empty:
                ct_mat = credit_tilt_etf.reindex(columns=etf_cols).values
                ct_panel = pd.DataFrame(
                    ct_mat[:, etf_idxs], index=credit_tilt_etf.index, columns=stocks_out)
                ct_panel.columns = ct_panel.columns.astype(ret_wide.columns.dtype)
                panels["credit_spread_tilt"] = ct_panel.reindex(ret_wide.index, method="ffill")

    return panels, sic_map, is_liquid_panel


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(panels: dict, ret_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Stack all (date × permno) panels into a long-format DataFrame.
    Rows are (date, permno) MultiIndex. Columns are feature names + 'target'.
    Target = cross-sectional percentile rank of next-month return.
    """
    print("  Stacking feature matrix...")

    # Stack each available panel
    series = []
    for col in FEATURE_COLS:
        panel = panels.get(col)
        if panel is None or (isinstance(panel, pd.DataFrame) and panel.empty):
            print(f"    [WARN] Feature '{col}' missing — filled with NaN")
            continue
        series.append(panel.stack(future_stack=True).rename(col))

    feat_df = pd.concat(series, axis=1)

    # Cross-sectionally winsorise each feature within each month
    def _cs_winsor(df: pd.DataFrame, pct: float = 0.01) -> pd.DataFrame:
        def _w(row):
            s = row.dropna()
            if len(s) < 10:
                return row
            lo, hi = s.quantile(pct), s.quantile(1 - pct)
            return row.clip(lo, hi)
        return df.groupby(level=0).transform(_w)

    feat_df = _cs_winsor(feat_df)
    feat_df = feat_df[~feat_df.index.duplicated(keep="first")]

    # Target: next-month return (shifted back one period)
    target_panel = ret_wide.shift(-1)
    target       = target_panel.stack(future_stack=True).rename("target")
    target       = target[~target.index.duplicated(keep="first")]

    # Target rank (percentile) within each month's cross-section
    rank_target = (target.groupby(level=0)
                   .rank(pct=True)
                   .rename("target_rank"))

    df = pd.concat([feat_df, target, rank_target], axis=1)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def run_xgb_backtest(
    feature_df:      pd.DataFrame,
    ret_wide:        pd.DataFrame,
    is_liquid_panel: pd.DataFrame,
    basket:          int  = 25,
    retrain_every:   int  = 3,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Walk-forward XGBoost backtest.

    Returns
    -------
    port_returns  : monthly return series (indexed by rebalance date)
    feat_importance : DataFrame of average feature importances across all models
    """
    dates = ret_wide.index
    all_importances = []
    port_rets  = []
    prev_hold  = set()
    model      = None

    pred_range = range(TRAIN_WINDOW, len(dates) - 1)
    print(f"\n  Walk-forward: {len(pred_range)} months "
          f"({dates[TRAIN_WINDOW].date()} → {dates[-2].date()}), "
          f"retraining every {retrain_every}m, basket={basket}")

    for step, i in enumerate(pred_range):
        rdate     = dates[i]
        next_date = dates[i + 1]

        # ── Liquidity filter ──────────────────────────────────────────────────
        if rdate in is_liquid_panel.index:
            liq_row  = is_liquid_panel.loc[rdate].fillna(False)
            universe = set(liq_row[liq_row].index)
        else:
            universe = set(ret_wide.columns)

        # ── Retrain on schedule ───────────────────────────────────────────────
        if step % retrain_every == 0:
            train_start = dates[max(0, i - TRAIN_WINDOW)]
            train_dates = dates[(dates >= train_start) & (dates < rdate)]
            train_slice = feature_df.loc[feature_df.index.get_level_values(0).isin(train_dates)]
            train_slice = train_slice.dropna(subset=FEATURE_COLS + ["target_rank"])

            if len(train_slice) >= 200:
                X_tr = train_slice[FEATURE_COLS]
                y_tr = train_slice["target_rank"]
                model = xgb.XGBRegressor(**XGB_PARAMS)
                model.fit(X_tr, y_tr)
                all_importances.append(
                    pd.Series(model.feature_importances_, index=FEATURE_COLS))

        if model is None:
            port_rets.append(np.nan)
            continue

        # ── Predict at rdate ──────────────────────────────────────────────────
        if rdate not in feature_df.index.get_level_values(0):
            port_rets.append(np.nan)
            continue

        pred_slice = feature_df.loc[rdate]
        pred_slice = pred_slice[pred_slice.index.isin(universe)]
        pred_slice = pred_slice[FEATURE_COLS].dropna(how="any")

        if len(pred_slice) < basket:
            port_rets.append(np.nan)
            continue

        scores = pd.Series(model.predict(pred_slice), index=pred_slice.index)
        top_n  = set(scores.nlargest(basket).index)

        # ── Transaction cost ──────────────────────────────────────────────────
        turnover = (len(top_n.symmetric_difference(prev_hold)) / (2 * basket)
                    if prev_hold else 1.0)
        tc_drag  = turnover * TC

        # ── Next-month portfolio return ───────────────────────────────────────
        stock_rets = ret_wide.loc[next_date, list(top_n)].dropna()
        if stock_rets.empty:
            port_rets.append(np.nan)
            continue

        port_rets.append(stock_rets.mean() - tc_drag)
        prev_hold = top_n

        if (step + 1) % 12 == 0:
            print(f"    {rdate.date()}  step {step+1}/{len(pred_range)}")

    curve = pd.Series(port_rets, index=dates[TRAIN_WINDOW:-1])

    feat_imp = (pd.concat(all_importances, axis=1).T
                .mean()
                .sort_values(ascending=False)
                .rename("importance")) if all_importances else pd.Series(dtype=float)

    return curve, feat_imp


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Walk-forward XGBoost stock ranker")
    parser.add_argument("--basket",          type=int, default=25)
    parser.add_argument("--retrain-monthly", action="store_true",
                        help="Retrain every month (slower but more responsive)")
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end",   default=END_DATE)
    args = parser.parse_args()

    retrain_every = 1 if args.retrain_monthly else 3

    print("\n" + "═" * 65)
    print("  XGBOOST WALK-FORWARD  —  Loading data")
    print("═" * 65)

    ret_wide = load_crsp_wide().loc[args.start:args.end]
    print(f"  CRSP: {ret_wide.shape[1]:,} stocks | "
          f"{ret_wide.index[0].date()} – {ret_wide.index[-1].date()}")

    panels, sic_map, is_liquid_panel = build_all_feature_panels(ret_wide)

    print(f"\n  Features available: "
          f"{[c for c in FEATURE_COLS if c in panels]}")
    missing = [c for c in FEATURE_COLS if c not in panels]
    if missing:
        print(f"  Missing (will be NaN): {missing}")

    print("\n  Building feature matrix...")
    feature_df = build_feature_matrix(panels, ret_wide)
    n_rows = feature_df.dropna(subset=FEATURE_COLS, how="all").shape[0]
    print(f"  Matrix: {n_rows:,} rows | {len(FEATURE_COLS)} features")

    # ── SPY benchmark ─────────────────────────────────────────────────────────
    spy_raw = yf.download("SPY", start=args.start, end="2025-06-01",
                           interval="1mo", auto_adjust=True, progress=False)["Close"]
    spy_ret = spy_raw.squeeze().pct_change().dropna()
    spy_ret.index = pd.to_datetime(spy_ret.index) + pd.offsets.MonthEnd(0)
    spy_ret = spy_ret.loc[args.start:args.end]
    spy_m   = compute_metrics(spy_ret, hold_months=1)

    # ── Run backtests for a few basket sizes ──────────────────────────────────
    BASKETS = [10, 25, 50]
    all_results = {}

    for b in BASKETS:
        print(f"\n{'═' * 65}")
        curve, feat_imp = run_xgb_backtest(
            feature_df, ret_wide, is_liquid_panel,
            basket=b, retrain_every=retrain_every,
        )
        m = compute_metrics(curve, hold_months=1)
        all_results[b] = {"curve": curve, "metrics": m, "feat_imp": feat_imp}
        curve.to_csv(RESULTS_DIR / f"xgb_top{b}_curve.csv")

    # ── Results table ─────────────────────────────────────────────────────────
    W = 68
    print("\n\n" + "═" * W)
    print("  XGBOOST MODEL — RESULTS  (1-month hold, SPY as benchmark)")
    print(f"  Train window: {TRAIN_WINDOW}m  |  Retrain every: {retrain_every}m")
    print("═" * W)
    rows = [
        ("Ann. Return",  "ann_return",   True),
        ("Ann. Vol",     "ann_vol",      False),
        ("Sharpe",       "sharpe",       False),
        ("Max Drawdown", "max_drawdown", True),
        ("Calmar",       "calmar",       False),
        ("Hit Rate %",   "hit_rate",     False),
        ("t-stat",       "t_stat",       False),
        ("N months",     "n_months",     False),
    ]
    basket_hdrs = "  ".join(f"Top {b:>3}" for b in BASKETS)
    print(f"  {'Metric':<20}  {basket_hdrs}     SPY")
    print("-" * W)
    for label, key, signed in rows:
        vals = []
        for b in BASKETS:
            v = all_results[b]["metrics"].get(key, float("nan"))
            vals.append(f"{v:>+7.2f}" if signed else f"{v:>7.3f}")
        spy_v = spy_m.get(key, float("nan"))
        spy_s = f"{spy_v:>+7.2f}" if signed else f"{spy_v:>7.3f}"
        print(f"  {label:<20}  {'  '.join(vals)}  {spy_s}")
    print("═" * W)

    # ── Feature importance ────────────────────────────────────────────────────
    imp = all_results[BASKETS[1]]["feat_imp"]
    if not imp.empty:
        print(f"\n  Feature importance (Top-{BASKETS[1]} model, avg across retrains):")
        for feat, score in imp.items():
            bar = "█" * int(score * 200)
            print(f"    {feat:<20s}  {score:.4f}  {bar}")

    # ── Save combined summary ─────────────────────────────────────────────────
    summary_rows = []
    for b in BASKETS:
        m = all_results[b]["metrics"]
        summary_rows.append({"basket": b, **m})
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "xgb_summary.csv", index=False)
    print(f"\n  Saved → data/results/xgb_summary.csv")
    print(f"  Saved → data/results/xgb_top{{10,25,50}}_curve.csv")


if __name__ == "__main__":
    main()
