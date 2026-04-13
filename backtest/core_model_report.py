"""
Deck-ready backtest report for the locked core model.

Outputs:
    - data/results/core_model_backtest_report.md
    - data/results/core_model_yearly_returns.csv
    - data/results/core_model_drawdowns.csv
    - data/results/core_model_monthly_curve.csv
    - data/results/core_model_holdings_latest.csv

Run:
    python backtest/core_model_report.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.core_model import ACTIVE_FACTORS, BASKET_SIZE, HOLD_MONTHS, build_core_inputs
from sectorscope.core_strategy import FULL_START, IS_END, OOS_START, active_oos_end, run_backtest_exact, build_spy_benchmark
from sectorscope.metrics import compute_metrics
from sectorscope.modeling import FACTOR_WEIGHTS_OPT
from sectorscope.model_config import CORE_MODEL


RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_monthly_portfolio_returns(ret_wide: pd.DataFrame, log_df: pd.DataFrame, hold_months: int) -> pd.Series:
    if log_df.empty:
        return pd.Series(dtype=float)

    dates = ret_wide.index.sort_values()
    rebal_dates = list(dates[::hold_months])
    # Build a direct map from each rebal_date to the *next* rebal_date.
    # Using the log-row index (i+1) is wrong when some periods produced NaN
    # and were not recorded in log_df — the row index then diverges from the
    # position in rebal_dates, causing wrong holding-window slices.
    rebal_date_to_next: dict[pd.Timestamp, pd.Timestamp] = {
        pd.Timestamp(rd): pd.Timestamp(rebal_dates[j + 1])
        for j, rd in enumerate(rebal_dates[:-1])
    }

    rows = log_df.sort_values("rebal_date").reset_index(drop=True)
    monthly_returns: dict[pd.Timestamp, float] = {}

    for _, row in rows.iterrows():
        rdate = pd.Timestamp(row["rebal_date"]).tz_localize(None) if pd.Timestamp(row["rebal_date"]).tzinfo else pd.Timestamp(row["rebal_date"])
        next_rdate = rebal_date_to_next.get(rdate)
        if next_rdate is None:
            continue
        holdings = [int(x) for x in str(row.get("holdings", "")).split(",") if x]
        if not holdings:
            continue
        period_slice = ret_wide.loc[(ret_wide.index > rdate) & (ret_wide.index <= next_rdate), holdings]
        if period_slice.empty:
            continue
        period_monthly = period_slice.mean(axis=1, skipna=True).dropna()
        for dt, ret in period_monthly.items():
            monthly_returns[pd.Timestamp(dt).tz_localize(None)] = float(ret)

    if not monthly_returns:
        return pd.Series(dtype=float)
    out = pd.Series(monthly_returns).sort_index()
    out.index = pd.DatetimeIndex(out.index).tz_localize(None).to_period("M").to_timestamp("M")
    return out.groupby(level=0).last().sort_index()


def load_name_map() -> tuple[pd.Series, pd.Series]:
    crsp = pd.read_parquet("data/fundamentals/crsp_monthly_returns.parquet", columns=["permno", "date", "ticker"])
    crsp["date"] = pd.to_datetime(crsp["date"])
    permno_to_ticker = crsp.sort_values("date").groupby("permno")["ticker"].last().str.upper()

    names = pd.Series(dtype=object)
    candidates = []
    for p in Path("data/etf_holdings").glob("*_holdings.parquet"):
        try:
            df = pd.read_parquet(p, columns=["ticker", "name"])
            candidates.append(df)
        except Exception:
            continue
    if candidates:
        tmp = pd.concat(candidates, ignore_index=True).dropna(subset=["ticker", "name"])
        names = tmp.assign(ticker=lambda x: x["ticker"].astype(str).str.upper()).drop_duplicates("ticker").set_index("ticker")["name"]
    return permno_to_ticker, names


def annual_return_table(model_monthly: pd.Series, spy_monthly: pd.Series) -> pd.DataFrame:
    df = pd.concat([model_monthly.rename("model"), spy_monthly.rename("spy")], axis=1).dropna(how="all")
    if df.empty:
        return pd.DataFrame()

    out = []
    for year, grp in df.groupby(df.index.year):
        row = {"year": int(year)}
        for col in ["model", "spy"]:
            s = grp[col].dropna()
            row[f"{col}_return"] = round(((1 + s).prod() - 1) * 100, 2) if not s.empty else None
        if row["model_return"] is not None and row["spy_return"] is not None:
            row["alpha_vs_spy"] = round(row["model_return"] - row["spy_return"], 2)
        else:
            row["alpha_vs_spy"] = None
        out.append(row)
    return pd.DataFrame(out)


def realized_metrics(period_returns: pd.Series, monthly_returns: pd.Series, hold_months: int) -> dict:
    pr = period_returns.dropna()
    mr = monthly_returns.dropna()
    if pr.empty:
        return {}

    total_return = ((1 + pr).prod() - 1) * 100
    avg_period = pr.mean() * 100
    median_period = pr.median() * 100
    period_vol = pr.std() * 100
    best_period = pr.max() * 100
    worst_period = pr.min() * 100
    hit_rate = (pr > 0).mean() * 100

    if not mr.empty:
        nav = (1 + mr).cumprod()
        max_dd = ((nav - nav.cummax()) / nav.cummax()).min() * 100
    else:
        nav = (1 + pr).cumprod()
        max_dd = ((nav - nav.cummax()) / nav.cummax()).min() * 100

    return {
        "total_return": round(total_return, 2),
        "avg_2m_return": round(avg_period, 2),
        "median_2m_return": round(median_period, 2),
        "period_vol": round(period_vol, 2),
        "best_2m_return": round(best_period, 2),
        "worst_2m_return": round(worst_period, 2),
        "hit_rate": round(hit_rate, 1),
        "max_drawdown": round(max_dd, 2),
        "n_periods": int(len(pr)),
        "n_months": int(len(pr) * hold_months),
    }


def realized_metrics_frame(model_period: pd.Series, spy_period: pd.Series, model_monthly: pd.Series, spy_monthly: pd.Series, hold_months: int) -> pd.DataFrame:
    model_metrics = realized_metrics(model_period, model_monthly, hold_months)
    spy_metrics = realized_metrics(spy_period, spy_monthly, hold_months)
    keys = [
        "total_return",
        "avg_2m_return",
        "median_2m_return",
        "period_vol",
        "best_2m_return",
        "worst_2m_return",
        "hit_rate",
        "max_drawdown",
        "n_periods",
        "n_months",
    ]
    return pd.DataFrame([{"metric": key, "model": model_metrics.get(key), "spy": spy_metrics.get(key)} for key in keys])


def drawdown_table(monthly_returns: pd.Series, top_n: int = 10) -> pd.DataFrame:
    s = monthly_returns.dropna()
    if s.empty:
        return pd.DataFrame()
    nav = (1 + s).cumprod()
    running_max = nav.cummax()
    dd = nav / running_max - 1

    episodes = []
    in_dd = False
    start = trough = end = None
    trough_dd = 0.0

    for dt, val in dd.items():
        if val < 0 and not in_dd:
            in_dd = True
            start = dt
            trough = dt
            trough_dd = val
        elif val < 0 and in_dd:
            if val < trough_dd:
                trough_dd = val
                trough = dt
        elif val >= 0 and in_dd:
            end = dt
            episodes.append({
                "start": start.strftime("%Y-%m-%d"),
                "trough": trough.strftime("%Y-%m-%d"),
                "recovery": end.strftime("%Y-%m-%d"),
                "drawdown_pct": round(trough_dd * 100, 2),
                "months_to_trough": (trough.to_period("M") - start.to_period("M")).n,
                "months_to_recover": (end.to_period("M") - start.to_period("M")).n,
            })
            in_dd = False

    if in_dd:
        episodes.append({
            "start": start.strftime("%Y-%m-%d"),
            "trough": trough.strftime("%Y-%m-%d"),
            "recovery": "open",
            "drawdown_pct": round(trough_dd * 100, 2),
            "months_to_trough": (trough.to_period("M") - start.to_period("M")).n,
            "months_to_recover": None,
        })

    return pd.DataFrame(episodes).sort_values("drawdown_pct").head(top_n)


def metrics_frame(model: pd.Series, spy: pd.Series, monthly_curve_model: pd.Series, monthly_curve_spy: pd.Series, hold_months: int, min_periods: int) -> pd.DataFrame:
    import math
    model_metrics = compute_metrics(model, hold_months=hold_months, min_periods=min_periods, monthly_curve=monthly_curve_model) or {}
    spy_metrics = compute_metrics(spy, hold_months=hold_months, min_periods=min_periods, monthly_curve=monthly_curve_spy) or {}

    keys = [
        "ann_return", "ann_vol", "sharpe", "sortino", "calmar",
        "max_drawdown", "hit_rate", "t_stat", "n_periods", "n_months",
    ]
    rows = []
    for key in keys:
        model_val = model_metrics.get(key)
        spy_val = spy_metrics.get(key)
        # Sortino is undefined (not zero) when all returns beat the risk-free
        # rate — show "N/A" rather than a blank cell.
        if key == "sortino":
            def _na_if_nan(v):
                if v is None:
                    return "N/A"
                try:
                    return "N/A" if math.isnan(float(v)) else v
                except (TypeError, ValueError):
                    return "N/A"
            model_val = _na_if_nan(model_val)
            spy_val = _na_if_nan(spy_val)
        rows.append({
            "metric": key,
            "model": model_val,
            "spy": spy_val,
        })
    return pd.DataFrame(rows)


def overlap_pair(left: pd.Series, right: pd.Series) -> tuple[pd.Series, pd.Series]:
    pair = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    if pair.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    return pair["left"], pair["right"]


def latest_holdings_table(log_df: pd.DataFrame) -> pd.DataFrame:
    if log_df.empty:
        return pd.DataFrame()
    latest = log_df.sort_values("rebal_date").iloc[-1]
    permnos = [int(x) for x in str(latest.get("holdings", "")).split(",") if x]
    permno_to_ticker, ticker_to_name = load_name_map()
    rows = []
    for permno in permnos:
        ticker = str(permno_to_ticker.get(permno, ""))
        rows.append({
            "rebal_date": pd.to_datetime(latest["rebal_date"]).strftime("%Y-%m-%d"),
            "permno": permno,
            "ticker": ticker,
            "company": ticker_to_name.get(ticker.upper(), ticker if ticker else str(permno)),
        })
    return pd.DataFrame(rows)


def backtest_detail_csv(
    log: pd.DataFrame,
    spy_curve: pd.Series,
    permno_to_ticker: pd.Series,
    ticker_to_name: pd.Series,
    window: str,
) -> pd.DataFrame:
    """
    One row per rebalance period with returns, cumulative NAV, tickers held.

    Columns
    -------
    window              : "IS" or "OOS"
    rebal_date          : portfolio formation date (month-end)
    period_end_date     : last date of holding window (next rebal_date)
    hold_months         : number of calendar months held
    n_stocks            : basket size
    turnover_pct        : one-way turnover vs prior period (%)
    raw_ret_pct         : gross equal-weight period return (%)
    tc_drag_bp          : transaction-cost drag in basis points
    net_ret_pct         : net period return after TC (%)
    spy_ret_pct         : SPY return over same window (%)
    excess_ret_pct      : net_ret - spy_ret (%)
    model_nav           : cumulative NAV of model (starts at 1)
    spy_nav             : cumulative NAV of SPY (starts at 1)
    tickers             : comma-separated list of ticker symbols held
    """
    if log.empty:
        return pd.DataFrame()

    rows = []
    model_nav = 1.0
    spy_nav = 1.0

    for _, row in log.sort_values("rebal_date").iterrows():
        rdate = pd.Timestamp(row["rebal_date"])
        net_ret = row["period_ret"] / 100          # decimal
        raw_ret = row.get("raw_ret", row["period_ret"]) / 100
        tc_bp   = row.get("tc_drag_bp", 0.0)

        # SPY return for this window
        spy_key = rdate
        spy_ret = float(spy_curve.get(spy_key, float("nan")))

        # Cumulative NAVs
        model_nav *= 1 + net_ret
        if not pd.isna(spy_ret):
            spy_nav *= 1 + spy_ret
        spy_nav_out = spy_nav if not pd.isna(spy_ret) else float("nan")

        # Resolve tickers
        permnos = [int(x) for x in str(row.get("holdings", "")).split(",") if x]
        tickers = [str(permno_to_ticker.get(p, p)) for p in permnos]

        # Infer period_end_date from the next row's rebal_date (or leave blank for last)
        rows.append({
            "window":           window,
            "rebal_date":       rdate.strftime("%Y-%m-%d"),
            "_rdate_ts":        rdate,
            "n_stocks":         int(row.get("n_stocks", len(permnos))),
            "turnover_pct":     float(row.get("turnover", float("nan"))),
            "raw_ret_pct":      round(raw_ret * 100, 4),
            "tc_drag_bp":       round(float(tc_bp), 1),
            "net_ret_pct":      round(net_ret * 100, 4),
            "spy_ret_pct":      round(spy_ret * 100, 4) if not pd.isna(spy_ret) else float("nan"),
            "excess_ret_pct":   round((net_ret - spy_ret) * 100, 4) if not pd.isna(spy_ret) else float("nan"),
            "model_nav":        round(model_nav, 6),
            "spy_nav":          round(spy_nav_out, 6),
            "tickers":          ", ".join(tickers),
        })

    # Fill in period_end_date (= next row's rebal_date)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("_rdate_ts").reset_index(drop=True)
    out["period_end_date"] = out["_rdate_ts"].shift(-1).dt.strftime("%Y-%m-%d").fillna("")
    out = out.drop(columns=["_rdate_ts"])

    # Reorder columns
    cols = [
        "window", "rebal_date", "period_end_date",
        "n_stocks", "turnover_pct", "raw_ret_pct", "tc_drag_bp",
        "net_ret_pct", "spy_ret_pct", "excess_ret_pct",
        "model_nav", "spy_nav", "tickers",
    ]
    return out[cols]


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"
    tmp = df.copy()
    tmp = tmp.where(pd.notnull(tmp), "")
    headers = [str(c) for c in tmp.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in tmp.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def main():
    oos_end = active_oos_end()
    ret_wide, panels, is_liq = build_core_inputs(oos_end=oos_end)
    is_idx = ret_wide.index[ret_wide.index <= IS_END]
    oos_idx = ret_wide.index[(ret_wide.index >= OOS_START) & (ret_wide.index <= oos_end)]

    curve_is, log_is = run_backtest_exact(
        ret_wide.reindex(is_idx),
        {k: v.reindex(is_idx) for k, v in panels.items()},
        is_liq.reindex(is_idx),
        basket_size=BASKET_SIZE,
        hold_months=HOLD_MONTHS,
        weights=FACTOR_WEIGHTS_OPT,
        score_weighted=False,
    )
    curve_oos, log_oos = run_backtest_exact(
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
    model_monthly_oos = build_monthly_portfolio_returns(ret_wide.reindex(oos_idx), log_oos, HOLD_MONTHS)
    spy_monthly = pd.read_parquet("data/prices/SPY.parquet")
    spy_dates = pd.to_datetime(spy_monthly["date"]) if "date" in spy_monthly.columns else pd.to_datetime(spy_monthly.index)
    spy_dates = pd.DatetimeIndex(spy_dates).tz_localize(None)
    spy_monthly = pd.Series(spy_monthly["close"].values, index=spy_dates).sort_index().resample("ME").last().pct_change()
    spy_monthly.index = pd.to_datetime(spy_monthly.index) + pd.offsets.MonthEnd(0)
    spy_monthly_oos = spy_monthly.reindex(model_monthly_oos.index)
    curve_is_aligned, spy_is_aligned = overlap_pair(curve_is, spy_is)
    curve_oos_aligned, spy_oos_aligned = overlap_pair(curve_oos, spy_oos)
    model_monthly_aligned, spy_monthly_aligned = overlap_pair(model_monthly_oos, spy_monthly_oos)
    curve_is_live = curve_is.dropna()
    curve_oos_live = curve_oos.dropna()
    effective_is_window = (
        f"{curve_is_live.index.min().strftime('%Y-%m-%d')} to {curve_is_live.index.max().strftime('%Y-%m-%d')}"
        if not curve_is_live.empty else "n/a"
    )
    effective_oos_window = (
        f"{curve_oos_live.index.min().strftime('%Y-%m-%d')} to {curve_oos_live.index.max().strftime('%Y-%m-%d')}"
        if not curve_oos_live.empty else "n/a"
    )

    is_realized = realized_metrics_frame(curve_is_aligned, spy_is_aligned, pd.Series(dtype=float), pd.Series(dtype=float), HOLD_MONTHS)
    oos_realized = realized_metrics_frame(curve_oos_aligned, spy_oos_aligned, model_monthly_aligned, spy_monthly_aligned, HOLD_MONTHS)
    monthly_realized = realized_metrics_frame(model_monthly_aligned, spy_monthly_aligned, model_monthly_aligned, spy_monthly_aligned, 1)
    is_metrics = metrics_frame(curve_is_aligned, spy_is_aligned, pd.Series(dtype=float), pd.Series(dtype=float), HOLD_MONTHS, 4)
    oos_metrics = metrics_frame(curve_oos_aligned, spy_oos_aligned, model_monthly_aligned, spy_monthly_aligned, HOLD_MONTHS, 2)
    monthly_metrics = metrics_frame(model_monthly_aligned, spy_monthly_aligned, model_monthly_aligned, spy_monthly_aligned, 1, 3)

    yearly = annual_return_table(model_monthly_aligned, spy_monthly_aligned)
    drawdowns = drawdown_table(model_monthly_aligned, top_n=10)
    latest_holdings = latest_holdings_table(log_oos)
    permno_to_ticker_map, ticker_to_name_map = load_name_map()
    detail_is = backtest_detail_csv(log_is, spy_is, permno_to_ticker_map, ticker_to_name_map, "IS")
    detail_oos = backtest_detail_csv(log_oos, spy_oos, permno_to_ticker_map, ticker_to_name_map, "OOS")
    backtest_detail = pd.concat([detail_is, detail_oos], ignore_index=True)

    monthly_curve = pd.concat(
        [
            model_monthly_oos.rename("model_monthly_return"),
            spy_monthly_oos.rename("spy_monthly_return"),
        ],
        axis=1,
    ).reset_index(names="date")

    robustness_path = RESULTS_DIR / "core_model_robustness.csv"
    robustness_text = ""
    if robustness_path.exists():
        rob = pd.read_csv(robustness_path)
        picked = rob[(rob["model"] == "three_factor") & (rob["basket_size"] == BASKET_SIZE) & (rob["hold_months"] == HOLD_MONTHS)]
        ranked = rob.sort_values(["post_shock_sharpe", "oos_sharpe", "is_sharpe"], ascending=False).reset_index(drop=True)
        if not picked.empty:
            rank = int(ranked.index[(ranked["model"] == "three_factor") & (ranked["basket_size"] == BASKET_SIZE) & (ranked["hold_months"] == HOLD_MONTHS)][0] + 1)
            robustness_text = (
                f"Locked specification ranks #{rank} in the current robustness grid "
                f"when ordered by post-shock Sharpe, then OOS Sharpe, then IS Sharpe."
            )

    report = "\n".join([
        f"# {CORE_MODEL.name} Backtest Report",
        "",
        "## Model Snapshot",
        f"- Model: {CORE_MODEL.description}",
        f"- Factors: {', '.join(ACTIVE_FACTORS)}",
        f"- Portfolio construction: {BASKET_SIZE} stocks, {HOLD_MONTHS}-month hold, top {CORE_MODEL.universe_size} by market cap",
        f"- Benchmark: {CORE_MODEL.benchmark}",
        f"- In-sample window: {FULL_START} to {IS_END}",
        f"- Out-of-sample window: {OOS_START} to {oos_end}",
        f"- Effective offline/local IS window with current sector ETF files: {effective_is_window}",
        f"- Effective offline/local OOS window: {effective_oos_window}",
        "",
        "## Executive Takeaways",
        f"- The locked three-factor specification is the active production research model used in the dashboard and core backtest path.",
        f"- OOS performance should be interpreted in the context of the 2025 tariff shock regime break; post-shock behavior has been materially stronger than the early-2025 tape.",
        f"- This report includes both rebalance-frequency metrics and a monthly path reconstruction for slide-deck presentation.",
        "- The current offline/local run is history-limited by local sector ETF price coverage beginning in 2024, so the IS block should be treated as a short validation window rather than a full historical training-era backtest.",
        f"- {robustness_text}" if robustness_text else "- Robustness ranking was unavailable because `data/results/core_model_robustness.csv` was not present.",
        "",
        "## In-Sample Realized Portfolio Metrics",
        markdown_table(is_realized),
        "",
        "## Out-of-Sample Realized Portfolio Metrics",
        markdown_table(oos_realized),
        "",
        "## Monthly Path Realized Metrics",
        markdown_table(monthly_realized),
        "",
        "## In-Sample Annualized Metrics",
        markdown_table(is_metrics),
        "",
        "## Out-of-Sample Annualized Metrics",
        markdown_table(oos_metrics),
        "",
        "## Monthly Path Annualized Metrics",
        markdown_table(monthly_metrics),
        "",
        "## Calendar-Year Returns",
        markdown_table(yearly),
        "",
        "## Worst Drawdowns",
        markdown_table(drawdowns),
        "",
        "## Latest Portfolio",
        markdown_table(latest_holdings),
        "",
        "## Notes For Slides",
        "- Use the Realized Portfolio Metrics sections as the primary deck numbers. They describe the actual realized 2-month portfolio path over the test window.",
        "- Use the Annualized Metrics sections only as secondary context, especially when the sample length is short.",
        "- The portfolio mechanically rebalances every 2 months and holds that basket through the full holding window.",
        "- If presenting 2025, explicitly call out the tariff-shock distortion and the stronger post-shock recovery regime.",
    ])

    (RESULTS_DIR / "core_model_backtest_report.md").write_text(report + "\n")
    yearly.to_csv(RESULTS_DIR / "core_model_yearly_returns.csv", index=False)
    drawdowns.to_csv(RESULTS_DIR / "core_model_drawdowns.csv", index=False)
    monthly_curve.to_csv(RESULTS_DIR / "core_model_monthly_curve.csv", index=False)
    latest_holdings.to_csv(RESULTS_DIR / "core_model_holdings_latest.csv", index=False)
    backtest_detail.to_csv(RESULTS_DIR / "core_model_backtest_detail.csv", index=False)

    # ── Portfolio summary CSV ─────────────────────────────────────────────────
    def _metrics_dict(m: dict, prefix: str) -> dict:
        return {
            f"{prefix}_ann_return":   m.get("ann_return"),
            f"{prefix}_ann_vol":      m.get("ann_vol"),
            f"{prefix}_sharpe":       m.get("sharpe"),
            f"{prefix}_sortino":      m.get("sortino"),
            f"{prefix}_calmar":       m.get("calmar"),
            f"{prefix}_max_drawdown": m.get("max_drawdown"),
            f"{prefix}_hit_rate":     m.get("hit_rate"),
            f"{prefix}_total_return": None,  # filled below
            f"{prefix}_n_periods":    m.get("n_periods"),
            f"{prefix}_n_months":     m.get("n_months"),
        }

    import math

    def _metrics_to_dict(m: dict, prefix: str, total_ret: float | None = None) -> dict:
        sortino = m.get("sortino")
        sortino_val = None if (sortino is None or (isinstance(sortino, float) and math.isnan(sortino))) else sortino
        return {
            f"{prefix}_total_return":   total_ret,
            f"{prefix}_ann_return":     m.get("ann_return"),
            f"{prefix}_ann_vol":        m.get("ann_vol"),
            f"{prefix}_sharpe":         m.get("sharpe"),
            f"{prefix}_sortino":        sortino_val,
            f"{prefix}_calmar":         m.get("calmar"),
            f"{prefix}_max_drawdown":   m.get("max_drawdown"),
            f"{prefix}_hit_rate":       m.get("hit_rate"),
            f"{prefix}_n_periods":      m.get("n_periods"),
            f"{prefix}_n_months":       m.get("n_months"),
        }

    from sectorscope.metrics import compute_metrics

    def _total_ret(s: pd.Series) -> float | None:
        s = s.dropna()
        return round(((1 + s).prod() - 1) * 100, 2) if not s.empty else None

    summary_rows = []
    for label, model_s, spy_s, monthly_m, monthly_s, hm, min_p in [
        ("IS",           curve_is_aligned,  spy_is_aligned,  pd.Series(dtype=float), pd.Series(dtype=float), HOLD_MONTHS, 4),
        ("OOS",          curve_oos_aligned, spy_oos_aligned, model_monthly_aligned,  spy_monthly_aligned,    HOLD_MONTHS, 2),
        ("OOS_monthly",  model_monthly_aligned, spy_monthly_aligned, model_monthly_aligned, spy_monthly_aligned, 1, 3),
    ]:
        mm = compute_metrics(model_s, hold_months=hm, min_periods=min_p, monthly_curve=monthly_m) or {}
        sm = compute_metrics(spy_s,   hold_months=hm, min_periods=min_p, monthly_curve=monthly_s) or {}
        row = {"window": label}
        row.update(_metrics_to_dict(mm, "model", _total_ret(model_s)))
        row.update(_metrics_to_dict(sm, "spy",   _total_ret(spy_s)))
        summary_rows.append(row)

    portfolio_summary = pd.DataFrame(summary_rows)
    portfolio_summary.to_csv(RESULTS_DIR / "core_model_portfolio_summary.csv", index=False)

    print(f"Saved → {RESULTS_DIR / 'core_model_backtest_report.md'}")
    print(f"Saved → {RESULTS_DIR / 'core_model_yearly_returns.csv'}")
    print(f"Saved → {RESULTS_DIR / 'core_model_drawdowns.csv'}")
    print(f"Saved → {RESULTS_DIR / 'core_model_monthly_curve.csv'}")
    print(f"Saved → {RESULTS_DIR / 'core_model_holdings_latest.csv'}")
    print(f"Saved → {RESULTS_DIR / 'core_model_backtest_detail.csv'}")
    print(f"Saved → {RESULTS_DIR / 'core_model_portfolio_summary.csv'}")


if __name__ == "__main__":
    main()
