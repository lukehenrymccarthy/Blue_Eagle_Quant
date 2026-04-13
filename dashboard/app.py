"""
dashboard/app.py
----------------
Single-page dashboard for the locked core equity model.

Run:
    python dashboard/app.py
"""

import glob
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from nicegui import ui

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.core_model import ACTIVE_FACTORS, BASKET_SIZE, HOLD_MONTHS, build_core_inputs
from sectorscope.core_strategy import (
    OOS_START,
    active_oos_end,
    run_backtest_exact,
)
from sectorscope.metrics import compute_metrics
from sectorscope.modeling import FACTOR_WEIGHTS_OPT
from sectorscope.utils import sic_to_etf as _sic_to_etf, zscore as _zscore


SECTOR_NAMES = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLE": "Energy",
    "XLP": "Consumer Staples",
    "XLY": "Consumer Discretionary",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLC": "Communication Services",
}

BG = "#f5f7f8"
PANEL = "#ffffff"
BORDER = "#d8e0e5"
TEXT = "#17212b"
MUTED = "#647684"
ACCENT = "#0f5f8c"
GREEN = "#137333"
RED = "#b42318"


def build_name_map() -> tuple[pd.Series, dict[str, str]]:
    crsp = pd.read_parquet("data/fundamentals/crsp_monthly_returns.parquet", columns=["permno", "date", "ticker"])
    crsp["date"] = pd.to_datetime(crsp["date"])
    permno_to_ticker = crsp.sort_values("date").groupby("permno")["ticker"].last().str.upper()

    ticker_to_name: dict[str, str] = {}
    for path in glob.glob("data/etf_holdings/*_holdings.parquet"):
        try:
            df = pd.read_parquet(path, columns=["ticker", "name"])
            for _, row in df.dropna(subset=["ticker", "name"]).iterrows():
                ticker_to_name[str(row["ticker"]).upper()] = str(row["name"]).strip()
        except Exception:
            continue
    return permno_to_ticker, ticker_to_name


def compound_return(series: pd.Series, tail_n: int | None = None) -> float | None:
    s = series.dropna()
    if tail_n is not None:
        s = s.tail(tail_n)
    if s.empty:
        return None
    return round(((1 + s).prod() - 1) * 100, 2)


def compound_return_from_date(series: pd.Series, start_date: pd.Timestamp) -> float | None:
    s = series.dropna()
    if s.empty:
        return None
    s = s.loc[s.index >= start_date]
    if s.empty:
        return None
    return round(((1 + s).prod() - 1) * 100, 2)


def load_local_spy_monthly_returns() -> pd.Series:
    path = Path("data/prices/SPY.parquet")
    if not path.exists():
        return pd.Series(dtype=float)

    spy = pd.read_parquet(path)
    if "close" not in spy.columns:
        return pd.Series(dtype=float)

    if "date" in spy.columns:
        dates = pd.to_datetime(spy["date"])
    else:
        dates = pd.to_datetime(spy.index)

    px = pd.Series(spy["close"].values, index=dates).sort_index()
    monthly = px.resample("ME").last().pct_change()
    monthly.index = pd.to_datetime(monthly.index) + pd.offsets.MonthEnd(0)
    return monthly


def build_monthly_portfolio_returns(ret_wide: pd.DataFrame, log_oos: pd.DataFrame, hold_months: int) -> pd.Series:
    if log_oos.empty:
        return pd.Series(dtype=float)

    dates = ret_wide.index.sort_values()
    rebal_dates = list(dates[::hold_months])
    rows = log_oos.sort_values("rebal_date").reset_index(drop=True)
    monthly_returns = {}

    for i, row in rows.iterrows():
        if i + 1 >= len(rebal_dates):
            break
        rdate = pd.to_datetime(row["rebal_date"])
        next_rdate = pd.to_datetime(rebal_dates[i + 1])

        holdings = []
        for raw in str(row.get("holdings", "")).split(","):
            try:
                holdings.append(int(raw))
            except Exception:
                continue
        if not holdings:
            continue

        period_slice = ret_wide.loc[(ret_wide.index > rdate) & (ret_wide.index <= next_rdate), holdings]
        if period_slice.empty:
            continue

        period_monthly = period_slice.mean(axis=1, skipna=True).dropna()
        for dt, ret in period_monthly.items():
            monthly_returns[pd.Timestamp(dt)] = float(ret)

    if not monthly_returns:
        return pd.Series(dtype=float)
    return pd.Series(monthly_returns).sort_index()


def build_period_benchmark(monthly_returns: pd.Series, dates: pd.Index, hold_months: int) -> pd.Series:
    if monthly_returns.empty:
        return pd.Series(dtype=float)

    rebal_dates = dates[::hold_months]
    bench_rets = []
    bench_idx = []
    for i, rd in enumerate(rebal_dates[:-1]):
        nrd = rebal_dates[i + 1]
        sl = monthly_returns.loc[(monthly_returns.index > rd) & (monthly_returns.index <= nrd)]
        bench_rets.append((1 + sl.fillna(0)).prod() - 1)
        bench_idx.append(rd)
    return pd.Series(bench_rets, index=bench_idx)


def build_chart_rows(portfolio_monthly: pd.Series, spy_monthly: pd.Series) -> list[dict]:
    chart = pd.concat(
        [
            portfolio_monthly.rename("portfolio_ret"),
            spy_monthly.rename("spy_ret"),
        ],
        axis=1,
    ).sort_index()
    chart = chart.dropna(subset=["portfolio_ret", "spy_ret"], how="any")
    rows = []
    for dt, row in chart.iterrows():
        rows.append({
            "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
            "portfolio_ret": float(row["portfolio_ret"]),
            "spy_ret": float(row["spy_ret"]),
        })
    return rows


def filter_chart_rows(rows: list[dict], window: str) -> list[dict]:
    if not rows:
        return []
    dates = pd.to_datetime([row["date"] for row in rows])
    end = dates.max()

    if window == "1M":
        start = end - pd.DateOffset(months=1)
    elif window == "6M":
        start = end - pd.DateOffset(months=6)
    elif window == "1Y":
        start = end - pd.DateOffset(years=1)
    elif window == "YTD":
        start = pd.Timestamp(year=end.year, month=1, day=1)
    elif window == "5Y":
        start = end - pd.DateOffset(years=5)
    else:
        start = dates.min()

    filtered = [row.copy() for row in rows if pd.Timestamp(row["date"]) >= start]
    if not filtered:
        filtered = [rows[-1].copy()]

    portfolio_level = 100.0
    spy_level = 100.0
    for row in filtered:
        port_ret = row.get("portfolio_ret")
        spy_ret = row.get("spy_ret")
        if port_ret is not None:
            portfolio_level *= 1 + port_ret
        if spy_ret is not None:
            spy_level *= 1 + spy_ret
        row["portfolio"] = round(portfolio_level, 2) if port_ret is not None else None
        row["spy"] = round(spy_level, 2) if spy_ret is not None else None
    return filtered


def build_dashboard_data():
    oos_end = active_oos_end()
    ret_wide, panels, is_liq = build_core_inputs(oos_end=oos_end)
    oos_idx = ret_wide.index[(ret_wide.index >= OOS_START) & (ret_wide.index <= oos_end)]

    curve_oos, log_oos = run_backtest_exact(
        ret_wide.reindex(oos_idx),
        {k: v.reindex(oos_idx) for k, v in panels.items()},
        is_liq.reindex(oos_idx),
        basket_size=BASKET_SIZE,
        hold_months=HOLD_MONTHS,
        weights=FACTOR_WEIGHTS_OPT,
        score_weighted=False,
    )
    portfolio_monthly = build_monthly_portfolio_returns(ret_wide.reindex(oos_idx), log_oos, HOLD_MONTHS)
    spy_monthly = load_local_spy_monthly_returns()
    chart_rows = build_chart_rows(portfolio_monthly, spy_monthly)
    chart_index = pd.to_datetime([row["date"] for row in chart_rows]) if chart_rows else pd.DatetimeIndex([])
    spy_monthly_overlap = spy_monthly.reindex(chart_index) if len(chart_index) else pd.Series(dtype=float)
    spy_oos = build_period_benchmark(
        load_local_spy_monthly_returns().reindex(oos_idx),
        ret_wide.reindex(oos_idx).index,
        HOLD_MONTHS,
    )

    full_log = log_oos.assign(window="Live").copy()
    full_log["rebal_date"] = pd.to_datetime(full_log["rebal_date"])
    full_log = full_log.sort_values("rebal_date", ascending=False).reset_index(drop=True)

    fund = pd.read_parquet("data/fundamentals/compustat_quarterly.parquet")
    sic_col = "siccd" if "siccd" in fund.columns else "sic"
    latest_sic = (
        fund.assign(permno=pd.to_numeric(fund["permno"], errors="coerce"))
        .dropna(subset=["permno"])
        .sort_values("datadate")
        .groupby("permno")[sic_col]
        .last()
    )
    sector_map = latest_sic.apply(_sic_to_etf)
    permno_to_ticker, ticker_to_name = build_name_map()
    factor_order = list(ACTIVE_FACTORS)
    weight_map = {k: FACTOR_WEIGHTS_OPT[k] for k in factor_order}
    weight_sum = sum(weight_map.values())
    weight_map = {k: v / weight_sum for k, v in weight_map.items()}

    scored_books: dict[pd.Timestamp, dict[int, dict]] = {}
    dates = ret_wide.reindex(oos_idx).index
    rebal_dates = dates[::HOLD_MONTHS]
    for rdate in rebal_dates[:-1]:
        universe = set(ret_wide.columns)
        if not is_liq.empty and rdate in is_liq.index:
            liq_row = is_liq.loc[rdate].fillna(False)
            universe = universe.intersection(liq_row[liq_row].index)

        scored = {}
        for fn in factor_order:
            panel = panels[fn]
            if rdate not in panel.index:
                continue
            row = panel.loc[rdate].dropna().replace([np.inf, -np.inf], np.nan).dropna()
            row = row[row.index.isin(universe)]
            z = _zscore(row)
            if len(z) > BASKET_SIZE:
                scored[fn] = z
        if len(scored) != len(factor_order):
            continue

        common = list(scored.values())[0].index
        for z in scored.values():
            common = common.intersection(z.index)
        if len(common) < BASKET_SIZE:
            continue

        mat = np.column_stack([scored[fn].reindex(common).values for fn in factor_order])
        w_arr = np.array([weight_map[fn] for fn in factor_order], dtype=float)
        composite = pd.Series(mat @ w_arr, index=common).sort_values(ascending=False)

        book = {}
        for permno in composite.head(BASKET_SIZE).index:
            book[int(permno)] = {
                "composite": float(composite.loc[permno]),
                "factors": {fn: float(scored[fn].loc[permno]) for fn in factor_order},
            }
        scored_books[pd.Timestamp(rdate)] = book

    def parse_holdings(row):
        if not isinstance(row.get("holdings"), str) or not row["holdings"]:
            return []
        out = []
        for raw in row["holdings"].split(","):
            try:
                permno = int(raw)
            except Exception:
                continue
            ticker = permno_to_ticker.get(permno, "")
            company = ticker_to_name.get(str(ticker).upper(), str(ticker).upper() if ticker else str(permno))
            sector_code = sector_map.get(permno)
            factor_snapshot = scored_books.get(pd.Timestamp(row["rebal_date"]), {}).get(permno, {})
            out.append({
                "permno": permno,
                "ticker": str(ticker).upper() if ticker else "",
                "company": company,
                "sector_name": SECTOR_NAMES.get(sector_code, "Unknown"),
                "composite": factor_snapshot.get("composite"),
                "factors": factor_snapshot.get("factors", {}),
            })
        return out

    full_log["parsed_holdings"] = full_log.apply(parse_holdings, axis=1)

    def calc_changes(df):
        prev = None
        changes = []
        for _, row in df.sort_values("rebal_date").iterrows():
            current = {h["permno"] for h in row["parsed_holdings"]}
            added = current if prev is None else current - prev
            removed = set() if prev is None else prev - current
            changes.append((row["rebal_date"], added, removed))
            prev = current
        cmap = {dt: {"added": added, "removed": removed} for dt, added, removed in changes}
        return df["rebal_date"].map(cmap)

    full_log["changes"] = calc_changes(full_log)
    latest = full_log.iloc[0] if not full_log.empty else None
    latest_chart_date = chart_index.max() if len(chart_index) else None
    ytd_start = (
        pd.Timestamp(year=latest_chart_date.year, month=1, day=1)
        if latest_chart_date is not None else None
    )

    portfolio_risk = compute_metrics(
        portfolio_monthly.reindex(chart_index) if len(chart_index) else portfolio_monthly,
        hold_months=1,
        min_periods=3,
        monthly_curve=portfolio_monthly.reindex(chart_index) if len(chart_index) else portfolio_monthly,
    ) or {}
    spy_risk = compute_metrics(
        spy_monthly_overlap,
        hold_months=1,
        min_periods=3,
        monthly_curve=spy_monthly_overlap,
    ) or {}

    recent = {
        "last_period_return": latest.get("period_ret") if latest is not None else None,
        "last_turnover": latest.get("turnover") if latest is not None else None,
        "last_raw_return": latest.get("raw_ret") if latest is not None else None,
        "ytd_return": compound_return_from_date(portfolio_monthly.reindex(chart_index), ytd_start) if ytd_start is not None and len(chart_index) else None,
        "six_month_return": compound_return(portfolio_monthly.reindex(chart_index), tail_n=6) if len(chart_index) else None,
        "spy_ytd_return": compound_return_from_date(spy_monthly_overlap, ytd_start) if ytd_start is not None and not spy_monthly_overlap.empty else None,
        "spy_six_month_return": compound_return(spy_monthly_overlap, tail_n=6) if not spy_monthly_overlap.empty else None,
        "portfolio_risk": portfolio_risk,
        "spy_risk": spy_risk,
    }

    return {
        "latest": latest,
        "full_log": full_log,
        "recent": recent,
        "chart_rows": chart_rows,
    }


DATA = build_dashboard_data()


def section(title: str):
    card = ui.column().classes("w-full gap-3").style(
        f"background:{PANEL}; border:1px solid {BORDER}; border-radius:16px; padding:18px;"
    )
    with card:
        ui.label(title).style(f"color:{TEXT}; font-size:16px; font-weight:700;")
    return card


def stat(label: str, value: str, color: str = TEXT):
    with ui.column().classes("gap-0"):
        ui.label(label).style(f"color:{MUTED}; font-size:11px; text-transform:uppercase; letter-spacing:0.06em;")
        ui.label(value).style(f"color:{color}; font-size:24px; font-weight:700;")


def render_holdings(container, holdings, added=None, removed=None):
    added = added or set()
    removed = removed or set()
    with container:
        if not holdings:
            ui.label("No holdings available.").style(f"color:{MUTED};")
            return
        for idx, h in enumerate(holdings, start=1):
            tag = ""
            tag_color = MUTED
            if h["permno"] in added:
                tag = "Added"
                tag_color = GREEN
            elif h["permno"] in removed:
                tag = "Removed"
                tag_color = RED
            with ui.row().classes("w-full items-start justify-between").style(
                f"padding:12px 0; border-bottom:1px solid {BORDER}; gap: 12px;"
            ):
                with ui.row().classes("items-start").style("gap: 12px;"):
                    ui.label(f"{idx:02d}").style(f"color:{MUTED}; width: 24px; font-size:12px;")
                    with ui.column().classes("gap-0"):
                        ui.label(h["company"]).style(f"color:{TEXT}; font-size:15px; font-weight:700;")
                        meta = " · ".join(x for x in [h["ticker"], h["sector_name"]] if x)
                        ui.label(meta).style(f"color:{MUTED}; font-size:12px;")
                        factor_bits = []
                        for fn in ACTIVE_FACTORS:
                            val = h.get("factors", {}).get(fn)
                            if val is not None:
                                factor_bits.append(f"{fn} {val:+.2f}")
                        if h.get("composite") is not None:
                            factor_bits.append(f"composite {h['composite']:+.2f}")
                        if factor_bits:
                            ui.label(" · ".join(factor_bits)).style(f"color:{MUTED}; font-size:11px;")
                if tag:
                    ui.label(tag).style(f"color:{tag_color}; font-size:12px; font-weight:700;")


def format_change_names(permnos, current_holdings, prior_holdings):
    if not permnos:
        return "None"

    name_map = {}
    for h in current_holdings or []:
        name_map[h["permno"]] = h["company"]
    for h in prior_holdings or []:
        name_map[h["permno"]] = h["company"]

    labels = []
    for permno in sorted(permnos):
        company = name_map.get(permno)
        labels.append(company if company else str(permno))
    return ", ".join(labels)


def render_performance_chart(chart_rows: list[dict]):
    windows = ["1M", "6M", "1Y", "YTD", "5Y"]
    default_window = "1Y" if len(chart_rows) > 6 else "YTD"
    state = {"window": default_window}

    with ui.column().classes("w-full gap-3"):
        with ui.row().classes("w-full items-center justify-between"):
            with ui.column().classes("gap-0"):
                ui.label("Performance").style(f"color:{TEXT}; font-size:16px; font-weight:700;")
                ui.label("Portfolio vs SPY, indexed to 100").style(f"color:{MUTED}; font-size:12px;")
            with ui.row().classes("items-center").style("gap: 8px; flex-wrap: wrap;") as button_row:
                buttons = {}
                for window in windows:
                    buttons[window] = ui.button(
                        window,
                        on_click=lambda e, w=window: update_chart(w),
                    ).props("unelevated")

        chart = ui.echart({
            "animation": False,
            "grid": {"left": 48, "right": 18, "top": 12, "bottom": 36},
            "tooltip": {"trigger": "axis"},
            "legend": {"bottom": 0, "textStyle": {"color": MUTED}},
            "xAxis": {"type": "category", "boundaryGap": False, "data": []},
            "yAxis": {
                "type": "value",
                "scale": True,
                "axisLabel": {"color": MUTED},
                "splitLine": {"lineStyle": {"color": "#e6ecef"}},
            },
            "series": [],
        }).classes("w-full").style("height: 320px;")

        def style_buttons():
            for window, button in buttons.items():
                if window == state["window"]:
                    button.style(
                        f"background:{TEXT}; color:#ffffff; border:1px solid {TEXT}; border-radius:999px; padding:0 12px;"
                    )
                else:
                    button.style(
                        f"background:{PANEL}; color:{MUTED}; border:1px solid {BORDER}; border-radius:999px; padding:0 12px;"
                    )

        def update_chart(window: str):
            state["window"] = window
            filtered = filter_chart_rows(chart_rows, window)
            chart.options["xAxis"]["data"] = [row["date"] for row in filtered]
            chart.options["series"] = [
                {
                    "name": "Portfolio",
                    "type": "line",
                    "smooth": True,
                    "showSymbol": False,
                    "lineStyle": {"width": 2.5, "color": ACCENT},
                    "data": [row["portfolio"] for row in filtered],
                },
                {
                    "name": "SPY",
                    "type": "line",
                    "smooth": True,
                    "showSymbol": False,
                    "lineStyle": {"width": 2.0, "color": "#9aa8b3"},
                    "data": [row["spy"] for row in filtered],
                },
            ]
            chart.update()
            style_buttons()

        update_chart(default_window)


@ui.page("/")
def main_page():
    ui.query("body").style(f"background:{BG}; color:{TEXT}; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;")

    latest = DATA["latest"]
    recent = DATA["recent"]
    full_log = DATA["full_log"]
    chart_rows = DATA["chart_rows"]

    with ui.column().classes("w-full").style("max-width: 1320px; margin: 0 auto; padding: 28px 20px 40px; gap: 16px;"):
        with ui.column().classes("w-full gap-2").style(
            f"background:{PANEL}; border:1px solid {BORDER}; border-radius:20px; padding:24px;"
        ):
            ui.label("Core Equity Model").style(f"color:{TEXT}; font-size:34px; font-weight:800;")
            ui.label("Current portfolio, recent performance, and rebalance history").style(
                f"color:{MUTED}; font-size:15px;"
            )
            ui.label("3 factors · 10 stocks · 2-month hold").style(
                f"color:{ACCENT}; font-size:13px; font-weight:700;"
            )

        with section("Portfolio Performance"):
            render_performance_chart(chart_rows)

        with ui.row().classes("w-full").style("gap: 16px; align-items: stretch;"):
            with section("Recent Portfolio Metrics").style("flex: 1;"):
                with ui.grid(columns=4).classes("w-full").style("gap: 18px;"):
                    stat("Last Period Return", f"{recent['last_period_return']:+.2f}%" if recent["last_period_return"] is not None else "n/a",
                         GREEN if (recent["last_period_return"] or 0) >= 0 else RED)
                    stat("Last Raw Return", f"{recent['last_raw_return']:+.2f}%" if recent["last_raw_return"] is not None else "n/a",
                         GREEN if (recent["last_raw_return"] or 0) >= 0 else RED)
                    stat("YTD Return", f"{recent['ytd_return']:+.2f}%" if recent["ytd_return"] is not None else "n/a",
                         GREEN if (recent["ytd_return"] or 0) >= 0 else RED)
                    stat("6-Month Return", f"{recent['six_month_return']:+.2f}%" if recent["six_month_return"] is not None else "n/a",
                         GREEN if (recent["six_month_return"] or 0) >= 0 else RED)
                    stat("Turnover", f"{recent['last_turnover']:.1f}%" if recent["last_turnover"] is not None else "n/a")
                    stat("Recent Sharpe", f"{recent['portfolio_risk'].get('sharpe', float('nan')):.3f}" if recent["portfolio_risk"] else "n/a")
                    stat("Recent Sortino", f"{recent['portfolio_risk'].get('sortino', float('nan')):.3f}" if recent["portfolio_risk"] else "n/a")
                    stat("Recent Calmar", f"{recent['portfolio_risk'].get('calmar', float('nan')):.3f}" if recent["portfolio_risk"] else "n/a")
            with section("Recent SPY Comparison").style("flex: 1;"):
                with ui.grid(columns=4).classes("w-full").style("gap: 18px;"):
                    stat("SPY YTD Return", f"{recent['spy_ytd_return']:+.2f}%" if recent["spy_ytd_return"] is not None else "n/a",
                         GREEN if (recent["spy_ytd_return"] or 0) >= 0 else RED)
                    stat("SPY 6-Month Return", f"{recent['spy_six_month_return']:+.2f}%" if recent["spy_six_month_return"] is not None else "n/a",
                         GREEN if (recent["spy_six_month_return"] or 0) >= 0 else RED)
                    stat("SPY Sharpe", f"{recent['spy_risk'].get('sharpe', float('nan')):.3f}" if recent["spy_risk"] else "n/a")
                    stat("SPY Sortino", f"{recent['spy_risk'].get('sortino', float('nan')):.3f}" if recent["spy_risk"] else "n/a")

        with ui.row().classes("w-full").style("gap: 16px; align-items: start;"):
            with section("Current Portfolio").style("flex: 1.1; min-height: 540px;"):
                if latest is not None:
                    ui.label(
                        f"Latest rebalance: {pd.to_datetime(latest['rebal_date']).strftime('%Y-%m-%d')}"
                    ).style(f"color:{MUTED}; font-size:12px;")
                    render_holdings(
                        ui.column().classes("w-full gap-0"),
                        latest["parsed_holdings"],
                        added=latest["changes"]["added"],
                        removed=latest["changes"]["removed"],
                    )
            with section("Recent Changes").style("flex: 0.9; min-height: 540px;"):
                prior_holdings = None
                for _, row in full_log.head(6).iterrows():
                    added = row["changes"]["added"]
                    removed = row["changes"]["removed"]
                    added_names = format_change_names(added, row["parsed_holdings"], prior_holdings)
                    removed_names = format_change_names(removed, row["parsed_holdings"], prior_holdings)
                    with ui.column().classes("w-full gap-1").style(f"padding:12px 0; border-bottom:1px solid {BORDER};"):
                        ui.label(pd.to_datetime(row["rebal_date"]).strftime("%Y-%m-%d")).style(
                            f"color:{TEXT}; font-size:14px; font-weight:700;"
                        )
                        ui.label(f"Turnover {row.get('turnover', 0):.1f}% · Period return {row.get('period_ret', 0):+.2f}%").style(
                            f"color:{MUTED}; font-size:12px;"
                        )
                        ui.label(f"Added: {added_names}").style(
                            f"color:{GREEN}; font-size:12px;"
                        )
                        ui.label(f"Removed: {removed_names}").style(
                            f"color:{RED}; font-size:12px;"
                        )
                    prior_holdings = row["parsed_holdings"]

        with section("Rebalance History"):
            ui.label("Select a rebalance to inspect the full 10-stock portfolio for that date.").style(
                f"color:{MUTED}; font-size:12px;"
            )
            history_rows = []
            row_lookup = {}
            for _, row in full_log.iterrows():
                row_id = pd.to_datetime(row["rebal_date"]).strftime("%Y-%m-%d")
                row_lookup[row_id] = row
                history_rows.append({
                    "row_id": row_id,
                    "date": row_id,
                    "turnover": f"{row.get('turnover', 0):.1f}%",
                    "period_ret": f"{row.get('period_ret', 0):+.2f}%",
                    "raw_ret": f"{row.get('raw_ret', 0):+.2f}%",
                    "n_stocks": row.get("n_stocks", BASKET_SIZE),
                })

            detail = ui.column().classes("w-full gap-0")

            def show_rebalance(e):
                row = row_lookup.get(e.args["row_id"])
                if row is None:
                    return
                detail.clear()
                with detail:
                    ui.label(f"Portfolio on {e.args['row_id']}").style(
                        f"color:{TEXT}; font-size:15px; font-weight:700; margin-top: 12px;"
                    )
                    render_holdings(
                        ui.column().classes("w-full gap-0"),
                        row["parsed_holdings"],
                        added=row["changes"]["added"],
                        removed=row["changes"]["removed"],
                    )

            ui.table(
                columns=[
                    {"name": "date", "label": "Rebalance", "field": "date", "align": "left"},
                    {"name": "turnover", "label": "Turnover", "field": "turnover", "align": "right"},
                    {"name": "period_ret", "label": "Period Return", "field": "period_ret", "align": "right"},
                    {"name": "raw_ret", "label": "Raw Return", "field": "raw_ret", "align": "right"},
                    {"name": "n_stocks", "label": "N", "field": "n_stocks", "align": "right"},
                ],
                rows=history_rows,
                row_key="row_id",
                on_select=show_rebalance,
                selection="single",
            ).props("dense flat bordered").classes("w-full")


ui.run(title="Core Equity Model", reload=False)
