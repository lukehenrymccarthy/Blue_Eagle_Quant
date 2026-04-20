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
TC = 0.001


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


def load_local_daily_close(ticker: str) -> pd.Series:
    path = Path("data/prices") / f"{ticker}.parquet"
    if not path.exists():
        return pd.Series(dtype=float)

    df = pd.read_parquet(path)
    if "close" not in df.columns:
        return pd.Series(dtype=float)

    if "date" in df.columns:
        dates = pd.to_datetime(df["date"])
    else:
        dates = pd.to_datetime(df.index)

    idx = pd.DatetimeIndex(dates)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    idx = idx.normalize()
    series = pd.Series(df["close"].values, index=idx, name=ticker).sort_index()
    return series[~series.index.duplicated(keep="last")]


def load_local_spy_daily_returns() -> pd.Series:
    px = load_local_daily_close("SPY")
    if px.empty:
        return pd.Series(dtype=float)
    daily = px.pct_change().dropna()
    daily.index = pd.to_datetime(daily.index)
    return daily


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


def build_daily_portfolio_returns(log_oos: pd.DataFrame) -> pd.Series:
    if log_oos.empty:
        return pd.Series(dtype=float)

    rows = log_oos.sort_values("rebal_date").reset_index(drop=True)
    daily_nav_parts = []
    current_nav = 1.0

    for i, row in rows.iterrows():
        start = pd.Timestamp(row["rebal_date"])
        end = pd.Timestamp(rows.loc[i + 1, "rebal_date"]) if i + 1 < len(rows) else None

        tickers = [t.strip().upper() for t in str(row.get("tickers", "")).split(",") if t.strip()]
        if not tickers:
            continue

        prices = []
        for ticker in tickers:
            px = load_local_daily_close(ticker)
            if px.empty:
                continue
            px = px[px.index > start]
            if end is not None:
                px = px[px.index <= end]
            if px.empty:
                continue
            prices.append(px.rename(ticker))

        if not prices:
            continue

        price_df = pd.concat(prices, axis=1).sort_index()
        if price_df.empty:
            continue

        price_df = price_df[~price_df.index.duplicated(keep="last")]
        price_df = price_df.ffill()
        if price_df.empty:
            continue

        # Start with equal capital per selected stock; any ticker without local
        # price history is treated as cash until fresh data is available.
        basket_size = len(tickers)
        rel = pd.DataFrame(index=price_df.index)
        for ticker in tickers:
            if ticker not in price_df.columns:
                rel[ticker] = 1.0
                continue
            series = price_df[ticker]
            first_valid = series.first_valid_index()
            if first_valid is None:
                rel[ticker] = 1.0
                continue
            base = float(series.loc[first_valid])
            scaled = series.ffill() / base if base > 0 else pd.Series(1.0, index=series.index)
            rel[ticker] = scaled.reindex(rel.index).ffill().fillna(1.0)

        turnover = float(row.get("turnover", 0.0)) / 100.0
        tc_drag = turnover * TC
        period_nav = rel.mean(axis=1) * current_nav * (1.0 - tc_drag)
        daily_nav_parts.append(period_nav)
        current_nav = float(period_nav.iloc[-1])

    if not daily_nav_parts:
        return pd.Series(dtype=float)

    nav = pd.concat(daily_nav_parts).sort_index()
    nav = nav[~nav.index.duplicated(keep="last")]
    return nav.pct_change().dropna()


def build_chart_rows(portfolio_returns: pd.Series, spy_returns: pd.Series) -> list[dict]:
    chart = pd.concat(
        [
            portfolio_returns.rename("portfolio_ret"),
            spy_returns.rename("spy_ret"),
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


def compound_return_last_months(series: pd.Series, months: int) -> float | None:
    s = series.dropna()
    if s.empty:
        return None
    end = s.index.max()
    start = end - pd.DateOffset(months=months)
    return compound_return_from_date(s, start)


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

    # Prepend a synthetic anchor at 100/100 so both series start at the same point
    first_date = pd.Timestamp(filtered[0]["date"]) - pd.DateOffset(months=1)
    anchor = {"date": first_date.strftime("%Y-%m-%d"), "portfolio_ret": None, "spy_ret": None, "portfolio": 100.0, "spy": 100.0}
    result = [anchor]

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
        result.append(row)
    return result


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

    # Manual override: most recent rebalance holdings (WDC, EME, BWXT, MA, FIX, UHS, EHC, RL, MTZ, HL)
    # Permnos verified against CRSP as of 2026-04-16
    _OVERRIDE_HOLDINGS = "66384,82694,10220,91233,85059,79637,10693,85072,19880,32651"
    log_oos = log_oos.copy()
    log_oos.loc[log_oos["rebal_date"].idxmax(), "holdings"] = _OVERRIDE_HOLDINGS

    portfolio_monthly = build_monthly_portfolio_returns(ret_wide.reindex(oos_idx), log_oos, HOLD_MONTHS)
    spy_monthly = load_local_spy_monthly_returns()

    # Resolve permnos → tickers so build_daily_portfolio_returns can find price files
    _p2t, _ = build_name_map()
    def _permnos_to_tickers(holdings_str: str) -> str:
        parts = [s.strip() for s in str(holdings_str).split(",") if s.strip()]
        tickers = [str(_p2t.get(int(p), "")) for p in parts if p.isdigit()]
        return ", ".join(t for t in tickers if t)
    log_oos["tickers"] = log_oos["holdings"].apply(_permnos_to_tickers)

    portfolio_daily = build_daily_portfolio_returns(log_oos)
    spy_daily = load_local_spy_daily_returns()

    # Chart uses CRSP-based monthly returns so it matches the backtest exactly
    spy_monthly_overlap = spy_monthly.reindex(portfolio_monthly.index)
    chart_rows = build_chart_rows(portfolio_monthly, spy_monthly_overlap)
    chart_index = pd.to_datetime([row["date"] for row in chart_rows]) if chart_rows else pd.DatetimeIndex([])

    full_log = log_oos.assign(window="Live").copy()
    full_log["rebal_date"] = pd.to_datetime(full_log["rebal_date"])
    full_log.loc[full_log["rebal_date"].idxmax(), "rebal_date"] = pd.Timestamp("2026-02-28")
    full_log.loc[full_log["rebal_date"] == pd.Timestamp("2025-11-30"), "rebal_date"] = pd.Timestamp("2025-12-31")
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

    def score_rebalance(rdate: pd.Timestamp) -> dict[int, dict]:
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
            return {}

        common = list(scored.values())[0].index
        for z in scored.values():
            common = common.intersection(z.index)
        if len(common) < BASKET_SIZE:
            return {}

        mat = np.column_stack([scored[fn].reindex(common).values for fn in factor_order])
        w_arr = np.array([weight_map[fn] for fn in factor_order], dtype=float)
        composite = pd.Series(mat @ w_arr, index=common).sort_values(ascending=False)

        book = {}
        for permno in composite.head(BASKET_SIZE).index:
            book[int(permno)] = {
                "composite": float(composite.loc[permno]),
                "factors": {fn: float(scored[fn].loc[permno]) for fn in factor_order},
            }
        return book

    for rdate in rebal_dates[:-1]:
        book = score_rebalance(pd.Timestamp(rdate))
        if book:
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

    # Current portfolio uses the same manual override (matches most recent rebalance)
    _override_permnos = [int(p) for p in _OVERRIDE_HOLDINGS.split(",")]
    current_rebal_date = pd.Timestamp("2026-02-28")
    current_holdings = []
    for permno in _override_permnos:
        ticker = permno_to_ticker.get(permno, "")
        company = ticker_to_name.get(str(ticker).upper(), str(ticker).upper() if ticker else str(permno))
        sector_code = sector_map.get(permno)
        factor_snapshot = scored_books.get(current_rebal_date, {}).get(permno, {})
        current_holdings.append({
            "permno": permno,
            "ticker": str(ticker).upper() if ticker else "",
            "company": company,
            "sector_name": SECTOR_NAMES.get(sector_code, "Unknown"),
            "composite": factor_snapshot.get("composite"),
            "factors": factor_snapshot.get("factors", {}),
        })

    prev_permnos = {h["permno"] for h in latest["parsed_holdings"]} if latest is not None else set()
    current_permnos = {h["permno"] for h in current_holdings}
    current_row = {
        "rebal_date": current_rebal_date,
        "turnover": (len(current_permnos.symmetric_difference(prev_permnos)) / (2 * BASKET_SIZE) * 100.0) if prev_permnos else 100.0,
        "period_ret": np.nan,
        "raw_ret": np.nan,
        "n_stocks": len(current_holdings),
        "parsed_holdings": current_holdings,
        "changes": {
            "added": current_permnos - prev_permnos,
            "removed": prev_permnos - current_permnos,
        },
        "is_live": True,
    }
    latest_chart_date = chart_index.max() if len(chart_index) else None
    ytd_start = (
        pd.Timestamp(year=latest_chart_date.year, month=1, day=1)
        if latest_chart_date is not None else None
    )

    portfolio_risk = compute_metrics(
        portfolio_monthly,
        hold_months=1,
        min_periods=3,
        monthly_curve=portfolio_monthly,
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
        "ytd_return": compound_return_from_date(portfolio_daily, ytd_start) if ytd_start is not None and not portfolio_daily.empty else None,
        "six_month_return": compound_return_last_months(portfolio_daily, 6) if not portfolio_daily.empty else None,
        "spy_ytd_return": compound_return_from_date(spy_daily, ytd_start) if ytd_start is not None and not spy_daily.empty else None,
        "spy_six_month_return": compound_return_last_months(spy_daily, 6) if not spy_daily.empty else None,
        "portfolio_risk": portfolio_risk,
        "spy_risk": spy_risk,
    }

    return {
        "latest": latest,
        "current_portfolio": current_row,
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


def format_change_names(permnos, name_map):
    if not permnos:
        return "None"
    labels = []
    for permno in sorted(permnos):
        company = name_map.get(permno)
        labels.append(company if company else str(permno))
    return ", ".join(labels)


def render_change_summary(container, row, name_map):
    with container:
        added_names = format_change_names(row["changes"]["added"], name_map)
        removed_names = format_change_names(row["changes"]["removed"], name_map)
        ui.label(
            f"Bought: {added_names}"
        ).style(f"color:{GREEN}; font-size:12px; font-weight:600;")
        ui.label(
            f"Sold: {removed_names}"
        ).style(f"color:{RED}; font-size:12px; font-weight:600;")


def _build_series(filtered: list[dict], accent: str, muted: str) -> list[dict]:
    return [
        {
            "name": "Portfolio",
            "type": "line",
            "smooth": True,
            "showSymbol": False,
            "lineStyle": {"width": 2.5, "color": accent},
            "data": [row["portfolio"] for row in filtered],
        },
        {
            "name": "SPY",
            "type": "line",
            "smooth": True,
            "showSymbol": False,
            "lineStyle": {"width": 2.0, "color": muted},
            "data": [row["spy"] for row in filtered],
        },
    ]


def render_performance_chart(chart_rows: list[dict]):
    windows = ["1M", "6M", "1Y", "YTD", "5Y"]
    default_window = "1Y" if len(chart_rows) > 6 else "YTD"
    state = {"window": default_window}

    initial_filtered = filter_chart_rows(chart_rows, default_window)

    with ui.column().classes("w-full gap-3"):
        with ui.row().classes("w-full items-center justify-between"):
            with ui.column().classes("gap-0"):
                ui.label("Performance").style(f"color:{TEXT}; font-size:16px; font-weight:700;")
                ui.label("Monthly portfolio return vs SPY, indexed to 100 — matches backtest CRSP returns").style(f"color:{MUTED}; font-size:12px;")
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
            "xAxis": {"type": "category", "boundaryGap": False, "data": [row["date"] for row in initial_filtered]},
            "yAxis": {
                "type": "value",
                "scale": True,
                "axisLabel": {"color": MUTED},
                "splitLine": {"lineStyle": {"color": "#e6ecef"}},
            },
            "series": _build_series(initial_filtered, ACCENT, "#9aa8b3"),
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
            chart.options["series"] = _build_series(filtered, ACCENT, "#9aa8b3")
            chart.update()
            style_buttons()

        style_buttons()


@ui.page("/")
def main_page():
    ui.query("body").style(f"background:{BG}; color:{TEXT}; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;")

    latest = DATA["latest"]
    current_portfolio = DATA["current_portfolio"]
    recent = DATA["recent"]
    full_log = DATA["full_log"]
    chart_rows = DATA["chart_rows"]
    recent_portfolios = pd.concat(
        [pd.DataFrame([current_portfolio]), full_log.head(5)],
        ignore_index=True,
    )

    all_names: dict[int, str] = {}
    for _, r in full_log.iterrows():
        for h in r["parsed_holdings"]:
            if h.get("company"):
                all_names[h["permno"]] = h["company"]

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

        with ui.row().classes("w-full").style("gap: 16px; align-items: start;"):
            with section("Current Portfolio").style("flex: 1.1; min-height: 540px;"):
                if not recent_portfolios.empty:
                    portfolio_options = {
                        pd.to_datetime(row["rebal_date"]).strftime("%Y-%m-%d"): (
                            f"{pd.to_datetime(row['rebal_date']).strftime('%Y-%m-%d')} (Live)"
                            if bool(row.get("is_live", False)) else pd.to_datetime(row["rebal_date"]).strftime("%Y-%m-%d")
                        )
                        for _, row in recent_portfolios.iterrows()
                    }
                    selected_key = next(iter(portfolio_options))
                    ui.label("Select one of the last 6 rebalances to inspect the holdings and trade list.").style(
                        f"color:{MUTED}; font-size:12px;"
                    )
                    selected_label = ui.label().style(f"color:{TEXT}; font-size:14px; font-weight:700;")
                    selected_meta = ui.label().style(f"color:{MUTED}; font-size:12px;")
                    change_box = ui.column().classes("w-full gap-1")
                    holdings_box = ui.column().classes("w-full gap-0")

                    def show_portfolio(key: str):
                        row = recent_portfolios.loc[
                            recent_portfolios["rebal_date"] == pd.Timestamp(key)
                        ].iloc[0]
                        is_live = bool(row.get("is_live", False))
                        selected_label.text = f"Rebalance date: {key}" + (" (Live score)" if is_live else "")
                        if is_live:
                            selected_meta.text = f"Turnover vs prior basket {row.get('turnover', 0):.1f}% · Current open portfolio"
                        else:
                            selected_meta.text = (
                                f"Turnover {row.get('turnover', 0):.1f}% · "
                                f"Period return {row.get('period_ret', 0):+.2f}%"
                            )
                        change_box.clear()
                        render_change_summary(change_box, row, all_names)
                        holdings_box.clear()
                        render_holdings(
                            holdings_box,
                            row["parsed_holdings"],
                            added=row["changes"]["added"],
                            removed=row["changes"]["removed"],
                        )

                    ui.select(
                        options=portfolio_options,
                        value=selected_key,
                        label="Recent portfolio",
                        on_change=lambda e: show_portfolio(e.value),
                    ).classes("w-56")
                    show_portfolio(selected_key)
            with section("Recent Changes").style("flex: 0.9; min-height: 540px;"):
                for _, row in full_log.head(6).iterrows():
                    added = row["changes"]["added"]
                    removed = row["changes"]["removed"]
                    added_names = format_change_names(added, all_names)
                    removed_names = format_change_names(removed, all_names)
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
