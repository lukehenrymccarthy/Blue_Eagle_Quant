"""
dashboard/app.py
-----------------
Two-page NiceGUI dashboard for the five-factor quant model.

  /           →  Sector ETF overview  (home page)
  /stocks     →  Top-25 / Bottom-25 individual stocks

Pages share a common header with navigation.

Run:
    python dashboard/app.py   →  http://localhost:8080
"""

import sys
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from nicegui import ui

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.five_factor_model import _sic_to_etf

# ── Sector / ETF metadata ─────────────────────────────────────────────────────
SECTOR_META = {
    "XLK":  {"name": "Technology",       "bg": "#dbeafe", "text": "#1d4ed8", "dot": "#3b82f6", "dark": "#1e3a8a"},
    "XLF":  {"name": "Financials",        "bg": "#d1fae5", "text": "#065f46", "dot": "#10b981", "dark": "#064e3b"},
    "XLV":  {"name": "Health Care",       "bg": "#fee2e2", "text": "#991b1b", "dot": "#ef4444", "dark": "#7f1d1d"},
    "XLI":  {"name": "Industrials",       "bg": "#fef3c7", "text": "#92400e", "dot": "#f59e0b", "dark": "#78350f"},
    "XLE":  {"name": "Energy",            "bg": "#ede9fe", "text": "#5b21b6", "dot": "#8b5cf6", "dark": "#4c1d95"},
    "XLP":  {"name": "Consumer Staples",  "bg": "#ccfbf1", "text": "#065f46", "dot": "#14b8a6", "dark": "#134e4a"},
    "XLY":  {"name": "Consumer Discr.",   "bg": "#ffedd5", "text": "#9a3412", "dot": "#f97316", "dark": "#7c2d12"},
    "XLB":  {"name": "Materials",         "bg": "#ecfccb", "text": "#3f6212", "dot": "#84cc16", "dark": "#365314"},
    "XLRE": {"name": "Real Estate",       "bg": "#fce7f3", "text": "#9d174d", "dot": "#ec4899", "dark": "#831843"},
    "XLU":  {"name": "Utilities",         "bg": "#f3f4f6", "text": "#374151", "dot": "#6b7280", "dark": "#1f2937"},
    "XLC":  {"name": "Comm. Services",    "bg": "#cffafe", "text": "#164e63", "dot": "#06b6d4", "dark": "#0c4a6e"},
}
ETF_NAMES = {k: v["name"] for k, v in SECTOR_META.items()}

SIGNAL_META = {
    "Strong Buy":  {"bg": "#064e3b", "text": "#ffffff", "icon": "▲▲"},
    "Buy":         {"bg": "#d1fae5", "text": "#065f46", "icon": "▲"},
    "Neutral":     {"bg": "#f3f4f6", "text": "#6b7280", "icon": "—"},
    "Sell":        {"bg": "#fee2e2", "text": "#991b1b", "icon": "▼"},
    "Strong Sell": {"bg": "#7f1d1d", "text": "#ffffff", "icon": "▼▼"},
    "N/A":         {"bg": "#f3f4f6", "text": "#9ca3af", "icon": "?"},
}

FACTOR_COLS = ["momentum_18m", "roe_ttm", "sector_rs_3m", "analyst_rev_3m", "macro_hy"]


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_etf_summary() -> pd.DataFrame:
    p = Path("data/etf_holdings/etf_summary.parquet")
    if not p.exists():
        raise FileNotFoundError("Run ingestion/etf_holdings.py first.")
    return pd.read_parquet(p)


def load_etf_holdings(etf: str) -> pd.DataFrame:
    p = Path(f"data/etf_holdings/{etf}_holdings.parquet")
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def load_scores() -> pd.DataFrame:
    p = Path("data/results/five_factor_scores_latest.csv")
    if not p.exists():
        raise FileNotFoundError("Run backtest/five_factor_model.py first.")
    df = pd.read_csv(p, index_col=0)
    if df.index.dtype == object:
        df.index = df.index.str.upper()
    else:
        df.index = df.index.astype(float)
        ibes = pd.read_parquet("data/analyst/ibes_signals.parquet",
                               columns=["permno", "ticker", "statpers"])
        ibes["statpers"] = pd.to_datetime(ibes["statpers"])
        ticker_map = (ibes.sort_values("statpers")
                      .groupby("permno")["ticker"].last()
                      .str.upper())
        df.index = ticker_map.reindex(df.index).values
        df = df[df.index.notna()]
    df.index.name = "ticker"
    return df.sort_values("composite", ascending=False)


def score_as_of() -> str:
    for ticker in ["SPY", "AAPL", "MSFT"]:
        p = Path(f"data/prices/{ticker}.parquet")
        try:
            idx = pd.to_datetime(pd.read_parquet(p, columns=[]).index)
            return idx.max().strftime("%B %d, %Y")
        except Exception:
            continue
    return "Latest available"


def macro_status() -> tuple[str, str, str]:
    try:
        from ingestion.fred_macro import load_macro_signals
        sig = load_macro_signals()
        s   = sig["hy_spread_widening"].dropna()
        roll_mean = s.rolling(36, min_periods=12).mean()
        roll_std  = s.rolling(36, min_periods=12).std().replace(0, float("nan"))
        latest    = (-((s - roll_mean) / roll_std)).dropna().iloc[-1]
        sign      = "+" if latest >= 0 else ""
        label     = "Bullish" if latest >= 0 else "Bearish"
        bg        = "#d1fae5" if latest >= 0 else "#fee2e2"
        fg        = "#065f46" if latest >= 0 else "#991b1b"
        return f"{label} {sign}{latest:.2f}σ", bg, fg
    except Exception:
        return "Unknown", "#f3f4f6", "#374151"


# ── Price history for stock popup ─────────────────────────────────────────────

def fetch_price_history(ticker: str, days: int = 365) -> list[list]:
    p = Path(f"data/prices/{ticker}.parquet")
    try:
        raw    = pd.read_parquet(p)
        raw.index = pd.to_datetime(raw.index)
        closes = (raw["close"] if "close" in raw.columns else raw.iloc[:, 0]).dropna().sort_index()
        closes = closes[closes.index >= closes.index[-1] - pd.Timedelta(days=days)]
        if len(closes) >= 30:
            return [[int(ts.timestamp() * 1000), round(float(v), 2)]
                    for ts, v in zip(closes.index, closes.values)]
    except Exception:
        pass
    try:
        end   = pd.Timestamp.today()
        start = end - pd.Timedelta(days=days + 10)
        raw   = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                            end=end.strftime("%Y-%m-%d"),
                            auto_adjust=True, progress=False)["Close"].dropna().squeeze()
        if len(raw) >= 10:
            return [[int(ts.timestamp() * 1000), round(float(v), 2)]
                    for ts, v in zip(raw.index, raw.values)]
    except Exception:
        pass
    return []


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HTML HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def factor_bar(value: float) -> str:
    lo, hi = -3.0, 5.0
    pct    = max(0.0, min(100.0, (value - lo) / (hi - lo) * 100))
    color  = "#10b981" if value >= 0 else "#ef4444"
    sign   = "+" if value >= 0 else ""
    return (
        f'<div style="display:flex;align-items:center;gap:5px;min-width:80px">'
        f'<div style="flex:0 0 40px;background:#e5e7eb;border-radius:3px;height:5px;overflow:hidden">'
        f'<div style="width:{pct:.0f}%;background:{color};height:5px;border-radius:3px"></div></div>'
        f'<span style="color:{color};font-size:11px;font-weight:600">{sign}{value:.2f}</span></div>'
    )


def composite_bar(value: float, hi: float = 3.0) -> str:
    pct   = max(0.0, min(100.0, value / hi * 100))
    color = "#3b82f6" if value >= 0 else "#f43f5e"
    return (
        f'<div style="display:flex;align-items:center;gap:6px;min-width:100px">'
        f'<div style="flex:0 0 56px;background:#e5e7eb;border-radius:4px;height:7px;overflow:hidden">'
        f'<div style="width:{pct:.0f}%;background:{color};height:7px;border-radius:4px"></div></div>'
        f'<strong style="font-size:12px">{value:.3f}</strong></div>'
    )


def sector_badge_html(etf: str) -> str:
    meta = SECTOR_META.get(etf, {"bg": "#f3f4f6", "text": "#374151", "name": etf})
    return (
        f'<span style="background:{meta["bg"]};color:{meta["text"]};'
        f'padding:2px 8px;border-radius:9999px;font-size:11px;font-weight:700">'
        f'{meta["name"]}</span>'
    )


def signal_badge_html(signal: str) -> str:
    m = SIGNAL_META.get(signal, SIGNAL_META["N/A"])
    return (
        f'<span style="background:{m["bg"]};color:{m["text"]};'
        f'padding:2px 10px;border-radius:4px;font-size:11px;font-weight:700;'
        f'letter-spacing:0.03em">{m["icon"]} {signal}</span>'
    )


def ticker_cell_html(ticker: str, rank: int, is_top: bool) -> str:
    bg    = "#f0f4ff" if is_top else "#fff0f0"
    color = "#4f46e5" if is_top else "#dc2626"
    return (
        f'<div style="display:flex;align-items:center;gap:7px">'
        f'<span style="background:{bg};color:{color};border-radius:4px;'
        f'padding:1px 5px;font-size:10px;font-weight:600">#{rank}</span>'
        f'<strong style="font-size:13px">{ticker}</strong></div>'
    )


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HEADER
# ══════════════════════════════════════════════════════════════════════════════

def make_header(active: str, as_of: str, macro_txt: str):
    """active: 'etf' or 'stocks'."""
    with ui.header().classes("bg-indigo-700 text-white px-6 py-0 flex items-center justify-between"):
        with ui.row().classes("items-center gap-3 py-3"):
            ui.icon("analytics", size="26px")
            with ui.column().classes("gap-0"):
                ui.label("SectorScope").classes("text-xl font-bold tracking-wide")
                ui.label("Five-Factor Quant Model  ·  Equal-Weight").classes(
                    "text-xs text-indigo-200"
                )
        with ui.row().classes("items-center gap-6"):
            # Nav tabs
            for label, href, key in [("Sector ETFs", "/", "etf"),
                                      ("Top / Bottom Stocks", "/stocks", "stocks"),
                                      ("Model Methodology", "/methodology", "methodology")]:
                cls = ("bg-white text-indigo-700 font-bold" if active == key
                       else "text-indigo-200 hover:text-white")
                ui.link(label, href).classes(
                    f"px-4 py-3 text-sm font-semibold no-underline {cls}"
                ).style("transition:color 0.15s")
            ui.label(f"As of {as_of}").classes("text-xs text-indigo-300 pl-4")


# ══════════════════════════════════════════════════════════════════════════════
# STOCK DETAIL POPUP  (pentagon radar + price history)
# ══════════════════════════════════════════════════════════════════════════════

def open_stock_popup(dialog: ui.dialog, row: dict) -> None:
    ticker    = row.get("_ticker", "—")
    sector    = row.get("_sector", "—")
    etf       = row.get("_etf",    "—")
    composite = row.get("_composite", 0.0)
    sign      = "+" if composite >= 0 else ""
    meta      = SECTOR_META.get(etf, {"bg": "#f3f4f6", "text": "#374151", "dot": "#6b7280"})

    scores = [
        round(row.get("_momentum_val",   0), 3),
        round(row.get("_roe_val",        0), 3),
        round(row.get("_sector_rs_val",  0), 3),
        round(row.get("_analyst_val",    0), 3),
        round(row.get("_macro_val",      0), 3),
    ]
    fill_color = "rgba(59,130,246,0.25)" if composite >= 0 else "rgba(239,68,68,0.20)"
    line_color = "#3b82f6" if composite >= 0 else "#ef4444"

    price_data = fetch_price_history(ticker)

    dialog.clear()
    with dialog, ui.card().classes("w-[920px] max-w-[96vw] p-0 overflow-hidden"):

        # Header
        with ui.row().classes("w-full items-center justify-between px-6 py-4").style(
            "background:#1e1b4b"
        ):
            with ui.row().classes("items-center gap-4"):
                ui.label(ticker).classes("text-2xl font-bold text-white tracking-wide")
                ui.element("span").style(
                    f"background:{meta['bg']};color:{meta['text']};"
                    f"padding:3px 12px;border-radius:9999px;font-size:12px;font-weight:700"
                ).text = sector
                with ui.column().classes("gap-0"):
                    ui.label(f"Composite: {sign}{composite:.4f}").classes(
                        "text-sm font-semibold"
                    ).style("color:#a5b4fc")
                    ui.label("5-Factor Equal-Weight").classes("text-xs").style("color:#818cf8")
            ui.button(icon="close", on_click=dialog.close).props("flat round color=white")

        with ui.row().classes("w-full gap-0 flex-wrap"):
            # Pentagon radar
            with ui.column().classes("flex-1 min-w-[380px] p-4 gap-1").style(
                "border-right:1px solid #f3f4f6"
            ):
                with ui.row().classes("items-center gap-2 mb-1"):
                    ui.icon("radar", size="16px").classes("text-indigo-500")
                    ui.label("Factor Z-Scores").classes("text-sm font-semibold text-gray-600")
                ui.highchart({
                    "chart": {"polar": True, "type": "area",
                              "backgroundColor": "transparent", "margin": [20, 20, 20, 20]},
                    "title": {"text": ""}, "credits": {"enabled": False}, "legend": {"enabled": False},
                    "tooltip": {"pointFormat": "<b>{point.y:.2f}σ</b>",
                                "headerFormat": "<span>{series.name}</span><br>"},
                    "pane": {"size": "80%"},
                    "xAxis": {
                        "categories": ["Momentum", "ROE TTM", "Sector RS", "Analyst Rev", "Macro HY"],
                        "tickmarkPlacement": "on", "lineWidth": 0,
                        "labels": {"style": {"fontSize": "11px", "fontWeight": "600", "color": "#374151"}},
                    },
                    "yAxis": {
                        "gridLineInterpolation": "polygon", "lineWidth": 0,
                        "min": -3, "max": 3, "tickInterval": 1,
                        "labels": {"style": {"fontSize": "10px", "color": "#9ca3af"}},
                    },
                    "series": [{
                        "name": "Z-Score", "data": scores, "pointPlacement": "on",
                        "color": line_color, "fillColor": fill_color, "lineWidth": 2,
                        "marker": {"enabled": True, "radius": 4,
                                   "fillColor": line_color, "lineWidth": 0},
                    }],
                }).classes("w-full h-72")
                # Score pills
                factor_map = [
                    ("Momentum",  row.get("_momentum_val",  0)),
                    ("ROE TTM",   row.get("_roe_val",       0)),
                    ("Sector RS", row.get("_sector_rs_val", 0)),
                    ("Analyst",   row.get("_analyst_val",   0)),
                    ("Macro HY",  row.get("_macro_val",     0)),
                ]
                with ui.row().classes("flex-wrap gap-2 mt-1"):
                    for fname, fval in factor_map:
                        pb  = "#dcfce7" if fval >= 0 else "#fee2e2"
                        pf  = "#166534" if fval >= 0 else "#991b1b"
                        sf  = "+" if fval >= 0 else ""
                        ui.element("div").style(
                            f"background:{pb};color:{pf};padding:2px 10px;"
                            f"border-radius:9999px;font-size:11px;font-weight:700"
                        ).text = f"{fname}  {sf}{fval:.2f}σ"

            # Price history
            with ui.column().classes("flex-1 min-w-[380px] p-4 gap-1"):
                with ui.row().classes("items-center justify-between mb-1"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("show_chart", size="16px").classes("text-indigo-500")
                        ui.label("Price History  (1 Year)").classes(
                            "text-sm font-semibold text-gray-600"
                        )
                    if price_data:
                        sp, ep = price_data[0][1], price_data[-1][1]
                        pct    = (ep / sp - 1) * 100 if sp else 0
                        pc     = "#10b981" if pct >= 0 else "#ef4444"
                        ui.label(f"${ep:.2f}  ({'+'if pct>=0 else''}{pct:.1f}% 1Y)").classes(
                            "text-sm font-bold"
                        ).style(f"color:{pc}")

                if price_data:
                    sp, ep   = price_data[0][1], price_data[-1][1]
                    pct_chg  = (ep / sp - 1) * 100 if sp else 0
                    lc       = "#10b981" if pct_chg >= 0 else "#ef4444"
                    grad_top = "rgba(16,185,129,0.20)" if pct_chg >= 0 else "rgba(239,68,68,0.20)"
                    grad_bot = "rgba(16,185,129,0.01)" if pct_chg >= 0 else "rgba(239,68,68,0.01)"
                    ui.highchart({
                        "chart": {"type": "area", "backgroundColor": "transparent",
                                  "margin": [10, 10, 40, 50], "zoomType": "x"},
                        "title": {"text": ""}, "credits": {"enabled": False}, "legend": {"enabled": False},
                        "tooltip": {"valueDecimals": 2, "valuePrefix": "$",
                                    "xDateFormat": "%b %d, %Y", "shared": True},
                        "xAxis": {"type": "datetime",
                                  "labels": {"style": {"fontSize": "10px", "color": "#9ca3af"}},
                                  "lineColor": "#e5e7eb", "tickColor": "#e5e7eb"},
                        "yAxis": {"title": {"text": ""},
                                  "labels": {"format": "${value}",
                                             "style": {"fontSize": "10px", "color": "#9ca3af"}},
                                  "gridLineColor": "#f3f4f6"},
                        "series": [{
                            "name": ticker, "data": price_data, "color": lc, "lineWidth": 2,
                            "fillColor": {
                                "linearGradient": {"x1": 0, "y1": 0, "x2": 0, "y2": 1},
                                "stops": [[0, grad_top], [1, grad_bot]],
                            },
                            "marker": {"enabled": False},
                        }],
                    }).classes("w-full h-80")
                else:
                    with ui.column().classes("w-full h-80 items-center justify-center gap-2"):
                        ui.icon("bar_chart_off", size="40px").classes("text-gray-300")
                        ui.label("Price data unavailable").classes("text-sm text-gray-400")

    dialog.open()


# ══════════════════════════════════════════════════════════════════════════════
# ETF HOLDINGS POPUP
# ══════════════════════════════════════════════════════════════════════════════

def open_etf_popup(dialog: ui.dialog, etf: str, etf_row: pd.Series) -> None:
    holdings = load_etf_holdings(etf)
    meta     = SECTOR_META.get(etf, {"name": etf, "bg": "#f3f4f6",
                                      "text": "#374151", "dot": "#6b7280", "dark": "#1f2937"})
    score    = etf_row.get("etf_score", float("nan"))
    signal   = etf_row.get("signal", "N/A")
    sig_meta = SIGNAL_META.get(signal, SIGNAL_META["N/A"])
    cov_w    = etf_row.get("coverage_w", 0)
    n_h      = etf_row.get("n_holdings", 0)
    n_s      = etf_row.get("n_scored", 0)
    score_z  = etf_row.get("score_z", float("nan"))

    dialog.clear()
    with dialog, ui.card().classes("w-[1100px] max-w-[98vw] p-0 overflow-hidden"):

        # ── Dialog header ──────────────────────────────────────────────────────
        with ui.row().classes("w-full items-center justify-between px-6 py-4").style(
            f"background:{meta['dark']}"
        ):
            with ui.row().classes("items-center gap-5 flex-wrap"):
                with ui.column().classes("gap-0"):
                    ui.label(etf).classes("text-3xl font-black text-white tracking-wider")
                    ui.label(meta["name"]).classes("text-sm text-white/70 font-medium")
                # Score
                with ui.column().classes("gap-0 pl-4").style(
                    "border-left:1px solid rgba(255,255,255,0.2)"
                ):
                    score_str = f"{score:+.4f}" if pd.notna(score) else "N/A"
                    ui.label(score_str).classes("text-xl font-bold text-white")
                    ui.label("Weighted composite").classes("text-xs text-white/60")
                # Signal badge
                ui.element("div").style(
                    f"background:{sig_meta['bg']};color:{sig_meta['text']};"
                    f"padding:6px 18px;border-radius:6px;font-size:14px;"
                    f"font-weight:800;letter-spacing:0.05em"
                ).text = f"{sig_meta['icon']} {signal}"
                # Stats
                with ui.column().classes("gap-0 pl-4").style(
                    "border-left:1px solid rgba(255,255,255,0.2)"
                ):
                    ui.label(f"{n_s} / {n_h} holdings scored").classes(
                        "text-xs text-white/70"
                    )
                    ui.label(f"{cov_w:.0f}% of ETF weight covered").classes(
                        "text-xs text-white/70"
                    )
                    if pd.notna(score_z):
                        ui.label(f"Cross-sectional z-score: {score_z:+.2f}").classes(
                            "text-xs text-white/70"
                        )
            ui.button(icon="close", on_click=dialog.close).props("flat round color=white")

        # ── Holdings table ─────────────────────────────────────────────────────
        if holdings.empty:
            with ui.column().classes("w-full items-center justify-center py-16 gap-2"):
                ui.icon("inventory_2", size="40px").classes("text-gray-300")
                ui.label("Holdings data not available.").classes("text-sm text-gray-400")
                ui.label("Run ingestion/etf_holdings.py to refresh.").classes(
                    "text-xs text-gray-300"
                )
            return

        # Build table rows
        tbl_rows = []
        for _, r in holdings.iterrows():
            comp = r.get("composite", None)
            price = r.get("price", None)

            comp_html = (
                factor_bar(float(comp)) if pd.notna(comp) else
                '<span style="color:#9ca3af;font-size:11px">Not scored</span>'
            )
            price_html = (
                f'<span style="font-weight:600">${float(price):.2f}</span>'
                if pd.notna(price) else
                '<span style="color:#9ca3af">—</span>'
            )
            weight_pct = float(r["weight"]) * 100

            tbl_rows.append({
                "rank":     int(r.get("rank", 0)),
                "ticker":   str(r["ticker"]),
                "name":     str(r["name"]),
                "weight":   f"{weight_pct:.2f}%",
                "price":    price_html,
                "score":    comp_html,
                "_comp_val": float(comp) if pd.notna(comp) else -999,
                "_w_val":    weight_pct,
            })

        tbl_cols = [
            {"name": "rank",   "label": "#",        "field": "rank",   "align": "center", "sortable": True},
            {"name": "ticker", "label": "Ticker",   "field": "ticker", "align": "left",   "sortable": True},
            {"name": "name",   "label": "Company",  "field": "name",   "align": "left",   "sortable": True},
            {"name": "weight", "label": "ETF Wt.",  "field": "weight", "align": "right",  "sortable": True},
            {"name": "price",  "label": "Price",    "field": "price",  "align": "right"},
            {"name": "score",  "label": "Composite Score", "field": "score", "align": "left"},
        ]

        with ui.column().classes("w-full px-4 pb-4"):
            tbl = ui.table(
                columns    = tbl_cols,
                rows       = tbl_rows,
                row_key    = "ticker",
                pagination = {"rowsPerPage": 30, "sortBy": "_w_val", "descending": True},
            ).classes("w-full text-sm")
            tbl.props("flat dense bordered")

            for col in ["price", "score"]:
                tbl.add_slot(
                    f"body-cell-{col}",
                    r'<q-td :props="props"><span v-html="props.value"></span></q-td>',
                )

            # Footer summary
            scored_holdings = holdings.dropna(subset=["composite"])
            with ui.row().classes("items-center justify-between pt-2 px-1"):
                ui.label(
                    f"{n_s}/{n_h} holdings scored  ·  "
                    f"{cov_w:.1f}% of ETF weight  ·  "
                    f"Remaining {100-cov_w:.1f}% not in current score universe"
                ).classes("text-xs text-gray-400")
                ui.label(
                    "Click column headers to sort  ·  "
                    "Run ingestion/etf_holdings.py to refresh"
                ).classes("text-xs text-gray-300")

    dialog.open()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1  —  SECTOR ETF OVERVIEW  (home)
# ══════════════════════════════════════════════════════════════════════════════

@ui.page("/")
def etf_page():
    as_of                         = score_as_of()
    macro_txt, macro_bg, macro_fg = macro_status()

    try:
        summary = load_etf_summary()
    except FileNotFoundError:
        make_header("etf", as_of, macro_txt)
        with ui.column().classes("w-full items-center justify-center py-32 gap-3"):
            ui.icon("hourglass_empty", size="48px").classes("text-gray-300")
            ui.label("ETF holdings not yet computed.").classes("text-lg text-gray-500")
            ui.label("Run:  python ingestion/etf_holdings.py").classes(
                "text-sm font-mono bg-gray-100 px-4 py-2 rounded text-gray-600"
            )
        return

    dialog = ui.dialog().props("maximized=false")
    make_header("etf", as_of, macro_txt)

    with ui.column().classes("w-full px-6 py-4 gap-4"):

        # ── KPI bar ───────────────────────────────────────────────────────────
        valid_summary = summary.dropna(subset=["etf_score"])
        buys  = (valid_summary["signal"].isin(["Strong Buy", "Buy"])).sum()
        sells = (valid_summary["signal"].isin(["Strong Sell", "Sell"])).sum()
        neutral = (valid_summary["signal"] == "Neutral").sum()
        best_etf  = valid_summary.loc[valid_summary["etf_score"].idxmax(), "etf"] if not valid_summary.empty else "—"
        worst_etf = valid_summary.loc[valid_summary["etf_score"].idxmin(), "etf"] if not valid_summary.empty else "—"

        with ui.row().classes("w-full gap-4 flex-wrap"):
            for label, value, icon, color in [
                ("Sectors Covered",  "11 SPDR ETFs",    "grid_view",    "#4f46e5"),
                ("Macro (HY Sprd)",  macro_txt,         "trending_up",  macro_fg),
                ("Buy Signals",      f"{buys} sectors",       "arrow_upward", "#10b981"),
                ("Sell Signals",     f"{sells} sectors",      "arrow_downward","#ef4444"),
                ("Top Sector",       best_etf,          "star",         "#059669"),
                ("Weak Sector",      worst_etf,         "star_border",  "#dc2626"),
            ]:
                with ui.card().classes("flex-1 min-w-36 shadow-sm"):
                    with ui.row().classes("items-center gap-3 px-1"):
                        ui.icon(icon, size="26px").style(f"color:{color}")
                        with ui.column().classes("gap-0"):
                            ui.label(value).classes("text-base font-bold").style(f"color:{color}")
                            ui.label(label).classes("text-xs text-gray-400")

        ui.label("Click any sector card to view full holdings  ·  "
                 "Signals = cross-sectional z-score of weighted composite across 11 sectors").classes(
            "text-xs text-gray-400 italic px-1"
        )

        # ── ETF grid ──────────────────────────────────────────────────────────
        # Sort: Strong Buy first, then by score descending
        signal_order = {"Strong Buy": 0, "Buy": 1, "Neutral": 2, "Sell": 3, "Strong Sell": 4, "N/A": 5}
        summary_sorted = summary.copy()
        summary_sorted["_sig_ord"] = summary_sorted["signal"].map(signal_order).fillna(5)
        summary_sorted = summary_sorted.sort_values(
            ["_sig_ord", "etf_score"], ascending=[True, False]
        )

        with ui.row().classes("w-full gap-4 flex-wrap"):
            for _, row in summary_sorted.iterrows():
                etf      = row["etf"]
                sector   = row["sector"]
                meta     = SECTOR_META.get(etf, {"bg": "#f3f4f6", "text": "#374151",
                                                   "dot": "#6b7280", "dark": "#1f2937"})
                score    = row.get("etf_score", float("nan"))
                score_z  = row.get("score_z",   float("nan"))
                signal   = row.get("signal",    "N/A")
                sig_meta = SIGNAL_META.get(signal, SIGNAL_META["N/A"])
                cov_w    = row.get("coverage_w",  0)
                n_h      = int(row.get("n_holdings", 0))
                n_s      = int(row.get("n_scored",   0))

                # Load top 3 holdings for preview
                holdings = load_etf_holdings(etf)
                top3     = []
                if not holdings.empty:
                    for _, hr in holdings.head(3).iterrows():
                        top3.append((str(hr["ticker"]), float(hr["weight"]) * 100))

                # Signal colours — stronger visual contrast
                SIG_COLORS = {
                    "Strong Buy":  {"card_bg": "#052e16", "banner_bg": "#16a34a",
                                    "banner_fg": "#ffffff", "score_fg": "#4ade80"},
                    "Buy":         {"card_bg": "#f0fdf4", "banner_bg": "#22c55e",
                                    "banner_fg": "#ffffff", "score_fg": "#15803d"},
                    "Neutral":     {"card_bg": "#f9fafb", "banner_bg": "#9ca3af",
                                    "banner_fg": "#ffffff", "score_fg": "#374151"},
                    "Sell":        {"card_bg": "#fff1f2", "banner_bg": "#f87171",
                                    "banner_fg": "#ffffff", "score_fg": "#dc2626"},
                    "Strong Sell": {"card_bg": "#450a0a", "banner_bg": "#dc2626",
                                    "banner_fg": "#ffffff", "score_fg": "#fca5a5"},
                    "N/A":         {"card_bg": "#f3f4f6", "banner_bg": "#d1d5db",
                                    "banner_fg": "#6b7280", "score_fg": "#9ca3af"},
                }
                sc = SIG_COLORS.get(signal, SIG_COLORS["N/A"])

                # z-score gauge: 0–100%, centre at 50%
                if pd.notna(score_z):
                    bar_pct = max(0.0, min(100.0, (score_z + 1.5) / 3.0 * 100))
                else:
                    bar_pct = 50.0

                score_str = f"{score:+.4f}" if pd.notna(score) else "N/A"
                z_str     = f"z = {score_z:+.2f}" if pd.notna(score_z) else ""

                with ui.card().classes(
                    "cursor-pointer hover:shadow-xl transition-all duration-200"
                ).style(
                    f"min-width:210px;flex:1;max-width:290px;overflow:hidden;"
                    f"border:2px solid {sc['banner_bg']};"
                    f"background:{sc['card_bg']}"
                ).on("click", lambda _, e=etf, r=row: open_etf_popup(dialog, e, r)):

                    # ── Signal banner (dominant element) ──────────────────────
                    with ui.element("div").style(
                        f"background:{sc['banner_bg']};padding:12px 14px 10px;"
                        f"display:flex;align-items:center;justify-content:space-between"
                    ):
                        # ETF ticker + sector
                        with ui.column().classes("gap-0"):
                            ui.label(etf).style(
                                f"font-size:22px;font-weight:900;color:#ffffff;"
                                f"letter-spacing:0.04em;line-height:1"
                            )
                            ui.label(sector).style(
                                "font-size:11px;color:rgba(255,255,255,0.75);font-weight:500"
                            )
                        # Large signal text
                        ui.label(f"{sig_meta['icon']} {signal}").style(
                            f"font-size:13px;font-weight:900;color:#ffffff;"
                            f"letter-spacing:0.06em;text-transform:uppercase;"
                            f"text-align:right;line-height:1.2"
                        )

                    # ── Score + gauge ──────────────────────────────────────────
                    with ui.column().classes("gap-1 px-3 pt-3 pb-1"):
                        with ui.row().classes("items-baseline gap-2"):
                            ui.label(score_str).style(
                                f"font-size:26px;font-weight:900;color:{sc['score_fg']};"
                                f"line-height:1"
                            )
                            ui.label(z_str).style(
                                "font-size:11px;font-weight:600;color:#9ca3af"
                            )

                        # Gauge bar
                        with ui.element("div").style(
                            "width:100%;background:rgba(0,0,0,0.10);"
                            "border-radius:4px;height:5px;margin:4px 0"
                        ):
                            ui.element("div").style(
                                f"width:{bar_pct:.0f}%;background:{sc['banner_bg']};"
                                f"height:5px;border-radius:4px;opacity:0.8"
                            )

                    # ── Holdings footer ────────────────────────────────────────
                    with ui.element("div").style(
                        f"padding:6px 12px 10px;border-top:1px solid rgba(0,0,0,0.07)"
                    ):
                        ui.label(f"{n_s}/{n_h} holdings  ·  {cov_w:.0f}% wt scored").style(
                            "font-size:10px;color:#9ca3af;margin-bottom:4px"
                        )
                        if top3:
                            tickers_html = "  ·  ".join(
                                f'<strong>{t}</strong> {w:.1f}%' for t, w in top3
                            )
                            ui.element("div").style(
                                "font-size:11px;color:#6b7280;"
                                "white-space:nowrap;overflow:hidden;text-overflow:ellipsis"
                            ).props(f'innerHTML="{tickers_html}"')

        # ── Legend ────────────────────────────────────────────────────────────
        with ui.card().classes("w-full shadow-sm"):
            with ui.row().classes("items-center gap-3 px-3 py-2 flex-wrap"):
                ui.icon("info_outline", size="15px").classes("text-gray-400")
                ui.label("Signal methodology:").classes(
                    "text-xs font-semibold text-gray-500"
                )
                for sig, desc in [
                    ("Strong Buy",  "z > +1.0"),
                    ("Buy",         "z > +0.3"),
                    ("Neutral",     "|z| ≤ 0.3"),
                    ("Sell",        "z < −0.3"),
                    ("Strong Sell", "z < −1.0"),
                ]:
                    sm = SIGNAL_META[sig]
                    with ui.row().classes("items-center gap-1"):
                        ui.element("span").style(
                            f"background:{sm['bg']};color:{sm['text']};"
                            f"padding:1px 7px;border-radius:3px;font-size:10px;font-weight:700"
                        ).text = sig
                        ui.label(f"= {desc}").classes("text-xs text-gray-400")
                ui.label("  z = cross-sectional z-score of weighted composite across 11 sectors").classes(
                    "text-xs text-gray-400"
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2  —  TOP-25 / BOTTOM-25 INDIVIDUAL STOCKS
# ══════════════════════════════════════════════════════════════════════════════

STOCK_TABLE_COLS = [
    {"name": "ticker_cell",  "label": "Ticker",      "field": "ticker_cell",  "align": "left"},
    {"name": "sector_cell",  "label": "Sector",      "field": "sector_cell",  "align": "left"},
    {"name": "composite",    "label": "Score",       "field": "composite",    "align": "left"},
    {"name": "momentum",     "label": "Mom 18m",     "field": "momentum",     "align": "left"},
    {"name": "roe",          "label": "ROE",         "field": "roe",          "align": "left"},
    {"name": "sector_rs",    "label": "Sector RS",   "field": "sector_rs",    "align": "left"},
    {"name": "analyst",      "label": "Analyst Rev", "field": "analyst",      "align": "left"},
    {"name": "macro",        "label": "Macro HY",    "field": "macro",        "align": "left"},
]
STOCK_HTML_COLS = [c["name"] for c in STOCK_TABLE_COLS]


def _fval(r, col, df):
    try:
        return float(r[col]) if col in df.columns and pd.notna(r[col]) else 0.0
    except Exception:
        return 0.0


def build_stock_rows(df: pd.DataFrame, is_top: bool) -> list[dict]:
    import pyarrow.parquet as pq
    fund_path  = Path("data/fundamentals/compustat_quarterly.parquet")
    schema_names = pq.read_schema(fund_path).names
    sic_col    = "siccd" if "siccd" in schema_names else "sic"
    fund       = pd.read_parquet(fund_path, columns=["permno", "datadate", sic_col])
    fund["datadate"] = pd.to_datetime(fund["datadate"])
    sic_latest = fund.sort_values("datadate").groupby("permno")[sic_col].last()
    etf_by_p   = sic_latest.apply(lambda s: _sic_to_etf(s) if pd.notna(s) else None)

    rows = []
    for i, (idx, r) in enumerate(df.iterrows()):
        ticker = str(idx).upper() if df.index.dtype == object else r.get("ticker", "—")
        etf    = etf_by_p.get(r.get("permno", -1), "—") or "—"
        sector = ETF_NAMES.get(etf, "Other")

        row = {
            "ticker_cell": ticker_cell_html(ticker, i + 1, is_top),
            "sector_cell": sector_badge_html(etf),
            "composite":   composite_bar(float(r["composite"])),
            "momentum":    factor_bar(_fval(r, "momentum_18m",   df)) if "momentum_18m"   in df.columns else "—",
            "roe":         factor_bar(_fval(r, "roe_ttm",        df)) if "roe_ttm"        in df.columns else "—",
            "sector_rs":   factor_bar(_fval(r, "sector_rs_3m",   df)) if "sector_rs_3m"   in df.columns else "—",
            "analyst":     factor_bar(_fval(r, "analyst_rev_3m", df)) if "analyst_rev_3m" in df.columns else "—",
            "macro":       factor_bar(_fval(r, "macro_hy",       df)) if "macro_hy"       in df.columns else "—",
            "_ticker":         ticker,
            "_sector":         sector,
            "_etf":            etf,
            "_composite":      round(float(r["composite"]), 4),
            "_momentum_val":   _fval(r, "momentum_18m",   df),
            "_roe_val":        _fval(r, "roe_ttm",        df),
            "_sector_rs_val":  _fval(r, "sector_rs_3m",   df),
            "_analyst_val":    _fval(r, "analyst_rev_3m", df),
            "_macro_val":      _fval(r, "macro_hy",       df),
        }
        rows.append(row)
    return rows


@ui.page("/stocks")
def stocks_page():
    as_of                         = score_as_of()
    macro_txt, macro_bg, macro_fg = macro_status()

    try:
        df = load_scores()
    except FileNotFoundError:
        make_header("stocks", as_of, macro_txt)
        with ui.column().classes("w-full items-center justify-center py-32 gap-3"):
            ui.icon("hourglass_empty", size="48px").classes("text-gray-300")
            ui.label("Scores not yet computed.").classes("text-lg text-gray-500")
            ui.label("Run:  python backtest/five_factor_model.py").classes(
                "text-sm font-mono bg-gray-100 px-4 py-2 rounded text-gray-600"
            )
        return

    top25    = df.head(25)
    bottom25 = df.tail(25).sort_values("composite", ascending=True)

    dialog   = ui.dialog().props("maximized=false")
    make_header("stocks", as_of, macro_txt)

    with ui.column().classes("w-full px-6 py-4 gap-4"):

        with ui.row().classes("w-full gap-4 flex-wrap"):
            for label, value, icon, color in [
                ("Model",       "5-Factor Equal-Weight",            "layers",        "#4f46e5"),
                ("Macro (F5)",  macro_txt,                          "trending_up",   macro_fg),
                ("Best Score",  f"{df['composite'].max():.3f}",     "arrow_upward",  "#10b981"),
                ("Worst Score", f"{df['composite'].min():.3f}",     "arrow_downward","#ef4444"),
                ("Universe",    f"{len(df):,} stocks scored",       "group",         "#0891b2"),
            ]:
                with ui.card().classes("flex-1 min-w-36 shadow-sm"):
                    with ui.row().classes("items-center gap-3 px-1"):
                        ui.icon(icon, size="26px").style(f"color:{color}")
                        with ui.column().classes("gap-0"):
                            ui.label(value).classes("text-base font-bold").style(f"color:{color}")
                            ui.label(label).classes("text-xs text-gray-400")

        ui.label("Click any row to view the factor pentagon chart and 1-year price history.").classes(
            "text-xs text-gray-400 italic px-1"
        )

        with ui.row().classes("w-full gap-4 items-start flex-wrap"):
            with ui.column().classes("flex-1 min-w-[580px] gap-4"):
                for rows, title, color, icon in [
                    (build_stock_rows(top25,    True),  "Top 25 — Strongest Signals",  "#10b981", "arrow_upward"),
                    (build_stock_rows(bottom25, False), "Bottom 25 — Weakest Signals", "#ef4444", "arrow_downward"),
                ]:
                    with ui.card().classes("w-full shadow-sm overflow-hidden"):
                        with ui.row().classes("items-center justify-between px-3 py-2").style(
                            f"border-left:4px solid {color}"
                        ):
                            with ui.row().classes("items-center gap-2"):
                                ui.icon(icon, size="18px").style(f"color:{color}")
                                ui.label(title).classes("text-sm font-semibold text-gray-700")
                            ui.label(f"{len(rows)} stocks").classes("text-xs text-gray-400")

                        tbl = ui.table(
                            columns    = STOCK_TABLE_COLS,
                            rows       = rows,
                            row_key    = "_ticker",
                            pagination = {"rowsPerPage": 25},
                        ).classes("w-full text-sm")
                        tbl.props("flat dense bordered")
                        for col in STOCK_HTML_COLS:
                            tbl.add_slot(
                                f"body-cell-{col}",
                                r'<q-td :props="props"><span v-html="props.value"></span></q-td>',
                            )

                        def make_handler(t):
                            def handler(e):
                                try:
                                    row = e.args[1] if isinstance(e.args, list) else e.args
                                    if isinstance(row, dict) and "_ticker" in row:
                                        open_stock_popup(dialog, row)
                                except Exception:
                                    pass
                            return handler

                        tbl.on("rowClick", make_handler(tbl))

        # Factor guide
        with ui.card().classes("w-full shadow-sm"):
            with ui.row().classes("items-center gap-2 px-2"):
                ui.icon("info_outline", size="14px").classes("text-gray-400")
                ui.label("Factor Guide").classes("text-xs font-semibold text-gray-500")
            with ui.row().classes("gap-6 px-2 pb-1 flex-wrap"):
                for name, desc in [
                    ("Momentum 18m", "18-month cumulative return, skip last month."),
                    ("ROE TTM",      "Return on equity, trailing 12 months (Compustat PIT via rdq)."),
                    ("Sector RS",    "Sector ETF 3m return vs SPY."),
                    ("Analyst Rev",  "3-month IBES consensus rec change."),
                    ("Macro HY",     "ICE BofA HY OAS rolling z-score, sign-flipped."),
                ]:
                    with ui.column().classes("gap-0 flex-1 min-w-40"):
                        ui.label(name).classes("text-xs font-bold text-indigo-600")
                        ui.label(desc).classes("text-xs text-gray-500")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3  —  MODEL METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════

def _load_equity_curves() -> dict:
    """Return dict of strategy → list of [timestamp_ms, nav] for Highcharts."""
    try:
        raw = pd.read_csv("data/results/twenty_year_equity_curves.csv", index_col=0)
        raw.index = pd.to_datetime(raw.index)
        # Convert cumulative-return index to NAV starting at 100
        out = {}
        for col in raw.columns:
            # Values are already cumulative NAV multipliers (1.0 = start); scale to $100
            series = raw[col] * 100
            out[col] = [[int(ts.timestamp() * 1000), round(float(v), 2)]
                        for ts, v in zip(series.index, series.values)]
        return out
    except Exception:
        return {}


def _load_backtest_summary() -> pd.DataFrame:
    try:
        return pd.read_csv("data/results/twenty_year_summary.csv", index_col=0)
    except Exception:
        return pd.DataFrame()


def _load_subperiod() -> pd.DataFrame:
    try:
        return pd.read_csv("data/results/twenty_year_subperiod.csv", index_col=0)
    except Exception:
        return pd.DataFrame()


@ui.page("/methodology")
def methodology_page():
    as_of                         = score_as_of()
    macro_txt, macro_bg, macro_fg = macro_status()
    make_header("methodology", as_of, macro_txt)

    equity_curves = _load_equity_curves()
    bt_summary    = _load_backtest_summary()
    subperiod     = _load_subperiod()

    # ── colour palette shared across sections ─────────────────────────────────
    FACTOR_DEFS = [
        {
            "id":     "momentum",
            "label":  "F1 — Momentum 18m-1m",
            "short":  "Momentum",
            "icon":   "trending_up",
            "color":  "#6366f1",
            "source": "CRSP monthly returns",
            "desc":   (
                "18-month cumulative return skipping the most recent month. "
                "Captures the well-documented price momentum anomaly — "
                "recent winners tend to continue outperforming over the next 6–12 months."
            ),
            "detail": "Winsorised at 1/99%, cross-sectionally z-scored each month.",
        },
        {
            "id":     "roe",
            "label":  "F2 — ROE TTM",
            "short":  "ROE TTM",
            "icon":   "account_balance",
            "color":  "#10b981",
            "source": "Compustat quarterly (PIT via rdq)",
            "desc":   (
                "Return on equity over the trailing twelve months. "
                "Screens for profitable, capital-efficient businesses. "
                "Point-in-time: uses the actual earnings release date (rdq) "
                "so no lookahead bias — only data publicly available at rebalance."
            ),
            "detail": "Staleness filter: values older than 15 months set to NaN.",
        },
        {
            "id":     "sector_rs",
            "label":  "F3 — Sector RS vs SPY 3m",
            "short":  "Sector RS",
            "icon":   "donut_large",
            "color":  "#f59e0b",
            "source": "SPDR sector ETF prices (yfinance)",
            "desc":   (
                "3-month cumulative return of the stock's SPDR sector ETF minus SPY's "
                "3-month return. Stocks in outperforming sectors inherit a tailwind; "
                "those in lagging sectors face a headwind. Captures sector rotation cycles."
            ),
            "detail": "SIC code → GICS-aligned sector ETF mapping (11 SPDR ETFs).",
        },
        {
            "id":     "analyst",
            "label":  "F4 — Analyst Rev 3m",
            "short":  "Analyst Rev",
            "icon":   "groups",
            "color":  "#ec4899",
            "source": "IBES consensus (WRDS)",
            "desc":   (
                "Change in mean analyst recommendation over the past 3 months. "
                "Upgrades signal improving fundamental outlook; downgrades signal deterioration. "
                "Uses statpers (consensus compilation date) as the PIT anchor."
            ),
            "detail": "Net upgrades (numup − numdown) and buy-% revision also computed.",
        },
        {
            "id":     "macro",
            "label":  "F5 — Macro HY Spread",
            "short":  "Macro HY",
            "icon":   "public",
            "color":  "#0ea5e9",
            "source": "FRED — ICE BofA HY OAS",
            "desc":   (
                "Rolling 36-month z-score of the 3-month change in the high-yield credit spread "
                "(ICE BofA HY OAS), sign-flipped. Narrowing spreads → risk-on → higher score; "
                "widening spreads → risk-off → lower score. Market-wide scalar applied equally "
                "to all stocks — acts as a macro timing overlay."
            ),
            "detail": "Sign-flipped: positive = bullish macro environment.",
        },
    ]

    with ui.column().classes("w-full px-6 py-5 gap-6"):

        # ── Hero ──────────────────────────────────────────────────────────────
        with ui.card().classes("w-full shadow-sm overflow-hidden"):
            with ui.element("div").style(
                "background:linear-gradient(135deg,#1e1b4b 0%,#312e81 60%,#4338ca 100%);"
                "padding:32px 40px"
            ):
                ui.label("Five-Factor Quantitative Model").style(
                    "font-size:28px;font-weight:900;color:#ffffff;letter-spacing:-0.01em"
                )
                ui.label(
                    "Equal-weight composite of five independent alpha signals — "
                    "momentum, quality, sector rotation, analyst sentiment, and macro regime."
                ).style("font-size:14px;color:rgba(255,255,255,0.70);margin-top:6px;max-width:700px")

            with ui.row().classes("gap-0 flex-wrap"):
                for label, value, border in [
                    ("Universe",        "~8,700 US equities",   True),
                    ("Rebalance",       "Monthly",              True),
                    ("Weighting",       "Equal weight (1/5)",   True),
                    ("PIT discipline",  "rdq + statpers",       True),
                    ("Backtest period", "2005 – 2024 (20 yr)",  False),
                ]:
                    with ui.element("div").style(
                        f"padding:16px 24px;{'border-right:1px solid #f3f4f6;' if border else ''}"
                        "flex:1;min-width:140px"
                    ):
                        ui.label(value).style(
                            "font-size:16px;font-weight:800;color:#1e1b4b"
                        )
                        ui.label(label).style("font-size:11px;color:#9ca3af;margin-top:2px")

        # ── Composite model diagram ───────────────────────────────────────────
        with ui.row().classes("w-full gap-4 items-start flex-wrap"):

            # Factor weight donut
            with ui.card().classes("shadow-sm flex-1 min-w-72"):
                with ui.row().classes("items-center gap-2 px-4 pt-4 pb-1"):
                    ui.icon("pie_chart", size="18px").classes("text-indigo-500")
                    ui.label("Factor Weights").classes("text-sm font-semibold text-gray-700")
                ui.highchart({
                    "chart": {
                        "type": "pie",
                        "backgroundColor": "transparent",
                        "margin": [0, 0, 0, 0],
                        "height": 260,
                    },
                    "title": {"text": ""},
                    "credits": {"enabled": False},
                    "tooltip": {"pointFormat": "<b>{point.percentage:.0f}%</b>"},
                    "plotOptions": {
                        "pie": {
                            "innerSize": "55%",
                            "dataLabels": {
                                "enabled": True,
                                "format": "<b>{point.name}</b>",
                                "style": {"fontSize": "11px", "fontWeight": "600"},
                                "distance": 14,
                            },
                            "startAngle": -90,
                        }
                    },
                    "series": [{
                        "name": "Weight",
                        "data": [
                            {"name": "Momentum",    "y": 5,    "color": "#6366f1"},
                            {"name": "ROE TTM",     "y": 23.75,"color": "#10b981"},
                            {"name": "Sector RS",   "y": 23.75,"color": "#f59e0b"},
                            {"name": "Analyst Rev", "y": 23.75,"color": "#ec4899"},
                            {"name": "Macro HY",    "y": 23.75,"color": "#3b82f6"},
                        ],
                    }],
                }).classes("w-full")
                ui.label("Momentum 5%  ·  ROE / Sector RS / Analyst Rev / Macro HY each 23.75%").style(
                    "font-size:11px;color:#9ca3af;padding:0 16px 14px;text-align:center"
                )

            # Pipeline flow diagram (HTML)
            with ui.card().classes("shadow-sm flex-1 min-w-96"):
                with ui.row().classes("items-center gap-2 px-4 pt-4 pb-2"):
                    ui.icon("account_tree", size="18px").classes("text-indigo-500")
                    ui.label("Signal Construction Pipeline").classes(
                        "text-sm font-semibold text-gray-700"
                    )
                with ui.element("div").style("padding:8px 20px 20px"):
                    step_style = (
                        "display:flex;align-items:center;gap:12px;padding:10px 14px;"
                        "border-radius:8px;margin-bottom:6px"
                    )
                    arrow_style = (
                        "text-align:center;color:#9ca3af;font-size:18px;"
                        "margin:0 0 6px 14px;line-height:1"
                    )
                    for bg, title_col, title, body_col, body in [
                        ("#ede9fe","#5b21b6","📥 Data Ingestion",        "#7c3aed",
                         "CRSP returns · Compustat (rdq PIT) · IBES (statpers PIT) · FRED · SSGA holdings"),
                        ("#fef3c7","#92400e","⚙️ Factor Computation",    "#b45309",
                         "F1 Momentum · F2 ROE · F3 Sector RS · F4 Analyst Rev · F5 Macro HY"),
                        ("#e0f2fe","#075985","📐 Normalisation",          "#0369a1",
                         "Winsorise 1/99% → cross-sectional z-score per factor per month"),
                        ("#d1fae5","#065f46","∑ Weighted Composite",      "#047857",
                         "Momentum 5% · ROE / Sector RS / Analyst Rev / Macro HY 23.75% each  ·  Partial scoring if ≥ 2 factors"),
                        ("#fee2e2","#991b1b","📡 ETF Signal",             "#dc2626",
                         "Weight-avg composite across holdings → cross-sectional z → Strong Buy / Sell"),
                    ]:
                        with ui.element("div").style(step_style + f"background:{bg}"):
                            with ui.row().classes("items-center gap-2 flex-wrap"):
                                ui.label(title).style(
                                    f"font-size:12px;font-weight:700;color:{title_col}"
                                )
                                ui.label(body).style(
                                    f"font-size:11px;color:{body_col}"
                                )
                        if bg != "#fee2e2":   # no arrow after last step
                            ui.label("↓").style(arrow_style)

        # ── Five factor cards ─────────────────────────────────────────────────
        with ui.row().classes("items-center gap-2 px-1"):
            ui.icon("layers", size="20px").classes("text-indigo-500")
            ui.label("The Five Factors").classes("text-lg font-bold text-gray-800")

        _F_WEIGHTS = {
            "momentum":  ("5%",    "#6366f1"),
            "roe":       ("23.75%","#10b981"),
            "sector_rs": ("23.75%","#f59e0b"),
            "analyst":   ("23.75%","#ec4899"),
            "macro":     ("23.75%","#3b82f6"),
        }
        with ui.row().classes("w-full gap-4 flex-wrap"):
            for f in FACTOR_DEFS:
                fw_pct, fw_col = _F_WEIGHTS.get(f["id"], ("20%", f["color"]))
                with ui.card().classes("shadow-sm flex-1 min-w-60").style(
                    f"border-top:4px solid {f['color']}"
                ):
                    with ui.column().classes("gap-2 p-4"):
                        with ui.row().classes("items-center gap-2 justify-between"):
                            with ui.row().classes("items-center gap-2"):
                                ui.icon(f["icon"], size="20px").style(f"color:{f['color']}")
                                ui.label(f["label"]).classes("text-sm font-bold text-gray-800")
                            ui.label(fw_pct).style(
                                f"background:{fw_col}22;color:{fw_col};"
                                "font-size:11px;font-weight:800;padding:2px 8px;"
                                "border-radius:99px;white-space:nowrap"
                            )
                        ui.label(f["desc"]).classes("text-xs text-gray-600").style(
                            "line-height:1.6"
                        )
                        with ui.element("div").style(
                            f"background:#f3f4f6;border-radius:6px;padding:8px 10px;margin-top:4px"
                        ):
                            ui.label(f"Source: {f['source']}").style(
                                "font-size:10px;font-weight:700;color:#6b7280"
                            )
                            ui.label(f["detail"]).style(
                                "font-size:10px;color:#9ca3af;margin-top:2px"
                            )

        # ── Signal thresholds ─────────────────────────────────────────────────
        with ui.card().classes("w-full shadow-sm"):
            with ui.row().classes("items-center gap-2 px-4 pt-4 pb-2"):
                ui.icon("flag", size="18px").classes("text-indigo-500")
                ui.label("Signal Generation — ETF-Level Thresholds").classes(
                    "text-sm font-semibold text-gray-700"
                )
            with ui.row().classes("gap-4 px-4 pb-4 flex-wrap items-stretch"):
                for sig, z_rule, explanation in [
                    ("Strong Buy",  "z > +1.0",  "Sector composite is more than 1σ above the cross-sectional mean — strong broad-based quality + momentum tailwind."),
                    ("Buy",         "z > +0.3",  "Sector composite is moderately above average — positive tilt across most factors."),
                    ("Neutral",     "|z| ≤ 0.3", "Sector composite is near the cross-sectional mean — no clear directional edge."),
                    ("Sell",        "z < −0.3",  "Sector composite is moderately below average — negative tilt across most factors."),
                    ("Strong Sell", "z < −1.0",  "Sector composite is more than 1σ below average — broad deterioration in quality, momentum, and analyst sentiment."),
                ]:
                    sm = SIGNAL_META[sig]
                    with ui.element("div").style(
                        "flex:1;min-width:160px;border-radius:8px;overflow:hidden;"
                        "border:1px solid #e5e7eb"
                    ):
                        with ui.element("div").style(
                            f"background:{sm['bg']};color:{sm['text']};"
                            "padding:8px 12px;letter-spacing:0.04em"
                        ):
                            ui.label(f"{sm['icon']} {sig}").style(
                                "font-size:13px;font-weight:900;display:block"
                            )
                            ui.label(z_rule).style(
                                "font-size:11px;font-weight:600;opacity:0.8"
                            )
                        ui.label(explanation).style(
                            "padding:8px 12px;font-size:11px;color:#6b7280;line-height:1.5;display:block"
                        )

        # ── 20-year backtest performance ──────────────────────────────────────
        with ui.card().classes("w-full shadow-sm"):
            with ui.row().classes("items-center justify-between px-4 pt-4 pb-2 flex-wrap gap-2"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("show_chart", size="18px").classes("text-indigo-500")
                    ui.label("20-Year Backtest  (2005 – 2024)").classes(
                        "text-sm font-semibold text-gray-700"
                    )
                # Summary pills
                if not bt_summary.empty:
                    with ui.row().classes("gap-3 flex-wrap"):
                        for strat, color in [("ML-Optimised Weights", "#6366f1"),
                                             ("Equal-Weight",         "#10b981"),
                                             ("SPY Buy-and-Hold",     "#f59e0b")]:
                            if strat in bt_summary.index:
                                r  = bt_summary.loc[strat]
                                ar = r.get("ann_return", float("nan"))
                                sh = r.get("sharpe",     float("nan"))
                                with ui.element("div").style(
                                    f"border-left:3px solid {color};"
                                    "padding:2px 10px;background:#f9fafb;border-radius:0 4px 4px 0"
                                ):
                                    ui.label(strat.replace(" Weights","").replace(" Buy-and-Hold","")).style(
                                        f"font-size:10px;font-weight:700;color:{color}"
                                    )
                                    ui.label(
                                        f"{ar:+.1f}% ann  ·  Sharpe {sh:.2f}"
                                        if pd.notna(ar) else "N/A"
                                    ).style("font-size:10px;color:#6b7280")

            if equity_curves:
                CURVE_COLORS = {
                    "ML-Optimised Weights": "#6366f1",
                    "Equal-Weight":         "#10b981",
                    "SPY_BH":               "#f59e0b",
                }
                CURVE_LABELS = {
                    "ML-Optimised Weights": "ML-Optimised",
                    "Equal-Weight":         "Equal-Weight",
                    "SPY_BH":               "SPY Buy & Hold",
                }
                series = []
                for name, data in equity_curves.items():
                    series.append({
                        "name":      CURVE_LABELS.get(name, name),
                        "data":      data,
                        "color":     CURVE_COLORS.get(name, "#9ca3af"),
                        "lineWidth": 2,
                        "marker":    {"enabled": False},
                    })
                ui.highchart({
                    "chart": {
                        "type": "line",
                        "backgroundColor": "transparent",
                        "margin": [20, 20, 50, 60],
                        "zoomType": "x",
                    },
                    "title": {"text": ""},
                    "credits": {"enabled": False},
                    "legend": {
                        "enabled": True,
                        "align": "right", "verticalAlign": "top",
                        "itemStyle": {"fontSize": "11px"},
                    },
                    "tooltip": {
                        "shared": True, "crosshairs": True,
                        "xDateFormat": "%b %Y",
                        "pointFormat": '<span style="color:{point.color}">●</span> {series.name}: <b>${point.y:.0f}</b><br>',
                        "headerFormat": "<b>{point.key}</b><br>",
                    },
                    "xAxis": {
                        "type": "datetime",
                        "labels": {"style": {"fontSize": "10px", "color": "#9ca3af"}},
                        "lineColor": "#e5e7eb", "tickColor": "#e5e7eb",
                    },
                    "yAxis": {
                        "title": {"text": "NAV (start = $100)", "style": {"fontSize": "11px"}},
                        "labels": {
                            "format": "${value:.0f}",
                            "style": {"fontSize": "10px", "color": "#9ca3af"},
                        },
                        "gridLineColor": "#f3f4f6",
                    },
                    "series": series,
                }).classes("w-full h-80 px-4 pb-4")
            else:
                with ui.column().classes("w-full h-40 items-center justify-center gap-2"):
                    ui.icon("bar_chart_off", size="36px").classes("text-gray-300")
                    ui.label("Run backtest/twenty_year_backtest.py to generate curves.").classes(
                        "text-sm text-gray-400"
                    )

        # ── Sub-period breakdown ──────────────────────────────────────────────
        if not subperiod.empty:
            with ui.card().classes("w-full shadow-sm"):
                with ui.row().classes("items-center gap-2 px-4 pt-4 pb-3"):
                    ui.icon("calendar_view_month", size="18px").classes("text-indigo-500")
                    ui.label("Performance Across Market Regimes").classes(
                        "text-sm font-semibold text-gray-700"
                    )
                with ui.element("div").style("overflow-x:auto;padding:0 16px 16px"):
                    # Build HTML table
                    hdr_style = (
                        "padding:8px 16px;font-size:11px;font-weight:700;"
                        "color:#6b7280;background:#f9fafb;text-align:right;"
                        "border-bottom:2px solid #e5e7eb;white-space:nowrap"
                    )
                    cell_style = (
                        "padding:8px 16px;font-size:12px;text-align:right;"
                        "border-bottom:1px solid #f3f4f6;white-space:nowrap"
                    )
                    period_style = (
                        "padding:8px 16px;font-size:12px;font-weight:700;"
                        "color:#374151;border-bottom:1px solid #f3f4f6;white-space:nowrap"
                    )

                    PERIOD_LABELS = {
                        "2006–2009  (GFC)":                "2006–2009  (GFC)",
                        "2010–2014  (Recovery)":            "2010–2014  (Recovery)",
                        "2015–2019  (Bull)":                "2015–2019  (Bull Market)",
                        "2020–2024  (COVID+Tightening)":    "2020–2024  (COVID + Tightening)",
                    }

                    col_map = {
                        "model_ann_ret": ("Model Ann. Return", "%"),
                        "model_sharpe":  ("Model Sharpe",      ""),
                        "model_max_dd":  ("Model Max DD",      "%"),
                        "spy_ann_ret":   ("SPY Ann. Return",   "%"),
                        "spy_sharpe":    ("SPY Sharpe",        ""),
                    }
                    display_cols = [c for c in col_map if c in subperiod.columns]

                    header_html = (
                        f'<th style="{hdr_style};text-align:left">Period</th>' +
                        "".join(
                            f'<th style="{hdr_style}">{col_map[c][0]}</th>'
                            for c in display_cols
                        )
                    )
                    rows_html = ""
                    for period, row in subperiod.iterrows():
                        label = PERIOD_LABELS.get(str(period).strip(), str(period))
                        cells = ""
                        for c in display_cols:
                            val  = row.get(c, float("nan"))
                            unit = col_map[c][1]
                            if pd.notna(val):
                                is_ret = unit == "%"
                                color  = "#059669" if val > 0 else "#dc2626"
                                sign   = "+" if val > 0 else ""
                                cells += (
                                    f'<td style="{cell_style};color:{color};font-weight:600">'
                                    f'{sign}{val:.1f}{unit}</td>'
                                )
                            else:
                                cells += f'<td style="{cell_style};color:#9ca3af">—</td>'
                        rows_html += f'<tr><td style="{period_style}">{label}</td>{cells}</tr>'

                    ui.element("div").props(
                        f'innerHTML="<table style=\'width:100%;border-collapse:collapse\'>'
                        f'<thead><tr>{header_html}</tr></thead>'
                        f'<tbody>{rows_html}</tbody></table>"'
                    )

        # ── Data sources + PIT note ───────────────────────────────────────────
        with ui.card().classes("w-full shadow-sm"):
            with ui.row().classes("items-center gap-2 px-4 pt-4 pb-2"):
                ui.icon("storage", size="18px").classes("text-indigo-500")
                ui.label("Data Sources & Point-in-Time Discipline").classes(
                    "text-sm font-semibold text-gray-700"
                )
            with ui.row().classes("gap-4 px-4 pb-4 flex-wrap"):
                for source, detail, icon, color in [
                    ("CRSP Monthly Stock File",
                     "Returns, prices, shares outstanding. ~9,700 US-listed stocks. Permno-based, survivorship-bias-free.",
                     "table_chart", "#6366f1"),
                    ("Compustat Quarterly",
                     "Balance sheet, income, cash flow. PIT anchor = rdq (actual earnings release date). Staleness filter: 15 months.",
                     "receipt_long", "#10b981"),
                    ("IBES Consensus",
                     "Analyst recommendations, upgrades/downgrades. PIT anchor = statpers (consensus compilation date).",
                     "people", "#ec4899"),
                    ("FRED",
                     "ICE BofA HY OAS, 10Y-2Y spread, Fed Funds Rate, CPI, UNRATE, PAYEMS. Monthly frequency.",
                     "language", "#0ea5e9"),
                    ("SSGA ETF Holdings",
                     "Full constituent lists + weights for 11 SPDR sector ETFs. Downloaded daily from SSGA public Excel files.",
                     "pie_chart", "#f59e0b"),
                    ("DuckDB Warehouse",
                     "Star-schema data warehouse (~157 MB). 12 tables, 4 views. Idempotent ETL pipeline; migration path to PostgreSQL.",
                     "database", "#8b5cf6"),
                ]:
                    with ui.card().classes("flex-1 min-w-52 shadow-none").style(
                        f"border:1px solid #e5e7eb;border-top:3px solid {color}"
                    ):
                        with ui.row().classes("items-center gap-2 p-3 pb-1"):
                            ui.icon(icon, size="16px").style(f"color:{color}")
                            ui.label(source).classes("text-xs font-bold text-gray-700")
                        ui.label(detail).classes("text-xs text-gray-500 px-3 pb-3").style(
                            "line-height:1.6"
                        )


# ══════════════════════════════════════════════════════════════════════════════
ui.run(
    title   = "SectorScope — Five-Factor Model",
    port    = 8080,
    reload  = False,
    favicon = "📊",
)
