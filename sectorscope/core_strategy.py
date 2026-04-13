from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from sectorscope.utils import zscore as _zscore

FULL_START = "2005-01-01"
IS_END = "2024-12-31"
OOS_START = "2025-04-01"
OOS_END = "2026-03-31"

UNIVERSE_SIZE = 1000
RISK_FREE_ANN = 0.0425
TC = 0.001


def _load_local_price_series(ticker: str) -> pd.Series:
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
    return pd.Series(df["close"].values, index=dates).sort_index()


def latest_local_oos_end() -> str:
    from ingestion.wrds_returns import load_returns as _lr, returns_wide

    rw = returns_wide(_lr())
    rw.index = pd.to_datetime(rw.index) + pd.offsets.MonthEnd(0)
    latest = rw.index.max()
    return latest.strftime("%Y-%m-%d")


def active_oos_end() -> str:
    latest = pd.Timestamp(latest_local_oos_end())
    configured = pd.Timestamp(OOS_END)
    return min(latest, configured).strftime("%Y-%m-%d")


def load_returns(oos_end: str | None = None) -> pd.DataFrame:
    from ingestion.wrds_returns import load_returns as _lr, returns_wide

    rw = returns_wide(_lr())
    # returns_wide already normalises to tz-naive month-end; apply MonthEnd(0)
    # again as a safety net and drop any remaining duplicates.
    idx = pd.to_datetime(rw.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    rw.index = idx.normalize() + pd.offsets.MonthEnd(0)
    rw = rw[~rw.index.duplicated(keep="last")]
    if oos_end is None:
        oos_end = active_oos_end()
    return rw.loc[FULL_START:oos_end]


def load_compustat() -> pd.DataFrame:
    df = pd.read_parquet("data/fundamentals/compustat_quarterly.parquet")
    df["available_date"] = pd.to_datetime(df["available_date"])
    df["datadate"] = pd.to_datetime(df["datadate"])
    df["permno"] = pd.to_numeric(df["permno"], errors="coerce")
    return df.dropna(subset=["permno"]).assign(permno=lambda x: x["permno"].astype(int))


def load_ibes() -> pd.DataFrame:
    path = Path("data/analyst/ibes_signals.parquet")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["statpers"] = pd.to_datetime(df["statpers"]) + pd.offsets.MonthEnd(0)
    df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
    return df


def run_backtest_exact(
    ret_wide: pd.DataFrame,
    panels: dict[str, pd.DataFrame],
    is_liquid: pd.DataFrame,
    basket_size: int,
    hold_months: int,
    weights: dict[str, float],
    score_weighted: bool = False,
) -> tuple[pd.Series, pd.DataFrame]:
    dates = ret_wide.index
    rebal_dates = dates[::hold_months]
    port_rets = []
    prev_hold = set()
    log_rows = []

    for i, rdate in enumerate(rebal_dates[:-1]):
        next_rdate = rebal_dates[i + 1]
        universe = set(ret_wide.columns)
        if not is_liquid.empty and rdate in is_liquid.index:
            liq_row = is_liquid.loc[rdate].fillna(False)
            universe = universe.intersection(liq_row[liq_row].index)

        scored = {}
        for fn, panel in panels.items():
            if rdate not in panel.index:
                continue
            row = panel.loc[rdate].dropna().replace([np.inf, -np.inf], np.nan).dropna()
            row = row[row.index.isin(universe)]
            z = _zscore(row)
            if len(z) > basket_size:
                scored[fn] = z

        if len(scored) != len(panels):
            port_rets.append(np.nan)
            continue

        common = list(scored.values())[0].index
        for z in scored.values():
            common = common.intersection(z.index)

        if len(common) < basket_size:
            port_rets.append(np.nan)
            continue

        factor_names = list(scored.keys())
        w_arr = np.array([weights.get(fn, 0.0) for fn in factor_names], dtype=float)
        if w_arr.sum() <= 0:
            port_rets.append(np.nan)
            continue
        w_arr = w_arr / w_arr.sum()
        mat = np.column_stack([scored[fn].reindex(common).values for fn in factor_names])
        composite = pd.Series(mat @ w_arr, index=common).sort_values(ascending=False)

        top_n = set(composite.head(basket_size).index)
        turnover = len(top_n.symmetric_difference(prev_hold)) / (2 * basket_size) if prev_hold else 1.0
        tc_drag = turnover * TC

        period_slice = ret_wide.loc[
            (ret_wide.index > rdate) & (ret_wide.index <= next_rdate),
            list(top_n),
        ].dropna(how="all", axis=1)
        if period_slice.empty:
            port_rets.append(np.nan)
            continue

        if score_weighted:
            live = [c for c in period_slice.columns if c in composite.index]
            if live:
                ranks = composite.reindex(live).rank()
                stock_w = (ranks / ranks.sum()).reindex(period_slice.columns).fillna(0)
                sw_monthly = period_slice.mul(stock_w, axis=1).sum(axis=1)
            else:
                sw_monthly = period_slice.mean(axis=1)
            raw_ret = (1 + sw_monthly).prod() - 1
        else:
            raw_ret = (1 + period_slice.mean(axis=1)).prod() - 1

        rf_period = (1 + RISK_FREE_ANN) ** (hold_months / 12) - 1
        period_ret = raw_ret - tc_drag + 0.0 * rf_period
        port_rets.append(period_ret)
        prev_hold = top_n

        log_rows.append({
            "rebal_date": rdate,
            "n_stocks": basket_size,
            "turnover": round(turnover * 100, 1),
            "period_ret": round(period_ret * 100, 2),
            "raw_ret": round(raw_ret * 100, 2),
            "tc_drag_bp": round(tc_drag * 10000, 1),
            "holdings": ",".join(str(x) for x in sorted(top_n)),
        })

    return pd.Series(port_rets, index=rebal_dates[:-1]), pd.DataFrame(log_rows)


def build_spy_benchmark(ret_wide: pd.DataFrame, hold_months: int) -> pd.Series:
    spy = _load_local_price_series("SPY")
    if spy.empty:
        end_date = (pd.to_datetime(ret_wide.index.max()) + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d")
        raw = yf.download(
            "SPY",
            start=str(ret_wide.index[0].date()),
            end=end_date,
            interval="1mo",
            auto_adjust=True,
            progress=False,
        )["Close"]
        spy = raw.squeeze() if isinstance(raw, pd.DataFrame) else raw
    spy = spy.resample("ME").last()
    spy.index = pd.to_datetime(spy.index) + pd.offsets.MonthEnd(0)
    spy_ret = spy.pct_change().reindex(ret_wide.index)
    dates = ret_wide.index
    rebal_dates = dates[::hold_months]
    bench_rets = []
    for i, rd in enumerate(rebal_dates[:-1]):
        nrd = rebal_dates[i + 1]
        sl = spy_ret.loc[(spy_ret.index > rd) & (spy_ret.index <= nrd)]
        sl = sl.dropna()
        if sl.empty:
            bench_rets.append(np.nan)
            continue
        bench_rets.append((1 + sl).prod() - 1)
    return pd.Series(bench_rets, index=rebal_dates[:-1])
