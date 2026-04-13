from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


SECTOR_ETFS = ["XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"]


def _load_local_price_panel(tickers: list[str]) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        path = Path("data/prices") / f"{ticker}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "close" not in df.columns:
            continue
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"])
        else:
            dates = pd.to_datetime(df.index)
        s = pd.Series(df["close"].values, index=dates, name=ticker).sort_index()
        frames.append(s)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


def _load_market_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    local = _load_local_price_panel(tickers)
    if not local.empty:
        monthly = local.loc[start:end].resample("ME").last()
        monthly.index = pd.to_datetime(monthly.index) + pd.offsets.MonthEnd(0)
        return monthly

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval="1mo",
        auto_adjust=True,
        progress=False,
    )["Close"]
    out = raw.squeeze() if isinstance(raw, pd.Series) else raw
    out.index = pd.to_datetime(out.index) + pd.offsets.MonthEnd(0)
    return out


def build_52wk_high(ret_wide: pd.DataFrame) -> pd.DataFrame:
    """
    52-week high proximity signal (George & Hwang 2004).
    """
    log_r = np.log1p(ret_wide.fillna(0))
    cum_log = log_r.cumsum()
    roll_max = cum_log.rolling(12, min_periods=6).max()
    proximity = np.exp(cum_log - roll_max)
    return proximity.shift(1)


def build_residual_52wk_high(ret_wide: pd.DataFrame, sic_map: pd.Series) -> pd.DataFrame:
    """
    Sector-demeaned 52-week high signal so stock-specific momentum is separated
    from the sector sleeve.
    """
    raw = build_52wk_high(ret_wide)

    def _demean_row(row: pd.Series) -> pd.Series:
        s = row.dropna()
        if len(s) < 10:
            return s
        sectors = sic_map.reindex(s.index).fillna("Unknown")
        sector_means = s.groupby(sectors).transform("mean")
        return s - sector_means

    return raw.apply(_demean_row, axis=1)


def build_earnings_yield(
    fund: pd.DataFrame,
    ret_wide: pd.DataFrame,
    sic_map: pd.Series,
) -> pd.DataFrame:
    """
    Earnings yield: TTM net income / market cap, sector-neutralized cross-sectionally.
    """
    fund = fund.copy()
    if "earnings_yield" not in fund.columns:
        fund["earnings_yield"] = fund["ibq_ttm"] / fund["market_cap"].replace(0, np.nan)

    piv = (
        fund.dropna(subset=["earnings_yield"])
        .sort_values("available_date")
        .pivot_table(index="available_date", columns="permno", values="earnings_yield", aggfunc="last")
    )
    piv.index = pd.to_datetime(piv.index)
    panel = piv.resample("ME").last().ffill().reindex(ret_wide.index, method="ffill")

    def _row_z(row: pd.Series) -> pd.Series:
        s = row.dropna().replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) < 20:
            return pd.Series(dtype=float)
        df = s.to_frame("v")
        df["sec"] = sic_map.reindex(df.index).fillna("Unknown")
        lo, hi = df["v"].quantile(0.01), df["v"].quantile(0.99)
        df["v"] = df["v"].clip(lo, hi)

        def _sector_zscore(group: pd.Series) -> pd.Series:
            return (group - group.mean()) / group.std() if len(group) >= 3 and group.std() > 0 else group * 0

        return df.groupby("sec")["v"].transform(_sector_zscore)

    return panel.apply(_row_z, axis=1)


def build_unemployment_macro(ret_wide: pd.DataFrame) -> pd.Series | None:
    raw_path = Path("data/macro/fred_raw.parquet")
    if not raw_path.exists():
        return None
    raw = pd.read_parquet(raw_path)
    raw.index = pd.to_datetime(raw.index)
    if "UNRATE" not in raw.columns:
        return None
    unrate = raw["UNRATE"].resample("ME").last().dropna()
    yoy = unrate.diff(12)
    roll_mean = yoy.rolling(36, min_periods=12).mean()
    roll_std = yoy.rolling(36, min_periods=12).std().replace(0, np.nan)
    macro_z = -((yoy - roll_mean) / roll_std).shift(1)
    return macro_z.reindex(ret_wide.index, method="ffill")


def build_vix_macro(ret_wide: pd.DataFrame) -> pd.Series | None:
    try:
        raw = yf.download(
            "^VIX",
            start="2004-01-01",
            end="2026-06-01",
            interval="1mo",
            auto_adjust=True,
            progress=False,
        )["Close"]
        vix = raw.squeeze().resample("ME").last().dropna()
        return vix.shift(1).reindex(ret_wide.index, method="ffill")
    except Exception:
        return None


def build_sector_rs_1m(ret_wide: pd.DataFrame, sic_map: pd.Series) -> pd.DataFrame:
    if sic_map.empty:
        return pd.DataFrame()
    stock_to_sector = sic_map.reindex(ret_wide.columns).dropna()
    if stock_to_sector.empty:
        return pd.DataFrame()

    # Build sector relative strength directly from stock returns so the factor
    # spans the full CRSP history instead of depending on short local ETF files.
    market_ret = ret_wide.mean(axis=1, skipna=True)
    sector_returns = {}
    for sector in sorted(stock_to_sector.unique()):
        members = stock_to_sector[stock_to_sector == sector].index
        if len(members) == 0:
            continue
        sector_returns[str(sector)] = ret_wide.reindex(columns=members).mean(axis=1, skipna=True)

    if not sector_returns:
        return pd.DataFrame()

    sector_ret_df = pd.DataFrame(sector_returns).reindex(ret_wide.index)
    sector_rs = sector_ret_df.sub(market_ret, axis=0).shift(1)

    matched = [(stock, sector) for stock, sector in stock_to_sector.items() if str(sector) in sector_rs.columns]
    if not matched:
        return pd.DataFrame()

    panel = pd.DataFrame(index=sector_rs.index)
    for stock, sector in matched:
        panel[stock] = sector_rs[str(sector)]
    panel.columns = panel.columns.astype(ret_wide.columns.dtype)
    return panel


def build_neg_dispersion(ibes: pd.DataFrame, ret_wide: pd.DataFrame) -> pd.DataFrame:
    if ibes.empty or "neg_dispersion" not in ibes.columns:
        return pd.DataFrame()

    piv = (
        ibes.dropna(subset=["neg_dispersion"])
        .pivot_table(index="statpers", columns="permno", values="neg_dispersion", aggfunc="last")
    )
    piv.index = pd.to_datetime(piv.index)
    piv.columns = piv.columns.astype(ret_wide.columns.dtype)
    return piv.resample("ME").last().ffill().reindex(ret_wide.index, method="ffill")


def build_sue(fund: pd.DataFrame, ret_wide: pd.DataFrame) -> pd.DataFrame:
    df = fund.dropna(subset=["permno", "epspxq", "datadate"]).copy()
    df = df.sort_values(["permno", "datadate"])
    df["eps_yoy"] = df.groupby("permno")["epspxq"].diff(4)
    df["eps_yoy_std"] = df.groupby("permno")["eps_yoy"].transform(lambda x: x.rolling(8, min_periods=4).std())
    df["sue"] = (df["eps_yoy"] / df["eps_yoy_std"].replace(0, np.nan)).clip(-10, 10)

    df["rdq"] = pd.to_datetime(df["rdq"]) if "rdq" in df.columns else pd.NaT
    df["signal_date"] = df["rdq"].fillna(pd.to_datetime(df["available_date"]))
    df["signal_date"] = pd.to_datetime(df["signal_date"]) + pd.offsets.MonthEnd(1)

    piv = (
        df.dropna(subset=["sue", "signal_date"])
        .sort_values("signal_date")
        .pivot_table(index="signal_date", columns="permno", values="sue", aggfunc="last")
    )
    piv.index = pd.to_datetime(piv.index)
    piv.columns = piv.columns.astype(ret_wide.columns.dtype)
    return piv.resample("ME").last().ffill(limit=3).reindex(ret_wide.index, method="ffill")


def build_liquidity_screen(
    fund: pd.DataFrame,
    ret_wide: pd.DataFrame,
    universe_size: int,
) -> pd.DataFrame:
    if "market_cap" not in fund.columns:
        return pd.DataFrame(True, index=ret_wide.index, columns=ret_wide.columns)

    mkt = (
        fund.dropna(subset=["market_cap"])
        .sort_values("available_date")
        .pivot_table(index="available_date", columns="permno", values="market_cap", aggfunc="last")
    )
    mkt.index = pd.to_datetime(mkt.index)
    mkt = mkt.resample("ME").last().ffill().reindex(ret_wide.index, method="ffill")

    def _top_k(row: pd.Series) -> pd.Series:
        s = row.dropna()
        if s.empty:
            return pd.Series(False, index=row.index)
        res = pd.Series(False, index=row.index)
        res.loc[s.nlargest(universe_size).index] = True
        return res

    return mkt.apply(_top_k, axis=1)


def build_factor_panels(
    ret_wide: pd.DataFrame,
    fund: pd.DataFrame,
    ibes: pd.DataFrame,
    sic_map: pd.Series,
    include_technical: bool = True,
    sector_neutral_momentum: bool = True,
) -> dict[str, pd.DataFrame]:
    panels: dict[str, pd.DataFrame] = {}

    if include_technical:
        if sector_neutral_momentum:
            panels["mom_52wk_high"] = build_residual_52wk_high(ret_wide, sic_map)
        else:
            panels["mom_52wk_high"] = build_52wk_high(ret_wide)
        panels["sector_rs_1m"] = build_sector_rs_1m(ret_wide, sic_map)

    panels["earnings_yield"] = build_earnings_yield(fund, ret_wide, sic_map)
    panels["sue"] = build_sue(fund, ret_wide)
    panels["neg_dispersion"] = build_neg_dispersion(ibes, ret_wide)

    return {name: panel for name, panel in panels.items() if not panel.empty}
