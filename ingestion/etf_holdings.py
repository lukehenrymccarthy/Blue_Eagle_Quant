"""
ingestion/etf_holdings.py
--------------------------
Downloads all 11 SPDR Select Sector ETF holdings from State Street Global
Advisors (SSGA) and joins them against the five-factor composite scores.

For each ETF:
  1. Downloads complete holdings + weights from SSGA public Excel file
  2. Joins composite scores (and per-factor z-scores) from five_factor_scores_latest.csv
  3. Fetches current prices via yfinance batch download
  4. Computes a weight-adjusted ETF-level composite score
  5. Generates a cross-sectional buy/sell signal across the 11 ETFs

Outputs
-------
  data/etf_holdings/{ticker}_holdings.parquet   — per-ETF holdings table
  data/etf_holdings/etf_summary.parquet         — ETF-level scores + signals

Run
---
  python ingestion/etf_holdings.py
"""

import sys
import time
import io
import re
import requests
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT_DIR = Path("data/etf_holdings")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SECTOR_ETFS = {
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLV":  "Health Care",
    "XLI":  "Industrials",
    "XLE":  "Energy",
    "XLP":  "Consumer Staples",
    "XLY":  "Consumer Discr.",
    "XLB":  "Materials",
    "XLRE": "Real Estate",
    "XLU":  "Utilities",
    "XLC":  "Comm. Services",
}

SSGA_URL = (
    "https://www.ssga.com/us/en/intermediary/library-content/products/"
    "fund-data/etfs/us/holdings-daily-us-en-{etf}.xlsx"
)

FACTOR_COLS = [
    "momentum_18m", "roe_ttm", "sector_rs_3m", "analyst_rev_3m", "macro_hy"
]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_scores() -> pd.DataFrame:
    """
    Load five-factor composite scores.
    Returns DataFrame indexed by uppercase TICKER with composite + factor cols.
    """
    p = Path("data/results/five_factor_scores_latest.csv")
    if not p.exists():
        raise FileNotFoundError(
            "five_factor_scores_latest.csv not found. "
            "Run backtest/five_factor_model.py first."
        )
    df = pd.read_csv(p, index_col=0)

    # Normalize index to uppercase ticker strings
    if df.index.dtype == object:
        # live_scoring format: index = ticker
        df.index = df.index.str.upper()
    else:
        # backtest format: index = permno (float) — map via IBES
        df.index = df.index.astype(float)
        ibes = pd.read_parquet(
            "data/analyst/ibes_signals.parquet",
            columns=["permno", "ticker", "statpers"],
        )
        ibes["statpers"] = pd.to_datetime(ibes["statpers"])
        ticker_map = (
            ibes.sort_values("statpers")
            .groupby("permno")["ticker"].last()
            .str.upper()
        )
        df.index = ticker_map.reindex(df.index).values
        df = df[df.index.notna()]

    df.index.name = "ticker"
    keep = ["composite"] + [c for c in FACTOR_COLS if c in df.columns]
    return df[keep].dropna(subset=["composite"])


def download_ssga_holdings(etf: str) -> pd.DataFrame:
    """
    Download full holdings from SSGA Excel file.
    Returns DataFrame: ticker, name, weight (as fraction, sums to ~1).
    """
    url = SSGA_URL.format(etf=etf.lower())
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        raw = pd.read_excel(io.BytesIO(resp.content), skiprows=4)
    except Exception as e:
        print(f"    [WARN] SSGA download failed for {etf}: {e}")
        return pd.DataFrame(columns=["ticker", "name", "weight"])

    raw = raw.rename(columns={
        "Ticker": "ticker", "Name": "name", "Weight": "weight"
    })
    raw = raw.dropna(subset=["ticker"])
    # Keep only equity rows (1-5 uppercase letters)
    raw = raw[raw["ticker"].astype(str).str.match(r"^[A-Z]{1,5}$")].copy()
    raw["ticker"] = raw["ticker"].str.upper().str.strip()
    raw["weight"] = pd.to_numeric(raw["weight"], errors="coerce").fillna(0) / 100
    raw["name"]   = raw["name"].astype(str).str.title().str.strip()

    # Normalise weights to sum to 1.0
    total = raw["weight"].sum()
    if total > 0:
        raw["weight"] = raw["weight"] / total

    return raw[["ticker", "name", "weight"]].reset_index(drop=True)


def fetch_prices(tickers: list[str]) -> pd.Series:
    """Batch-fetch latest adjusted close price for a list of tickers."""
    if not tickers:
        return pd.Series(dtype=float, name="price")
    try:
        raw = yf.download(
            tickers, period="5d", auto_adjust=True, progress=False
        )["Close"]
        if isinstance(raw, pd.Series):
            raw = raw.to_frame(name=tickers[0])
        prices = raw.ffill().iloc[-1].rename("price")
        prices.index = prices.index.str.upper()
        return prices
    except Exception as e:
        print(f"    [WARN] price fetch error: {e}")
        return pd.Series(dtype=float, name="price")


def compute_etf_score(holdings: pd.DataFrame) -> float:
    """
    Weighted average composite score over scored holdings only.
    Renormalises weights so they sum to 1 across the scored subset.
    """
    scored = holdings.dropna(subset=["composite"])
    if scored.empty:
        return np.nan
    w = scored["weight"] / scored["weight"].sum()
    return float((scored["composite"] * w).sum())


def signal_label(z: float) -> str:
    if pd.isna(z):   return "N/A"
    if z >  1.0:     return "Strong Buy"
    if z >  0.3:     return "Buy"
    if z > -0.3:     return "Neutral"
    if z > -1.0:     return "Sell"
    return "Strong Sell"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    W = 65
    print("\n" + "═" * W)
    print("  ETF HOLDINGS INGESTION  —  11 SPDR Sector ETFs")
    print("  Source: SSGA daily holdings Excel (full constituent list)")
    print("═" * W + "\n")

    # Load five-factor scores
    print("Loading composite scores...")
    try:
        scores = load_scores()
        print(f"  {len(scores):,} stocks with composite scores\n")
    except Exception as e:
        print(f"  [ERROR] {e}")
        return

    all_summaries = []

    for etf, sector in SECTOR_ETFS.items():
        print(f"  {etf}  ({sector})")

        # ── Step 1: Download holdings ──────────────────────────────────
        holdings = download_ssga_holdings(etf)
        if holdings.empty:
            print(f"    [SKIP] No holdings returned\n")
            continue
        print(f"    {len(holdings)} holdings  |  weight sum: {holdings['weight'].sum():.1%}")

        # ── Step 2: Join composite scores ──────────────────────────────
        score_cols = ["composite"] + [c for c in FACTOR_COLS if c in scores.columns]
        holdings = holdings.merge(
            scores[score_cols].reset_index(),
            on="ticker", how="left",
        )

        scored_n = holdings["composite"].notna().sum()
        scored_w = holdings.loc[holdings["composite"].notna(), "weight"].sum()
        print(f"    Scored: {scored_n}/{len(holdings)} holdings  "
              f"({scored_w:.1%} of ETF weight)")

        # ── Step 3: Current prices ─────────────────────────────────────
        prices = fetch_prices(holdings["ticker"].tolist())
        holdings["price"] = holdings["ticker"].map(prices.to_dict())

        # ── Step 4: ETF-level score ────────────────────────────────────
        etf_score = compute_etf_score(holdings)
        print(f"    Weighted composite: {etf_score:+.4f}" if pd.notna(etf_score)
              else "    Weighted composite: N/A")

        # ── Step 5: Add rank and save ──────────────────────────────────
        holdings = holdings.sort_values("weight", ascending=False).reset_index(drop=True)
        holdings.insert(0, "rank", holdings.index + 1)
        holdings.to_parquet(OUT_DIR / f"{etf}_holdings.parquet", index=False)

        all_summaries.append({
            "etf":          etf,
            "sector":       sector,
            "n_holdings":   len(holdings),
            "n_scored":     int(scored_n),
            "coverage_w":   round(scored_w * 100, 1),
            "etf_score":    round(etf_score, 4) if pd.notna(etf_score) else np.nan,
        })
        print()
        time.sleep(0.4)   # be respectful to SSGA

    # ── Build ETF summary with cross-sectional signals ─────────────────────
    summary = pd.DataFrame(all_summaries)
    valid   = summary["etf_score"].dropna()

    if len(valid) >= 3:
        mu  = valid.mean()
        std = valid.std()
        summary["score_z"] = (
            (summary["etf_score"] - mu) / std if std > 0 else 0.0
        )
    else:
        summary["score_z"] = np.nan

    summary["signal"] = summary["score_z"].apply(signal_label)

    out = OUT_DIR / "etf_summary.parquet"
    summary.to_parquet(out, index=False)

    # ── Print summary table ────────────────────────────────────────────────
    print("═" * W)
    print("  RESULTS  (sorted by weighted composite score)")
    print("═" * W)
    hdr = f"  {'ETF':<6}  {'Sector':<22}  {'Score':>8}  {'Z':>6}  {'Signal':<12}  {'Cov%':>5}"
    print(hdr)
    print("  " + "─" * (W - 2))

    for _, row in summary.sort_values("etf_score", ascending=False).iterrows():
        score_s = f"{row['etf_score']:+.4f}" if pd.notna(row["etf_score"]) else "   N/A"
        z_s     = f"{row['score_z']:+.2f}"   if pd.notna(row.get("score_z")) else "  N/A"
        print(f"  {row['etf']:<6}  {row['sector']:<22}  {score_s:>8}  "
              f"{z_s:>6}  {row['signal']:<12}  {row['coverage_w']:>4.0f}%")

    print("═" * W)
    print(f"\n  Saved → data/etf_holdings/{{ETF}}_holdings.parquet")
    print(f"  Saved → {out}")
    print("  Done.\n")


if __name__ == "__main__":
    main()
