"""
ingestion/massive_prices.py
---------------------------
Pulls daily OHLCV price data for a list of tickers using the Massive SDK
(formerly Polygon.io). Saves each ticker as a parquet file in data/prices/.

Massive SDK docs: https://massive.com/docs
"""

import os
import time
import pandas as pd
from pathlib import Path
from datetime import date
from dotenv import load_dotenv
from massive import RESTClient
from tqdm import tqdm

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY    = os.getenv("MASSIVE_API_KEY")
START_DATE = "2010-01-01"
END_DATE   = date.today().isoformat()
OUTPUT_DIR = Path("data/prices")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Russell 1000 constituent tickers — replace with your full universe list.
# For a real pull, load from a CSV or from the WRDS index membership table.
TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "GOOGL", "BRK.B", "JPM", "UNH", "XOM",
    # ... add full Russell 1000 list here
]

# Sector ETFs (11 GICS sectors via SPDR)
SECTOR_ETFS = [
    "XLC", "XLY", "XLP", "XLE", "XLF",
    "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU",
]


def fetch_ticker(client: RESTClient, ticker: str) -> pd.DataFrame:
    """
    Pull daily aggregate bars for one ticker.
    Returns a DataFrame with columns: date, open, high, low, close, volume, vwap.
    """
    aggs = []

    # list_aggs handles pagination automatically
    for agg in client.list_aggs(
        ticker=ticker,
        multiplier=1,
        timespan="day",
        from_=START_DATE,
        to=END_DATE,
        adjusted=True,      # split/dividend adjusted
        sort="asc",
        limit=50000,        # max per request — SDK auto-paginates
    ):
        aggs.append({
            "date":   pd.to_datetime(agg.timestamp, unit="ms").date(),
            "open":   agg.open,
            "high":   agg.high,
            "low":    agg.low,
            "close":  agg.close,
            "volume": agg.volume,
            "vwap":   agg.vwap,
        })

    if not aggs:
        return pd.DataFrame()

    df = pd.DataFrame(aggs)
    df["ticker"] = ticker
    df = df.set_index("date").sort_index()
    return df


def run():
    client = RESTClient(api_key=API_KEY)
    all_tickers = TICKERS + SECTOR_ETFS

    for ticker in tqdm(all_tickers, desc="Pulling prices"):
        out_path = OUTPUT_DIR / f"{ticker}.parquet"

        # Skip if already pulled today
        if out_path.exists():
            mtime = pd.Timestamp(out_path.stat().st_mtime, unit="s").date()
            if mtime == date.today():
                continue

        try:
            df = fetch_ticker(client, ticker)
            if not df.empty:
                df.to_parquet(out_path)
        except Exception as e:
            print(f"  [WARN] {ticker}: {e}")

        # Polite rate limit pause (Massive free tier = 5 req/min; paid = higher)
        time.sleep(0.15)

    print(f"\nDone. {len(all_tickers)} tickers saved to {OUTPUT_DIR}/")


def load_prices(tickers: list[str] = None) -> pd.DataFrame:
    """
    Load saved price data into a single wide DataFrame of closing prices.
    Index = date, columns = ticker symbols.
    """
    frames = {}
    files = list(OUTPUT_DIR.glob("*.parquet"))

    for f in files:
        ticker = f.stem
        if tickers and ticker not in tickers:
            continue
        df = pd.read_parquet(f)
        frames[ticker] = df["close"]

    if not frames:
        raise FileNotFoundError("No price files found. Run run() first.")

    prices = pd.DataFrame(frames)
    prices.index = pd.to_datetime(prices.index)
    return prices.sort_index()


if __name__ == "__main__":
    run()
