# SectorScope — Three-Factor Equity Model

A systematic US equity model that holds 10 large-cap stocks, rebalanced every two months. It selects stocks using three signals: earnings surprise (SUE), price momentum, and sector relative strength.

---

## What This Model Does

Every two months the model:
1. Scores the top 1,000 US stocks by market cap across three factors
2. Picks the 10 highest-scoring stocks
3. Holds them equally weighted until the next rebalance

**Backtest results (2005–2024):** 14% annualised return vs 10.7% for SPY, with a lower max drawdown (-37% vs -46%).

**Live results (Apr 2025–Mar 2026):** 23.8% total return vs 18.6% for SPY.

---

## Quickstart — See the Current Portfolio and Results

Run this one command from the project root:

```bash
python show_portfolio.py
```

This will print:
- In-sample backtest metrics (2005–2024)
- Live out-of-sample metrics (Apr 2025–Mar 2026)
- Each 2-month holding period's return vs SPY
- The current 10-stock portfolio scored against today's data

> **Note:** The first run takes about 30 seconds while it loads and scores the data.

---

## Prerequisites

### 1. Python packages

```bash
pip install pandas numpy pyarrow yfinance python-dotenv
```

### 2. Data files

The model reads from pre-built data files that live in `data/`. These are **not included in the repo** (they come from WRDS, a paid academic data service). You need:

| File | What it is |
|---|---|
| `data/fundamentals/crsp_monthly_returns.parquet` | Monthly stock returns (CRSP) |
| `data/fundamentals/compustat_quarterly.parquet` | Quarterly earnings data (Compustat) |
| `data/prices/SPY.parquet` | SPY price history for benchmarking |

If you have WRDS access, pull fresh data with:

```bash
python ingestion/refresh_2025.py
```

---

## Running the Full Backtest

To regenerate all backtest results and save them to `data/results/`:

```bash
# Full backtest report (IS + OOS metrics, holdings history, monthly curve)
python backtest/core_model_report.py

# Clean one-year OOS summary (Apr 2025 – Mar 2026)
python backtest/oos_annual_summary.py
```

Output files written to `data/results/`:

| File | Contents |
|---|---|
| `core_model_backtest_report.md` | Full markdown report with all metrics |
| `core_model_portfolio_summary.csv` | IS and OOS metrics, model vs SPY |
| `oos_annual_summary.csv` | One-year OOS summary table |
| `core_model_backtest_detail.csv` | Every period: returns, holdings, NAV |
| `core_model_monthly_curve.csv` | Month-by-month return path |
| `core_model_holdings_latest.csv` | Most recent portfolio |

---

## The Three Factors

**SUE — Standardised Unexpected Earnings (80% weight)**
Measures how much a company beat or missed earnings expectations, scaled by the historical volatility of its own earnings. Companies that beat estimates tend to keep outperforming for several months as the market slowly absorbs the news. This is the primary driver of the model.

**Sector Relative Strength (15% weight)**
Measures whether the stock's sector is currently attracting capital inflows. Sectors in favour tend to stay in favour over the short term. This acts as a macro filter — it avoids stocks in sectors that are broadly out of favour even if the stock itself scores well.

**Residual 52-Week-High Momentum (5% weight)**
Measures how close a stock is trading to its 52-week high, after stripping out the sector's contribution. Stocks near their 52-week high on a sector-adjusted basis tend to continue outperforming. This is a minor tiebreaker.

---

## Project Structure

```
show_portfolio.py              ← START HERE: current portfolio + results
backtest/
  core_model.py                ← runs the IS + OOS backtest
  core_model_report.py         ← generates the full report and CSVs
  core_model_robustness.py     ← tests different basket sizes and hold periods
  oos_annual_summary.py        ← one-year OOS summary
sectorscope/
  core_strategy.py             ← backtest engine and data loading
  factors.py                   ← SUE, momentum, sector RS signal construction
  metrics.py                   ← Sharpe, Sortino, Calmar, drawdown calculations
  modeling.py                  ← factor weights
ingestion/
  wrds_returns.py              ← pulls CRSP monthly returns from WRDS
  wrds_fundamentals.py         ← pulls Compustat quarterly data from WRDS
  refresh_2025.py              ← incremental data refresh (run periodically)
data/
  fundamentals/                ← parquet files for returns and earnings
  prices/                      ← SPY and sector ETF price files
  results/                     ← all backtest output CSVs and reports
archive/                       ← old research and prior model iterations
```

---

## Keeping Data Current

The model requires monthly CRSP return data to score new periods. To refresh:

```bash
python ingestion/refresh_2025.py
```

Then re-run the backtest and portfolio scorer:

```bash
python backtest/core_model_report.py
python backtest/oos_annual_summary.py
python show_portfolio.py
```

---

## Key Numbers at a Glance

| | Model | SPY |
|---|---|---|
| **IS Ann. Return** (2005–2024) | 14.0% | 10.7% |
| **IS Sharpe** | 0.59 | 0.43 |
| **IS Sortino** | 0.57 | 0.36 |
| **IS Calmar** | 0.38 | 0.23 |
| **IS Max Drawdown** | -36.8% | -46.0% |
| **OOS Total Return** (Apr 2025–Mar 2026) | 23.8% | 18.6% |
| **OOS Sharpe** | 1.64 | 1.36 |
| **OOS Max Drawdown** | -1.8% | -5.8% |

> OOS Sortino and Calmar are not reported — the 11-month live window is too short for those ratios to be statistically meaningful. Use the IS figures as the primary risk benchmarks.
