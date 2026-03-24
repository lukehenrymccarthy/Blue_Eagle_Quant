"""
analysis/fundamental_ic.py
---------------------------
Cross-sectional IC (Information Coefficient) test for fundamental signals.

Unlike macro IC (time-series correlation vs market return), fundamental IC is
computed cross-sectionally: each month, rank all stocks by the signal, rank them
by next-month return, then compute the Spearman rank correlation. Average that
correlation over time.

Metrics reported per signal:
  IC_mean    - mean monthly rank IC  (target: > 0.02 is solid for fundamentals)
  IC_std     - standard deviation of monthly IC
  IC_IR      - IC_mean / IC_std  (consistency; target: > 0.3)
  t_stat     - t-statistic on IC_mean
  hit_rate   - % of months where IC > 0 (signal correct direction)
  IC_1m      - forward 1-month IC (same as IC_mean)
  IC_3m      - forward 3-month IC (does the signal persist?)

Usage:
    cd sectorscope_project
    python analysis/fundamental_ic.py
    python analysis/fundamental_ic.py --plot
    python analysis/fundamental_ic.py --top 10
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore")

FUND_PATH    = Path("data/fundamentals/compustat_quarterly.parquet")
RETURNS_PATH = Path("data/fundamentals/crsp_monthly_returns.parquet")
START_DATE   = "2005-01-01"
END_DATE     = "2024-12-31"

# Signals to test — add any column from compustat_quarterly.parquet here
SIGNALS = {
    "roe_ttm":        "ROE (TTM)",
    "fcf_yield":      "FCF Yield",
    "earnings_yield": "Earnings Yield (1/PE)",
    "pb_ratio":       "Price-to-Book (raw)",
    "neg_pb":         "Value (neg P/B)",        # low P/B = cheap = good → negate
    "fcf_ttm":        "FCF (TTM, $M)",
    "gross_margin":   "Gross Margin",           # computed below
    "asset_turnover": "Asset Turnover",         # computed below
    "accruals":       "Accruals (lower=better)",# computed below
    "leverage":       "Leverage (neg)",         # lt/at, negated
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & SIGNAL ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_engineer(fund_path: Path) -> pd.DataFrame:
    """Load fundamentals and add derived signals not in the raw file."""
    f = pd.read_parquet(fund_path)
    f["available_date"] = pd.to_datetime(f["available_date"])
    f["datadate"]       = pd.to_datetime(f["datadate"])

    # ── Derived signals ───────────────────────────────────────────────────────
    # Gross margin: (revenue - cost_proxy) / revenue
    # Best proxy with available cols: ibq_ttm / saleq (operating margin proxy)
    f["gross_margin"] = np.where(
        f["saleq"] > 0,
        f["ibq_ttm"] / (f["saleq"] * 4),   # annualised sales proxy
        np.nan,
    )

    # Asset turnover: sales / total assets (efficiency)
    f["asset_turnover"] = np.where(
        f["atq"] > 0,
        (f["saleq"] * 4) / f["atq"],
        np.nan,
    )

    # Accruals: net income - operating cash flow (lower = better quality earnings)
    # Scaled by average assets to make it cross-sectionally comparable
    f["accruals"] = np.where(
        f["atq"] > 0,
        (f["ibq_ttm"] - f["oancfq_ttm"]) / f["atq"],
        np.nan,
    )

    # Value factor: negate P/B so HIGH score = cheap stock
    f["neg_pb"] = -f["pb_ratio"]

    # Leverage: total liabilities / total assets (negated so low leverage scores high)
    f["leverage"] = np.where(
        f["atq"] > 0,
        -(f["ltq"] / f["atq"]),
        np.nan,
    )

    return f


def load_monthly_returns(path: Path) -> pd.DataFrame:
    """Monthly stock returns: permno × date × ret."""
    ret = pd.read_parquet(path, columns=["permno", "date", "ret"])
    ret["date"] = pd.to_datetime(ret["date"]) + pd.offsets.MonthEnd(0)
    return ret.dropna(subset=["ret"])


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-SECTIONAL IC ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def get_pit_snapshot(fund: pd.DataFrame, rebalance_date: pd.Timestamp) -> pd.DataFrame:
    """
    Point-in-time snapshot: for each permno, take the most recent quarter
    where available_date <= rebalance_date.
    """
    eligible = fund[fund["available_date"] <= rebalance_date]
    if eligible.empty:
        return pd.DataFrame()
    return (
        eligible.sort_values("datadate")
        .groupby("permno")
        .last()
        .reset_index()[["permno"] + list(SIGNALS.keys())]
    )


def cross_sectional_rank_ic(
    signal: pd.Series,
    fwd_ret: pd.Series,
) -> float:
    """Spearman rank IC between signal and forward return for one month."""
    df = pd.concat([signal, fwd_ret], axis=1).dropna()
    if len(df) < 20:
        return np.nan
    r, _ = stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1])
    return r


def run_ic_test(
    fund: pd.DataFrame,
    returns: pd.DataFrame,
    start: str = START_DATE,
    end: str   = END_DATE,
) -> dict[str, pd.Series]:
    """
    Walk forward month by month, computing cross-sectional rank IC for each signal.
    Returns a dict: signal_name → pd.Series of monthly IC values.
    """
    # Monthly return pivot: date → permno → ret
    ret_wide = returns.pivot(index="date", columns="permno", values="ret")
    rebalance_dates = ret_wide.loc[start:end].index

    ic_series = {sig: [] for sig in SIGNALS}
    dates_used = []

    for i, rdate in enumerate(tqdm(rebalance_dates[:-1], desc="Months")):
        next_date = rebalance_dates[i + 1]

        # PIT fundamentals at this rebalance date
        snap = get_pit_snapshot(fund, rdate)
        if snap.empty or len(snap) < 30:
            continue

        snap = snap.set_index("permno")

        # Next-month returns
        if next_date not in ret_wide.index:
            continue
        fwd_ret = ret_wide.loc[next_date].dropna()

        dates_used.append(rdate)
        for sig in SIGNALS:
            if sig not in snap.columns:
                ic_series[sig].append(np.nan)
                continue
            ic = cross_sectional_rank_ic(snap[sig], fwd_ret)
            ic_series[sig].append(ic)

    idx = pd.DatetimeIndex(dates_used)
    return {sig: pd.Series(ic_series[sig], index=idx) for sig in SIGNALS}


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def summarise(ic_dict: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for sig, series in ic_dict.items():
        s = series.dropna()
        if len(s) < 12:
            continue
        mean = s.mean()
        std  = s.std()
        ir   = mean / std if std > 0 else np.nan
        t    = mean / (std / np.sqrt(len(s))) if std > 0 else np.nan
        hit  = (s > 0).mean()

        rows.append({
            "signal":    sig,
            "label":     SIGNALS[sig],
            "IC_mean":   round(mean, 4),
            "IC_std":    round(std,  4),
            "IC_IR":     round(ir,   3),
            "t_stat":    round(t,    2),
            "hit_rate":  round(hit,  3),
            "n_months":  len(s),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("IC_IR", ascending=False)
        .reset_index(drop=True)
    )


def print_table(summary: pd.DataFrame, n: int = None):
    top = summary.head(n) if n else summary
    top = top.copy()
    top.index = range(1, len(top) + 1)
    top.index.name = "rank"

    fmt = top.copy()
    fmt["IC_mean"]  = fmt["IC_mean"].map("{:+.4f}".format)
    fmt["IC_std"]   = fmt["IC_std"].map("{:.4f}".format)
    fmt["IC_IR"]    = fmt["IC_IR"].map("{:+.3f}".format)
    fmt["t_stat"]   = fmt["t_stat"].map("{:+.2f}".format)
    fmt["hit_rate"] = fmt["hit_rate"].map("{:.1%}".format)

    print("\n" + "─" * 95)
    print("  FUNDAMENTAL SIGNAL IC LEADERBOARD  (ranked by IC_IR)")
    print("  IC_IR > 0.3 = usable   |t| > 2.0 = significant   hit_rate > 52% = directionally stable")
    print("─" * 95)
    print(fmt[["label", "IC_mean", "IC_std", "IC_IR", "t_stat", "hit_rate", "n_months"]].to_string())
    print("─" * 95)


def print_recommendations(summary: pd.DataFrame):
    qualified = summary[summary["t_stat"].abs() >= 1.65].copy()

    print("\n  RECOMMENDED SIGNALS FOR FUNDAMENTAL FACTOR\n")
    if qualified.empty:
        print("  No signals cleared |t| ≥ 1.65. Use top signals as weak priors.\n")
        qualified = summary.head(3)

    for _, row in qualified.head(5).iterrows():
        direction = "positive (high = buy)" if row["IC_mean"] > 0 else "NEGATE  (high = avoid)"
        print(f"  ✓  {row['label']:<28s}  IC={row['IC_mean']:+.4f}  "
              f"IR={row['IC_IR']:+.3f}  t={row['t_stat']:+.2f}  → {direction}")
    print()


def plot_ic(ic_dict: dict[str, pd.Series], summary: pd.DataFrame, n: int = 6):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    top_sigs = summary.head(n)["signal"].tolist()
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=True)
    fig.patch.set_facecolor("#0f1117")

    colors = ["#4fc3f7", "#81c784", "#ffb74d", "#f06292", "#ce93d8", "#80cbc4"]

    for i, (sig, ax) in enumerate(zip(top_sigs, axes)):
        s    = ic_dict[sig].dropna()
        roll = s.rolling(12).mean()
        c    = colors[i % len(colors)]
        label = SIGNALS[sig]
        ir    = summary.loc[summary["signal"] == sig, "IC_IR"].values[0]

        ax.set_facecolor("#0f1117")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#333333")

        ax.bar(s.index, s.values, color=c, alpha=0.25, width=20)
        ax.plot(roll.index, roll.values, color=c, linewidth=1.5,
                label=f"{label}  (IR={ir:+.3f})")
        ax.axhline(0, color="#555555", linewidth=0.7)
        ax.legend(loc="upper left", facecolor="#1a1d26",
                  edgecolor="#333333", labelcolor="white", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    axes[-1].set_xlabel("Date", color="#aaaaaa", fontsize=9)
    fig.suptitle("Fundamental Signal Monthly IC (bars) + 12m Rolling Mean (line)",
                 color="white", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top",  type=int, default=None, help="Rows to show (default: all)")
    parser.add_argument("--plot", action="store_true",    help="Plot rolling IC for top signals")
    args = parser.parse_args()

    print("Loading data...")
    fund    = load_and_engineer(FUND_PATH)
    returns = load_monthly_returns(RETURNS_PATH)
    print(f"  Fundamentals : {fund['permno'].nunique():,} firms, "
          f"{fund['available_date'].min().date()} – {fund['available_date'].max().date()}")
    print(f"  Returns      : {returns['permno'].nunique():,} stocks, "
          f"{returns['date'].min().date()} – {returns['date'].max().date()}")

    print("\nRunning cross-sectional IC test (this takes ~1-2 min)...")
    ic_dict = run_ic_test(fund, returns)

    summary = summarise(ic_dict)
    print_table(summary, n=args.top)
    print_recommendations(summary)

    if args.plot:
        plot_ic(ic_dict, summary)

    return summary, ic_dict


if __name__ == "__main__":
    main()
