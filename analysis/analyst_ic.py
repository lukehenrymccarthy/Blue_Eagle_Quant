"""
analysis/analyst_ic.py
-----------------------
Cross-sectional IC analysis for IBES analyst rating signals.
Same methodology as fundamental_ic.py — each month, rank stocks by signal,
rank by next-month return, compute Spearman rank IC, average over time.

Signals tested:
  neg_meanrec      - consensus bullishness (negated IBES mean rec)
  buy_pct          - % analysts with buy/strong buy
  net_upgrades     - upgrades minus downgrades this month
  rev_3m           - 3-month recommendation improvement
  buy_pct_rev_3m   - 3-month change in buy%
  coverage         - analyst count (attention)
  coverage_chg_3m  - change in analyst count
  neg_dispersion   - conviction (negated rating std dev)

Usage:
    python analysis/analyst_ic.py
    python analysis/analyst_ic.py --plot
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore")

ANALYST_PATH = Path("data/analyst/ibes_signals.parquet")
RETURNS_PATH = Path("data/fundamentals/crsp_monthly_returns.parquet")
START_DATE   = "2005-01-01"
END_DATE     = "2024-12-31"

SIGNALS = {
    "neg_meanrec":      "Consensus Bullishness (neg mean rec)",
    "buy_pct":          "Buy % (analyst coverage)",
    "net_upgrades":     "Net Upgrades (up - down)",
    "rev_3m":           "Rec Revision Momentum (3m)",
    "buy_pct_rev_3m":   "Buy % Change (3m)",
    "coverage":         "Analyst Coverage (count)",
    "coverage_chg_3m":  "Coverage Change (3m)",
    "neg_dispersion":   "Analyst Conviction (neg dispersion)",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    signals = pd.read_parquet(ANALYST_PATH)
    signals["statpers"] = pd.to_datetime(signals["statpers"]) + pd.offsets.MonthEnd(0)
    signals["permno"]   = signals["permno"].astype(int)

    returns = pd.read_parquet(RETURNS_PATH, columns=["permno", "date", "ret"])
    returns["date"] = pd.to_datetime(returns["date"]) + pd.offsets.MonthEnd(0)
    returns = returns.dropna(subset=["ret"])

    return signals, returns


# ─────────────────────────────────────────────────────────────────────────────
# IC ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def cross_sectional_rank_ic(signal: pd.Series, fwd_ret: pd.Series) -> float:
    df = pd.concat([signal, fwd_ret], axis=1).dropna()
    if len(df) < 20:
        return np.nan
    r, _ = stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1])
    return r


def run_ic_test(signals: pd.DataFrame, returns: pd.DataFrame) -> dict:
    ret_wide = returns.pivot(index="date", columns="permno", values="ret")
    rebalance_dates = ret_wide.loc[START_DATE:END_DATE].index

    ic_series  = {sig: [] for sig in SIGNALS}
    dates_used = []

    for i, rdate in enumerate(tqdm(rebalance_dates[:-1], desc="Months")):
        next_date = rebalance_dates[i + 1]

        # Most recent signal snapshot at or before this rebalance date
        snap = (
            signals[signals["statpers"] <= rdate]
            .sort_values("statpers")
            .groupby("permno")
            .last()
            [list(SIGNALS.keys())]
        )

        if len(snap) < 30 or next_date not in ret_wide.index:
            continue

        fwd_ret = ret_wide.loc[next_date].dropna()
        dates_used.append(rdate)

        for sig in SIGNALS:
            ic = cross_sectional_rank_ic(snap[sig], fwd_ret)
            ic_series[sig].append(ic)

    idx = pd.DatetimeIndex(dates_used)
    return {sig: pd.Series(ic_series[sig], index=idx) for sig in SIGNALS}


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def summarise(ic_dict: dict) -> pd.DataFrame:
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

        # Best 3-year rolling IC_IR window (regime stability check)
        roll_ir = (s.rolling(36).mean() / s.rolling(36).std()).dropna()
        best_ir = roll_ir.max() if len(roll_ir) else np.nan
        worst_ir = roll_ir.min() if len(roll_ir) else np.nan

        rows.append({
            "signal":    sig,
            "label":     SIGNALS[sig],
            "IC_mean":   round(mean,     4),
            "IC_std":    round(std,      4),
            "IC_IR":     round(ir,       3),
            "t_stat":    round(t,        2),
            "hit_rate":  round(hit,      3),
            "best_3y_IR":  round(best_ir,  3) if not np.isnan(best_ir) else np.nan,
            "worst_3y_IR": round(worst_ir, 3) if not np.isnan(worst_ir) else np.nan,
            "n_months":  len(s),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("IC_IR", ascending=False)
        .reset_index(drop=True)
    )


def print_table(summary: pd.DataFrame):
    top = summary.copy()
    top.index = range(1, len(top) + 1)
    top.index.name = "rank"

    fmt = top.copy()
    fmt["IC_mean"]    = fmt["IC_mean"].map("{:+.4f}".format)
    fmt["IC_std"]     = fmt["IC_std"].map("{:.4f}".format)
    fmt["IC_IR"]      = fmt["IC_IR"].map("{:+.3f}".format)
    fmt["t_stat"]     = fmt["t_stat"].map("{:+.2f}".format)
    fmt["hit_rate"]   = fmt["hit_rate"].map("{:.1%}".format)
    fmt["best_3y_IR"] = fmt["best_3y_IR"].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "  nan")
    fmt["worst_3y_IR"]= fmt["worst_3y_IR"].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "  nan")

    print("\n" + "─" * 110)
    print("  ANALYST SIGNAL IC LEADERBOARD  (ranked by IC_IR)")
    print("  IC_IR > 0.3 = usable   |t| > 2.0 = significant   best/worst 3y IR shows regime stability")
    print("─" * 110)
    print(fmt[["label", "IC_mean", "IC_std", "IC_IR", "t_stat",
               "hit_rate", "best_3y_IR", "worst_3y_IR"]].to_string())
    print("─" * 110)


def print_recommendations(summary: pd.DataFrame):
    qualified = summary[summary["t_stat"].abs() >= 1.65].copy()

    print("\n  RECOMMENDED SIGNALS FOR ANALYST FACTOR\n")
    if qualified.empty:
        print("  No signals cleared |t| ≥ 1.65.\n")
        qualified = summary.head(3)

    for _, row in qualified.head(5).iterrows():
        direction = "use as-is" if row["IC_mean"] > 0 else "NEGATE"
        print(f"  ✓  {row['label']:<38s}  IC={row['IC_mean']:+.4f}  "
              f"IR={row['IC_IR']:+.3f}  t={row['t_stat']:+.2f}  → {direction}")
    print()


def print_vs_fundamental(analyst_summary: pd.DataFrame, fund_ir: float = 0.60):
    """Compare best analyst IR against the fundamental factor benchmark."""
    best = analyst_summary.iloc[0]
    print(f"  CONTEXT: Best fundamental signal IC_IR ≈ {fund_ir:.2f} (FCF TTM)")
    print(f"           Best analyst signal IC_IR     = {best['IC_IR']:+.3f} ({best['label']})")
    if abs(best["IC_IR"]) >= fund_ir * 0.5:
        print("  → Analyst signals are competitive — worth including in composite.")
    else:
        print("  → Analyst signals are weaker than fundamentals — use with lower weight.")
    print()


def plot_ic(ic_dict: dict, summary: pd.DataFrame, n: int = 4):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    top_sigs = summary.head(n)["signal"].tolist()
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), sharex=True)
    fig.patch.set_facecolor("#0f1117")
    if n == 1:
        axes = [axes]

    colors = ["#4fc3f7", "#81c784", "#ffb74d", "#f06292"]

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
    fig.suptitle("Analyst Signal Monthly IC (bars) + 12m Rolling Mean (line)",
                 color="white", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    print("Loading data...")
    signals, returns = load_data()
    print(f"  IBES signals: {signals['permno'].nunique():,} permnos, "
          f"{signals['statpers'].min().date()} – {signals['statpers'].max().date()}")
    print(f"  Returns     : {returns['permno'].nunique():,} stocks")

    print("\nRunning cross-sectional IC test...")
    ic_dict = run_ic_test(signals, returns)

    summary = summarise(ic_dict)
    print_table(summary)
    print_recommendations(summary)
    print_vs_fundamental(summary)

    if args.plot:
        plot_ic(ic_dict, summary)

    return summary, ic_dict


if __name__ == "__main__":
    main()
