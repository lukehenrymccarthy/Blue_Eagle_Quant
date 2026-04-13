"""
analysis/macro_ic.py
--------------------
IC (Information Coefficient) test for every FRED macro signal.

Since macro signals are market-wide scalars (not cross-sectional), IC here means
correlation between the signal and the NEXT month's equal-weighted market return.
Signals that consistently predict the market direction are worth keeping.

Metrics reported per signal:
  IC_1m       - Pearson correlation with next-month market return
  RankIC_1m   - Spearman rank correlation (more robust to outliers)
  t_stat      - t-statistic on IC_1m (|t| > 2 ≈ significant)
  IC_IR       - IC / std(IC)  (information ratio — consistency matters)
  IC_3m       - Pearson correlation with next-3-month market return
  hit_rate    - fraction of months where signal correctly called direction

Usage:
    cd sectorscope_project
    python analysis/macro_ic.py
    python analysis/macro_ic.py --plot        # show bar chart of top signals
    python analysis/macro_ic.py --top 15      # show more signals
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

MACRO_PATH   = Path("data/macro/fred_signals.parquet")
RETURNS_PATH = Path("data/fundamentals/crsp_monthly_returns.parquet")
START_DATE   = "2005-01-01"   # align with CRSP availability


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_market_returns() -> pd.Series:
    """
    Equal-weighted average monthly return across all CRSP stocks.
    This is what the macro factor is trying to predict.
    """
    ret = pd.read_parquet(RETURNS_PATH, columns=["date", "ret"])
    ret["date"] = pd.to_datetime(ret["date"])
    mkt = (
        ret.dropna(subset=["ret"])
        .groupby("date")["ret"]
        .mean()
        .rename("mkt_ret")
    )
    mkt.index = mkt.index + pd.offsets.MonthEnd(0)
    return mkt.sort_index()


def load_signals() -> pd.DataFrame:
    sig = pd.read_parquet(MACRO_PATH)
    sig.index = pd.to_datetime(sig.index) + pd.offsets.MonthEnd(0)
    return sig.sort_index()


# ─────────────────────────────────────────────────────────────────────────────
# IC COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_forward_returns(mkt: pd.Series) -> pd.DataFrame:
    """Build 1-month and 3-month forward return series."""
    fwd = pd.DataFrame(index=mkt.index)
    fwd["fwd_1m"] = mkt.shift(-1)
    fwd["fwd_3m"] = (
        (1 + mkt.shift(-1)) * (1 + mkt.shift(-2)) * (1 + mkt.shift(-3)) - 1
    )
    return fwd


def ic_stats(signal: pd.Series, forward: pd.Series) -> dict:
    """
    Compute IC statistics between one signal and one forward return series.
    Both series must share the same monthly index.
    """
    df = pd.concat([signal, forward], axis=1).dropna()
    if len(df) < 24:
        return {}

    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values

    pearson_r, pearson_p  = stats.pearsonr(x, y)
    spearman_r, _         = stats.spearmanr(x, y)
    t_stat                = pearson_r * np.sqrt((len(df) - 2) / (1 - pearson_r**2 + 1e-12))
    hit_rate              = np.mean(np.sign(x) == np.sign(y))

    return {
        "n_obs":     len(df),
        "IC_1m":     round(pearson_r, 4),
        "RankIC_1m": round(spearman_r, 4),
        "t_stat":    round(t_stat, 2),
        "p_value":   round(pearson_p, 4),
        "hit_rate":  round(hit_rate, 3),
    }


def run_ic_test(
    signals: pd.DataFrame,
    mkt: pd.Series,
    start: str = START_DATE,
) -> pd.DataFrame:
    """
    Run IC test for every signal column.
    Returns a DataFrame ranked by |t_stat|.
    """
    signals = signals.loc[start:]
    mkt     = mkt.loc[start:]
    fwd     = compute_forward_returns(mkt)

    rows = []
    for col in signals.columns:
        sig = signals[col]

        stats_1m = ic_stats(sig, fwd["fwd_1m"])
        stats_3m = ic_stats(sig, fwd["fwd_3m"])

        if not stats_1m:
            continue

        # Rolling IC over trailing 36 months → IC IR (consistency)
        roll_ic = []
        dates = signals.loc[start:].index
        for i in range(36, len(dates)):
            window_sig = sig.iloc[i - 36 : i]
            window_fwd = fwd["fwd_1m"].iloc[i - 36 : i]
            df_w = pd.concat([window_sig, window_fwd], axis=1).dropna()
            if len(df_w) >= 18:
                r, _ = stats.pearsonr(df_w.iloc[:, 0], df_w.iloc[:, 1])
                roll_ic.append(r)

        ic_ir = (np.mean(roll_ic) / (np.std(roll_ic) + 1e-9)) if roll_ic else np.nan

        rows.append({
            "signal":     col,
            "IC_1m":      stats_1m["IC_1m"],
            "RankIC_1m":  stats_1m["RankIC_1m"],
            "t_stat":     stats_1m["t_stat"],
            "p_value":    stats_1m["p_value"],
            "hit_rate":   stats_1m["hit_rate"],
            "IC_IR":      round(ic_ir, 3) if not np.isnan(ic_ir) else np.nan,
            "IC_3m":      stats_3m.get("IC_1m", np.nan),
            "n_obs":      stats_1m["n_obs"],
        })

    results = (
        pd.DataFrame(rows)
        .assign(abs_t=lambda d: d["t_stat"].abs())
        .sort_values("abs_t", ascending=False)
        .drop(columns="abs_t")
        .reset_index(drop=True)
    )
    results.index += 1
    results.index.name = "rank"
    return results


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def print_table(results: pd.DataFrame, n: int = 10):
    top = results.head(n).copy()
    top["IC_1m"]     = top["IC_1m"].map("{:+.4f}".format)
    top["RankIC_1m"] = top["RankIC_1m"].map("{:+.4f}".format)
    top["t_stat"]    = top["t_stat"].map("{:+.2f}".format)
    top["IC_IR"]     = top["IC_IR"].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "  nan")
    top["IC_3m"]     = top["IC_3m"].map(lambda x: f"{x:+.4f}" if pd.notna(x) else "  nan")
    top["hit_rate"]  = top["hit_rate"].map("{:.1%}".format)

    print("\n" + "─" * 95)
    print(f"  MACRO SIGNAL IC LEADERBOARD  (top {n} by |t-stat|, predicting EW market return)")
    print(f"  |t| > 2.0 = significant at ~95%   |t| > 1.65 = significant at ~90%")
    print("─" * 95)
    print(top[["signal", "IC_1m", "RankIC_1m", "t_stat", "p_value",
               "IC_IR", "IC_3m", "hit_rate"]].to_string())
    print("─" * 95)


def print_recommendations(results: pd.DataFrame):
    """Suggest which signals to use in the macro factor."""
    sig_results = results[results["t_stat"].abs() >= 1.65].copy()

    print("\n  RECOMMENDED SIGNALS FOR MACRO FACTOR")
    print("  (|t| ≥ 1.65, ranked by IC_IR for consistency)\n")

    if sig_results.empty:
        print("  No signals cleared the t-stat threshold.")
        print("  Consider using the top 3 by |t-stat| anyway as weak priors.\n")
        sig_results = results.head(3)

    # For the factor, we want signals that are directionally consistent.
    # Positive IC = higher signal → higher market return (use as-is)
    # Negative IC = higher signal → lower market return (negate it in the factor)
    for _, row in sig_results.assign(abs_ir=sig_results["IC_IR"].abs()).sort_values("abs_ir", ascending=False).head(5).iterrows():
        direction = "positive" if row["IC_1m"] > 0 else "NEGATE (negative IC)"
        print(f"  ✓  {row['signal']:<30s}  IC={row['IC_1m']:+.4f}  "
              f"t={row['t_stat']:+.2f}  IR={row['IC_IR']:+.3f}  → {direction}")

    print()


def plot_ic(results: pd.DataFrame, n: int = 20):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    top = results.head(n).copy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#0f1117")

    for ax in axes:
        ax.set_facecolor("#0f1117")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#333333")

    # ── Left: IC_1m bar chart ────────────────────────────────────────────────
    colors = ["#4fc3f7" if v > 0 else "#f06292" for v in top["IC_1m"]]
    axes[0].barh(top["signal"][::-1], top["IC_1m"][::-1], color=colors[::-1], alpha=0.85)
    axes[0].axvline(0, color="#555555", linewidth=0.8)
    axes[0].set_title("IC (1-month forward)", color="white", fontsize=11)
    axes[0].set_xlabel("Pearson IC", color="#aaaaaa", fontsize=9)
    axes[0].xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # ── Right: t-stat bar chart ──────────────────────────────────────────────
    colors_t = ["#4fc3f7" if v > 0 else "#f06292" for v in top["t_stat"]]
    axes[1].barh(top["signal"][::-1], top["t_stat"][::-1], color=colors_t[::-1], alpha=0.85)
    axes[1].axvline(0,    color="#555555", linewidth=0.8)
    axes[1].axvline(1.65, color="#ffb74d", linewidth=1.0, linestyle="--", alpha=0.6, label="|t|=1.65")
    axes[1].axvline(-1.65, color="#ffb74d", linewidth=1.0, linestyle="--", alpha=0.6)
    axes[1].axvline(2.0,  color="#81c784", linewidth=1.0, linestyle="--", alpha=0.6, label="|t|=2.0")
    axes[1].axvline(-2.0, color="#81c784", linewidth=1.0, linestyle="--", alpha=0.6)
    axes[1].set_title("t-statistic", color="white", fontsize=11)
    axes[1].set_xlabel("t-stat", color="#aaaaaa", fontsize=9)
    axes[1].legend(facecolor="#1a1d26", edgecolor="#333333", labelcolor="white", fontsize=8)

    fig.suptitle(
        "Macro Signal IC vs Equal-Weighted Market Return (1-month forward)",
        color="white", fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top",  type=int, default=10, help="Rows to display")
    parser.add_argument("--plot", action="store_true",  help="Show bar chart")
    args = parser.parse_args()

    print("Loading data...")
    mkt     = load_market_returns()
    signals = load_signals()
    print(f"  Market returns : {len(mkt)} months  ({mkt.index.min().date()} – {mkt.index.max().date()})")
    print(f"  Macro signals  : {signals.shape[1]} columns")

    print("\nRunning IC tests...")
    results = run_ic_test(signals, mkt)

    print_table(results, n=args.top)
    print_recommendations(results)

    if args.plot:
        plot_ic(results, n=min(args.top, 25))

    return results


if __name__ == "__main__":
    main()
