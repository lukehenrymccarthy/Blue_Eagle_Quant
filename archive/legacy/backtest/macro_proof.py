"""
macro_proof.py
==============
Rigorous multi-test empirical validation of unemployment YOY as a macro/regime
overlay factor in a multi-factor equity model.

Produces output CSVs and PNGs consumed by the dashboard.

Sections:
  A  — Predictive correlations (IC vs all candidate signals)
  B  — Regime bucket analysis (quintile + tercile)
  C  — Rolling stability & subperiod analysis
  D  — Horse race: candidate ranking by OOS IC
  E  — Incremental value / OLS regression with controls
  F  — Overlay implementation comparison

Usage:
  python backtest/macro_proof.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

# ── Project root on sys.path so internal imports work ─────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Constants ─────────────────────────────────────────────────────────────────
IS_START   = "2010-01-01"
IS_END     = "2024-12-31"
OOS_START  = "2025-01-01"

RESULTS_DIR = ROOT / "data" / "results" / "macro_proof"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ROLLING_WINDOW = 48   # months for rolling correlation

# Candidate signals to compare against unrate_yoy
CANDIDATES = [
    "unrate_yoy",
    "unrate_level",
    "unrate_3m",
    "icsa_yoy",
    "hy_spread",
    "yield_curve",
    "indpro_yoy",
]

# Forward return horizons to test
FWD_HORIZONS = ["fwd_1m", "fwd_3m", "fwd_6m", "fwd_12m"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_fred() -> pd.DataFrame:
    """Load FRED macro parquet and return a month-end indexed DataFrame."""
    path = ROOT / "data" / "macro" / "fred_raw.parquet"
    fred = pd.read_parquet(path)
    fred.index = pd.to_datetime(fred.index) + pd.offsets.MonthEnd(0)
    fred = fred.sort_index()
    return fred


def build_signals(fred: pd.DataFrame) -> pd.DataFrame:
    """
    Construct candidate macro/regime signals from raw FRED data.

    All columns are lagged 1 month (.shift(1)) to reflect real-world release
    timing: the signal is known at end of prior month and used the following
    month (no look-ahead).

    Parameters
    ----------
    fred : pd.DataFrame
        Monthly FRED data with a DatetimeIndex aligned to month-end.

    Returns
    -------
    pd.DataFrame
        Columns: unrate_yoy, unrate_level, unrate_3m, icsa_yoy,
                 hy_spread, yield_curve, indpro_yoy
    """
    df = pd.DataFrame(index=fred.index)

    # Unemployment YOY change (12-month difference in percentage points)
    df["unrate_yoy"]   = fred["UNRATE"].diff(12)

    # Unemployment level
    df["unrate_level"] = fred["UNRATE"]

    # Unemployment 3-month change
    df["unrate_3m"]    = fred["UNRATE"].diff(3)

    # Initial claims YOY % change
    df["icsa_yoy"] = (
        fred["ICSA"].diff(12) / fred["ICSA"].shift(12) * 100
    )

    # High-yield OAS spread (credit risk proxy)
    df["hy_spread"]    = fred["BAMLH0A0HYM2"]

    # Yield curve slope (10Y minus 2Y)
    df["yield_curve"]  = fred["T10Y2Y"]

    # Industrial production YOY % change (activity proxy)
    df["indpro_yoy"]   = fred["INDPRO"].pct_change(12) * 100

    # Lag all signals by 1 month — signal known at end of prior month
    df = df.shift(1)

    return df


def build_forward_returns(mkt: pd.Series) -> pd.DataFrame:
    """
    Build forward-looking return and risk targets from a monthly return series.

    Parameters
    ----------
    mkt : pd.Series
        Monthly equal-weight market return series (month-end indexed).

    Returns
    -------
    pd.DataFrame
        Columns: fwd_1m, fwd_3m, fwd_6m, fwd_12m, fwd_vol_3m, fwd_dd_6m
    """
    df = pd.DataFrame(index=mkt.index)

    # Simple 1-month forward return
    df["fwd_1m"]  = mkt.shift(-1)

    # Compounded multi-month forward returns (shift so the window ends n months out)
    for n, col in [(3, "fwd_3m"), (6, "fwd_6m"), (12, "fwd_12m")]:
        df[col] = (
            (1 + mkt)
            .rolling(n)
            .apply(np.prod, raw=True)
            .shift(-n)
            - 1
        )

    # Annualised forward realised volatility (next 3 months)
    df["fwd_vol_3m"] = mkt.rolling(3).std().shift(-3) * np.sqrt(12)

    # Forward maximum drawdown over next 6 months
    price = (1 + mkt).cumprod()
    fwd_dd = pd.Series(index=mkt.index, dtype=float)
    for i in range(len(price)):
        window = price.iloc[i : i + 6]
        if len(window) < 2:
            fwd_dd.iloc[i] = np.nan
        else:
            rolling_max = window.cummax()
            dd_series = (window - rolling_max) / rolling_max
            fwd_dd.iloc[i] = dd_series.min()
    df["fwd_dd_6m"] = fwd_dd

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. SECTION A — Predictive Correlations
# ══════════════════════════════════════════════════════════════════════════════

def _ic_stats(x: pd.Series, y: pd.Series):
    """
    Compute Pearson and Spearman IC + Pearson t-stat and p-value.
    Returns (pearson_r, spearman_r, t_stat, p_val) or NaNs on failure.
    """
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 10:
        return np.nan, np.nan, np.nan, np.nan
    x_c, y_c = df.iloc[:, 0], df.iloc[:, 1]
    pr, pp = pearsonr(x_c, y_c)
    sr, _  = spearmanr(x_c, y_c)
    n = len(df)
    t = pr * np.sqrt(n - 2) / np.sqrt(max(1 - pr**2, 1e-12))
    return pr, sr, t, pp


def section_a_predictive(
    signals: pd.DataFrame,
    fwd_rets: pd.DataFrame,
    is_mask: pd.Series,
    oos_mask: pd.Series,
) -> pd.DataFrame:
    """
    Compute Pearson and Spearman ICs for each candidate signal × forward horizon,
    split by IS and OOS periods.

    Notes
    -----
    Negative correlation with fwd_1m is expected for unrate_yoy (higher
    unemployment deterioration predicts weaker equity returns ahead).
    """
    rows = []
    for sig in CANDIDATES:
        if sig not in signals.columns:
            continue
        for hor in FWD_HORIZONS:
            if hor not in fwd_rets.columns:
                continue

            is_pr, is_sr, is_t, is_p = _ic_stats(
                signals.loc[is_mask, sig], fwd_rets.loc[is_mask, hor]
            )
            oos_pr, oos_sr, oos_t, oos_p = _ic_stats(
                signals.loc[oos_mask, sig], fwd_rets.loc[oos_mask, hor]
            )

            rows.append(dict(
                signal=sig,
                horizon=hor,
                is_pearson=round(is_pr, 4) if not np.isnan(is_pr) else np.nan,
                is_spearman=round(is_sr, 4) if not np.isnan(is_sr) else np.nan,
                is_tstat=round(is_t, 3) if not np.isnan(is_t) else np.nan,
                is_pval=round(is_p, 4) if not np.isnan(is_p) else np.nan,
                oos_pearson=round(oos_pr, 4) if not np.isnan(oos_pr) else np.nan,
                oos_spearman=round(oos_sr, 4) if not np.isnan(oos_sr) else np.nan,
                oos_tstat=round(oos_t, 3) if not np.isnan(oos_t) else np.nan,
            ))

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 3. SECTION B — Regime Bucket Analysis
# ══════════════════════════════════════════════════════════════════════════════

def _bucket_stats(
    bucket_label: str,
    idx: pd.Index,
    fwd_rets: pd.DataFrame,
) -> dict:
    """Summarise forward return statistics for a set of date indices."""
    sub = fwd_rets.loc[idx].dropna(subset=["fwd_1m"])
    n = len(sub)
    if n == 0:
        return dict(bucket=bucket_label, n_obs=0)

    price = (1 + sub["fwd_1m"]).cumprod()
    peak  = price.cummax()
    dd    = ((price - peak) / peak).min()

    return dict(
        bucket=bucket_label,
        n_obs=n,
        avg_fwd_1m=sub["fwd_1m"].mean(),
        avg_fwd_3m=sub["fwd_3m"].dropna().mean() if "fwd_3m" in sub.columns else np.nan,
        avg_fwd_12m=sub["fwd_12m"].dropna().mean() if "fwd_12m" in sub.columns else np.nan,
        avg_fwd_vol_3m=sub["fwd_vol_3m"].dropna().mean() if "fwd_vol_3m" in sub.columns else np.nan,
        hit_rate_1m=(sub["fwd_1m"] > 0).mean(),
        max_dd_6m=sub["fwd_dd_6m"].dropna().mean() if "fwd_dd_6m" in sub.columns else np.nan,
        std_fwd_1m=sub["fwd_1m"].std(),
    )


def section_b_regimes(
    signals: pd.DataFrame,
    fwd_rets: pd.DataFrame,
    mkt: pd.Series,
    is_mask: pd.Series,
) -> dict:
    """
    Analyse market returns across unrate_yoy quintiles and terciles.

    Returns
    -------
    dict with keys:
        quintile_stats  — pd.DataFrame (5 rows)
        tercile_stats   — pd.DataFrame (3 rows)
        quintile_labels — pd.Series (labelled IS dates)
        tercile_labels  — pd.Series (labelled IS dates)
    """
    sig_is = signals.loc[is_mask, "unrate_yoy"].dropna()

    # ── Quintiles ─────────────────────────────────────────────────────────────
    q_labels = pd.qcut(sig_is, q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])

    q_rows = []
    for bucket in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        idx = q_labels[q_labels == bucket].index
        idx = idx.intersection(fwd_rets.index)
        q_rows.append(_bucket_stats(bucket, idx, fwd_rets))
    quintile_stats = pd.DataFrame(q_rows)

    # ── Terciles ──────────────────────────────────────────────────────────────
    t_labels = pd.qcut(sig_is, q=3, labels=["Low", "Neutral", "High"])

    t_rows = []
    for bucket in ["Low", "Neutral", "High"]:
        idx = t_labels[t_labels == bucket].index
        idx = idx.intersection(fwd_rets.index)
        t_rows.append(_bucket_stats(bucket, idx, fwd_rets))
    tercile_stats = pd.DataFrame(t_rows)

    return dict(
        quintile_stats=quintile_stats,
        tercile_stats=tercile_stats,
        quintile_labels=q_labels,
        tercile_labels=t_labels,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. SECTION C — Rolling Stability
# ══════════════════════════════════════════════════════════════════════════════

def section_c_rolling(
    signals: pd.DataFrame,
    fwd_rets: pd.DataFrame,
    window: int = ROLLING_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rolling Pearson correlation and regression beta between unrate_yoy and
    fwd_1m. Also computes subperiod statistics.

    Returns
    -------
    rolling_df  : pd.DataFrame — columns: date, rolling_corr, rolling_beta,
                  beta_ci_lo, beta_ci_hi
    subperiod_df: pd.DataFrame — per-subperiod correlation stats
    """
    combined = pd.concat(
        [signals["unrate_yoy"], fwd_rets["fwd_1m"]], axis=1
    ).dropna()

    dates  = combined.index
    corrs  = []
    betas  = []
    ci_lo  = []
    ci_hi  = []

    for i in range(len(dates)):
        if i < window - 1:
            corrs.append(np.nan)
            betas.append(np.nan)
            ci_lo.append(np.nan)
            ci_hi.append(np.nan)
            continue

        sub = combined.iloc[i - window + 1 : i + 1]
        x   = sub.iloc[:, 0]
        y   = sub.iloc[:, 1]

        r, _ = pearsonr(x, y)
        corrs.append(r)

        # OLS beta with confidence interval
        X = sm.add_constant(x)
        try:
            res = sm.OLS(y, X).fit(cov_type="HC3")
            betas.append(res.params.iloc[1])
            ci = res.conf_int(alpha=0.05)
            ci_lo.append(ci.iloc[1, 0])
            ci_hi.append(ci.iloc[1, 1])
        except Exception:
            betas.append(np.nan)
            ci_lo.append(np.nan)
            ci_hi.append(np.nan)

    rolling_df = pd.DataFrame(dict(
        date=dates,
        rolling_corr=corrs,
        rolling_beta=betas,
        beta_ci_lo=ci_lo,
        beta_ci_hi=ci_hi,
    )).set_index("date")

    # ── Subperiod breakdown ────────────────────────────────────────────────────
    subperiods = [
        ("2010-2014", "2010-01-01", "2014-12-31"),
        ("2015-2019", "2015-01-01", "2019-12-31"),
        ("2020-2022", "2020-01-01", "2022-12-31"),
        ("2023-2024", "2023-01-01", "2024-12-31"),
    ]

    sub_rows = []
    for label, start, end in subperiods:
        mask = (combined.index >= start) & (combined.index <= end)
        sub  = combined.loc[mask]
        if len(sub) < 5:
            continue
        x, y = sub.iloc[:, 0], sub.iloc[:, 1]
        pr, pp = pearsonr(x, y)
        sr, _  = spearmanr(x, y)
        n = len(sub)
        t = pr * np.sqrt(n - 2) / np.sqrt(max(1 - pr**2, 1e-12))

        # Correlations with longer horizons — slice fwd_rets by date range
        sp_date_mask = (fwd_rets.index >= start) & (fwd_rets.index <= end)
        fwd3  = fwd_rets.loc[sp_date_mask, "fwd_3m"].dropna()
        fwd12 = fwd_rets.loc[sp_date_mask, "fwd_12m"].dropna()
        sig3  = signals.loc[fwd3.index, "unrate_yoy"].dropna()
        sig12 = signals.loc[fwd12.index, "unrate_yoy"].dropna()

        pr3  = pearsonr(sig3, fwd3.loc[sig3.index])[0] if len(sig3) >= 5 else np.nan
        pr12 = pearsonr(sig12, fwd12.loc[sig12.index])[0] if len(sig12) >= 5 else np.nan

        sub_rows.append(dict(
            subperiod=label,
            n_obs=n,
            corr_fwd_1m=round(pr, 4),
            corr_fwd_3m=round(pr3, 4) if not np.isnan(pr3) else np.nan,
            corr_fwd_12m=round(pr12, 4) if not np.isnan(pr12) else np.nan,
            t_stat=round(t, 3),
            p_val=round(pp, 4),
        ))

    subperiod_df = pd.DataFrame(sub_rows)
    return rolling_df, subperiod_df


# ══════════════════════════════════════════════════════════════════════════════
# 5. SECTION D — Horse Race
# ══════════════════════════════════════════════════════════════════════════════

def section_d_horse_race(
    signals: pd.DataFrame,
    fwd_rets: pd.DataFrame,
    is_mask: pd.Series,
    oos_mask: pd.Series,
) -> pd.DataFrame:
    """
    Score and rank all candidate signals by predictive quality.

    Metrics computed per signal:
      - predictive_ic_1m   : IS Pearson IC vs fwd_1m
      - predictive_ic_12m  : IS Pearson IC vs fwd_12m
      - oos_predictive_ic  : OOS Pearson IC vs fwd_1m  (primary ranking key)
      - downside_ic        : IS IC vs fwd_vol_3m  (negative = predicts lower vol)
      - stability_score    : mean absolute rolling 48M correlation (IS)
    """
    rows = []
    for sig in CANDIDATES:
        if sig not in signals.columns:
            continue

        ic_1m,  _, t1,  _ = _ic_stats(signals.loc[is_mask, sig],  fwd_rets.loc[is_mask, "fwd_1m"])
        ic_12m, _, t12, _ = _ic_stats(signals.loc[is_mask, sig],  fwd_rets.loc[is_mask, "fwd_12m"])
        ic_oos, _, toos,_ = _ic_stats(signals.loc[oos_mask, sig], fwd_rets.loc[oos_mask, "fwd_1m"])
        ic_vol, _, _,   _ = _ic_stats(signals.loc[is_mask, sig],  fwd_rets.loc[is_mask, "fwd_vol_3m"])

        # Rolling stability: mean |rolling 48M corr| over IS period
        combo = pd.concat([signals[sig], fwd_rets["fwd_1m"]], axis=1).dropna()
        combo = combo.loc[is_mask.index[is_mask].intersection(combo.index)]
        roll_corr = []
        for i in range(ROLLING_WINDOW, len(combo) + 1):
            sub = combo.iloc[i - ROLLING_WINDOW : i]
            r, _ = pearsonr(sub.iloc[:, 0], sub.iloc[:, 1])
            roll_corr.append(abs(r))
        stability = np.mean(roll_corr) if roll_corr else np.nan

        rows.append(dict(
            signal=sig,
            predictive_ic_1m=round(ic_1m, 4)  if not np.isnan(ic_1m)  else np.nan,
            predictive_ic_12m=round(ic_12m, 4) if not np.isnan(ic_12m) else np.nan,
            oos_predictive_ic=round(ic_oos, 4) if not np.isnan(ic_oos) else np.nan,
            downside_ic=round(ic_vol, 4)        if not np.isnan(ic_vol) else np.nan,
            stability_score=round(stability, 4) if not np.isnan(stability) else np.nan,
            is_selected=(sig == "unrate_yoy"),
        ))

    df = pd.DataFrame(rows)
    df = df.sort_values("oos_predictive_ic").reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6. SECTION E — Incremental Value / OLS Regression
# ══════════════════════════════════════════════════════════════════════════════

def section_e_incremental(
    signals: pd.DataFrame,
    fwd_rets: pd.DataFrame,
    is_mask: pd.Series,
) -> pd.DataFrame:
    """
    OLS regression of fwd_3m on macro predictors with robust SE (HC3).

    Reports: coefficient, t-stat, p-value, VIF for each predictor.
    Also reports univariate R² vs full-model R².

    Independence finding: if unrate_yoy coefficient is significant even with
    controls, it carries information orthogonal to other macro variables.
    """
    regressors = ["unrate_yoy", "hy_spread", "yield_curve", "indpro_yoy"]
    y_col = "fwd_3m"

    combo = pd.concat(
        [signals[regressors], fwd_rets[y_col]], axis=1
    ).loc[is_mask].dropna()

    y = combo[y_col]
    X = combo[regressors]

    # ── Full model (HC3 robust SE) ─────────────────────────────────────────────
    X_const = sm.add_constant(X)
    res_full = sm.OLS(y, X_const).fit(cov_type="HC3")

    # ── Univariate R² for unrate_yoy ──────────────────────────────────────────
    X_uni = sm.add_constant(combo[["unrate_yoy"]])
    res_uni = sm.OLS(y, X_uni).fit()

    # ── VIF ───────────────────────────────────────────────────────────────────
    vif_vals = {}
    for i, col in enumerate(regressors):
        vif_vals[col] = variance_inflation_factor(X.values, i)

    rows = []
    for pred in regressors:
        coef  = res_full.params.get(pred, np.nan)
        tstat = res_full.tvalues.get(pred, np.nan)
        pval  = res_full.pvalues.get(pred, np.nan)
        rows.append(dict(
            predictor=pred,
            coef=round(coef, 5),
            t_stat=round(tstat, 3),
            p_value=round(pval, 4),
            vif=round(vif_vals.get(pred, np.nan), 2),
        ))

    reg_df = pd.DataFrame(rows)
    reg_df.attrs["r2_univariate"] = round(res_uni.rsquared, 5)
    reg_df.attrs["r2_full"]       = round(res_full.rsquared, 5)
    reg_df.attrs["n_obs"]         = len(combo)
    reg_df.attrs["model_summary"] = str(res_full.summary())

    return reg_df


# ══════════════════════════════════════════════════════════════════════════════
# 7. SECTION F — Overlay Logic Comparison
# ══════════════════════════════════════════════════════════════════════════════

def _rolling_zscore(series: pd.Series, window: int = 36) -> pd.Series:
    """Rolling z-score using trailing `window`-month mean and std."""
    mu  = series.rolling(window, min_periods=12).mean()
    sig = series.rolling(window, min_periods=12).std()
    return (series - mu) / sig.replace(0, np.nan)


def _portfolio_stats(rets: pd.Series, rf_ann: float = 0.0) -> dict:
    """Annualised performance statistics for a monthly return series."""
    if len(rets.dropna()) < 2:
        return dict(ann_return=np.nan, ann_vol=np.nan, sharpe=np.nan,
                    max_drawdown=np.nan, calmar=np.nan, hit_rate=np.nan)
    r = rets.dropna()
    ann_ret  = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol  = r.std() * np.sqrt(12)
    sharpe   = (ann_ret - rf_ann) / ann_vol if ann_vol > 0 else np.nan
    price    = (1 + r).cumprod()
    peak     = price.cummax()
    dd       = (price - peak) / peak
    max_dd   = dd.min()
    calmar   = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
    hit_rate = (r > 0).mean()
    return dict(
        ann_return=round(ann_ret, 5),
        ann_vol=round(ann_vol, 5),
        sharpe=round(sharpe, 4),
        max_drawdown=round(max_dd, 5),
        calmar=round(calmar, 4),
        hit_rate=round(hit_rate, 4),
    )


def section_f_overlay_logic(
    signals: pd.DataFrame,
    mkt: pd.Series,
    is_mask: pd.Series,
    oos_mask: pd.Series,
) -> pd.DataFrame:
    """
    Compare four overlay implementations applied to the market return series.

    Implementations:
      1. Binary      — 0% if unrate_yoy in Q5 else 100%
      2. Stepped     — 70%/85%/100% by z-score thresholds (current model)
      3. Continuous  — linear taper based on z-score
      4. No overlay  — always 100%

    Returns
    -------
    pd.DataFrame with one row per (implementation, window).
    Also stores NAV series as .attrs["nav_series"] for plotting (IS only).
    """
    sig = signals["unrate_yoy"]
    zscore = _rolling_zscore(sig, window=36)

    # Quintile boundaries computed on IS period
    is_sig = sig.loc[is_mask].dropna()
    q5_threshold = is_sig.quantile(0.80)   # top quintile cut-off

    rows = []
    nav_series = {}  # IS NAV curves keyed by implementation name

    for window_name, mask in [("IS", is_mask), ("OOS", oos_mask)]:
        mkt_w  = mkt.loc[mask].dropna()
        sig_w  = sig.loc[mask]
        zsc_w  = zscore.loc[mask]

        # ── 1. Binary ─────────────────────────────────────────────────────────
        binary_exp = (sig_w.fillna(0) < q5_threshold).astype(float)
        r_binary   = mkt_w * binary_exp.reindex(mkt_w.index).fillna(1.0)

        # ── 2. Stepped (z-score thresholds: >2σ → 70%, 1-2σ → 85%, else 100%) ─
        def stepped_exposure(z):
            if pd.isna(z):
                return 1.0
            if z > 2.0:
                return 0.70
            elif z > 1.0:
                return 0.85
            return 1.0

        stepped_exp = zsc_w.reindex(mkt_w.index).apply(stepped_exposure)
        r_stepped   = mkt_w * stepped_exp

        # ── 3. Continuous linear ─────────────────────────────────────────────
        # exposure = 1 - 0.5 * clip(z/2, -1, 1)
        cont_exp  = 1.0 - 0.5 * (zsc_w.reindex(mkt_w.index).fillna(0) / 2).clip(-1, 1)
        r_cont    = mkt_w * cont_exp

        # ── 4. No overlay ────────────────────────────────────────────────────
        r_none = mkt_w.copy()

        for name, r in [
            ("binary", r_binary),
            ("stepped", r_stepped),
            ("continuous", r_cont),
            ("no_overlay", r_none),
        ]:
            stats = _portfolio_stats(r)
            stats["implementation"] = name
            stats["window"] = window_name
            rows.append(stats)

            if window_name == "IS":
                nav_series[name] = (1 + r).cumprod()

    df = pd.DataFrame(rows)[
        ["implementation", "window", "ann_return", "ann_vol",
         "sharpe", "max_drawdown", "calmar", "hit_rate"]
    ]
    df.attrs["nav_series"] = nav_series
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 8. VERDICT
# ══════════════════════════════════════════════════════════════════════════════

def compute_verdict(results: dict) -> str:
    """
    Honest, data-driven verdict on whether to keep unrate_yoy in the model.

    Logic (in priority order):
      1. If OOS IC < 0  → "Drop it"
      2. If OOS IC > 0 but hy_spread or yield_curve OOS IC is consistently
         larger → "Use X instead"
      3. If stepped overlay reduces max_dd meaningfully (>5pp IS OR >2pp OOS)
         AND IS IC > 0 → "Keep as regime overlay"
      4. Else → "Keep but only as risk-control overlay — weak predictive content"
    """
    corr_df    = results.get("predictive_correlations", pd.DataFrame())
    overlay_df = results.get("overlay_logic", pd.DataFrame())

    bullets = []

    # OOS IC for unrate_yoy vs fwd_1m
    oos_ic = np.nan
    if not corr_df.empty:
        row = corr_df[
            (corr_df["signal"] == "unrate_yoy") &
            (corr_df["horizon"] == "fwd_1m")
        ]
        if not row.empty:
            oos_ic = row["oos_pearson"].iloc[0]

    # Max drawdown comparison: stepped vs no_overlay
    dd_improvement_is  = 0.0
    dd_improvement_oos = 0.0
    if not overlay_df.empty:
        def get_dd(impl, window):
            r = overlay_df[
                (overlay_df["implementation"] == impl) &
                (overlay_df["window"] == window)
            ]
            return r["max_drawdown"].iloc[0] if not r.empty else np.nan

        no_dd_is   = get_dd("no_overlay", "IS")
        step_dd_is = get_dd("stepped",    "IS")
        no_dd_oos  = get_dd("no_overlay", "OOS")
        step_dd_oos= get_dd("stepped",    "OOS")

        if not np.isnan(no_dd_is) and not np.isnan(step_dd_is):
            dd_improvement_is  = abs(no_dd_is)  - abs(step_dd_is)   # >0 = improvement
        if not np.isnan(no_dd_oos) and not np.isnan(step_dd_oos):
            dd_improvement_oos = abs(no_dd_oos) - abs(step_dd_oos)

    # Best alternative signal OOS IC
    best_alt_ic  = np.nan
    best_alt_sig = ""
    if not corr_df.empty:
        alt_df = corr_df[
            (corr_df["signal"] != "unrate_yoy") &
            (corr_df["horizon"] == "fwd_1m")
        ].copy()
        if not alt_df.empty:
            best_row = alt_df.loc[alt_df["oos_pearson"].abs().idxmax()]
            best_alt_ic  = best_row["oos_pearson"]
            best_alt_sig = best_row["signal"]

    # ── Decision logic ────────────────────────────────────────────────────────
    if np.isnan(oos_ic):
        verdict = "INCONCLUSIVE — insufficient OOS data"
        bullets.append("OOS period too short to compute reliable IC.")

    elif oos_ic < 0:
        verdict = "DROP IT"
        bullets.append(
            f"OOS predictive IC = {oos_ic:.4f} (negative) — signal inverts out-of-sample."
        )
        bullets.append("Suggests the historical relationship was period-specific or noisy.")

    elif (
        not np.isnan(best_alt_ic) and
        abs(best_alt_ic) > abs(oos_ic) * 1.5 and
        best_alt_sig in ("hy_spread", "yield_curve")
    ):
        verdict = f"USE {best_alt_sig.upper()} INSTEAD"
        bullets.append(
            f"unrate_yoy OOS IC = {oos_ic:.4f} but {best_alt_sig} OOS IC = {best_alt_ic:.4f} (50%+ stronger)."
        )
        bullets.append("Prefer the higher-IC alternative for the regime overlay.")

    elif (
        oos_ic > 0 and
        (dd_improvement_is > 0.05 or dd_improvement_oos > 0.02)
    ):
        is_pct = dd_improvement_is * 100
        oos_pct = dd_improvement_oos * 100
        verdict = "KEEP AS REGIME OVERLAY"
        bullets.append(
            f"OOS IC positive ({oos_ic:.4f}) and stepped overlay reduces max_dd "
            f"by {is_pct:.1f}pp IS / {oos_pct:.1f}pp OOS."
        )
        bullets.append("Signal has both directional and risk-management utility.")

    else:
        verdict = "KEEP BUT ONLY AS RISK-CONTROL OVERLAY — weak predictive content"
        bullets.append(f"OOS IC = {oos_ic:.4f} (positive but marginal).")
        bullets.append(
            f"Drawdown improvement: {dd_improvement_is*100:.1f}pp IS, "
            f"{dd_improvement_oos*100:.1f}pp OOS."
        )
        bullets.append(
            "Signal does not consistently predict returns but reduces tail risk."
        )

    lines = [
        "VERDICT: " + verdict,
        "",
        "Supporting evidence:",
    ] + [f"  • {b}" for b in bullets]

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 9. PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_regime_buckets(quintile_stats: pd.DataFrame) -> None:
    """
    Bar chart: Q1–Q5 quintile labels vs average forward 1-month return.
    Green bars for positive, red for negative; error bars = std / sqrt(n).
    """
    df   = quintile_stats.copy()
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["green" if v >= 0 else "red" for v in df["avg_fwd_1m"]]
    yerr   = df["std_fwd_1m"] / np.sqrt(df["n_obs"].clip(lower=1))

    ax.bar(
        df["bucket"], df["avg_fwd_1m"] * 100,
        color=colors, alpha=0.75, edgecolor="black",
        yerr=yerr * 100, capsize=4, error_kw=dict(elinewidth=1, ecolor="black")
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(
        "Market Returns by Unemployment YOY Quintile\n(Q1 = improving, Q5 = deteriorating)",
        fontsize=12
    )
    ax.set_xlabel("Quintile (unrate_yoy)", fontsize=11)
    ax.set_ylabel("Avg Forward 1M Return (%)", fontsize=11)
    ax.tick_params(labelsize=10)

    # Annotate n_obs
    for _, row in df.iterrows():
        ax.text(
            row["bucket"], (row["avg_fwd_1m"] * 100) + (0.15 if row["avg_fwd_1m"] >= 0 else -0.35),
            f"n={int(row['n_obs'])}", ha="center", fontsize=8, color="dimgray"
        )

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "regime_buckets.png", dpi=150)
    plt.close()


def plot_rolling_correlation(
    rolling_df: pd.DataFrame,
    is_end: str = IS_END,
    oos_start: str = OOS_START,
) -> None:
    """
    48M rolling Pearson correlation between unrate_yoy and fwd_1m.
    Shaded regions for IS and OOS periods; horizontal 0-line.
    """
    fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(
        rolling_df.index, rolling_df["rolling_corr"],
        color="steelblue", linewidth=1.5, label="48M rolling Pearson r"
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)

    # Shade IS region
    is_e = pd.to_datetime(is_end)
    ax.axvspan(
        rolling_df.index.min(), is_e,
        alpha=0.06, color="steelblue", label="In-sample"
    )
    # Shade OOS region
    oos_s = pd.to_datetime(oos_start)
    if oos_s <= rolling_df.index.max():
        ax.axvspan(
            oos_s, rolling_df.index.max(),
            alpha=0.08, color="darkorange", label="Out-of-sample"
        )

    ax.set_title(
        "Rolling 48M Correlation: unrate_yoy vs Forward 1M Market Return",
        fontsize=12
    )
    ax.set_ylabel("Pearson r", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "rolling_correlation.png", dpi=150)
    plt.close()


def plot_overlay_comparison(nav_series: dict) -> None:
    """
    Cumulative NAV curves for 4 overlay implementations (IS period only).
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    style_map = {
        "no_overlay":  dict(color="black",      linestyle="--",  linewidth=1.5),
        "stepped":     dict(color="steelblue",  linestyle="-",   linewidth=2.0),
        "binary":      dict(color="firebrick",  linestyle="-.",  linewidth=1.5),
        "continuous":  dict(color="darkorange", linestyle=":",   linewidth=1.5),
    }
    label_map = {
        "no_overlay": "No overlay (100%)",
        "stepped":    "Stepped (z-score thresholds)",
        "binary":     "Binary (0% if Q5)",
        "continuous": "Continuous linear",
    }

    for name, nav in nav_series.items():
        s = style_map.get(name, {})
        ax.plot(nav.index, nav.values, label=label_map.get(name, name), **s)

    ax.set_title(
        "Overlay Strategy Comparison — Cumulative NAV (In-Sample)",
        fontsize=12
    )
    ax.set_ylabel("Cumulative NAV (start = 1.0)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "overlay_comparison.png", dpi=150)
    plt.close()


def plot_horse_race(horse_race_df: pd.DataFrame) -> None:
    """
    Horizontal bar chart of all candidates ranked by OOS predictive IC.
    Selected signal (unrate_yoy) highlighted.
    """
    df = horse_race_df.dropna(subset=["oos_predictive_ic"]).copy()
    df = df.sort_values("oos_predictive_ic")

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.7)))

    colors = [
        "steelblue" if row["is_selected"] else "lightgray"
        for _, row in df.iterrows()
    ]
    bars = ax.barh(
        df["signal"], df["oos_predictive_ic"],
        color=colors, edgecolor="black", alpha=0.85
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    # Annotate bar values
    for bar, val in zip(bars, df["oos_predictive_ic"]):
        offset = 0.002 if val >= 0 else -0.002
        ax.text(
            val + offset, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left" if val >= 0 else "right",
            fontsize=8
        )

    ax.set_title(
        "Signal Horse Race — OOS Predictive IC vs fwd_1m\n(blue = selected: unrate_yoy)",
        fontsize=11
    )
    ax.set_xlabel("OOS Pearson IC", fontsize=11)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "horse_race.png", dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 10. HELPERS — pretty printing
# ══════════════════════════════════════════════════════════════════════════════

def _header(title: str) -> str:
    line = "═" * 50
    return f"\n{line}\n  {title}\n{line}"


def _fmt(df: pd.DataFrame, float_fmt: str = ".4f") -> str:
    return df.to_string(index=False, float_format=lambda x: f"{x:{float_fmt}}")


# ══════════════════════════════════════════════════════════════════════════════
# 11. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    matplotlib.use("Agg")  # non-interactive backend for file rendering

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    fred     = load_fred()
    signals  = build_signals(fred)

    from backtest.five_factor_model import load_crsp_wide
    ret_wide = load_crsp_wide()
    mkt      = ret_wide.mean(axis=1).rename("mkt_ew")
    mkt.index = pd.to_datetime(mkt.index) + pd.offsets.MonthEnd(0)
    mkt      = mkt.sort_index()

    fwd_rets = build_forward_returns(mkt)

    # ── Align to common index ─────────────────────────────────────────────────
    common_idx = signals.index.intersection(fwd_rets.index).intersection(mkt.index)
    signals    = signals.loc[common_idx]
    fwd_rets   = fwd_rets.loc[common_idx]
    mkt        = mkt.loc[common_idx]

    # ── IS / OOS masks ────────────────────────────────────────────────────────
    is_mask  = (common_idx >= IS_START)  & (common_idx <= IS_END)
    oos_mask = (common_idx >= OOS_START)

    is_mask  = pd.Series(is_mask,  index=common_idx)
    oos_mask = pd.Series(oos_mask, index=common_idx)

    print(f"  IS  period: {common_idx[is_mask].min().date()} — {common_idx[is_mask].max().date()} ({is_mask.sum()} months)")
    print(f"  OOS period: {common_idx[oos_mask].min().date() if oos_mask.any() else 'n/a'} — {common_idx[oos_mask].max().date() if oos_mask.any() else 'n/a'} ({oos_mask.sum()} months)")

    results = {}

    # ── Section A ─────────────────────────────────────────────────────────────
    print(_header("A. PREDICTIVE CORRELATIONS"))
    corr_df = section_a_predictive(signals, fwd_rets, is_mask, oos_mask)
    print(_fmt(corr_df))
    corr_df.to_csv(RESULTS_DIR / "predictive_correlations.csv", index=False)
    results["predictive_correlations"] = corr_df

    # ── Section B ─────────────────────────────────────────────────────────────
    print(_header("B. REGIME BUCKET ANALYSIS"))
    regime = section_b_regimes(signals, fwd_rets, mkt, is_mask)
    print("\nQuintile stats:")
    print(_fmt(regime["quintile_stats"]))
    print("\nTercile stats:")
    print(_fmt(regime["tercile_stats"]))
    regime["quintile_stats"].to_csv(RESULTS_DIR / "regime_buckets.csv", index=False)
    regime["tercile_stats"].to_csv(RESULTS_DIR / "regime_terciles.csv", index=False)
    results["regime"] = regime

    # ── Section C ─────────────────────────────────────────────────────────────
    print(_header("C. ROLLING STABILITY"))
    rolling_df, subperiod_df = section_c_rolling(signals, fwd_rets)
    print("Subperiod breakdown:")
    print(_fmt(subperiod_df))
    rolling_df.to_csv(RESULTS_DIR / "rolling_stability.csv")
    subperiod_df.to_csv(RESULTS_DIR / "subperiod_stability.csv", index=False)
    results["rolling_stability"] = rolling_df
    results["subperiod_stability"] = subperiod_df

    # ── Section D ─────────────────────────────────────────────────────────────
    print(_header("D. HORSE RACE"))
    horse_df = section_d_horse_race(signals, fwd_rets, is_mask, oos_mask)
    print(_fmt(horse_df))
    horse_df.to_csv(RESULTS_DIR / "horse_race.csv", index=False)
    results["horse_race"] = horse_df

    # ── Section E ─────────────────────────────────────────────────────────────
    print(_header("E. INCREMENTAL REGRESSION"))
    reg_df = section_e_incremental(signals, fwd_rets, is_mask)
    print(f"  Univariate R² (unrate_yoy): {reg_df.attrs['r2_univariate']:.5f}")
    print(f"  Full-model R²:              {reg_df.attrs['r2_full']:.5f}")
    print(f"  N observations:             {reg_df.attrs['n_obs']}")
    print(_fmt(reg_df))
    reg_df.to_csv(RESULTS_DIR / "incremental_regression.csv", index=False)
    # Append R² metadata as a footer row for CSV consumers
    meta = pd.DataFrame([
        dict(predictor="_r2_univariate", coef=reg_df.attrs["r2_univariate"],
             t_stat=np.nan, p_value=np.nan, vif=np.nan),
        dict(predictor="_r2_full", coef=reg_df.attrs["r2_full"],
             t_stat=np.nan, p_value=np.nan, vif=np.nan),
    ])
    pd.concat([reg_df, meta], ignore_index=True).to_csv(
        RESULTS_DIR / "incremental_regression.csv", index=False
    )
    results["incremental_regression"] = reg_df

    # ── Section F ─────────────────────────────────────────────────────────────
    print(_header("F. OVERLAY LOGIC COMPARISON"))
    overlay_df = section_f_overlay_logic(signals, mkt, is_mask, oos_mask)
    print(_fmt(overlay_df))
    overlay_df.to_csv(RESULTS_DIR / "overlay_logic.csv", index=False)
    results["overlay_logic"] = overlay_df

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(_header("VERDICT"))
    verdict_text = compute_verdict(results)
    print(verdict_text)
    (RESULTS_DIR / "verdict.txt").write_text(verdict_text)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_regime_buckets(regime["quintile_stats"])
    plot_rolling_correlation(rolling_df)
    plot_overlay_comparison(overlay_df.attrs["nav_series"])
    plot_horse_race(horse_df)
    print(f"  Plots saved to {RESULTS_DIR}/")

    print(f"\nAll outputs saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
