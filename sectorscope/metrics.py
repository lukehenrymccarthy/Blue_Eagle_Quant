import numpy as np
import pandas as pd

def compute_metrics(r: pd.Series, hold_months: int = 1, risk_free_ann: float = 0.05,
                    min_periods: int = None, monthly_curve: pd.Series = None) -> dict:
    """
    Annualised performance metrics for a series of holding-period returns.
    Properly scales Sharpe and vol for non-monthly periods.
    min_periods: override minimum sample size (default: max(6, periods_per_year)).
    monthly_curve: optional monthly return series used for max drawdown — more
                   accurate than computing drawdown on aggregated period returns,
                   which can miss intra-period losses that recover by period end.
    """
    r = pd.Series(r.values, index=r.index).dropna() if hasattr(r, 'values') else r.dropna()
    periods_per_year = 12 / hold_months
    threshold = min_periods if min_periods is not None else max(6, int(periods_per_year))
    if len(r) < threshold:
        return {}

    mu  = float(r.mean())
    std = float(r.std())
    rf_per_period = (1 + risk_free_ann) ** (hold_months / 12) - 1
    ann_ret = (1 + mu)  ** periods_per_year - 1
    ann_vol = std * np.sqrt(periods_per_year)
    sharpe  = (mu - rf_per_period) / std * np.sqrt(periods_per_year) if std > 0 else np.nan

    downside = r[r < rf_per_period] - rf_per_period
    down_std = float(np.sqrt((downside ** 2).mean())) if len(downside) > 0 else np.nan
    sortino  = (mu - rf_per_period) / down_std * np.sqrt(periods_per_year) if down_std and down_std > 0 else np.nan

    # Use monthly curve for drawdown when available — period-level drawdown
    # misses intra-period losses that recover by the end of the hold window.
    dd_series = monthly_curve.dropna() if monthly_curve is not None and len(monthly_curve) > 0 else r
    nav    = (1 + dd_series).cumprod()
    max_dd = float(((nav - nav.cummax()) / nav.cummax()).min())
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    hit    = float((r > 0).mean())
    t_stat = mu / (std / np.sqrt(len(r))) if std > 0 else np.nan

    return {
        "ann_return":   round(ann_ret * 100, 2),
        "ann_vol":      round(ann_vol  * 100, 2),
        "sharpe":       round(sharpe,   3),
        "sortino":      round(sortino,  3) if not np.isnan(sortino) else np.nan,
        "max_drawdown": round(max_dd   * 100, 2),
        "calmar":       round(calmar,   3),
        "hit_rate":     round(hit      * 100, 1),
        "t_stat":       round(t_stat,   2),
        "n_periods":    len(r),
        "n_months":     len(r) * hold_months,
    }
