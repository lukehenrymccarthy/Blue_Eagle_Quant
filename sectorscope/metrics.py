import numpy as np
import pandas as pd

def compute_metrics(r: pd.Series, hold_months: int = 1, risk_free_ann: float = 0.05) -> dict:
    """
    Annualised performance metrics for a series of holding-period returns.
    Properly scales Sharpe and vol for non-monthly periods.
    """
    r = r.dropna()
    periods_per_year = 12 / hold_months
    if len(r) < max(6, int(periods_per_year)):
        return {}

    rf_per_period = (1 + risk_free_ann) ** (hold_months / 12) - 1
    ann_ret = (1 + r.mean()) ** periods_per_year - 1
    ann_vol = r.std() * np.sqrt(periods_per_year)
    sharpe  = (r.mean() - rf_per_period) / r.std() * np.sqrt(periods_per_year) \
              if r.std() > 0 else np.nan

    nav    = (1 + r).cumprod()
    max_dd = ((nav - nav.cummax()) / nav.cummax()).min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    hit    = (r > 0).mean()
    t_stat = r.mean() / (r.std() / np.sqrt(len(r))) if r.std() > 0 else np.nan

    return {
        "ann_return":   round(ann_ret * 100, 2),
        "ann_vol":      round(ann_vol  * 100, 2),
        "sharpe":       round(sharpe,   3),
        "max_drawdown": round(max_dd   * 100, 2),
        "calmar":       round(calmar,   3),
        "hit_rate":     round(hit      * 100, 1),
        "t_stat":       round(t_stat,   2),
        "n_periods":    len(r),
        "n_months":     len(r) * hold_months,
    }
