# three_factor_core_10x2 Backtest Report

## Model Snapshot
- Model: Locked live-research model built around SUE, residual 52-week-high momentum, and sector relative strength with a 10-stock, 2-month hold.
- Factors: sue, mom_52wk_high, sector_rs_1m
- Portfolio construction: 10 stocks, 2-month hold, top 1000 by market cap
- Benchmark: SPY
- In-sample window: 2005-01-01 to 2024-12-31
- Out-of-sample window: 2025-04-01 to 2026-03-31
- Effective offline/local IS window with current sector ETF files: 2005-11-30 to 2024-09-30
- Effective offline/local OOS window: 2025-04-30 to 2025-12-31

## Executive Takeaways
- The locked three-factor specification is the active production research model used in the dashboard and core backtest path.
- OOS performance should be interpreted in the context of the 2025 tariff shock regime break; post-shock behavior has been materially stronger than the early-2025 tape.
- This report includes both rebalance-frequency metrics and a monthly path reconstruction for slide-deck presentation.
- The current offline/local run is history-limited by local sector ETF price coverage beginning in 2024, so the IS block should be treated as a short validation window rather than a full historical training-era backtest.
- Locked specification ranks #21 in the current robustness grid when ordered by post-shock Sharpe, then OOS Sharpe, then IS Sharpe.

## In-Sample Realized Portfolio Metrics
| metric | model | spy |
| --- | --- | --- |
| total_return | 1106.82 | 589.25 |
| avg_2m_return | 2.43 | 1.91 |
| median_2m_return | 3.04 | 2.91 |
| period_vol | 6.72 | 6.27 |
| best_2m_return | 28.29 | 18.07 |
| worst_2m_return | -18.18 | -22.33 |
| hit_rate | 67.5 | 72.8 |
| max_drawdown | -36.84 | -45.96 |
| n_periods | 114.0 | 114.0 |
| n_months | 228.0 | 228.0 |

## Out-of-Sample Realized Portfolio Metrics
| metric | model | spy |
| --- | --- | --- |
| total_return | 35.45 | 24.78 |
| avg_2m_return | 6.35 | 4.61 |
| median_2m_return | 5.88 | 4.4 |
| period_vol | 5.05 | 4.69 |
| best_2m_return | 14.74 | 11.75 |
| worst_2m_return | 2.18 | 0.27 |
| hit_rate | 100.0 | 100.0 |
| max_drawdown | -0.78 | -0.86 |
| n_periods | 5.0 | 5.0 |
| n_months | 10.0 | 10.0 |

## Monthly Path Realized Metrics
| metric | model | spy |
| --- | --- | --- |
| total_return | 35.91 | 24.78 |
| avg_2m_return | 3.16 | 2.26 |
| median_2m_return | 2.25 | 2.18 |
| period_vol | 3.23 | 2.25 |
| best_2m_return | 9.53 | 6.28 |
| worst_2m_return | -0.78 | -0.86 |
| hit_rate | 90.0 | 90.0 |
| max_drawdown | -0.78 | -0.86 |
| n_periods | 10.0 | 10.0 |
| n_months | 10.0 | 10.0 |

## In-Sample Annualized Metrics
| metric | model | spy |
| --- | --- | --- |
| ann_return | 14.01 | 10.69 |
| ann_vol | 16.45 | 15.35 |
| sharpe | 0.589 | 0.427 |
| sortino | 0.567 | 0.357 |
| calmar | 0.38 | 0.233 |
| max_drawdown | -36.84 | -45.96 |
| hit_rate | 67.5 | 72.8 |
| t_stat | 3.86 | 3.25 |
| n_periods | 114.0 | 114.0 |
| n_months | 228.0 | 228.0 |

## Out-of-Sample Annualized Metrics
| metric | model | spy |
| --- | --- | --- |
| ann_return | 43.93 | 30.43 |
| ann_vol | 12.37 | 11.48 |
| sharpe | 2.684 | 1.983 |
| sortino | N/A | 22.486 |
| calmar | 55.963 | 35.213 |
| max_drawdown | -0.78 | -0.86 |
| hit_rate | 100.0 | 100.0 |
| t_stat | 2.81 | 2.2 |
| n_periods | 5 | 5.0 |
| n_months | 10 | 10.0 |

## Monthly Path Annualized Metrics
| metric | model | spy |
| --- | --- | --- |
| ann_return | 44.51 | 30.43 |
| ann_vol | 11.18 | 7.8 |
| sharpe | 2.955 | 2.851 |
| sortino | 11.008 | 8.36 |
| calmar | 56.706 | 35.213 |
| max_drawdown | -0.78 | -0.86 |
| hit_rate | 90.0 | 90.0 |
| t_stat | 3.1 | 3.17 |
| n_periods | 10.0 | 10.0 |
| n_months | 10.0 | 10.0 |

## Calendar-Year Returns
| year | model_return | spy_return | alpha_vs_spy |
| --- | --- | --- | --- |
| 2025 | 27.77 | 24.04 | 3.73 |
| 2026 | 6.37 | 0.6 | 5.77 |

## Worst Drawdowns
| start | trough | recovery | drawdown_pct | months_to_trough | months_to_recover |
| --- | --- | --- | --- | --- | --- |
| 2025-12-31 | 2025-12-31 | 2026-01-31 | -0.78 | 0 | 1 |

## Latest Portfolio
| rebal_date | permno | ticker | company |
| --- | --- | --- | --- |
| 2025-12-31 | 23744 | CR | CR |
| 2025-12-31 | 32678 | HEI | HEI |
| 2025-12-31 | 50876 | LLY | Eli Lilly + Co |
| 2025-12-31 | 53613 | MU | Micron Technology Inc |
| 2025-12-31 | 62308 | GL | Globe Life Inc |
| 2025-12-31 | 79637 | UHS | Universal Health Services B |
| 2025-12-31 | 85059 | FIX | Comfort Systems Usa Inc |
| 2025-12-31 | 85072 | RL | Ralph Lauren Corp |
| 2025-12-31 | 91233 | MA | Mastercard Inc   A |
| 2025-12-31 | 92602 | PM | Philip Morris International |

## Notes For Slides
- Use the Realized Portfolio Metrics sections as the primary deck numbers. They describe the actual realized 2-month portfolio path over the test window.
- Use the Annualized Metrics sections only as secondary context, especially when the sample length is short.
- The portfolio mechanically rebalances every 2 months and holds that basket through the full holding window.
- If presenting 2025, explicitly call out the tariff-shock distortion and the stronger post-shock recovery regime.
