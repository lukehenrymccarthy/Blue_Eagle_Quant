-- =============================================================================
-- SectorScope  —  Example Analytical Queries
-- =============================================================================
-- Connect:  python3 -c "import duckdb; conn = duckdb.connect('data/sectorscope.duckdb')"
-- DuckDB CLI:  duckdb data/sectorscope.duckdb
-- =============================================================================


-- ─────────────────────────────────────────────────────────────────────────────
-- 1.  CURRENT ETF SIGNALS  (use the pre-built view)
-- ─────────────────────────────────────────────────────────────────────────────

SELECT
    etf_ticker,
    sector_name,
    signal,
    ROUND(weighted_composite, 4)  AS score,
    ROUND(score_z, 2)             AS z,
    n_scored,
    ROUND(coverage_weight_pct, 1) AS covered_pct
FROM v_latest_etf_signals
ORDER BY weighted_composite DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- 2.  TOP-20 STOCKS BY COMPOSITE SCORE (current snapshot)
-- ─────────────────────────────────────────────────────────────────────────────

SELECT
    ticker,
    sector_name,
    ROUND(composite, 3)      AS score,
    ROUND(momentum_18m, 3)   AS momentum,
    ROUND(roe_ttm, 3)        AS roe,
    ROUND(sector_rs_3m, 3)   AS sector_rs,
    ROUND(analyst_rev_3m, 3) AS analyst_rev,
    ROUND(macro_hy, 3)       AS macro
FROM v_latest_scores
ORDER BY composite DESC
LIMIT 20;


-- ─────────────────────────────────────────────────────────────────────────────
-- 3.  HOLDINGS DRILL-DOWN  (e.g. XLE — Energy sector)
-- ─────────────────────────────────────────────────────────────────────────────

SELECT
    holding_ticker,
    holding_name,
    ROUND(weight * 100, 2)       AS weight_pct,
    ROUND(price, 2)              AS price,
    ROUND(composite_score, 3)    AS score,
    ROUND(momentum_18m, 3)       AS momentum,
    ROUND(roe_ttm, 3)            AS roe
FROM v_holdings_with_scores
WHERE etf_ticker = 'XLE'
ORDER BY weight DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- 4.  BACKTEST STRATEGY COMPARISON
-- ─────────────────────────────────────────────────────────────────────────────

SELECT
    strategy,
    period_label,
    ROUND(ann_return * 100, 1)    AS ann_ret_pct,
    ROUND(ann_vol * 100, 1)       AS ann_vol_pct,
    ROUND(sharpe, 3)              AS sharpe,
    ROUND(max_drawdown * 100, 1)  AS max_dd_pct
FROM fact_backtest_metrics
ORDER BY strategy, period_start;


-- ─────────────────────────────────────────────────────────────────────────────
-- 5.  CUMULATIVE NAV  (compound portfolio growth over full history)
-- ─────────────────────────────────────────────────────────────────────────────

SELECT
    strategy,
    rebalance_date,
    ROUND(period_return * 100, 2)       AS period_ret_pct,
    ROUND(compound_nav_minus1 * 100, 2) AS cumulative_ret_pct
FROM v_portfolio_performance
ORDER BY strategy, rebalance_date;


-- ─────────────────────────────────────────────────────────────────────────────
-- 6.  POINT-IN-TIME FUNDAMENTALS  (no lookahead — uses rdq as anchor)
-- ─────────────────────────────────────────────────────────────────────────────
-- "What ROE was available for AAPL as of 2020-01-31?"

SELECT
    ds.ticker,
    f.datadate,
    f.rdq               AS report_date,
    ROUND(f.roe_ttm, 4) AS roe_ttm,
    ROUND(f.pb_ratio, 2) AS pb_ratio,
    f.market_cap        AS market_cap_k
FROM fact_fundamentals f
JOIN dim_stock ds ON ds.permno = f.permno
WHERE ds.ticker = 'AAPL'
  AND f.rdq <= DATE '2020-01-31'          -- only data released by this date
ORDER BY f.rdq DESC
LIMIT 4;


-- ─────────────────────────────────────────────────────────────────────────────
-- 7.  SECTOR ROTATION SIGNAL HISTORY  (monthly ETF scores over time)
-- ─────────────────────────────────────────────────────────────────────────────

SELECT
    es.etf_ticker,
    de.sector_name,
    es.score_date,
    ROUND(es.weighted_composite, 4) AS score,
    ROUND(es.score_z, 2)            AS z,
    es.signal
FROM fact_etf_scores es
JOIN dim_etf de ON de.etf_ticker = es.etf_ticker
ORDER BY es.score_date DESC, es.weighted_composite DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- 8.  ANALYST SENTIMENT TREND  (for a specific stock)
-- ─────────────────────────────────────────────────────────────────────────────

SELECT
    a.statpers,
    a.mean_rec,
    ROUND(a.buy_pct * 100, 1)      AS buy_pct,
    a.num_analysts,
    ROUND(a.rev_3m, 4)             AS rec_rev_3m,
    ROUND(a.buy_pct_rev_3m, 4)     AS buy_pct_rev_3m
FROM fact_analyst a
JOIN dim_stock ds ON ds.permno = a.permno
WHERE ds.ticker = 'NVDA'
ORDER BY a.statpers DESC
LIMIT 12;


-- ─────────────────────────────────────────────────────────────────────────────
-- 9.  MACRO ENVIRONMENT DASHBOARD  (last 12 months)
-- ─────────────────────────────────────────────────────────────────────────────

SELECT
    date,
    ROUND(t10y2y_spread, 3)     AS yield_curve,
    yield_curve_inverted,
    ROUND(fed_funds_rate, 3)    AS fed_rate,
    ROUND(cpi_mom * 100, 2)     AS cpi_mom_pct,
    ROUND(hy_spread, 1)         AS hy_spread_bps,
    ROUND(macro_factor_z, 3)    AS macro_z
FROM fact_macro
ORDER BY date DESC
LIMIT 12;


-- ─────────────────────────────────────────────────────────────────────────────
-- 10. CROSS-SECTIONAL FACTOR ANALYTICS  (factor z-score distributions)
-- ─────────────────────────────────────────────────────────────────────────────

SELECT
    ds.sector_etf,
    de.sector_name,
    COUNT(*)                           AS n_stocks,
    ROUND(AVG(fs.composite), 3)        AS avg_composite,
    ROUND(AVG(fs.momentum_18m), 3)     AS avg_momentum,
    ROUND(AVG(fs.roe_ttm), 3)          AS avg_roe,
    ROUND(AVG(fs.analyst_rev_3m), 3)   AS avg_analyst_rev,
    ROUND(STDDEV(fs.composite), 3)     AS std_composite
FROM fact_factor_scores fs
JOIN dim_stock ds ON ds.permno = fs.permno
JOIN dim_etf de   ON de.etf_ticker = ds.sector_etf
WHERE fs.score_date = (SELECT MAX(score_date) FROM fact_factor_scores)
GROUP BY ds.sector_etf, de.sector_name
ORDER BY avg_composite DESC;


-- ─────────────────────────────────────────────────────────────────────────────
-- 11. LARGE-CAP SCREEN  (score > 0.5, market cap > $10B)
-- ─────────────────────────────────────────────────────────────────────────────

SELECT
    ds.ticker,
    de.sector_name,
    ROUND(fs.composite, 3)          AS score,
    ROUND(f.market_cap / 1e6, 1)    AS mktcap_bn,
    ROUND(f.pb_ratio, 2)            AS pb,
    ROUND(f.roe_ttm * 100, 1)       AS roe_pct
FROM fact_factor_scores fs
JOIN dim_stock ds       ON ds.permno = fs.permno
JOIN dim_etf de         ON de.etf_ticker = ds.sector_etf
JOIN (
    -- Latest fundamentals per stock
    SELECT permno, market_cap, pb_ratio, roe_ttm, rdq
    FROM fact_fundamentals
    QUALIFY ROW_NUMBER() OVER (PARTITION BY permno ORDER BY rdq DESC) = 1
) f ON f.permno = fs.permno
WHERE fs.score_date = (SELECT MAX(score_date) FROM fact_factor_scores)
  AND fs.composite > 0.5
  AND f.market_cap > 10e6     -- $10B+ (market_cap stored in thousands)
ORDER BY fs.composite DESC
LIMIT 20;


-- ─────────────────────────────────────────────────────────────────────────────
-- 12. PRICE PERFORMANCE RELATIVE TO SECTOR ETF  (1-year)
-- ─────────────────────────────────────────────────────────────────────────────

WITH stock_ret AS (
    SELECT
        r.permno,
        ds.ticker,
        ds.sector_etf,
        SUM(LN(1 + COALESCE(r.ret, 0))) AS log_ret_1y
    FROM fact_returns r
    JOIN dim_stock ds ON ds.permno = r.permno
    WHERE r.date >= CURRENT_DATE - INTERVAL 365 DAYS
    GROUP BY r.permno, ds.ticker, ds.sector_etf
),
etf_ret AS (
    SELECT
        p.ticker AS etf_ticker,
        (p2.close / p1.close - 1) AS etf_ret_1y
    FROM fact_etf_prices p1
    JOIN fact_etf_prices p2
        ON p1.ticker = p2.ticker
       AND p1.date   = (SELECT MIN(date) FROM fact_etf_prices
                        WHERE date >= CURRENT_DATE - INTERVAL 365 DAYS)
       AND p2.date   = (SELECT MAX(date) FROM fact_etf_prices)
    WHERE p1.ticker IN (SELECT etf_ticker FROM dim_etf)
)
SELECT
    s.ticker,
    s.sector_etf,
    ROUND((EXP(s.log_ret_1y) - 1) * 100, 1)  AS stock_ret_1y_pct,
    ROUND(e.etf_ret_1y * 100, 1)              AS etf_ret_1y_pct,
    ROUND((EXP(s.log_ret_1y) - 1 - e.etf_ret_1y) * 100, 1) AS excess_pct
FROM stock_ret s
JOIN etf_ret e ON e.etf_ticker = s.sector_etf
ORDER BY excess_pct DESC
LIMIT 20;
