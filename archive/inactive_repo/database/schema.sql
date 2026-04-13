-- =============================================================================
-- SectorScope Data Warehouse  —  DuckDB Schema
-- =============================================================================
-- Design principles
--   • Star schema: fact tables join to dimension tables via surrogate keys
--   • All dates stored as DATE; prices/returns as DOUBLE
--   • PIT-safe: raw source dates preserved (datadate vs rdq, statpers, etc.)
--   • Partitioned by time where DuckDB benefits (returns, prices)
--   • Metadata columns (loaded_at) on every table for lineage tracking
--
-- Migration path to PostgreSQL
--   • Replace DOUBLE with NUMERIC(20,8) for strict precision
--   • Replace auto-generated sequences with SERIAL / BIGSERIAL
--   • COPY or pg_dump from DuckDB CSV exports
-- =============================================================================


-- ─────────────────────────────────────────────────────────────────────────────
-- DIMENSION TABLES
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS dim_stock (
    permno          INTEGER     PRIMARY KEY,
    ticker          VARCHAR(10),        -- most recent ticker from CRSP/IBES
    company_name    VARCHAR(255),
    sic_code        INTEGER,
    sector_etf      VARCHAR(5),         -- XLK, XLF, etc. (derived from SIC)
    exchange_code   SMALLINT,           -- CRSP exchcd: 1=NYSE, 2=AMEX, 3=NASDAQ
    share_class     SMALLINT,           -- CRSP shrcd
    first_obs_date  DATE,               -- earliest date in fact_returns
    last_obs_date   DATE,               -- latest date in fact_returns
    loaded_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dim_etf (
    etf_ticker      VARCHAR(5)  PRIMARY KEY,
    sector_name     VARCHAR(50),
    index_tracked   VARCHAR(100),       -- e.g. "S&P Technology Select Sector"
    inception_date  DATE,
    loaded_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Populate dim_etf with the 11 SPDR sector ETFs
INSERT OR IGNORE INTO dim_etf (etf_ticker, sector_name, index_tracked) VALUES
    ('XLK',  'Technology',         'S&P Technology Select Sector'),
    ('XLF',  'Financials',         'S&P Financial Select Sector'),
    ('XLV',  'Health Care',        'S&P Health Care Select Sector'),
    ('XLI',  'Industrials',        'S&P Industrial Select Sector'),
    ('XLE',  'Energy',             'S&P Energy Select Sector'),
    ('XLP',  'Consumer Staples',   'S&P Consumer Staples Select Sector'),
    ('XLY',  'Consumer Discr.',    'S&P Consumer Discretionary Select Sector'),
    ('XLB',  'Materials',          'S&P Materials Select Sector'),
    ('XLRE', 'Real Estate',        'S&P Real Estate Select Sector'),
    ('XLU',  'Utilities',          'S&P Utilities Select Sector'),
    ('XLC',  'Comm. Services',     'S&P Communication Services Select Sector');

CREATE TABLE IF NOT EXISTS dim_date (
    date_id         DATE        PRIMARY KEY,
    year            SMALLINT,
    month           SMALLINT,
    quarter         SMALLINT,
    week_of_year    SMALLINT,
    is_month_end    BOOLEAN,
    is_quarter_end  BOOLEAN,
    is_year_end     BOOLEAN
);


-- ─────────────────────────────────────────────────────────────────────────────
-- FACT — CRSP MONTHLY RETURNS
-- ─────────────────────────────────────────────────────────────────────────────
-- Source: WRDS CRSP msf (monthly stock file)
-- Grain:  one row per (permno, month-end date)

CREATE TABLE IF NOT EXISTS fact_returns (
    permno          INTEGER     NOT NULL,
    date            DATE        NOT NULL,
    ret             DOUBLE,             -- total return (with dividends)
    retx            DOUBLE,             -- ex-dividend return
    price           DOUBLE,             -- absolute value of closing price
    shares_out      DOUBLE,             -- shares outstanding (thousands)
    market_cap      DOUBLE,             -- price × shares_out (thousands $)
    loaded_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (permno, date)
);


-- ─────────────────────────────────────────────────────────────────────────────
-- FACT — COMPUSTAT QUARTERLY FUNDAMENTALS  (point-in-time)
-- ─────────────────────────────────────────────────────────────────────────────
-- Source: WRDS Compustat Fundamentals Quarterly (funda_q)
-- Grain:  one row per (permno, fiscal quarter end date)
-- PIT:    rdq = actual earnings release date (use for no-lookahead queries)

CREATE TABLE IF NOT EXISTS fact_fundamentals (
    permno          INTEGER     NOT NULL,
    datadate        DATE        NOT NULL,   -- fiscal quarter end
    rdq             DATE,                   -- actual report/filing date (PIT anchor)
    -- Balance sheet
    total_assets    DOUBLE,                 -- atq
    total_liabilities DOUBLE,              -- ltq
    common_equity   DOUBLE,                -- ceqq
    -- Income / cash flow
    net_income_q    DOUBLE,                -- ibq  (quarterly)
    net_income_ttm  DOUBLE,                -- ibq_ttm
    oper_cf_ttm     DOUBLE,                -- oancfq_ttm
    capex_ttm       DOUBLE,                -- capxq_ttm
    revenue_q       DOUBLE,                -- saleq
    eps_ttm         DOUBLE,                -- epspxq_ttm
    -- Derived metrics
    roe_ttm         DOUBLE,                -- net_income_ttm / avg_equity
    fcf_ttm         DOUBLE,                -- oper_cf_ttm - capex_ttm
    fcf_yield       DOUBLE,                -- fcf_ttm / market_cap
    earnings_yield  DOUBLE,                -- eps_ttm / price
    pb_ratio        DOUBLE,                -- price / (equity / shares)
    market_cap      DOUBLE,
    loaded_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (permno, datadate)
);


-- ─────────────────────────────────────────────────────────────────────────────
-- FACT — IBES ANALYST CONSENSUS  (point-in-time via statpers)
-- ─────────────────────────────────────────────────────────────────────────────
-- Source: WRDS IBES Summary Statistics
-- Grain:  one row per (permno, statistics period end date)
-- PIT:    statpers is when IBES compiled the consensus (safe to use as-of date)

CREATE TABLE IF NOT EXISTS fact_analyst (
    permno          INTEGER     NOT NULL,
    statpers        DATE        NOT NULL,   -- consensus compilation date (PIT)
    ticker          VARCHAR(10),
    mean_rec        DOUBLE,                 -- 1=Strong Buy … 5=Strong Sell
    median_rec      DOUBLE,
    num_analysts    SMALLINT,
    num_upgrades    SMALLINT,               -- numup  (vs prior period)
    num_downgrades  SMALLINT,               -- numdown
    buy_pct         DOUBLE,                 -- fraction with Buy/Strong Buy
    hold_pct        DOUBLE,
    sell_pct        DOUBLE,
    net_upgrades    DOUBLE,                 -- upgrades - downgrades
    rev_3m          DOUBLE,                 -- 3-month recommendation change
    buy_pct_rev_3m  DOUBLE,                 -- 3-month buy-pct change
    coverage_chg_3m DOUBLE,                 -- analyst count change 3m
    loaded_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (permno, statpers)
);


-- ─────────────────────────────────────────────────────────────────────────────
-- FACT — FRED MACRO SIGNALS  (monthly)
-- ─────────────────────────────────────────────────────────────────────────────
-- Source: FRED via fredapi; pre-processed in ingestion/fred_macro.py
-- Grain:  one row per calendar month-end

CREATE TABLE IF NOT EXISTS fact_macro (
    date                DATE        PRIMARY KEY,
    -- Yield curve
    t10y2y_spread       DOUBLE,
    t10y2y_3m_chg       DOUBLE,
    t10y2y_zscore       DOUBLE,
    yield_curve_inverted BOOLEAN,
    -- Fed funds rate
    fed_funds_rate      DOUBLE,
    fed_funds_3m_chg    DOUBLE,
    -- 10-year Treasury
    dgs10               DOUBLE,
    dgs10_3m_chg        DOUBLE,
    -- Inflation
    cpi_mom             DOUBLE,
    cpi_3m_chg          DOUBLE,
    cpi_zscore          DOUBLE,
    -- Labour market
    unrate              DOUBLE,
    unrate_mom          DOUBLE,
    payems_mom          DOUBLE,
    -- HY credit spread (key macro factor)
    hy_spread           DOUBLE,             -- ICE BofA HY OAS level (bps)
    hy_spread_3m_chg    DOUBLE,             -- 3-month change
    hy_spread_zscore    DOUBLE,             -- rolling 36m z-score
    hy_spread_widening  DOUBLE,             -- pre-computed signal used in model
    -- Model macro factor (rolling z, sign-flipped; used as F5)
    macro_factor_z      DOUBLE,
    loaded_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ─────────────────────────────────────────────────────────────────────────────
-- FACT — FIVE-FACTOR MODEL SCORES  (monthly snapshot)
-- ─────────────────────────────────────────────────────────────────────────────
-- Grain: one row per (ticker, score_date)
-- Append-only: new rows added each month; old rows preserved for history

CREATE TABLE IF NOT EXISTS fact_factor_scores (
    ticker          VARCHAR(10) NOT NULL,
    score_date      DATE        NOT NULL,
    permno          INTEGER,
    -- Cross-sectional z-scores (each factor winsorised 1/99, z-scored)
    momentum_18m    DOUBLE,
    roe_ttm         DOUBLE,
    sector_rs_3m    DOUBLE,
    analyst_rev_3m  DOUBLE,
    macro_hy        DOUBLE,                 -- time-series z-score (same for all)
    composite       DOUBLE,                 -- equal-weight average of above
    loaded_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, score_date)
);


-- ─────────────────────────────────────────────────────────────────────────────
-- FACT — ETF DAILY PRICES
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fact_etf_prices (
    ticker          VARCHAR(10) NOT NULL,
    date            DATE        NOT NULL,
    close           DOUBLE,
    volume          DOUBLE,
    loaded_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, date)
);


-- ─────────────────────────────────────────────────────────────────────────────
-- FACT — ETF HOLDINGS  (point-in-time snapshot per refresh)
-- ─────────────────────────────────────────────────────────────────────────────
-- Grain: (etf_ticker, as_of_date, holding_ticker)

CREATE TABLE IF NOT EXISTS fact_etf_holdings (
    etf_ticker      VARCHAR(5)  NOT NULL,
    as_of_date      DATE        NOT NULL,
    holding_ticker  VARCHAR(10) NOT NULL,
    holding_name    VARCHAR(200),
    weight          DOUBLE,                 -- fraction of ETF (sums to ~1)
    price           DOUBLE,                 -- last close at time of snapshot
    composite_score DOUBLE,                 -- model score at as_of_date
    momentum_18m    DOUBLE,
    roe_ttm         DOUBLE,
    sector_rs_3m    DOUBLE,
    analyst_rev_3m  DOUBLE,
    macro_hy        DOUBLE,
    loaded_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (etf_ticker, as_of_date, holding_ticker)
);


-- ─────────────────────────────────────────────────────────────────────────────
-- FACT — ETF AGGREGATE SCORES & SIGNALS  (monthly snapshot)
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fact_etf_scores (
    etf_ticker          VARCHAR(5)  NOT NULL,
    score_date          DATE        NOT NULL,
    weighted_composite  DOUBLE,             -- weighted avg of holding scores
    score_z             DOUBLE,             -- cross-sectional z across 11 ETFs
    signal              VARCHAR(15),        -- Strong Buy / Buy / Neutral / Sell / Strong Sell
    n_holdings          SMALLINT,
    n_scored            SMALLINT,
    coverage_weight_pct DOUBLE,             -- % of ETF weight that is scored
    loaded_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (etf_ticker, score_date)
);


-- ─────────────────────────────────────────────────────────────────────────────
-- FACT — BACKTEST RETURNS  (one row per rebalance period per strategy)
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fact_backtest_returns (
    strategy        VARCHAR(50) NOT NULL,   -- e.g. 'ML-Optimised', 'Equal-Weight', 'SPY_BH'
    rebalance_date  DATE        NOT NULL,
    period_return   DOUBLE,                 -- holding-period net return
    loaded_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (strategy, rebalance_date)
);


-- ─────────────────────────────────────────────────────────────────────────────
-- FACT — BACKTEST SUMMARY METRICS  (one row per strategy × sub-period)
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fact_backtest_metrics (
    strategy        VARCHAR(50) NOT NULL,
    period_label    VARCHAR(30) NOT NULL,   -- 'full', '2006-2009', etc.
    period_start    DATE,
    period_end      DATE,
    ann_return      DOUBLE,
    ann_vol         DOUBLE,
    sharpe          DOUBLE,
    max_drawdown    DOUBLE,
    calmar          DOUBLE,
    hit_rate        DOUBLE,
    t_stat          DOUBLE,
    total_return    DOUBLE,
    n_periods       SMALLINT,
    loaded_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (strategy, period_label)
);


-- ─────────────────────────────────────────────────────────────────────────────
-- VIEWS  —  commonly used joins pre-built for convenience
-- ─────────────────────────────────────────────────────────────────────────────

CREATE OR REPLACE VIEW v_latest_scores AS
    SELECT
        fs.ticker,
        fs.score_date,
        fs.composite,
        fs.momentum_18m,
        fs.roe_ttm,
        fs.sector_rs_3m,
        fs.analyst_rev_3m,
        fs.macro_hy,
        ds.company_name,
        ds.sector_etf,
        de.sector_name
    FROM fact_factor_scores fs
    LEFT JOIN dim_stock  ds ON ds.permno = fs.permno
    LEFT JOIN dim_etf    de ON de.etf_ticker = ds.sector_etf
    WHERE fs.score_date = (SELECT MAX(score_date) FROM fact_factor_scores);


CREATE OR REPLACE VIEW v_latest_etf_signals AS
    SELECT
        es.etf_ticker,
        de.sector_name,
        es.score_date,
        es.weighted_composite,
        es.score_z,
        es.signal,
        es.n_holdings,
        es.n_scored,
        es.coverage_weight_pct
    FROM fact_etf_scores es
    JOIN dim_etf de ON de.etf_ticker = es.etf_ticker
    WHERE es.score_date = (SELECT MAX(score_date) FROM fact_etf_scores);


CREATE OR REPLACE VIEW v_holdings_with_scores AS
    SELECT
        h.etf_ticker,
        de.sector_name,
        h.as_of_date,
        h.holding_ticker,
        h.holding_name,
        h.weight,
        h.price,
        h.composite_score,
        h.momentum_18m,
        h.roe_ttm,
        h.sector_rs_3m,
        h.analyst_rev_3m,
        h.macro_hy
    FROM fact_etf_holdings h
    JOIN dim_etf de ON de.etf_ticker = h.etf_ticker
    WHERE h.as_of_date = (SELECT MAX(as_of_date) FROM fact_etf_holdings);


CREATE OR REPLACE VIEW v_portfolio_performance AS
    SELECT
        strategy,
        rebalance_date,
        period_return,
        SUM(1 + period_return) OVER (
            PARTITION BY strategy
            ORDER BY rebalance_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) - 1                      AS cumulative_return,
        EXP(SUM(LN(1 + period_return)) OVER (
            PARTITION BY strategy
            ORDER BY rebalance_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )) - 1                     AS compound_nav_minus1
    FROM fact_backtest_returns
    ORDER BY strategy, rebalance_date;
