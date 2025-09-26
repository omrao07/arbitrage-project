-- catalog/lakehouse.sql
-- Master catalog for the Hyper-OS Lakehouse
-- Compatible with DuckDB, Spark, BigQuery, or Trino

-- =========================================================
-- 1. EQUITIES (prices, factors, fundamentals)
-- =========================================================
CREATE TABLE IF NOT EXISTS equities_prices (
    ts TIMESTAMP NOT NULL,              -- event time (UTC)
    ticker STRING NOT NULL,             -- e.g. "AAPL"
    source STRING,                      -- polygon/yahoo
    open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
    volume BIGINT,
    adj_close DOUBLE,
    ingest_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
PARTITIONED BY (DATE(ts));

CREATE TABLE IF NOT EXISTS equities_factors (
    ts DATE NOT NULL,
    ticker STRING NOT NULL,
    factor STRING NOT NULL,             -- e.g. "momentum", "quality"
    value DOUBLE,
    ingest_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
PARTITIONED BY (ts);

CREATE TABLE IF NOT EXISTS equities_fundamentals (
    ts DATE NOT NULL,
    ticker STRING NOT NULL,
    metric STRING NOT NULL,             -- e.g. "ROE", "EPS"
    value DOUBLE,
    currency STRING,
    ingest_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
PARTITIONED BY (ts);

-- =========================================================
-- 2. FX (spot, cross, vol surface)
-- =========================================================
CREATE TABLE IF NOT EXISTS fx_rates (
    ts TIMESTAMP NOT NULL,
    base STRING NOT NULL,               -- e.g. "USD"
    quote STRING NOT NULL,              -- e.g. "JPY"
    rate DOUBLE,
    source STRING,
    ingest_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
PARTITIONED BY (DATE(ts));

CREATE TABLE IF NOT EXISTS fx_vol_surface (
    ts DATE NOT NULL,
    pair STRING NOT NULL,               -- e.g. "EURUSD"
    tenor STRING NOT NULL,              -- e.g. "1M", "3M"
    strike DOUBLE,
    vol DOUBLE,
    ingest_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
PARTITIONED BY (ts);

-- =========================================================
-- 3. MACRO (FRED, WorldBank, IMF)
-- =========================================================
CREATE TABLE IF NOT EXISTS macro_series (
    ts DATE NOT NULL,
    source STRING NOT NULL,             -- "FRED", "WorldBank"
    series_id STRING NOT NULL,          -- e.g. "CPIAUCSL", "NY.GDP.MKTP.CD"
    country STRING,
    value DOUBLE,
    ingest_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
PARTITIONED BY (ts, source);

-- =========================================================
-- 4. NEWS + SIGNALS
-- =========================================================
CREATE TABLE IF NOT EXISTS news_feed (
    ts TIMESTAMP NOT NULL,
    source STRING NOT NULL,             -- bloomberg, reuters, rss
    headline STRING,
    body STRING,
    sentiment DOUBLE,
    tags ARRAY<STRING>,
    ingest_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
PARTITIONED BY (DATE(ts), source);

-- =========================================================
-- 5. PORTFOLIO + TRADING
-- =========================================================
CREATE TABLE IF NOT EXISTS portfolio_positions (
    ts TIMESTAMP NOT NULL,
    account_id STRING,
    ticker STRING,
    qty DOUBLE,
    avg_price DOUBLE,
    market_value DOUBLE,
    ingest_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
PARTITIONED BY (DATE(ts));

CREATE TABLE IF NOT EXISTS portfolio_pnl (
    ts TIMESTAMP NOT NULL,
    account_id STRING,
    realized DOUBLE,
    unrealized DOUBLE,
    gross DOUBLE,
    net DOUBLE,
    ingest_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
PARTITIONED BY (DATE(ts));

CREATE TABLE IF NOT EXISTS portfolio_orders (
    ts TIMESTAMP NOT NULL,
    order_id STRING,
    account_id STRING,
    ticker STRING,
    side STRING,                         -- BUY/SELL
    qty DOUBLE,
    price DOUBLE,
    status STRING,                       -- NEW/FILLED/CANCELLED
    ingest_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
PARTITIONED BY (DATE(ts));

-- =========================================================
-- 6. RISK + POLICY
-- =========================================================
CREATE TABLE IF NOT EXISTS risk_scenarios (
    ts TIMESTAMP NOT NULL,
    scenario_id STRING,
    description STRING,
    shock JSON,                          -- e.g. {"rate+100bp": -5%}
    pnl_impact DOUBLE,
    ingest_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS risk_var (
    ts TIMESTAMP NOT NULL,
    portfolio_id STRING,
    var_95 DOUBLE,
    var_99 DOUBLE,
    stressed DOUBLE,
    ingest_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS policy_audit (
    ts TIMESTAMP NOT NULL,
    user_id STRING,
    action STRING,
    resource STRING,
    status STRING,
    ingest_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =========================================================
-- END
-- =========================================================