-- =========================================
-- DATABASE
-- =========================================
CREATE DATABASE IF NOT EXISTS hedge;

USE hedge;

-- =========================================
-- RAW TICKS (trades / quotes normalized)
-- =========================================
CREATE TABLE IF NOT EXISTS ticks
(
    ts          DateTime64(3)      COMMENT 'Event time (ms)',
    symbol      LowCardinality(String),
    venue       LowCardinality(String),
    region      LowCardinality(String),
    side        Enum8('na' = 0, 'buy' = 1, 'sell' = -1),
    price       Float64,
    size        Float64,
    raw         String              COMMENT 'Compact JSON payload',
    ingest_ts   DateTime           DEFAULT now() COMMENT 'Ingest wall time'
)
ENGINE = MergeTree
PARTITION BY toDate(ts)
ORDER BY (symbol, ts)
SETTINGS index_granularity = 8192;

-- Optional retention for hot storage (keep 30 days)
ALTER TABLE ticks MODIFY TTL ts + INTERVAL 30 DAY DELETE;

-- =========================================
-- 1-MIN CANDLES (aggregated from ticks)
-- =========================================
CREATE TABLE IF NOT EXISTS candles_1m
(
    bucket      DateTime            COMMENT 'Start of minute',
    symbol      LowCardinality(String),
    venue       LowCardinality(String),
    open        Float64,
    high        Float64,
    low         Float64,
    close       Float64,
    volume      Float64
)
ENGINE = MergeTree
PARTITION BY toDate(bucket)
ORDER BY (symbol, bucket)
SETTINGS index_granularity = 8192;

-- Materialized view to build 1m candles automatically
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_ticks_to_1m
TO candles_1m
AS
SELECT
    toStartOfMinute(ts)                          AS bucket,
    symbol,
    anyLast(venue)                               AS venue,
    argMin(price, ts)                            AS open,
    max(price)                                   AS high,
    min(price)                                   AS low,
    argMax(price, ts)                            AS close,
    sum(size)                                    AS volume
FROM ticks
GROUP BY bucket, symbol;

-- =========================================
-- LAST PRICE SNAPSHOT (fast lookup)
-- =========================================
-- Store as aggregate state + finalize view for ultra fast last-price queries.
CREATE TABLE IF NOT EXISTS last_price_state
(
    symbol  LowCardinality(String),
    state   AggregateFunction(argMax, Float64, DateTime64(3))
)
ENGINE = AggregatingMergeTree
ORDER BY symbol;

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_ticks_to_last_price
TO last_price_state
AS
SELECT
    symbol,
    argMaxState(price, ts) AS state
FROM ticks
GROUP BY symbol;

-- Readable view that exposes finalized last price + timestamp.
CREATE OR REPLACE VIEW last_price AS
SELECT
    symbol,
    finalizeAggregation(state).1 AS price,
    toDateTime64(finalizeAggregation(state).2, 3) AS ts
FROM last_price_state;

-- =========================================
-- ORDERS (post‑risk, pre‑/post‑fill audit)
-- =========================================
CREATE TABLE IF NOT EXISTS orders
(
    ts          DateTime64(3),
    order_id    String,
    strategy    LowCardinality(String),
    symbol      LowCardinality(String),
    region      LowCardinality(String),
    venue       LowCardinality(String),
    side        Enum8('buy' = 1, 'sell' = -1),
    qty         Float64,
    typ         LowCardinality(String),             -- market|limit|...
    limit_price Nullable(Float64),
    status      LowCardinality(String) DEFAULT 'accepted',  -- accepted|rejected|routed
    reason      String DEFAULT ''                   -- rejection reason if any
)
ENGINE = MergeTree
PARTITION BY toDate(ts)
ORDER BY (order_id, ts);

-- =========================================
-- FILLS (executions)
-- =========================================
CREATE TABLE IF NOT EXISTS fills
(
    ts              DateTime64(3),
    fill_id         String,
    order_id        String,
    strategy        LowCardinality(String),
    symbol          LowCardinality(String),
    region          LowCardinality(String),
    venue           LowCardinality(String),
    side            Enum8('buy' = 1, 'sell' = -1),
    qty             Float64,
    price           Float64,
    realized_delta  Float64 DEFAULT 0,
    meta            String  DEFAULT ''               -- optional JSON
)
ENGINE = MergeTree
PARTITION BY toDate(ts)
ORDER BY (order_id, ts);

-- =========================================
-- POSITIONS SNAPSHOTS (optional periodic)
-- =========================================
CREATE TABLE IF NOT EXISTS positions_snap
(
    ts          DateTime64(3),
    symbol      LowCardinality(String),
    strategy    LowCardinality(String) DEFAULT '',
    qty         Float64,
    avg_price   Float64,
    realized_pnl Float64
)
ENGINE = ReplacingMergeTree(ts)
PARTITION BY toDate(ts)
ORDER BY (symbol, strategy, ts);

-- =========================================
-- PNL SNAPSHOTS (portfolio)
-- =========================================
CREATE TABLE IF NOT EXISTS pnl_snap
(
    ts          DateTime64(3),
    realized    Float64,
    unrealized  Float64,
    total       Float64
)
ENGINE = ReplacingMergeTree(ts)
ORDER BY ts;

-- =========================================
-- SIMPLE DERIVED VIEWS
-- =========================================
CREATE OR REPLACE VIEW strategy_daily_pnl AS
SELECT
    toDate(ts) AS d,
    strategy,
    sum(realized_delta) AS realized
FROM fills
GROUP BY d, strategy
ORDER BY d, strategy;

CREATE OR REPLACE VIEW symbol_turnover_daily AS
SELECT
    toDate(ts) AS d,
    symbol,
    sum(abs(price * qty)) AS notional
FROM fills
GROUP BY d, symbol
ORDER BY d, symbol;

-- =========================================
-- INDEX / PROJECTION EXAMPLES (optional)
-- =========================================
-- Speeds up per-symbol time range queries on ticks:
ALTER TABLE ticks ADD PROJECTION IF NOT EXISTS ticks_by_symbol_ts
(
    SELECT *
    ORDER BY (symbol, ts)
);

-- =========================================
-- GRANTS (optional local user for read-only dashboards)
-- =========================================
-- CREATE USER IF NOT EXISTS ro IDENTIFIED WITH no_password;
-- GRANT SELECT ON hedge.* TO ro;