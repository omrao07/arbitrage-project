# Hyper-OS Lakehouse — Schemas & Data Contracts

**Version:** 1  
**Last updated:** 2025-09-15

This document is the human-readable contract for datasets defined in:
- `catalog/lakehouse.sql` (DDL)
- `catalog/registry.yaml` (dataset registry)

Each dataset lists: columns (name • type • nullability), semantics, partitioning, lineage (upstream), freshness, and DQ checks.

---

## Equities

### `equities_prices`
**Purpose:** OHLCV price history for listed equities (daily/intraday aggregates).  
**Upstream:** Polygon (`equities_polygon.py`), Yahoo (UI/backup).  
**Storage:** `s3://hyper-lakehouse/equities/prices/` (parquet) • Partition: `DATE(ts)`.

| Column | Type | Null | Description |
|---|---|---|---|
| ts | TIMESTAMP | NO | Event time (UTC) of the bar close. |
| ticker | STRING | NO | Ticker symbol, e.g., `AAPL`. |
| source | STRING | YES | Ingest source: `polygon`, `yahoo`. |
| open | DOUBLE | YES | Session/bar open. |
| high | DOUBLE | YES | High. |
| low | DOUBLE | YES | Low. |
| close | DOUBLE | YES | Close. |
| volume | BIGINT | YES | Trade volume. |
| adj_close | DOUBLE | YES | Adjusted close if available. |
| ingest_ts | TIMESTAMP | YES | Ingestion timestamp. |

**Freshness:** daily (EOD) + intraday for supported symbols.  
**DQ:** `ts` monotonic within `ticker`, non-negative `volume`, high≥open/close≥low, no duplicate (ticker, ts, source).

---

### `equities_factors`
**Purpose:** Factor exposures per security and date (e.g., momentum, quality).  
**Upstream:** Internal factor pipeline (research models).  
**Storage:** `s3://hyper-lakehouse/equities/factors/` • Partition: `ts`.

| Column | Type | Null | Description |
|---|---|---|---|
| ts | DATE | NO | Effective date (UTC). |
| ticker | STRING | NO | Security identifier. |
| factor | STRING | NO | Factor name: `momentum`, `quality`, `value`, etc. |
| value | DOUBLE | YES | Factor exposure (z-score or standardized value). |
| ingest_ts | TIMESTAMP | YES | Ingestion time. |

**Freshness:** daily.  
**DQ:** Unique key `(ts, ticker, factor)`, value finite, winsorization within ±6σ (applied upstream).

---

### `equities_fundamentals`
**Purpose:** Point-in-time fundamentals (statements & ratios).  
**Upstream:** SEC/Refinitiv/Polygon fundamentals; curated by `build_curated.py`.  
**Storage:** `s3://hyper-lakehouse/equities/fundamentals/` • Partition: `ts`.

| Column | Type | Null | Description |
|---|---|---|---|
| ts | DATE | NO | Statement effective date (filing/as-of). |
| ticker | STRING | NO | Security identifier. |
| metric | STRING | NO | Metric/ratio key, e.g., `ROE`, `EPS`, `EV_EBITDA`. |
| value | DOUBLE | YES | Numeric value (original units). |
| currency | STRING | YES | ISO currency code, if applicable. |
| ingest_ts | TIMESTAMP | YES | Ingestion time. |

**Freshness:** quarterly (statements) + ad-hoc (restatements).  
**DQ:** Unique `(ts, ticker, metric)`, currency present for monetary metrics, unit normalization in curated layer.

---

## Foreign Exchange (FX)

### `fx_rates`
**Purpose:** Spot/cross FX rates.  
**Upstream:** Yahoo/Oanda via `fx_yahoo.py`.  
**Storage:** `s3://hyper-lakehouse/fx/rates/` • Partition: `DATE(ts)`.

| Column | Type | Null | Description |
|---|---|---|---|
| ts | TIMESTAMP | NO | Quote time (UTC). |
| base | STRING | NO | Base currency ISO, e.g., `USD`. |
| quote | STRING | NO | Quote currency ISO, e.g., `JPY`. |
| rate | DOUBLE | YES | Spot rate (base→quote). |
| source | STRING | YES | `yahoo`, `oanda`, etc. |
| ingest_ts | TIMESTAMP | YES | Ingestion time. |

**Freshness:** intraday.  
**DQ:** Unique `(ts, base, quote, source)`, rate>0, cross-symmetry checks vs. triangulation (tolerance 10⁻⁶).

---

### `fx_vol_surface`
**Purpose:** Implied vol surface snapshots for FX options.  
**Upstream:** Bloomberg/Reuters adapters.  
**Storage:** `s3://hyper-lakehouse/fx/vol_surface/` • Partition: `ts`.

| Column | Type | Null | Description |
|---|---|---|---|
| ts | DATE | NO | Surface as-of date (UTC). |
| pair | STRING | NO | FX pair, e.g., `EURUSD`. |
| tenor | STRING | NO | Option tenor, e.g., `1M`, `3M`. |
| strike | DOUBLE | NO | Strike or delta proxy as configured. |
| vol | DOUBLE | YES | Implied volatility (decimal, e.g., 0.12). |
| ingest_ts | TIMESTAMP | YES | Ingestion time. |

**Freshness:** daily.  
**DQ:** No negative vols; surface arbitrage sanity (butterfly/calendar) flagged upstream.

---

## Macro

### `macro_series`
**Purpose:** Macro time series (FRED, World Bank, IMF).  
**Upstream:** `macro_feed.py` (FRED & World Bank).  
**Storage:** `s3://hyper-lakehouse/macro/series/` • Partition: `ts`, `source`.

| Column | Type | Null | Description |
|---|---|---|---|
| ts | DATE | NO | Observation date (UTC). |
| source | STRING | NO | `FRED`, `WorldBank`, `IMF`. |
| series_id | STRING | NO | Series code, e.g., `CPIAUCSL`, `NY.GDP.MKTP.CD`. |
| country | STRING | YES | Country name/ISO code if applicable. |
| value | DOUBLE | YES | Numeric observation. |
| ingest_ts | TIMESTAMP | YES | Ingestion time. |

**Freshness:** monthly/quarterly depending on series.  
**DQ:** Unique `(ts, source, series_id, country)`, non-decreasing real-time vintages where available.

---

## News

### `news_feed`
**Purpose:** News headlines & bodies with NLP sentiment and tags.  
**Upstream:** `news_bridge.py` (Bloomberg, Reuters, RSS).  
**Storage:** `s3://hyper-lakehouse/news/feed/` • Partition: `DATE(ts)`, `source`.

| Column | Type | Null | Description |
|---|---|---|---|
| ts | TIMESTAMP | NO | Publish time (UTC). |
| source | STRING | NO | `bloomberg`, `reuters`, `rss`. |
| headline | STRING | YES | Headline text. |
| body | STRING | YES | Body text (may be truncated by feed terms). |
| sentiment | DOUBLE | YES | Model score [-1, +1] or [0,1] (documented in model card). |
| tags | ARRAY<STRING> | YES | Extracted entities/topics. |
| ingest_ts | TIMESTAMP | YES | Ingestion time. |

**Freshness:** realtime.  
**DQ:** Deduplicate by (ts, source, headline hash), PII scrub per `policy/` configs, language detection enrichment.

---

## Portfolio & Trading

### `portfolio_positions`
**Purpose:** Positions snapshot per account and ticker.  
**Upstream:** Execution engine / broker adapters.  
**Storage:** `s3://hyper-lakehouse/portfolio/positions/` • Partition: `DATE(ts)`.

| Column | Type | Null | Description |
|---|---|---|---|
| ts | TIMESTAMP | NO | Snapshot time (UTC). |
| account_id | STRING | YES | Internal account identifier. |
| ticker | STRING | YES | Security symbol. |
| qty | DOUBLE | YES | Position quantity. |
| avg_price | DOUBLE | YES | Average cost. |
| market_value | DOUBLE | YES | Market value at `ts`. |
| ingest_ts | TIMESTAMP | YES | Ingestion time. |

**Freshness:** intraday (hourly or on change).  
**DQ:** market_value ≈ qty×price (tolerance), non-negative not enforced (shorts allowed).

---

### `portfolio_pnl`
**Purpose:** Realized/unrealized P&L metrics.  
**Upstream:** Risk/PnL attribution jobs.  
**Storage:** `s3://hyper-lakehouse/portfolio/pnl/` • Partition: `DATE(ts)`.

| Column | Type | Null | Description |
|---|---|---|---|
| ts | TIMESTAMP | NO | Measurement time. |
| account_id | STRING | YES | Account id. |
| realized | DOUBLE | YES | Realized PnL. |
| unrealized | DOUBLE | YES | Unrealized PnL. |
| gross | DOUBLE | YES | Gross PnL. |
| net | DOUBLE | YES | Net PnL after fees/costs. |
| ingest_ts | TIMESTAMP | YES | Ingestion time. |

**Freshness:** intraday EOD roll-ups.  
**DQ:** Net = Gross − fees (tolerance), continuity across days.

---

### `portfolio_orders`
**Purpose:** Order lifecycle history.  
**Upstream:** OMS/EMS events.  
**Storage:** `s3://hyper-lakehouse/portfolio/orders/` • Partition: `DATE(ts)`.

| Column | Type | Null | Description |
|---|---|---|---|
| ts | TIMESTAMP | NO | Event time (UTC). |
| order_id | STRING | YES | Unique order id. |
| account_id | STRING | YES | Account id. |
| ticker | STRING | YES | Symbol. |
| side | STRING | YES | `BUY` / `SELL`. |
| qty | DOUBLE | YES | Quantity. |
| price | DOUBLE | YES | Limit/exec price. |
| status | STRING | YES | `NEW`, `PARTIAL`, `FILLED`, `CANCELLED`, etc. |
| ingest_ts | TIMESTAMP | YES | Ingestion time. |

**Freshness:** realtime.  
**DQ:** Status transitions valid; (order_id, ts) unique.

---

## Risk & Policy

### `risk_scenarios`
**Purpose:** Scenario shocks and PnL impacts.  
**Upstream:** Risk engine / scenario runner.  
**Storage:** `s3://hyper-lakehouse/risk/scenarios/`.

| Column | Type | Null | Description |
|---|---|---|---|
| ts | TIMESTAMP | NO | Run time (UTC). |
| scenario_id | STRING | YES | Scenario name/id. |
| description | STRING | YES | Human description. |
| shock | JSON | YES | Shock map (e.g., `{ "UST10Y_bps": 100 }`). |
| pnl_impact | DOUBLE | YES | Estimated PnL impact. |
| ingest_ts | TIMESTAMP | YES | Ingestion time. |

**Freshness:** daily or on-demand.  
**DQ:** JSON schema validation of `shock`; pnl_impact finite.

---

### `risk_var`
**Purpose:** Portfolio Value-at-Risk metrics.  
**Upstream:** Risk engine.  
**Storage:** `s3://hyper-lakehouse/risk/var/`.

| Column | Type | Null | Description |
|---|---|---|---|
| ts | TIMESTAMP | NO | As-of time. |
| portfolio_id | STRING | YES | Portfolio id. |
| var_95 | DOUBLE | YES | 95% VaR. |
| var_99 | DOUBLE | YES | 99% VaR. |
| stressed | DOUBLE | YES | Stressed VaR. |
| ingest_ts | TIMESTAMP | YES | Ingestion time. |

**Freshness:** daily.  
**DQ:** VaR values non-negative; var_99 ≥ var_95.

---

### `policy_audit`
**Purpose:** Compliance & security audit trail.  
**Upstream:** Policy engine / guards.  
**Storage:** `s3://hyper-lakehouse/policy/audit/`.

| Column | Type | Null | Description |
|---|---|---|---|
| ts | TIMESTAMP | NO | Event time. |
| user_id | STRING | YES | Actor id (user/service). |
| action | STRING | YES | Action verb, e.g., `LIMIT_UPDATE`. |
| resource | STRING | YES | Resource id/path. |
| status | STRING | YES | `SUCCESS` / `FAILURE`. |
| ingest_ts | TIMESTAMP | YES | Ingestion time. |

**Freshness:** realtime.  
**DQ:** (ts, user_id, action, resource) uniqueness best-effort; PII redaction enforced upstream.

---

## Conventions

- **Timestamps:** All stored in **UTC**. Partitioning uses `DATE(ts)` where applicable.  
- **Ids & Symbols:** Text fields are case-normalized upstream (`ticker` upper-case, `pair` like `EURUSD`).  
- **Types:** Use `DOUBLE` for continuous numeric, `BIGINT` for counts, `STRING` for textual.  
- **PII & Compliance:** See `policy/*.yaml`. News/body fields may be truncated per provider terms.  
- **Quality Gates:** Great Expectations suite in `data-fabric/quality/` runs pre-publish.

---

## Example Queries

```sql
-- 1) Pull EOD prices for a basket
SELECT ts::date AS d, ticker, close
FROM equities_prices
WHERE DATE(ts) BETWEEN DATE '2025-01-01' AND DATE '2025-03-31'
  AND ticker IN ('AAPL','MSFT','NVDA')
ORDER BY 1,2;

-- 2) Join factors to prices
SELECT p.ts::date AS d, p.ticker, p.close, f.value AS momentum_z
FROM equities_prices p
JOIN equities_factors f
  ON f.ts = p.ts::date AND f.ticker = p.ticker AND f.factor = 'momentum';

-- 3) FX conversion using spot
SELECT p.ts, p.ticker, p.close,
       p.close / r.rate AS close_usd
FROM equities_prices p
JOIN fx_rates r
  ON r.base = 'USD' AND r.quote = 'JPY'
 AND DATE(r.ts) = DATE(p.ts);