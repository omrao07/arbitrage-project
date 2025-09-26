# Meta / Glossary

Authoritative glossary and conventions for the catalog. Keep it simple, unambiguous, and vendor-agnostic.

---

## 1) Core objects

**Dataset**  
A declarative spec that describes one table/stream: schema, keys, frequency, endpoints, lineage, tags. Stored as `*.yaml` under `catalog/<vendor>/**/datasets/`.

**Vendor**  
Source namespace: `bloomberg`, `koyfin`, `hammer_pro`, `internal`.

**Endpoint**  
How to fetch the dataset: `bpipe | blp | rest | file`. Carries `path` + `params`.

**Registry**  
In-memory index built from all dataset specs; supports queries by id, vendor, tags, frequency, lineage.

---

## 2) Naming & casing

- **dataset id**: `snake_case` and vendor-prefixed if external, e.g. `blp_eod_prices`, `koyfin_factor_scores`.  
- **column names**: `snake_case` (canonical), stable across vendors.  
- **tags**: short kebab or snake, e.g. `prices`, `fx`, `vol`, `fundamentals`.  
- **tenors**: `1W`, `1M`, `3M`, `6M`, `1Y` (uppercase).  
- **fx pair**: `EURUSD` (BASE+QUOTE, 6 letters, uppercase).  
- **MIC**: ISO 10383, uppercase (e.g., `XNAS`).  
- **currency**: ISO 4217 (e.g., `USD`).

---

## 3) Time rules

- **`dt`** is a **business date** (`YYYY-MM-DD`).  
- **`ts`** is an **instant** in UTC (ISO8601); store/transport as UTC only.  
- **Freshness** = `now_utc - last_success_ts`.  
- No future timestamps unless explicitly stated (use `no_future_date` macro).

---

## 4) Keys, partitions, lineage

- **primary_key**: minimal set that uniquely identifies a row, e.g. `[dt, ticker]`, `[ts, ticker]`, `[dt, pair, tenor]`.
- **partitions**: physical layout hints, e.g. `[dt]`, `[dt, exchange]`.
- **lineage**: upstream dataset ids used to derive this dataset (enables rebuild order and impact analysis).

---

## 5) Canonical columns (common)

| Column        | Type     | Notes                                      |
|---------------|----------|--------------------------------------------|
| `dt`          | date     | trading/business date                      |
| `ts`          | timestamp| event time (UTC)                            |
| `ticker`      | string   | internal symbol                             |
| `currency`    | string   | ISO 4217                                    |
| `px_open`     | float64  |                                            |
| `px_high`     | float64  |                                            |
| `px_low`      | float64  |                                            |
| `px_last`     | float64  | close/last                                  |
| `px_adj`      | float64  | split/div adjusted close                    |
| `volume`      | float64  | shares/contracts                            |
| `vwap`        | float64  | volume-weighted average price               |
| `ret_1d`      | float64  | simple return                               |
| `pair`        | string   | fx pair (e.g., EURUSD)                      |
| `tenor`       | string   | `1W/1M/3M/6M/1Y/...`                        |
| `delta`       | float64  | option delta 10/25/50 (as percent point)    |
| `vol`         | float64  | implied vol (0–1 or fraction)               |
| `rr`          | float64  | risk reversal                               |
| `bf`          | float64  | butterfly                                   |

> Full canonical list and aliases live in **`meta/fields/types.yaml`**.

---

## 6) Frequencies

`tick | 1m | 5m | hourly | daily | weekly | monthly | ad-hoc`

- **dataset.frequency** describes the natural cadence of the source, not your batch window.
- Intraday bars imply `ts` primary key; daily bars imply `dt`.

---

## 7) Quality macros (semantic names)

- `not_null` – value must be present.  
- `non_negative` – value ≥ 0 if present.  
- `pct_0_1` – 0 ≤ value ≤ 1 if present.  
- `no_future_date` – not beyond current UTC time/date.  
- `ohlc_bounds` – `low ≤ {open,last} ≤ high` (when all present).

These names appear in dataset column definitions; validators enforce them.

---

## 8) FX & options terms

- **Pair**: `EURUSD` = 1 EUR priced in USD.  
- **Tenor**: time to expiry (1W, 1M, …).  
- **Delta**: option delta bucket (10/25/50).  
- **RR (Risk Reversal)**: `vol(call) − vol(put)` at same delta/tenor.  
- **BF (Butterfly)**: `0.5*(vol(call)+vol(put)) − vol(ATM)`; vendor formulas vary — record vendor definition in dataset description.  
- **Forward points (`fwd_pts`)** in pips; **forward rate (`fwd_rate`)** may be vendor-provided or computed.

---

## 9) Symbology

- `ticker` is internal canonical.  
- Map vendor ids (BBG, RIC, ISIN) via `symbology/mapping.yaml`.  
- Include `mic` and `exchange_code` when vendor requires routing.

---

## 10) Endpoints

Kinds:

- `bpipe` – Bloomberg B-PIPE fields/universe.  
- `blp` – Bloomberg Desktop/Server API request.  
- `rest` – HTTP JSON/CSV API.  
- `file` – static files (SFTP, S3, GCS, local).

Each endpoint can carry `path` and `params` (universe references, calendars, tz, auth keys, etc.).

---

## 11) Units

- Monetary: `USD/EUR/GBP/JPY/INR`.  
- Rates: `pct` (0–1 fraction unless dataset states percent points), `bp` (basis points).  
- FX: `pips` for forward points.  
- Size: `shares`, `contracts`.  
- Time: `years` where numeric tenor is used.

Be explicit in each dataset; do not mix fractions and percent points within a column.

---

## 12) Dataset lifecycle

`draft → active → deprecated → retired`

- **draft**: schema may change, backfills allowed.  
- **active**: schema frozen (backward compatible only).  
- **deprecated**: write-only for sunset window.  
- **retired**: frozen; available for historical reads.

Keep lifecycle in the dataset `description` or an extension field if needed.

---

## 13) Example dataset skeleton

```yaml
id: blp_eod_prices
title: Bloomberg EOD Prices
vendor: bloomberg
source: bpipe
version: "1.0.0"
frequency: daily
primary_key: [dt, ticker]
partitions: [dt]
columns:
  - { name: dt,        type: date,      quality: [not_null, no_future_date] }
  - { name: ticker,    type: string,    quality: [not_null] }
  - { name: px_last,   type: float64,   unit: USD, quality: [not_null, non_negative] }
  - { name: volume,    type: float64,   unit: shares, quality: [non_negative] }
endpoints:
  - kind: bpipe
    path: "PX_LAST,VOLUME"
    params:
      universe_ref: "bloomberg/baseline/symbology/mapping.yaml"
      calendar: "exchange"
      tz: "UTC"
tags: [prices, eod, baseline] 
