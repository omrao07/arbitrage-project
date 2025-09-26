---

## 1) Nodes & dependencies

| Dataset ID              | Source     | Frequency | Primary Key                         | Partitions | Upstream deps                     |
|-------------------------|------------|-----------|-------------------------------------|------------|-----------------------------------|
| `refdata_issuers`       | internal   | ad-hoc    | `issuer_id`                         | none       | —                                 |
| `credit/indices.universe`| internal  | ad-hoc    | `index_code, series, version`       | none       | —                                 |
| `blp_cds_spreads`       | bpipe      | daily     | `dt, entity_id, tenor_yrs`          | `dt`       | `refdata_issuers`                 |
| `blp_credit_indices`    | bpipe      | daily     | `dt, index_code, series, version`   | `dt`       | `credit/indices.universe`         |
| `blp_corp_oas`          | bpipe      | daily     | `dt, ticker`                        | `dt`       | `refdata_instruments` (security master) |
| `blp_ratings_history`   | blp        | ad-hoc    | `ts, issuer_id, agency`             | `dt`       | `refdata_issuers`                 |
| `internal_cds_curves`   | derived    | daily     | `dt, entity_id, tenor_yrs`          | `dt`       | `blp_cds_spreads`                 |

> Canonical fields and aliases are defined in `credit/fields.yaml` and `meta/fields/types.yaml`.

---

# 2) Column-level lineage (selected)

 `blp_cds_spreads`

- `spread_bp`  ← Bloomberg `CDS_SPREAD_MID` → **bp**
- `upfront_pct`← `CDS_UPFRONT_MID` → **fraction** (0–1)
- `recovery`   ← `RECOVERY_RATE` → **fraction** (0–1)
- `tenor_yrs`  ← vendor tenor buckets mapped to numeric years: {1,3,5,7,10}
- `entity_id`  ← join `BBG Ticker / RED code` → `refdata_issuers.entity_id`

 `blp_credit_indices`

- `spread_bp`  ← `CDS_INDEX_SPREAD_MID`
- `total_return` ← `TOTAL_RETURN_INDEX` (document if rebased)
- `(index_code, series, version)` from `credit/indices.universe`

 `blp_corp_oas`

- `oas_bp` ← `OAS_SPREAD_MID`
- `z_spread` ← `Z_SPREAD_MID`
- `yield` ← `YLD_YTM_MID`
- `duration` ← `MOD_DUR_MID`, `convexity` ← `CONVEXITY_MID`
- `ticker` ← security master mapping (internal)

 `blp_ratings_history`

- `rating_moody`, `rating_sp` are **per-dataset** normalized strings; point-in-time via `ts`.  
- SCD2 semantics downstream if you snapshot by `issuer_id, agency`.

 `internal_cds_curves`

- Inputs: `blp_cds_spreads.{spread_bp, tenor_yrs, recovery}` at `dt, entity_id`.
- Output: `hazard, survival` via bootstrap method (documented in `method`).

---

## 3) Schedules & SLAs

| Dataset                | Window (UTC)     | SLA (freshness) | Late data policy                    |
|-----------------------|------------------|------------------|-------------------------------------|
| `blp_cds_spreads`     | T (NY close +2h) | ≤ 24h           | Re-pull T-2 if any T gaps detected  |
| `blp_credit_indices`  | T (NY close +2h) | ≤ 24h           | Same as above                       |
| `blp_corp_oas`        | T (NY close +2h) | ≤ 24h           | Same as above                       |
| `blp_ratings_history` | event-driven     | N/A              | Backfill by event time range        |
| `internal_cds_curves` | after CDS ready  | ≤ 24h           | Rebuild when base re-ingested       |

Freshness thresholds align with `meta/providers.yaml → ops.alerts.freshness_min`.

---

## 4) Rebuild order & triggers

1. `refdata_issuers`, `credit/indices.universe` (on change)  
2. `blp_cds_spreads`, `blp_credit_indices`, `blp_corp_oas` (daily)  
3. `internal_cds_curves` (triggered when 1 or more of step 2 updated for the same `dt`)

**Trigger rule:**  

---

## 5) Backfill strategy

- **Point backfills**: specify `[start_dt, end_dt]` per dataset; for `internal_cds_curves` recompute from CDS spreads only for touched dates.
- **Idempotency**: write to a temp path `.../_tmp/run_id` → validate → atomic move to final partition.
- **Reconciliation**: after backfill, run parity checks (counts, nulls, min/max deltas) vs. previous snapshot.

---

## 6) Quality gates

Apply the following at **ingest** (vendor → bronze) and **normalize** (bronze → silver):

- `dt`: not null, not in future.
- `spread_bp`, `oas_bp`, `z_spread`, `yield`, `duration`, `convexity`: `non_negative` (nullable allowed).
- `upfront_pct`, `recovery`, `survival`: `pct_0_1`.
- `tenor_yrs`: `non_negative`, in allowed set for CDS (1,3,5,7,10).
- **Cross-row**: for `blp_credit_indices`, prevent series/version regressions (monotone series roll date).
- **Join health**: `% unmatched` when mapping vendor entity → `entity_id` must be < 0.5% (alert otherwise).

---

## 7) Symbology & keys

- **Issuer / entity mapping**: vendor RED/BBG → `entity_id` in `refdata_issuers`. Store both for audit.
- **Credit indices**: freeze `(index_code, series, version)` tuples in `credit/indices.universe`; do not infer from strings.
- **Bonds**: `ticker` is your internal security key (not vendor mnemonic); keep CUSIP/ISIN as attributes, not keys.

---

## 8) Versioning & reproducibility

- Semantic version per dataset (`version: "1.0.0"`).
- Any **schema breaking** change bumps MAJOR and requires a new dataset id OR a migration plan.
- Keep **method** id in `internal_cds_curves.method` to tie to the exact bootstrap settings (recovery assumption, interpolation, day count).

---

## 9) Rollbacks

- Maintain **daily snapshots** or partition copies: `.../dataset/_snapshots/YYYYMMDD/`.
- Rollback = pointer flip of the partition’s manifest to previous snapshot (no rewrite).

---

## 10) Example orchestration (pseudo)

```yaml
jobs:
  cds_ingest:
    dataset: blp_cds_spreads
    params: { date: ${D} }
  cds_curves:
    dataset: internal_cds_curves
    depends_on: [cds_ingest]
    params: { date: ${D} }

  indices_ingest:
    dataset: blp_credit_indices
    params: { date: ${D} }

  corp_oas_ingest:
    dataset: blp_corp_oas
    params: { date: ${D} }

  ratings_pull:
    dataset: blp_ratings_history
    params: { from_ts: ${TS0}, to_ts: ${TS1} }
