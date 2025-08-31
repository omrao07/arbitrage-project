
# Macro LLM Monitor — Design & Runbook

> Observability, evaluation, and alerting for LLM-driven macroeconomic intelligence (data releases, policy moves, and market-impact narratives).

---

## 1) Goals & Scope

**Problem:** We run LLM jobs that summarize macro releases (CPI, NFP, GDP), parse policy communications, and produce risk classifications. We need **reliable monitoring** for quality, latency, cost, and drift.

**Primary goals**
- Detect **hallucinations** and **data mismatches** (e.g., wrong CPI YoY).
- Track **latency**, **token usage**, and **$ cost** per job and per source.
- Measure **quality** (factuality, calibration, extraction accuracy).
- Provide **real-time alerts** on failures, outliers, and content policy risks.
- Offer dashboards for **SLOs** and **release-day health**.

**Out of scope (v1):**
- Model training pipelines (only covers inference and post-LLM checks).
- Data vendor retries beyond a simple backoff policy.

---

## 2) Architecture (high level)

```
Vendor/API Feeds (BLS, BEA, FRED, ECB, BoE, etc.)
      │
      ├── Ingestor (release calendar + fetchers)
      │       • Schedules via Cron/Celery/K8s CronJobs
      │       • Caches canonical JSON for each release
      │
      ├── Validator (schema & checksum)
      │       • Ensures fields present; computes release_hash
      │
      ├── LLM Worker(s)
      │       • Prompt templates per release type (CPI/NFP/FOMC/etc.)
      │       • Output: summary, bullet points, prob. views, risks
      │
      ├── Post-LLM Guards
      │       • Fact checks against canonical JSON (thresholds)
      │       • Regex/key-phrase checks; numeric reconciliation
      │       • PII/policy filter; risk taxonomy classification
      │
      ├── Scoring + Metrics Emitter
      │       • factuality_score, brier, extraction_accuracy
      │       • latency_s, tokens_prompt/completion, cost_usd
      │
      └── Storage + Alerts + Dashboard
              • Redis/Kafka (events) → Parquet/S3 + Postgres (facts)
              • Prometheus metrics → Grafana
              • Alertmanager/PagerDuty/Slack
```

**Services**
- `macro-ingestor` (calendar + fetch)
- `macro-llm` (inference)
- `macro-guard` (post checks)
- `macro-monitor` (metrics/alerts, this doc)
- Optional: `macro-api` (serve latest validated summaries to UI)

---

## 3) Data contracts

### 3.1 Canonical release JSON (examples: CPI, NFP)
```json
{
  "release_type": "CPI",
  "source": "BLS",
  "period": "2025-08",
  "timestamp": "2025-09-10T12:30:02Z",
  "values": {
    "headline_mom": 0.2,
    "headline_yoy": 2.6,
    "core_mom": 0.3,
    "core_yoy": 2.4
  },
  "metadata": {
    "calendar_id": "cpi-us",
    "release_url": "https://example",
    "release_hash": "sha256:..."
  }
}
```

### 3.2 LLM output envelope
```json
{
  "job_id": "cpi-2025-08-abcd",
  "release_ref": {"release_type":"CPI","period":"2025-08"},
  "summary": "US CPI rose 0.2% m/m (2.6% y/y). Core accelerated slightly...",
  "bullets": [
    "Energy drag faded; shelter sticky",
    "Goods deflation narrowing; services moderate"
  ],
  "structured": {
    "stance": "slightly hawkish",
    "prob_cut_next_meeting": 0.18,
    "market_impact": {"UST2Y": "bearish", "DXY": "bullish", "SPX": "neutral"}
  },
  "extracted": {
    "headline_mom": "0.2%",
    "headline_yoy": "2.6%",
    "core_mom": "0.3%",
    "core_yoy": "2.4%"
  },
  "tokens": {"prompt": 1420, "completion": 210},
  "latency_s": 4.7,
  "cost_usd": 0.021
}
```

---

## 4) Guardrails & Scoring

### 4.1 Numeric reconciliation
- Convert `extracted.*` to numeric.
- Compare with canonical `values.*` within **tolerance** (e.g., ±0.01pp).
- Compute **factuality_score ∈ [0,1]** (1 = all fields consistent).

### 4.2 Policy & content checks
- Disallow PII, trading recommendations without risk disclaimer.
- Disallow model hallucinations: if confidence below threshold or missing fields → **degrade to template** summary.

### 4.3 Probabilistic calibration (if forecasts supplied)
- If we output `prob_*` (e.g., `prob_cut_next_meeting`), track **Brier score** after outcome is known.
- Maintain a **reliability diagram** per horizon.

### 4.4 Extraction accuracy
- Regex-based slot filling for standard fields (CPI, NFP, GDP).
- `extraction_accuracy = matched_slots / total_slots`.

### 4.5 Aggregate quality index
```
quality_index = 0.5*factuality_score + 0.3*extraction_accuracy + 0.2*(1 - normalized_brier)
```

---

## 5) Metrics (Prometheus)

| Metric | Type | Labels | Description |
|---|---|---|---|
| `llm_requests_total` | Counter | `model,release_type,source` | Number of LLM calls |
| `llm_latency_seconds` | Histogram | `model,release_type` | End-to-end latency |
| `llm_tokens_prompt_total` | Counter | `model,release_type` | Prompt tokens |
| `llm_tokens_completion_total` | Counter | `model,release_type` | Completion tokens |
| `llm_cost_usd_total` | Counter | `model,release_type` | Accumulated USD spend |
| `llm_factuality_score` | Gauge | `release_type` | Per-job factuality score |
| `llm_extraction_accuracy` | Gauge | `release_type` | Slots matched ratio |
| `llm_quality_index` | Gauge | `release_type` | Aggregate quality |
| `llm_guard_fail_total` | Counter | `reason,release_type` | Guardrail failures |
| `llm_alerts_total` | Counter | `severity,release_type` | Alerts emitted |

**SLOs**
- **Availability:** `P(guard_pass) ≥ 99.5%` on release days.
- **Latency:** p95 `llm_latency_seconds ≤ 8s` during 30m release window.
- **Quality:** p50 factuality ≥ 0.98, p95 ≥ 0.95.
- **Cost:** daily budget caps per release type.

---

## 6) Alerting

**Immediate (page)**
- `guard_pass == false` on any tier-1 release (CPI/NFP/FOMC).
- p95 latency > SLO for 3 consecutive jobs during release window.
- Cost rate > budget per hour.

**Ticket/Slack (warn)**
- factuality_score ∈ [0.9, 0.95).
- extraction_accuracy ∈ [0.8, 0.9).
- No data fetched by T+3 minutes of scheduled time.

**Silencing rules**
- Silence alerts for a release if the data vendor status page marks outage.

---

## 7) Dashboards (Grafana spec)

**Panels**
1. **Release Day Health**: bar for guard_pass rate by release_type.
2. **Latency**: p50/p95 line for `llm_latency_seconds` (by model).
3. **Cost & Tokens**: stacked area for cost & tokens over time.
4. **Quality**: factuality_score distribution + extraction_accuracy trend.
5. **Failures**: table of last 20 guard fails with reasons and payload links.
6. **Calibration**: reliability diagram (post-event).

**Useful PromQL**
```promql
sum(rate(llm_cost_usd_total[1h])) by (release_type)
histogram_quantile(0.95, sum(rate(llm_latency_seconds_bucket[5m])) by (le, release_type))
avg(llm_factuality_score) by (release_type)
```

---

## 8) Storage schemas

### 8.1 Postgres tables
```sql
CREATE TABLE macro_release (
  id TEXT PRIMARY KEY,
  release_type TEXT NOT NULL,
  period TEXT NOT NULL,
  timestamp TIMESTAMPTZ NOT NULL,
  source TEXT NOT NULL,
  values_json JSONB NOT NULL,
  metadata JSONB
);

CREATE TABLE llm_job (
  job_id TEXT PRIMARY KEY,
  release_id TEXT REFERENCES macro_release(id),
  model TEXT,
  latency_s DOUBLE PRECISION,
  tokens_prompt INTEGER,
  tokens_completion INTEGER,
  cost_usd NUMERIC,
  output_json JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE llm_guard (
  job_id TEXT PRIMARY KEY REFERENCES llm_job(job_id),
  factuality_score DOUBLE PRECISION,
  extraction_accuracy DOUBLE PRECISION,
  quality_index DOUBLE PRECISION,
  pass BOOLEAN,
  reasons TEXT[],
  created_at TIMESTAMPTZ DEFAULT now()
);
```

### 8.2 Object store (S3)
- `s3://macro/ingest/{release_type}/{period}/canonical.json`
- `s3://macro/llm/{release_type}/{period}/{job_id}.json`
- `s3://macro/guard/{release_type}/{period}/{job_id}.json`

---

## 9) Prompt templates (sketch)

```jinja
System: You are a macro analyst. Use only the provided data. If a figure is missing, say "not available".
User:
Release: {{ release_type }} ({{ period }})
Canonical JSON:
{{ canonical_json | tojson }}

Tasks:
1) Summarize in 3 bullets focusing on what's new vs. previous release.
2) Extract fields: headline_mom, headline_yoy, core_mom, core_yoy.
3) Provide a stance (dovish/neutral/hawkish) and probability of rate cut next meeting (0..1).

Output JSON with keys: summary, bullets[], extracted{}, structured{}.
```

---

## 10) Guard checks — pseudocode

```python
def reconcile_numbers(extracted, canonical, tol=0.01):
    mismatches = []
    score = 1.0
    for k, v in canonical.items():
        if k in extracted:
            try:
                x = float(str(extracted[k]).strip('%'))/100 if '%' in str(extracted[k]) else float(extracted[k])
            except Exception:
                mismatches.append((k, "parse_error")); score -= 0.25; continue
            diff = abs(x - float(v))
            if diff > tol:
                mismatches.append((k, f"diff={diff:.4f}"))
                score -= 0.25
        else:
            mismatches.append((k, "missing")); score -= 0.25
    return max(0.0, score), mismatches

def guard(job, canonical):
    score, errs = reconcile_numbers(job["extracted"], canonical["values"])
    extraction = matched_slots(job["extracted"], canonical["values"])
    pass_ = score >= 0.95 and extraction >= 0.95
    return {
      "factuality_score": score,
      "extraction_accuracy": extraction,
      "quality_index": 0.5*score + 0.5*extraction,
      "pass": pass_,
      "reasons": [e[0] for e in errs]
    }
```

---

## 11) Deployment & Config

- **Kubernetes**: separate Deployments for ingestor, llm, guard, monitor.
- **Env** (examples):
  - `MACRO_SLO_LATENCY_P95=8`
  - `MACRO_BUDGET_DAILY_USD=15`
  - `PROM_PORT=9102`
  - `S3_BUCKET=macro`
- **RBAC**: read-only for ingestors; write for storage; network policy to LLM endpoints only.
- **Secrets**: API keys via K8s Secrets & mounted env.

---

## 12) Runbook (on-call)

**1. Release failure (guard_pass=false on CPI/NFP)**  
- Check `macro-ingestor` logs for fetch errors.  
- Compare `canonical.json` vs `extracted` in last `llm_job`.  
- If mismatch due to vendor lag: rerun job with latest payload.  
- If parsing regression: rollback to previous extraction prompt.  
- Acknowledge PagerDuty; attach root cause in ticket.

**2. Latency SLO breach**  
- Inspect model queue depth; scale `macro-llm` replicas.  
- Switch to a **faster model** for surge window (feature flag).  
- Enable **summary-lite** template (shorter tokens).

**3. Cost surge**  
- Verify no runaway retries.  
- Enforce token caps; enable compression of context.  
- Throttle low-priority releases via rate limiter.

**4. Data vendor outage**  
- Mark release as `degraded=true`, silence alerts, post status in Slack.  
- Backfill when vendor recovers; re-enable guards.

---

## 13) Backtesting & QA

- Replay last 12 months of releases; compute factuality/extraction distributions.  
- Plot calibration (Brier) for meeting outcomes where we emitted probabilities.  
- Maintain a **golden set** of releases w/ expected JSON outputs.  
- Add CI check: run 5 sample releases → require `quality_index ≥ 0.95` to pass.

---

## 14) Roadmap

- Add **RAG** with policy statements & prior meeting minutes.  
- Add **multilingual** support (ECB/BoJ) with translation quality checks.  
- Deeper **entity-level impact** mapping to markets (rates, FX, equity sectors).  
- Human-in-the-loop review for tier-1 releases.

---

## 15) Appendix — Example OpenMetrics exposition

```
# HELP llm_quality_index Aggregate quality score
# TYPE llm_quality_index gauge
llm_quality_index{release_type="CPI"} 0.98
llm_latency_seconds_sum{release_type="CPI"} 47.1
llm_latency_seconds_count{release_type="CPI"} 10
llm_cost_usd_total{release_type="CPI"} 0.21
```
