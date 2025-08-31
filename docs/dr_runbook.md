# 🚨 Disaster Recovery Runbook — Hedge Fund Platform

> **Purpose:** Ensure rapid recovery of trading platform during outages, broker failures, data feed issues, or risk events.

---

## 1. Incident Classification

- **P1 (Critical)** — Trading halted, broker disconnected, Redis/DB down, kill switch engaged.
- **P2 (High)** — Data feed degraded, latency >100ms, partial broker disconnect.
- **P3 (Medium)** — Dashboards unavailable, minor worker crash.
- **P4 (Low)** — Cosmetic/UI issues, non-trading component down.

---

## 2. Core Principles

1. **Protect Capital** — risk manager + kill switch always first.
2. **Preserve State** — ensure ledger and Redis are snapshotted.
3. **Failover Fast** — switch to backup adapters or feeds.
4. **Communicate** — log incident, notify ops/team immediately.

---

## 3. Recovery Steps

### 3.1 Broker Outage
- Trigger **Kill Switch**:
  ```bash
  redis-cli HSET policy:kill_switch enabled true
  ```
- Reroute via **backup adapter** (e.g., IBKR → Zerodha, Zerodha → Paper).
- Verify order flow:
  ```bash
  redis-cli XLEN orders.validated
  redis-cli XLEN orders.fills
  ```

### 3.2 Data Feed Failure
- Switch registry to backup in `registry.yaml`:
  ```yaml
  primary_feed: yahoo
  backup_feed: moneycontrol
  ```
- Restart ingestion workers:
  ```bash
  kubectl rollout restart deploy/ws-gateway
  ```
- Enable `candle_aggregator.py` gap-fill.

### 3.3 Redis Crash
- Restart Redis container:
  ```bash
  docker restart redis
  ```
- Replay from **ledger**:
  ```bash
  python backend/ops/replay_with_params.py --source ledger.db
  ```

### 3.4 Postgres Crash
- Restore from snapshot:
  ```bash
  pg_restore -U hedgefund_user -d hedgefund /backups/pg/latest.dump
  ```

### 3.5 Risk Breach
- Kill switch auto-engages.
- Flatten positions via `hedger.py`:
  ```bash
  python backend/ops/hedger.py --flatten-all
  ```
- Run scenario generator to assess exposure:
  ```bash
  python backend/ops/scenario_generator.py --all
  ```

---

## 4. Monitoring & Alerts

- **Grafana Alerts**:
  - Latency > 100ms
  - Redis memory > 80%
  - VaR/ES > limits
- **Slack/Email Notifications** tied to `alerts.yaml`

---

## 5. DR Drills

- Monthly **kill switch drill** (simulate 5% intraday drawdown).
- Quarterly **broker failover drill** (IBKR → Paper adapter).
- Semi-annual **data feed failover** (Yahoo → Moneycontrol).

---

## 6. Communication

- Incident report logged in `runbooks/incidents/`.
- Notify:
  - Ops team
  - Risk officer
  - Stakeholders (summary only)

---

✅ With this DR runbook, system resilience is ensured even during **broker outages, data feed crashes, or catastrophic risk events**.
