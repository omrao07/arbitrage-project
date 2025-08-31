# 📝 Hedge Fund Platform — Runbook

> **Purpose:** How to **run**, **monitor**, and **troubleshoot** your arbitrage + multi-strategy hedge fund stack (engine, risk, brokers, WS, workers, UI).

---

## 0) System Map (high level)

```
[Data Feeds]
  ├─ Market: brokers, venues, altdata
  └─ News: Yahoo, Moneycontrol, Twitter

          ▼
[Ingest / WS Gate]  (ws_gateway.py, ws_candles.py, ws_orderbook.py, ws_greeks.py)
          │  Redis Streams (ticks, books, orders, fills, alerts)
          ▼
[Strategy Engine]   (strategy_runner.py, Strategy subclasses)
          │  -> orders.incoming
          ▼
[Risk Manager]      (risk_manager.py: limits, VaR/ES, kill switch, policy engine)
          │  -> orders.validated
          ▼
[Router / OMS]      (router.py, market_maker.py, cost_model.py, pricer.py)
          │  -> broker adapters (IBKR, Zerodha, Paper)
          ▼
[Broker Fills] ----> reconciler.py -> ledger.py

Side workers:
- Analyst / Sentiment / Scenario workers (insights_worker.py, sentiment_worker.py, scenario_worker.py)
- Backtester + TCA + Notebooks
- Dashboards (Next.js / Bolt)
```

---

## 1) Prereqs

- **Python** 3.10+ & **Node** 18+
- **Docker** & **docker compose**
- (Optional) **kubectl** & a Kubernetes cluster
- Create `.env` (for local dev) with at least:
  ```
  REDIS_PASSWORD=redispass123
  POSTGRES_USER=hedgefund_user
  POSTGRES_PASSWORD=supersecretpassword
  OPENAI_API_KEY=...
  HUGGINGFACE_TOKEN=...
  ```

---

## 2) Start the Stack

### 2.1 Local (bare-metal dev)
```bash
# Backend deps
pip install -r requirements.txt

# Frontend dev
cd frontend
npm install
npm run dev
# -> http://localhost:3000
```

### 2.2 Docker Compose (recommended for full stack)
```bash
docker compose -f docker-compose.yaml -f docker-compose.override.yaml up --build
# Services: api, ws, analyst-worker, scenario-worker, sentiment-worker,
#           redis, postgres, kafka, zookeeper, prometheus, grafana
```

### 2.3 Kubernetes (prod-like)
```bash
# Secrets & core infra
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/prom_grafana.yaml

# App services
kubectl apply -f k8s/api_depl.yaml
kubectl apply -f k8s/ws_depl.yaml
kubectl apply -f k8s/analyst_worker.yaml
kubectl apply -f k8s/sentiment_worker.yaml
kubectl apply -f k8s/scenario_worker.yaml
```

---

## 3) Smoke Tests

### 3.1 Health
```bash
# API health (FastAPI/Flask)
curl -s http://localhost:8000/health

# WS gateway TCP/HTTP
nc -vz localhost 8765
```

### 3.2 Redis / DB
```bash
redis-cli -a "$REDIS_PASSWORD" ping
PGPASSWORD=$POSTGRES_PASSWORD psql -h localhost -U $POSTGRES_USER -d hedgefund -c '\dt'
```

### 3.3 WS streams quick check
```bash
# Subscribe to ticks (if you have a ws->redis forwarder writing ticks)
redis-cli -a "$REDIS_PASSWORD" XINFO STREAM ticks.live
redis-cli -a "$REDIS_PASSWORD" XLEN ticks.live
```

---

## 4) Daily Operations

### 4.1 Pre-Open Checklist
- [ ] **Secrets loaded**: `kubectl get secrets | grep hedgefund`  
- [ ] **Redis up**: `redis-cli -a "$REDIS_PASSWORD" ping`  
- [ ] **Brokers connected**: `kubectl logs deploy/router -f | grep "Connected"`  
- [ ] **Risk config** loaded (`configs/risk.yaml`)  
- [ ] **Kill switch** set to SAFE: `redis-cli HGET policy:kill_switch enabled` → `false`  
- [ ] **Dashboards live**: Grafana `PnL / Risk / Latency` show data

### 4.2 During Market Hours
- Monitor:
  - **Latency p95** < 50ms (ingest→signal→risk→route→oms)
  - **Order reject rate** < 0.5%
  - **VaR/ES** within policy bands (alerts fire if breached)
- Actions:
  - Toggle strategy via `HSET strategy:enabled <name> true|false`
  - Adjust **Policy Engine** (exposure caps, region blocks)

### 4.3 Post-Close
```bash
# Backtest replay for the day
python backend/backtester.py --date $(date +%F)

# Reconcile fills vs OMS
python backend/ops/reconciler.py --date $(date +%F)

# Generate daily PDF/MD
python backend/reports/report_generator.py --date $(date +%F)
```

---

## 5) Troubleshooting Matrix

| Symptom | Quick Check | Likely Cause | Fix |
|---|---|---|---|
| **No orders leaving engine** | `XLEN orders.incoming` | Strategy not publishing / disabled | Enable strategy, check `on_tick` exceptions in logs |
| **Orders rejected by risk** | `XLEN orders.rejected` & reason | Limit breach / missing symbol config | Update `policy.yaml`, `registry.yaml`; re-enable |
| **Latency spike** | Grafana latency panel | Broker slow / network | Switch venue, reduce batch, enable `latency_adapter` |
| **PnL drift vs broker** | Run `reconciler.py` | Missing fees/slippage | Update `cost_model.py`, backfill fees; reprice |
| **WS dashboard empty** | `kubectl logs deploy/hedgefund-ws` | WS not bound / CORS | Check port `8765`, origin, service selector |
| **Risk dashboard blank** | `kubectl logs deploy/risk-manager` | Timeseries store down | Verify Postgres; restart risk manager |

---

## 6) Incident Runbooks

### 6.1 Broker Outage
1. **Kill switch** partial: `HSET policy:route_override VENUE secondary`  
2. Reroute via **paper broker** or backup adapter.  
3. Validate fills loopback with `order_store.py` + `reconciler.py`.

### 6.2 Data Feed Degradation
1. Switch **primary→backup** in `registry.yaml` (Yahoo/Moneycontrol/Alt).  
2. Reduce frequency in `throttle.py`, enable **gap-fill** (candle aggregator).  
3. Watch **model drift** flags (insight agent).

### 6.3 Risk Breach (VaR/ES/Drawdown)
1. Trigger **Kill switch**: `HSET policy:kill_switch enabled true`.  
2. Flatten via `hedger.py` (close positions).  
3. Snapshot `ledger.py`, file incident summary in `runbooks/incidents/`.

---

## 7) Deployments & Rollbacks

### 7.1 Deploy
```bash
# Build & tag
docker build -t your-registry/hedgefund-api:$(git rev-parse --short HEAD) -f backend/Dockerfile backend
docker push your-registry/hedgefund-api:$(git rev-parse --short HEAD)

# Update Kubernetes
kubectl set image deploy/hedgefund-api api=your-registry/hedgefund-api:<tag>
kubectl rollout status deploy/hedgefund-api
```

### 7.2 Rollback
```bash
kubectl rollout undo deploy/hedgefund-api
kubectl rollout undo deploy/hedgefund-ws
```

---

## 8) Observability

- **Logs**
  ```bash
  kubectl logs deploy/hedgefund-api -f --tail=200
  kubectl logs deploy/hedgefund-ws -f
  ```
- **Metrics**
  - Prometheus targets healthy
  - Grafana dashboards:
    - `01_Latency`
    - `02_Risk_VaR_ES`
    - `03_PnL_Attribution`
