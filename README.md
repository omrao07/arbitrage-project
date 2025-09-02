
# HF Platform

A **production-ready hedge fund platform backbone**, designed Bloomberg‑style for resilience, monitoring, and compliance.  
This repo contains workers, shared libraries, risk models, sentiment analysis, infra configs, dashboards, and tests — everything except APIs, adapters, and broker connectors.

---

## 🚀 Features

- **Workers**
  - Analyst, Scenario, Sentiment workers (with DLQ + retries)
  - Shared bootstrap (tracing, entitlements, Redis connection)

- **Shared libraries**
  - `envelope.py` – standard message schema & idempotency
  - `dlq.py` – retry + dead letter queue handler
  - `otel.py` – OpenTelemetry tracing & metrics
  - `audit.py` – append‑only audit log writer
  - `entitlements.py` – role/desk‑based allow/deny
  - `calendars.py` – region‑aware trading calendars

- **Research**
  - Almgren‑Chriss execution model
  - RL Execution Agent (DDQN, env, evaluator)
  - VaR/ES & stress scenarios
  - Sentiment utilities (news, transcripts, NLP)

- **Infra as Code**
  - Kubernetes deployments with probes, HPA, PDB, anti‑affinity
  - Prometheus + Grafana dashboards & alert rules
  - Chaos & DR job manifests

- **Monitoring & SLOs**
  - OpenTelemetry integrated across workers
  - Prometheus metrics for latency, error, DLQ rates
  - Grafana dashboards (`worker_slo.json`, `exec_latency.json`)

- **Testing & CI/CD**
  - Unit + integration tests (`pytest`, `ruff`, `mypy`, `black`)
  - GitHub Actions pipeline (`ci.yml`)
  - Dockerfiles for workers
  - Makefile for dev, build, and deploy

- **UI Components**
  - Depth chart, Card, Utils
  - Pages: Overview, Risk Matrix

---

## 📂 Repo Structure

```
hf-platform/
├─ configs/        # YAML configs for workers, calendars, entitlements
├─ platform/       # Shared libs (envelope, dlq, otel, audit, entitlements, calendars)
├─ workers/        # Analyst, scenario, sentiment workers
├─ research/       # Execution models, risk, sentiment
├─ monitoring/     # Dashboards & exporters
├─ k8s/            # Kubernetes manifests (deployments, alerts, jobs)
├─ docker/         # Dockerfiles
├─ ui/             # React/TSX components (overview, risk matrix)
├─ tests/          # Unit + integration tests
├─ pyproject.toml  # Dependencies & tooling
├─ Makefile        # Dev/build/deploy commands
└─ README.md
```

---

## 🔧 Setup

1. **Install dependencies**
   ```bash
   pip install -e .[dev]
   ```

2. **Lint & test**
   ```bash
   make lint
   make test
   ```

3. **Run locally with Docker**
   ```bash
   make docker-build
   make docker-run
   ```

4. **Deploy to Kubernetes**
   ```bash
   make k8s-apply
   ```

---

## 📊 Monitoring

- Prometheus scrapes metrics from all workers.  
- Grafana dashboards show worker latency, DLQ rates, execution metrics.  
- Alert rules in `k8s/alerts/alert_rules.yaml` trigger PagerDuty/Slack.  

---

## 🛡️ Compliance & Safety

- Audit logs stamped with hash, timestamp, policy ID.  
- Entitlement policies applied per desk/role.  
- Region calendars & risk guardrails enforced at runtime.  
- DLQs capture poison messages for analysis.  

---

## 📌 Roadmap

- Add APIs (FastAPI/WS) for external access.  
- Add adapters (market data feeds, vendors).  
- Add broker connectors (execution venues).  
- Extend RL execution with continuous action PPO/SAC.  

---

## 📜 License

Private / internal use only.
