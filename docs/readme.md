# 🏦 Arbitrage Alpha Platform

A **modular hedge-fund style trading research & execution system**, built with:
- **Python back-end**: strategy engines, risk manager, portfolio allocator.
- **React/TS front-end**: dashboards (overview, risk matrix, exec console, strategy monitor).
- **Alt-data ingestion**: macro pipelines, satellite/energy/shipping signals, LLM macro monitor.
- **Full infra**: Redis bus, Prometheus/Grafana monitoring, Docker/K8s deploy.

---

## 📂 Repo Structure

---

## 🚀 Features

- **71+ strategies** (macro, carry, credit, ETF arb, regional alpha).
- **Dynamic risk manager**:
  - VaR, stress tests, scenario shocks.
  - Box & sector caps, gross/net guardrails.
- **Allocator suite**:
  - Equal weight, inverse-vol, mean-variance with sector caps.
  - Full test coverage (`test_allocator.py`).
- **Execution layer**:
  - Paper + live adapters (IBKR, Alpaca, Zerodha).
  - DLQ handling + retry policies (`test_dlq.py`).
- **Governance**:
  - Signed envelopes for signal transport (`test_envelope.py`).
  - LLM macro monitor (`docs/macro_llm_monitor.md`).
- **Dashboards**:
  - Overview cards, ticker tape, execution console, correlation galaxy, risk matrix, streaming heatmaps.

---

## 🧪 Tests

All major modules ship with **unit tests** (pytest + unittest bridge):

- `test_allocator.py` → Portfolio optimization, sector caps, stability.
- `test_risk.py` → Risk register, SLA/overdue detection, CSV I/O.
- `test_strategies.py` → Strategy CRUD, search, guardrails, rebalance.
- `test_exec.py` → Subprocess sandbox runner with redaction & timeout.
- `test_dlq.py` → Dead-letter queue semantics, retries, poison pills.
- `test_envelope.py` → Secure message envelopes with HMAC + replay protection.

Run everything:

```bash
pytest -q