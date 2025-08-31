# 🎬 Demo Script — Hedge Fund Trading Platform

## 1. Intro (1–2 min)
- “This is a **multi-strategy hedge fund platform** I’ve built from scratch.  
  It covers **data ingestion, alpha strategies, risk management, OMS/router, broker adapters, and live dashboards**.”
- “I’ll show how the system ingests **real-time data**, runs strategies, enforces **risk limits**, and visualizes **PnL, VaR/ES, and trades**.”

---

## 2. Startup (2 min)
1. **Bring up services**  
   ```bash
   docker compose up
   ```
   - Redis, API, WebSocket, Risk Manager, Strategy Engine, Workers, Frontend.
2. Show **health checks**:  
   - `curl http://localhost:8000/health` → “healthy”  
   - Redis ping, logs of broker adapter connection.

---

## 3. Data Ingestion (2–3 min)
- “The system connects to **Yahoo Finance & Moneycontrol** for market + news feeds.”
- Open **Redis stream** viewer:
  ```bash
  redis-cli XLEN ticks.live
  ```
- Show `ws_candles.py` pushing candlestick data.
- Explain **altdata ingestion** (satellite, shipping, sentiment).

---

## 4. Strategy Engine (3–4 min)
- Show `strategy_base.py` and a **sample strategy** (`ExampleBuyTheDip`).
- Run:
  ```bash
  python backend/engine/strategy_runner.py --stream ticks.live
  ```
- Show live logs:
  - Strategies consuming ticks
  - Publishing orders → `orders.incoming`.

---

## 5. Risk Manager (3 min)
- Open **Grafana dashboard**:
  - VaR, Expected Shortfall, Latency metrics.
- Trigger a **stress scenario**:
  ```bash
  python backend/ops/scenario_generator.py --shock flash_crash
  ```
- Show **kill switch** activating if drawdown > threshold.

---

## 6. Router & Broker Adapters (2–3 min)
- Show how orders are routed via `router.py`.
- Display logs from `paper_broker.py` → simulating fills.
- Show reconciliation via `reconciler.py` and `ledger.py`.

---

## 7. Dashboards (5 min)
- Walk through frontend:
  - **PnL Dashboard** → daily PnL, attribution by strategy/region.
  - **Options Chain** → Greeks, Vol Surface, F&O toggle.
  - **Risk Dashboard** → VaR, ES, stress scenarios.
  - **Terminal** view → candlestick chart, orderbook, trade log.

---

## 8. Advanced Features (optional 3–5 min)
- **AI integration**: `sentiment_ai.py` analyzing news → signals.
- **Scenario workers**: Monte Carlo, liquidity spiral, contagion graph.
- **Explainable trades**: `explainable_trades.py` shows why a trade was made.
- **Voice commands** (if enabled): “flatten positions” → routes through policy engine.

---

## 9. Wrap Up (2 min)
- “This platform is essentially a **mini hedge fund in a box**:
  - Multiple alpha models
  - Real-time OMS + risk
  - Rich dashboards
  - Extensible to live brokers (IBKR, Zerodha).”
- “It’s designed to scale from **paper trading → production** with minimal changes.”

---

✅ **End of Demo** — total ~20–25 minutes.  
