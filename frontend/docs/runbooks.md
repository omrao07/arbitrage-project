# Hedge Fund Platform – Runbooks

This document contains **standard operating procedures (SOPs)** for running the hedge fund platform.  
It covers **startup, monitoring, recovery, and shutdown**.  

---

## 1. Environment Setup
- Ensure `.env` is populated with correct values:
  - `REDIS_HOST`, `REDIS_PORT`
  - Broker API keys (`IBKR_KEY`, `ZERODHA_KEY`, `BINANCE_KEY`…)
  - Data API keys (Yahoo, Moneycontrol, News APIs)
- Run database migrations (if using Postgres/Timescale).
- Start Redis:
  ```bash
  redis-server