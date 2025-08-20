# Hedge Fund-Style Arbitrage Platform — Runbook

## 1. Overview
This platform implements **72+ arbitrage strategies** across equities, commodities, FX, crypto, and derivatives, divided into:
- **Alpha strategies** (pure alpha, news & sentiment, policy signals, etc.)
- **Diversified strategies** (macro, cross-asset, statistical arbitrage)

It supports:
- Real-time market data ingestion
- Strategy signal generation
- Risk management & allocation
- Order routing to execution brokers (live or paper)
- Recording & replay for backtesting
- Dashboard integration with Bolt

---

## 2. Prerequisites

### System Requirements
- Python 3.10+
- VS Code (recommended for development)
- Redis (caching & pub/sub)
- ClickHouse (trade/data storage)
- GitHub account for version control
- Bolt account for deployment/display

### API Access (Optional for Live)
You’ll need API keys (stored in `.env`) for:
- Binance (crypto)
- Alpaca (stocks, paper trading)
- IBKR / Zerodha / OANDA (if going live)
- News / sentiment providers (if enabled)

**Note:** Paper trading can be done without paid APIs using simulated feeds.

---

## 3. Repository Structure

```plaintext
backend/
  ├── config/             # YAML configs for regions, strategy catalog
  ├── core/               # Aggregator, allocator, risk manager, OMS, etc.
  ├── data/               # Data streams, adapters, recorders, replayers
  ├── execution/          # Broker adapters, execution engine
  ├── risk/               # Risk modules
  ├── strategies/
  │     ├── alpha/        # 32 hedge-fund-style alpha strategies
  │     └── diversified/  # 40+ diversified strategies
  ├── storage/            # ClickHouse integration
frontend/                 # Bolt-ready dashboard
.env                      # API keys and secrets