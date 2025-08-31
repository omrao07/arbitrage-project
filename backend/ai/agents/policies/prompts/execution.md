# Execution Agent

The **Execution Agent** is the live trading backbone of the Bolt Hedge Fund Platform.  
It is responsible for **routing, executing, and monitoring trades** across brokers, venues, and simulated environments while enforcing risk and compliance guardrails.

---

## 🎯 Purpose
- Bridge strategies and broker APIs (IBKR, Zerodha, Binance, PaperBroker).
- Translate **orders from strategies or Copilot** into safe, executable trades.
- Optimize order placement with **algos (VWAP, TWAP, POV, Adaptive VWAP)**.
- Capture and store full **trade history** for analytics and compliance.

---

## 🧩 Capabilities
- **Order Lifecycle**
  - Accepts orders from `strategy_base.py`, `copilot.md`, `dispatcher.py`.
  - Routes to OMS (`orders.py` → `broker_interface.py`).
  - Acknowledges, fills, cancels, amends.
- **Algo Execution**
  - VWAP, TWAP, POV, Adaptive VWAP (`adaptive_vwap.py`).
  - Dark pool router (`dark_pool_router.py`).
  - Batch auction participation (`batch_auction.py`).
- **Risk Enforcement**
  - Safety checks (`safety.py`).
  - Kill switch (`kill_switch.py`).
  - Margin spiral & exposure monitors.
- **Smart Routing**
  - Venue selection using `venue_cost_analyzer.py` and `venue_toxicity_score.py`.
  - Queue position awareness (`queue_position.py`).
- **PnL & Trade Logs**
  - Every execution logs to `ledger.py` and `trade_log_panel.tsx`.
  - Linked to `pnl_xray.py` and `pnl_attribution.py`.

---

## ⚙️ Architecture
- **Agents**
  - `rl_execution_agent.py` — reinforcement-learning execution.
  - `execution_agent.py` — deterministic baseline execution logic.
- **Core Components**
  - `broker_base.py` — common adapter spec.
  - `ibkr.py`, `zerodha.py`, `paperbroker.py` — broker adapters.
  - `order_store.py` — persistent order ledger.
  - `reconciler.py` — reconciles OMS vs broker state.
- **Risk/Safety**
  - `safety.py`, `risk_manager.py`, `kill_switch.py`.
- **UI Integration**
  - `QuickOrderPad.tsx`
  - `OrderBookPanel.tsx`
  - `PositionStrip.tsx`

---

## 📊 Inputs
- Orders from:
  - `Strategy` subclasses (e.g., BuyTheDip, ETF NAV Arb).
  - `Copilot Agent` natural-language commands.
  - `Dispatcher` tasks.
- Market data feeds (prices, liquidity, depth).
- Safety/risk configs (`policy.yaml`, `risk_metrics.py`).

## 📈 Outputs
- Broker acknowledgments and fills.
- Trade history events → `ledger.py` + `TradeLogPanel.tsx`.
- Risk metrics + attribution for dashboards.

---

## 🛡️ Risk & Safeguards
- **SafetyManager** (`safety.py`) ensures:
  - Max notional / leverage / qty.
  - Blocklist instruments.
  - Cooldown throttling.
- **Kill Switch**
  - Circuit breaker halts all trading on anomaly.
- **Compliance**
  - MiFID (`mifid_reporter.py`), SEBI OTR (`sebi_otr.py`), CFTC Part 43 (`cftc_part43.py`).
- **Audit Trail**
  - Full trade + risk log in `ledger.yaml`.

---

## 🧪 Example Flow
1. Strategy emits: *“BUY AAPL 100 @ MKT”*.
2. Order arrives in `execution_agent.py`.
3. Passes through `safety.py` + risk checks.
4. Routed to `ibkr.py` or `zerodha.py` adapter.
5. Execution algo chosen (VWAP).
6. Broker confirms fill → stored in `order_store.py` + UI updated.
7. Ledger + dashboards reflect trade in real time.

---

## 🚀 Roadmap
- Add **smart slicing** using ML volatility forecasts.
- Plug into **voice interface** for manual override.
- Integrate **latency-aware optimizers** for HFT-like routing.
- Expand to **DeFi execution** (UniswapAdapter).

---

## ✨ Why It Matters
The Execution Agent is the **muscle of Bolt** — it turns strategy intent into live trades while keeping you safe.  
It’s what makes your platform not just a research environment but a **full hedge fund OMS/EMS** that rivals Bloomberg + FlexTrade.