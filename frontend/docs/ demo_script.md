# Hedge Fund Platform Demo Script

This script walks through a **10-minute demo** of the hedge fund system.  
The goal: show **end-to-end capability** (data → signal → execution → risk → dashboard) in a **clear, bulletproof flow**.

---

## 1. Introduction (1 min)
- “This is my hedge fund simulation and trading platform.”  
- Built to replicate the **end-to-end infrastructure of a global fund**:  
  - **Data ingestion** (market, news, alt-data)  
  - **70+ strategies** (alpha + arbitrage + statistical + macro)  
  - **Execution layer** (OMS, SOR, broker adapters)  
  - **Risk management** (VaR, stress tests, liquidity spirals, contagion graphs)  
  - **Dashboards** for PnL, Greeks, ESG, and literacy mode.  

---

## 2. Data & News Ingestion (1 min)
- Show **live data feeds**: Yahoo Finance + Moneycontrol (for India).  
- “Here you can see real-time market ticks flowing into Redis streams.”  
- Show **news ingestion**: RSS parsing with sentiment analysis.  
- Key point: *Market + news → unified signal bus*.

---

## 3. Strategy Layer (2 min)
- Open `strategy_base.py` → highlight one **example strategy** (EMA crossover / Buy-the-dip).  
- Then show **Strategy Builder UI** (`strategy-builder.tsx`):
  - Build rule: *IF RSI < 30 AND Sentiment > 0 → BUY*.  
  - Export JSON → feed into backend.  
- Point: “Non-coders can build hedge-fund grade strategies visually.”

---

## 4. Execution Layer (1.5 min)
- Show order flow:  
  - **Strategy → Risk Manager → OMS → Broker Adapter → Fill → PnL**.  
- Demo:
  1. Trigger a toy order from Strategy Builder.  
  2. Show order appear in `orders.incoming` stream.  
  3. Risk checks pass, OMS routes it.  
  4. Broker adapter (paper mode or IBKR/Zerodha in live).  
- Show **trade history UI** (`trade-history.tsx`).

---

## 5. Risk Management (2 min)
- Open **Risk Dashboard**:  
  - VaR, drawdown, volatility, Greeks.  
- Switch to **Stress Sandbox** (`stress-sandbox.tsx`):
  - Apply preset: *2008 Crisis* → watch portfolio drop.  
  - Apply *COVID Crash* → see liquidity spiral & PnL shocks.  
- Show **Contagion Graph** (networks of shocks across banks/sovereigns).  
- Key point: *Real hedge funds obsess over risk → I recreated that infrastructure*.

---

## 6. Advanced Features (1.5 min)
- **Dark Pool X-Ray** (`dark-pool-xray.tsx`):  
  - Show off-exchange volumes, venue shares, price improvement.  
- **Options Dashboard**:  
  - Option chain, Greeks, vol surface.  
- **Alternative data** modules: satellite lights, shipping traffic, credit card spend.  
- **AI Sentiment Engine**: highlights how news moves signals.

---

## 7. Dashboards & Literacy Mode (1 min)
- Show **PnL Attribution** dashboard (per strategy / per region).  
- Show **Literacy Mode**:  
  - Click “Explain risk” → AI explainer breaks down what VaR means in plain English.  
  - Designed for teaching finance in communities.  

---

## 8. Closing (30s)
- “This isn’t a toy bot — it’s a **full hedge fund operating system**.  
  - Live data feeds  
  - Real broker connectivity  
  - Strategy builder  
  - Risk + Stress testing  
  - Dashboards + explainers  
- I built this as a **student project**, but it mirrors **how real hedge funds operate**.”

---

# Tips
- **Golden Path**: Demo exactly the sequence above → don’t click random tabs.  
- **Keep explanations short** — focus on *visual wow*.  
- Have a **backup demo mode** (pre-recorded PnL curve, contagion graph screenshots) in case live feed fails.  