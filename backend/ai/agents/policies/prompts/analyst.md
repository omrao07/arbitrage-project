# Analyst Agent

The **Analyst Agent** is the core research and market–intelligence layer inside the Bolt Hedge Fund Platform.  
It functions as a Bloomberg–style analyst that continuously digests raw feeds, processes alternative data, and generates structured insights.

---

## 🔎 Purpose
- Transform raw **market + alternative data** into actionable insights.
- Act as the **bridge** between unstructured news/sentiment and structured strategies.
- Support **both discretionary traders and autonomous strategies**.

---

## 🧩 Capabilities
- **Market Research**
  - Real-time screening of equities, FX, crypto, futures.
  - Factor analysis: momentum, value, carry, sentiment.
- **Alt-Data Processing**
  - News ingestion (Yahoo Finance, MoneyControl, Reuters RSS).
  - Satellite imagery, shipping traffic, credit-card spend, social chatter.
- **Sentiment Radar**
  - Transformer-based classification of bullish/bearish/neutral tone.
  - Cross-language support.
- **Explainer Mode**
  - “Explain this trade/idea” → returns rationale, features, and risks.
- **Scenario Builder**
  - Stress-tests impact of events (e.g., Fed hike, oil shock, INR depreciation).

---

## ⚙️ Architecture
- **Agents Layer**
  - `analyst_agent.py` — base class + pipelines.
  - `insight_agent.py` — extracts and structures insights.
  - `query_agent.py` — answers ad-hoc questions from the UI.
  - `explainer_agent.py` — generates explainability narratives.
- **Pipelines**
  - Ingestors (`altdata_ingestor.py`, `news_yahoo.py`, `news_moneycontrol.py`).
  - Feature store (`feature_store.py`).
  - Knowledge graph (`knowledge_graph.py`).
- **Storage**
  - Insights indexed in vector DB for retrieval.
  - Metadata logged into `ledger.py`.

---

## 📊 Inputs
- Market data (via adapters: IBKR, Zerodha, Binance, PaperBroker).
- Alt-data feeds (RSS, APIs, scrapers).
- Historical datasets (backtests, research mode).

## 📈 Outputs
- Insight JSON payloads: `{symbol, factors, sentiment, rationale}`.
- UI components:
  - `SentimentRadar.tsx`
  - `ExplainTrade.tsx`
  - `AnalystPanel.tsx`
- Feeds dispatcher (`dispatcher.py`) for downstream strategy agents.

---

## 🛡️ Risk & Safeguards
- CSRF + session validation (`sessions.py`).
- Rate-limiting for API calls (avoid overloading providers).
- Bias awareness: reports include **confidence scores** + caveats.
- Compliance hooks: MiFID reporter, SEBI OTR, CFTC Part 43.

---

## 🧪 Example Flow
1. News arrives via `news_yahoo.py`.
2. Analyst Agent parses → extracts tickers/entities.
3. Sentiment AI scores article.
4. Result passed to:
   - **Trader UI** → sentiment heatmap / ticker panel.
   - **Strategy Allocator** → optional signal enrichment.

---

## 🗂️ Demo Queries
- *“What’s the impact of OPEC cuts on Reliance Industries?”*  
- *“Explain the trade in AAPL on 2025-08-20.”*  
- *“Show me sentiment for NIFTY50 last 24h.”*

---

## 🚀 Roadmap
- Expand ESG factor integration (`esg_factor.py`).
- Integrate multilingual news feeds (Hindi, Japanese, Chinese).
- Add GPT-based **“Analyst Copilot”** for portfolio managers.
- Link with **Scenario Sandbox** to simulate policy shocks.

---

## ✨ Why It Matters
The Analyst Agent is what makes Bolt feel like a **Bloomberg Terminal + Hedge Fund Research Desk in one**.  
It’s the *narrative engine* that makes your trading platform not just execute, but **explain and justify** decisions.