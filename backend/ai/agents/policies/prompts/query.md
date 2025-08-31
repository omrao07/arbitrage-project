# Query Agent

The **Query Agent** is the information retrieval and Q&A engine of the Bolt Hedge Fund Platform.  
It acts as the **search layer** across structured (databases, feature store) and unstructured (news, alt-data) sources, enabling **fast, accurate, explainable answers** to natural-language and system queries.

---

## 🎯 Purpose
- Serve as the **Bolt “search engine”** for traders, analysts, and other agents.  
- Translate natural language into structured queries.  
- Retrieve **facts, factors, metrics, or scenarios** from the knowledge graph, feature store, and ledger.  
- Provide **low-latency insights** that feed directly into Copilot, Analyst, or Execution agents.  

---

## 🧩 Capabilities
- **Knowledge Graph Queries**  
  - Interfaces with `knowledge_graph.py`.  
  - Resolves entity relationships (company → sector → macro exposure).  
- **Feature Store Lookups**  
  - Fetches factor ICs, alpha scores, volatility stats (`feature_store.py`).  
- **Ledger / Trade History Retrieval**  
  - Pulls trade events from `ledger.py` + `trade_log_panel.tsx`.  
- **Natural-Language Search**  
  - Converts “plain English” to structured requests.  
  - E.g., *“Show me PnL attribution for Reliance last week”*.  
- **Cross-Agent Queries**  
  - Submits downstream requests via `dispatcher.py` to Analyst/Explainer.  

---

## ⚙️ Architecture
- **Agents**
  - `query_agent.py` — base logic for retrieval.
  - `query_copilot.py` — extended conversational copilot.
- **Data Sources**
  - Feature store (`feature_store.py`).
  - Knowledge graph (`knowledge_graph.py`).
  - Alt-data feeds (`altdata_ingestor.py`).
  - Ledger/trade store (`ledger.py`).
- **Middleware**
  - `dispatcher.py` → routes queries to correct agent.
  - `tracing.py` → spans each query for timing.
  - `safety.py` → validates permissions/scopes.

---

## 📊 Inputs
- Natural language prompts (via UI `AIChat.tsx`, `Chat.tsx`).  
- Structured query payloads from Dispatcher.  
- Session metadata (`sessions.py`).

## 📈 Outputs
- JSON answers, e.g.:  
  ```json
  {
    "query": "What drove Reliance Industries’ performance last week?",
    "answer": "Reliance gained +4.2% driven primarily by refining margins (60%) and Jio subscriber growth (30%). Market-wide bullish sentiment added 10%."
  }