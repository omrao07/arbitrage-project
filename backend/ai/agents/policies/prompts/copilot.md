# Copilot Agent

The **Copilot Agent** is the interactive assistant layer inside the Bolt Hedge Fund Platform.  
It behaves like an **AI research partner + trade assistant**, enabling natural language interaction with the platform’s data, strategies, and execution stack.

---

## 🎯 Purpose
- Provide a **chat-first interface** for Bolt.
- Translate **natural language → structured queries/commands**.
- Act as an orchestrator between **Analyst, Execution, and Insight agents**.
- Lower the barrier to access by making Bolt feel like a **Bloomberg Terminal with ChatGPT inside**.

---

## 🧩 Capabilities
- **Query Routing**
  - Directs questions to the right agent (Analyst, Execution, Swarm, etc.).
  - Uses `dispatcher.py` to submit/track tasks.
- **Knowledge Retrieval**
  - Connects to `knowledge_graph.py` and `feature_store.py`.
  - Surfaces historical insights, factor IC, alt-data anomalies.
- **Trade Copilot**
  - Execute trades via `execution_agent.py` (paper/live).
  - Supports one-click strategies: VWAP, TWAP, POV, Adaptive VWAP.
- **Explainer Mode**
  - Generates reasoning behind signals (`explainer_agent.py`).
- **Scenario Copilot**
  - Hooks into `scenario_generator.py` and `crisis_theatre.py` to stress-test portfolios.
- **Research Partner**
  - Directly query notebooks: `factors_ic.ipynb`, `optimizer_playground.ipynb`.

---

## ⚙️ Architecture
- **Core Agents**
  - `query_copilot.py` — main orchestration logic.
  - `conversation.py` — manages dialogue context.
  - `memory.py` — persistent profile + facts for personalization.
- **Dispatcher**
  - Bridges `dispatcher.py` with natural-language requests.
- **Safety**
  - Guardrails enforced via `safety.py` before executing trades.
- **Tracing**
  - `tracing.py` spans wrap all calls for observability.

---

## 📊 Inputs
- Natural language prompts from UI (`AIChat.tsx`, `Chat.tsx`).
- Session metadata (`sessions.py`).
- Analyst/Execution agent results.

## 📈 Outputs
- Natural-language responses.
- Structured JSON commands (trade, query, scenario).
- UI component feeds:
  - `ExplainTrade.tsx`
  - `SentimentRadar.tsx`
  - `TradeLogPanel.tsx`

---

## 🛡️ Risk & Safeguards
- **Execution Safeguards**
  - All trades pass through `safety.py`.
  - Circuit-breaker: max notional, leverage, cooldown.
- **Compliance**
  - Hooks to MiFID, SEBI OTR, CFTC reporting.
- **Explainability**
  - Each trade triggered by Copilot also logs rationale via `trade_explainability.py`.

---

## 🧪 Example Queries
- *“Show me today’s sentiment on Reliance Industries and execute a VWAP buy order for 10k INR.”*
- *“Explain why NIFTY fell yesterday using news + factor analysis.”*
- *“Simulate a Fed hike scenario on my portfolio.”*
- *“What are my PnL drivers for this week?”*

---

## 🚀 Roadmap
- Add **voice interface** (`voice_interface.py`) for spoken queries.
- Integrate **multimodal analysis** (charts + text explanations together).
- Enable **multi-agent “swarm copilot”** (analyst + execution + sentiment agents collaborating).
- Fine-tuned **financial LLM** backend (GGUF/OpenAI local hybrid).

---

## ✨ Why It Matters
The Copilot Agent is the **human-facing intelligence layer** of Bolt.  
It transforms the platform from just a **hedge fund execution stack** into an **interactive AI-powered Bloomberg Terminal** that **explains, executes, and explores** with you.