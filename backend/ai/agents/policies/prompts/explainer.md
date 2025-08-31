# Explainer Agent

The **Explainer Agent** is the explainability layer of the Bolt Hedge Fund Platform.  
It provides **transparent, human-readable justifications** for signals, strategies, and trades — turning raw algorithmic actions into narratives that portfolio managers, auditors, and regulators can trust.

---

## 🎯 Purpose
- Bridge the **“black-box” gap** between advanced models and human oversight.  
- Ensure every trade has a **traceable rationale** and can be explained in plain English.  
- Generate narratives that enhance **compliance, trust, and investor confidence**.  

---

## 🧩 Capabilities
- **Trade Explainability**
  - Consumes trade events (`trade_explainability.py`, `pnl_xray.py`).
  - Explains why a trade was triggered (factor signals, news drivers, market context).
- **Signal Transparency**
  - Works with `strategy_base.py` → annotates “why strategy fired”.
  - Surfaces factor weights, alpha sources, sentiment inputs.
- **PnL & Attribution**
  - Hooks into `pnl_attribution.py` and `risk_explainer.py`.
  - Explains *which drivers* contributed to gains/losses.
- **Narrative Generation**
  - Translates metrics into **dashboard-ready text** for:
    - `ExplainTrade.tsx`
    - `VisualTradeHistory.tsx`
    - `AnalystPanel.tsx`
- **Regulatory Justification**
  - Provides plain-language logs for MiFID/SEBI/CFTC reporting.
  - Connects to `ledger.yaml` for immutable storage.

---

## ⚙️ Architecture
- **Agents**
  - `explainer_agent.py` — main reasoning orchestrator.
  - `trade_explainability.py` — trade-specific narratives.
  - `risk_explainer.py` — risk/margin/VAR explainability.
- **Inputs**
  - Trades (`execution_agent.py`).
  - Signals (`strategy_dsl.py`, `strategy_base.py`).
  - Risk metrics (`var_engine.py`, `stress_attribution.py`).
- **Outputs**
  - Explanations → `ExplainTrade.tsx` (UI).
  - Reports → `report_generator.py`, `playbooks.md`.

---

## 📊 Inputs
- Trade events (order + fill).
- Strategy signals (alpha scores, sentiment weights).
- Market context (prices, volatility, liquidity).
- Alternative data triggers (news, ESG factors, social sentiment).

## 📈 Outputs
- JSON narratives like:  
  ```json
  {
    "trade_id": "T20250829-XYZ",
    "explanation": "Buy order triggered by mean-reversion signal after 2% dip below 20d average, reinforced by positive news sentiment on Reliance Industries."
  }