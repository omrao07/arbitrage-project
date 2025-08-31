# 📑 Model Cards Guide — Hedge Fund Platform

> **Purpose:**  
> This guide standardizes how to document models (ML, statistical, rule-based) used in the hedge fund platform.  
> A **Model Card** gives context: what a model is, why it exists, how it was trained/tested, and where it should/shouldn’t be used.

---

## 1. Template

Each model card should follow this structure:

### 🔹 Model Name
- **File/Path**: `backend/models/<file>.py`
- **Category**: [Sentiment | Risk | Forecasting | Alpha | Execution | Compliance]

### 🔹 Overview
- **Short description** of what the model does.
- **Intended use** cases.
- **Not intended for** (limitations).

### 🔹 Data
- **Input data** used (feeds, datasets, altdata sources).
- **Data transformations / features**.
- **Training / testing split** (if ML-based).
- **Update frequency**.

### 🔹 Method
- **Approach**: ML algorithm / statistical method / rules.
- **Baseline comparison**: how it improves on simple heuristic.
- **Assumptions** (e.g., normality, stationarity, independence).

### 🔹 Evaluation
- **Metrics** (Accuracy, IC, IR, Sharpe, RMSE, Precision/Recall, etc.).
- **Backtest results** (PnL, hit ratio, drawdowns).
- **Stress tests** (market shock, low-liquidity regime).

### 🔹 Risks & Limitations
- Biases in data.
- Overfitting possibility.
- Regime sensitivity.
- Dependency on external data (e.g., Yahoo/Moneycontrol feed).

### 🔹 Governance
- **Owner**: who maintains this model.
- **Last updated**: YYYY-MM-DD.
- **Validation**: reviewed by risk/research team.
- **Kill switch**: conditions under which it must be disabled.

---

## 2. Example

### Model Name: `sentiment_ai.py`
- **File**: `backend/ai/sentiment_ai.py`
- **Category**: Sentiment / NLP

**Overview**  
Uses Transformer-based NLP to parse Yahoo Finance & Moneycontrol news headlines. Generates a sentiment score ∈ [-1,1] that feeds into strategies and risk overlays.

**Data**  
- Input: RSS feeds, scraped news headlines.  
- Features: TF-IDF + embeddings from `finBERT`.  
- Train/Val Split: 80/20 (finetuned on 50k financial news samples).  
- Update: real-time, re-trained monthly.

**Method**  
- Base model: BERT → finBERT.  
- Fine-tuned on labeled finance dataset.  
- Produces sentiment + confidence.

**Evaluation**  
- Accuracy: 82% on validation set.  
- Backtest: added +3% annualized return to momentum strategies.  
- Stress: performed poorly in “COVID headline shock” (overly negative).

**Risks & Limitations**  
- Biased toward US financial language.  
- Struggles with sarcasm or regional news.  
- Dependency on news availability.

**Governance**  
- Owner: Quant Research Team  
- Last updated: 2025-08-01  
- Validation: Reviewed quarterly by risk.  
- Kill switch: Disable if IC < 0.02 for >30 days.

---

## 3. Best Practices
- Always include **metrics + limitations**, not just hype.  
- Keep **last updated** current (automation via Git hooks recommended).  
- Store **all model cards** under `/docs/model_cards/`.  
- Cross-link with **risk dashboard** so governance is transparent.

---

✅ With model cards in place, your platform demonstrates **professional governance + transparency**, like a real hedge fund or quant research lab.