# 🌍 Macro LLM Monitor

A monitoring & governance document for **Large Language Model (LLM)–driven macro signals** in our trading & research stack.  
This serves as the control panel spec + playbook for oversight, auditing, and iteration.

---

## 1. Purpose

- Track LLM-generated **macro commentary, forecasts, and event interpretations**.  
- Ensure **transparency, risk limits, and human-in-the-loop validation**.  
- Detect and mitigate **hallucinations, bias, and drift** in model outputs.  
- Provide **real-time dashboards** to traders and risk managers.  

---

## 2. Signal Sources

- **Economic releases**: CPI, NFP, PMIs, Fed/ECB/BOJ statements.  
- **Policy chatter**: FOMC minutes, fiscal policy, geopolitical speeches.  
- **Media / sentiment**: curated RSS, Twitter/X, Bloomberg headlines.  
- **Research uploads**: PDFs, analyst notes, transcripts.  

LLM pipelines summarize, classify, and produce structured `macro_signal.json` objects.

---

## 3. Output Schema

```json
{
  "timestamp": "2025-09-07T12:00:00Z",
  "source": "FOMC statement",
  "theme": "monetary_policy",
  "signal": "hawkish",
  "confidence": 0.82,
  "impact": {
    "asset": "US 2Y Treasury",
    "direction": "yields_up",
    "horizon": "0-3d"
  },
  "explanation": "Fed emphasized inflation persistence; reduced rate-cut probability."
}