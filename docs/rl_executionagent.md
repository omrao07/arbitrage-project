# 🤖 RL Execution Agent

Design + governance notes for the **Reinforcement Learning–driven execution agent** in the Arbitrage Alpha platform.  
The agent learns **optimal trade scheduling, order slicing, and venue selection** under real-time market constraints.

---

## 1. Purpose

- Improve **execution quality** vs static VWAP/TWAP.  
- Balance **market impact, slippage, latency, and fill risk**.  
- Adapt to changing **liquidity regimes** (normal, stressed, fragmented).  
- Act as a **co-pilot**: bounded by hard risk limits, supervised by the Risk Manager.

---

## 2. Architecture

### Components

- **State features**:
  - Order book depth (L1–L5 snapshots).
  - Trade tape (prints, aggressor flags).
  - Market volatility + spreads.
  - Agent’s remaining inventory & time-to-horizon.

- **Action space**:
  - Slice size (% of remaining order).
  - Aggressiveness (passive quote vs crossing spread).
  - Venue choice (if multi-exchange).
  - Delay / skip action (wait for better liquidity).

- **Reward function**:
  - Negative slippage vs benchmark (VWAP / arrival price).
  - Penalize large market impact.
  - Penalize unfilled remainder at horizon.
  - Small bonus for low variance in execution.

- **Training loop**:
  - Simulated fills in **market replay environment**.
  - Policy gradient or DQN (discrete bins).
  - Risk shaping (clip rewards beyond guardrails).

- **Deployment**:
  - Policy model exported (`rl_exec_policy.pt`).  
  - Inference engine in `backend/engine/execution_agent.py`.  
  - Logs streamed to Redis channel `CHAN_EXEC_AGENT`.

---

## 3. Guardrails & Risk Limits

- **Max slice size**: ≤ 5% ADV or ≤ 20% of remaining inventory.  
- **Latency cap**: must act within 100 ms of state update.  
- **Kill-switch**: global stop if P&L drawdown > X% intraday.  
- **Fallback**: revert to VWAP schedule if model confidence < 0.6 or drift detected.  

---

## 4. Monitoring Metrics

| Category         | Metric                              | Threshold / Alert |
|------------------|-------------------------------------|------------------|
| **Slippage**     | Avg exec vs arrival price           | > 10 bps → alert |
| **Impact**       | % of trades moving mid by > 1 tick  | > 5% → alert     |
| **Fill rate**    | % of target size executed           | < 95% → review   |
| **Latency**      | Decision → child order submission   | > 100ms → alert  |
| **Drift**        | KL-div between current policy & ref | > 0.15 → retrain |
| **Fallback rate**| % orders falling back to VWAP       | > 20% → review   |

---

## 5. Human Oversight

- **Live console**: Execution tape with agent actions, confidence scores.  
- **Risk review**: Daily slippage attribution vs static VWAP baseline.  
- **Shadow mode**: New policies run in **paper-trading shadow** before promotion.  
- **Approval**: Only PM or risk officer may switch policy to `live`.

---

## 6. Research Notes

- Start with **discrete action bins** → stable, interpretable.  
- Incorporate **market regime classification** as context (calm vs stressed).  
- Try **offline RL** on historical fill datasets.  
- Explore **multi-agent setups**:
  - Agent 1 = US equities, Agent 2 = FX, etc.  
  - Negotiation loop for cross-asset hedges.

---

## 7. Dashboard Spec

- **Execution heatmap**: Slippage vs time-of-day.  
- **Policy confidence histogram**.  
- **Fallback tape**: VWAP vs RL side-by-side.  
- **Latency tracker**: decision → order send.  
- **Reward drift**: reward distribution over time.

---

## 8. Future Enhancements

- Dynamic **limit-order placement** with micro-alpha signals.  
- **Adversarial stress testing** (flash crash, illiquidity).  
- Add **self-play** between multiple execution agents.  
- Integrate **quantum optimizer** for multi-venue scheduling (experimental).  

---

## 9. Contacts

- **RL Research Lead**: [Name / Slack handle]  
- **Execution Infra**: [Name / Slack handle]  
- **Risk Oversight**: [Name / Slack handle]  

---

_Last updated: 2025-09-07_