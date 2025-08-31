
# RL Execution Agent — Design, Safety, and Runbook

> Learning-based order execution for minimizing implementation shortfall under market impact and risk constraints.

---

## 1) Goals & Scope

**Problem.** We want an agent that breaks parent orders into child orders (slice size, price, timing, and routing) to minimize **implementation shortfall (IS)** vs a benchmark (arrival, VWAP, or AC trajectory) while respecting **risk, impact, and compliance** constraints.

**Goals**
- Reduce IS vs. baselines (TWAP/VWAP/Almgren–Chriss) by 5–20% depending on regime/liquidity.
- Keep tail risk controlled: p95/p99 IS and **CVaR** within budget.
- Respect hard limits: participation caps, price bands, notional/inventory, venue restrictions.
- Robust to regime shifts; degrade gracefully to rule-based policies.

**Out of scope (v1)**
- Smart order routing across fragmented lit/dark venues with complex microstructure alpha.
- Position-taking or alpha generation; execution only, flat end-of-horizon inventory target.

---

## 2) MDP Formulation

Let a parent order \(Q\) shares be executed over horizon \(T\) with discrete decision times \(t=0,\dots,N-1\).

### 2.1 State \(s_t\)
Feature vector with stable scaling (z-score/robust). Suggested groups:

- **Market microstructure**
  - Top-k LOB levels: \((p^{bid}_i, q^{bid}_i, p^{ask}_i, q^{ask}_i)_{i=1..k}\)
  - Recent trades: last m prints (price, size, side); imbalance; queue position estimate
  - Short-term realized vol, spread, depth, order arrival rates
- **Order context**
  - Remaining inventory \(x_t\), time remaining \(\tau_t\), target trajectory deviance (vs AC/TWAP)
  - Participation so far, venue fill rates, cancel/replace counts
- **Risk & regime**
  - Regime embedding (from **GNN correlations** clusters), intraday seasonality bucket
  - News/sentiment flags (from **sentiment worker**), volatility regime
- **Constraints snapshot**
  - Hard caps: max child size, participation, price band, min resting time, max cancels

> Normalize/clip features to bounded ranges to stabilize training. Persist feature schema in a **Feature Store**.

### 2.2 Action \(a_t\)
Combination of **size, price, timing, and type**. Two representations:

- **Discrete param head** (safe default):
  - `size_bucket` ∈ {0.25%, 0.5%, 1%, 2% of ADV or of remaining}
  - `aggression` ∈ {post-passive @ -2, -1, 0 ticks; IOC @ +0, +1, +2 ticks}
  - `wait` ∈ {0s, 2s, 5s, 10s}
  - `type` ∈ {limit, marketable limit, IOC}
- **Continuous** (later): \(q_t \in [0, q_{max}]\), price offset \(\delta p \in [-b, +b]\).

**Routing**: optional categorical over venues; v1 can keep fixed venue priority.

### 2.3 Reward \(r_t\)
We minimize IS and penalize risk/impact/violations.

\[
r_t = -\Delta \text{IS}_t - \lambda_{risk}\,x_t^2\,\sigma_t\,\Delta t - \lambda_{impact}\,c_{tmp}(v_t) - \lambda_{perm}\,c_{perm}(q_t) - \sum_j \lambda_{viol,j}\,\mathbb{1}\{\text{violation}_j\}
\]

- **IS**: difference between execution price and benchmark (arrival or VWAP slice).
- **Temporary/permanent impact**: e.g., AC costs with \(\eta, \gamma\) (see our **Almgren–Chriss** notebook).
- **Risk**: inventory variance proxy.
- **Constraint violations**: participation, price band, cancels, throttles.

Terminal bonus: \(r_N = -\text{IS}_{total} - \alpha \cdot \text{slippage\_variance}\).

### 2.4 Transition
Event-driven environment using **historical L2/L3 replays** or a **simulator** (below).

---

## 3) Environment & Simulator

Two backends:

1) **Market Replay** (preferred for OPE/validation)  
   - Play back recorded L2/prints.  
   - Deterministic fills via simple queue model + crossed/marketable logic.  
   - Latency model (submit→ack, market data delay).  
   - Fees/rebates, tick size, price bands.

2) **Stochastic LOB Simulator** (training augmentation)  
   - Order arrivals as Hawkes/Poisson; cancellation rates; depth distributions.  
   - Impact model: **temporary** \(\eta v\) and **permanent** \(\gamma q\).  
   - Repricing events; spread widening under stress.

**Fill model** (simplified): FIFO at each price; queued volume ahead + arrivals; probability of fill within `wait` horizon.

**Config** ties to earlier work:
- Use `market_impact_almgren.ipynb` parameters (\(\eta,\gamma,\sigma\)).
- Regime sampling from **GNN clusters** (to vary liquidity/vol).

---

## 4) Algorithms

Start with **safe, sample-efficient** approaches:

- **Behavior Cloning (BC)** warm-start from historical child order logs.
- **Conservative Q-Learning (CQL)** for **offline RL**, avoiding overestimation.
- **Dueling Double DQN** for discrete policy online fine-tuning (small action grid).
- **Risk-aware objectives**: CVaR-DQN or distributional RL (Quantile Regression DQN) to tame tails.
- **Policy constraints** via **Action Masking** (see §6).

Optional:
- **BCQ/IQL** if dataset is purely offline and diverse.
- **PPO** with KL clamps if moving to continuous actions later.

---

## 5) Training Pipeline

```
raw logs ─► ETL ─► feature store ─► (BC pretrain) ─► offline RL (CQL) ─►
shadow online (read-only decisions) ─► A/B canary ─► full prod
```

- **Replay Buffer**: prioritized by TD error; separate buffer for rare regimes (thin markets).
- **Curriculum**: start with small orders and liquid names; expand to harder scenarios.
- **Hyperparams (starting)**: 
  - Dueling DDQN: lr=2e-4, γ=0.995, target_sync=2s, batch=256, replay=200k, ε: 0.1→0.01.
  - CQL α=1.0, temperature=1.0; L2 weight 1e-4.
- **Normalization**: online feature normalizer with exponential decay; locked per regime bucket.

---

## 6) Safety & Compliance

**Hard action masks** at inference:
- Max participation (e.g., 10% rolling window).  
- Price band: limit orders within ±b ticks of reference.  
- Child size ≤ q_max, cancel rate throttle.  
- Kill-switch if latency/venue down or if remaining x_t breaches schedule drift.

**Fallbacks**
- Degrade to **Almgren–Chriss** schedule (our existing implementation) or TWAP.
- Freeze to passive posting only in extreme volatility or news flags.

**Safe exploration** (if online learning enabled)
- Epsilon only on **non-critical** axes (e.g., wait time), and gated by guard.  
- Shadow mode collects counterfactuals; A/B canary <10% flow.

**Auditability**
- Deterministic seeding, full action trace, features snapshot, policy hash per decision.  
- Write payloads to `STREAM_RESEARCH` and store parquet logs for replay.

---

## 7) Off-Policy Evaluation (OPE)

- **DM (Direct Method)** using learned value model on held-out replays.  
- **IPS/WIS** where action propensities are known (from historical policy).  
- **Doubly Robust (DR)** estimator for bias-variance tradeoff.  
- Report **CIs** via bootstrapping across days/names.

---

## 8) Metrics & SLOs

**Per-order metrics**
- IS vs arrival / VWAP / AC benchmark (mean, p50/p95/p99).  
- Fill rate, time-to-fill, partials, cancels per child, venue hit rates.  
- Tail risk: **CVaR@95** of per-order IS.  
- Impact proxies: post-trade drift, footprint.

**Prometheus (suggested)**
- `exec_is_bps{policy}` (histogram)  
- `exec_tail_cvar_bps{policy}` (gauge)  
- `exec_child_rate{type}` (counter)  
- `exec_violations_total{kind}` (counter)  
- `exec_decision_latency_ms` (histogram)

**SLOs**
- p95 decision latency < 50ms.  
- Violations per 1k decisions < 1.  
- p95 IS not worse than AC baseline by more than 2 bps during canary.

---

## 9) Interfaces & Deployment

### 9.1 Request/Response over Redis Streams
- **Inbound**: `STREAM_ORDERS` (parent order envelope).  
- **Outbound**: `STREAM_FILLS` (child orders placed / fills), `STREAM_RESEARCH` (traces).  
- **UI Pub/Sub**: `CHAN_ANALYST` for overlays and diagnostics.

**Parent order payload (example)**
```json
{
  "order_id": "ABC-123",
  "symbol": "AAPL",
  "side": "sell",
  "qty": 250000,
  "arrival_px": 187.53,
  "start_ts": 1693564800,
  "end_ts": 1693568400,
  "constraints": {"participation": 0.1, "price_band_bps": 5, "child_max": 15000}
}
```

**Child order command**
```json
{"parent_id":"ABC-123","type":"limit","qty":8000,"px":187.48,"ttl_ms":3000,"venue":"XNAS"}
```

### 9.2 Services
- `exec-agent` (policy server, gRPC/WS/Redis).  
- `execution-simulator` (training + OPE).  
- `policy-trainer` (offline/online).  
- `risk-guard` (masks + kill-switch).

### 9.3 Config (YAML sketch)
```yaml
policy:
  type: "dueling_ddqn"
  checkpoint: "s3://exec/policies/ddqn_v3.pt"
  action_set:
    size_bp_of_remaining: [25, 50, 100, 200]
    aggression_ticks: [-2, -1, 0, +1, +2]
    wait_seconds: [0, 2, 5, 10]
risk:
  participation_cap: 0.10
  price_band_ticks: 5
  child_max: 15000
  fallback: "almgren_chiss"
```

---

## 10) Pseudocode (Environment + Agent)

```python
# Environment step
def step(action):
    # apply action mask already validated by risk-guard
    place_child_order(action)
    fills, book_updates = advance_sim_or_replay(dt=action.wait)
    pnl_delta, is_delta = compute_costs(fills, benchmark)
    impact_cost = eta * v + gamma * q  # AC-style
    risk_pen = lam_risk * (inv**2) * sigma * dt
    reward = -is_delta - impact_cost - risk_pen - penalties
    next_state = featurize(book_updates, inventory, time_left, constraints)
    done = time_left <= 0 or inventory <= 0
    return next_state, reward, done, info
```

```python
# Dueling Double DQN (discrete action space)
for step in range(T):
    a = policy.act(s, epsilon, mask=risk_guard.action_mask(s))
    s2, r, done, info = env.step(a)
    replay.add(s, a, r, s2, done)
    policy.learn(replay.sample())
    s = s2
    if done: break
```

---

## 11) Evaluation Protocol

- **Backtest** on last 12 months across liquidity buckets, both buy & sell.  
- **Stress** with scenario worker (vol-up ×2, crash -10%, spreads widen).  
- **Shadow live** for ≥ 2 weeks: take decisions but do not place; compute **counterfactual IS**.  
- **A/B canary**: 10% flow vs AC baseline; gate on SLOs.

Stat tests: paired t-test or Wilcoxon on per-order IS; plot uplift distribution.

---

## 12) Monitoring & Runbook

- Dashboards: IS distributions, tail metrics, violation counts, decision latency, policy version.  
- Alerts (PagerDuty): violation spike, decision latency p95 > SLO, IS degradation vs baseline > 3 bps sustained.

**Incident steps**
1. Flip **feature flag** to fallback (AC or TWAP).  
2. Quarantine policy version; roll back to last green.  
3. Inspect traces (features, action, Q-values, mask).  
4. Reproduce with simulator using stored replays.  
5. File RCA; add unit test in regression suite.

---

## 13) Data & Storage

- **Parquet** logs: per-decision features, action, mask, Q-values, rewards.  
- **S3**: policy checkpoints, training datasets, evaluation reports.  
- **Postgres**: per-order summary table: IS stats, constraints usage, incidents.

---

## 14) Security & Compliance

- AuthN to exchanges/brokers via vault-managed creds.  
- Guard markets where short-sale/locate applies.  
- Capture approvals for policy updates; **4-eyes** deployment for production.  
- Full audit trail retained ≥ 7 years (jurisdiction dependent).

---

## 15) Roadmap

- Continuous actions (PPO/SAC) with price-time-dose parameterization.  
- Multi-venue routing with queue-position estimation.  
- Meta-controller to pick between policy/AC/TWAP by regime.  
- Model-based RL with learned impact dynamics.  
- Trade-size aware reward shaping using **Expected Shortfall**.

---

## 16) References / Links (internal)

- `market_impact_almgren.ipynb` (impact model & efficient frontier)  
- `analyst_worker.yaml` (streams & routing used here)  
- `scenario_worker.yaml` (stress hooks)  
- `gnn_correlations.py` (regime embeddings)
