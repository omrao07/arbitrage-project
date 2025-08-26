# Feature Overview

This document lists the key features of the arbitrage + macro simulation project.  
It is organized by **layer**: data ingestion, simulation, agents, infrastructure, and extensions.

---

## 📡 Data & Ingestion

- **Multi-Adapter Signal Bus**  
  - Redis-backed `streams.py` + `signal_bus.py` for pub/sub & state.  
  - Adapters for **brokers, venues, synthetic feeds**.  

- **Alternative Data Connectors** (`altdata.yaml`)  
  - Wearables: `apple.py`, `fitbit.py`, `health_ingest.py`  
  - Logistics: `shipping_ais.py`, `satelites.py`  
  - Web & Social: `reddit.py`, `tiktok.py`, `discord.py`, `x.py`, `web_trends.py`  
  - Climate: `ecmwf.py`, `noaa.py`, `climate_signals.py`, `hazards.py`

- **Sentiment Analysis** (`sentiment_model.py`)  
  - Ingests posts, runs basic NLP scoring, publishes normalized z-scores.  

- **Configurable Sources**  
  - `fed.yaml`, `ecb.yaml`, `rbi.yaml` map central bank data into signals.  
  - `synthetic.yaml` for synthetic assets, `venues.yaml` for execution venues.  

---

## 🧮 Simulation & Scenarios

- **Policy Simulator (`policy_sim.py`)**  
  - Multi-regime model (`expansion`, `slowdown`, `inflation`, `crisis`)  
  - Tracks rates, risk_z, infl_z, liq_z, and proxies.  

- **Shock Models (`shock_models.py`)**  
  - Jump diffusion (Poisson rate surprises)  
  - Hawkes-like risk clusters  
  - Regime-conditional shocks  
  - Volatility spikes, liquidity drains  
  - FX gap + cross-asset propagator  

- **Scenario Runner (`scenarios.py`)**  
  - Timeline of blocks: regime + shocks + repeats.  
  - JSON/YAML external scenarios:
    - `covid_202.json` (COVID panic)  
    - `flash_crash.json` (intraday equity crash)  
    - `lehman_2008.json` (systemic collapse)  
  - Presets: `soft_landing`, `hard_landing`, `stagflation`, `crisis_liquidity`.  

- **Runner CLI (`runner.py`)**  
  - Execute plain sims or scenario files.  
  - Attach shocks (`shocks.yaml`).  
  - Output JSON, JSONL, CSV; publish to live bus.  

---

## 🧠 Agents & Swarm

- **Agent Base (`agents/base.py`)**  
  - Common interface: `propose()`, `risk()`, `explain()`.  

- **Domain Agents** (`agents/`)  
  - `fx.py`, `equities.py`, `crypto.py`, `commodities.py`.  
  - Each negotiates positions in a **swarm loop** (`coordinator.py`).  

- **Alpha Libraries** (`alpha/`)  
  - Inspired by Dalio (`dalio.py`), Buffett (`buffet.py`), Druckmiller (`druck.py`).  
  - Mixers (`mixer.py`) for combining signals.  

- **Macro Gauge (`economy.py`)**  
  - Normalizes growth_z, infl_z, risk_z, liq_z into a **GI quadrant** regime.  
  - Labels: Goldilocks, Reflation, Stagflation, Disinflation.  
  - Policy stance maps for Fed, ECB, RBI.  

- **Explainers (`explainer.py`)**  
  - Generate human-readable narratives from signals/regimes.  

---

## ⚙️ Infrastructure & Risk

- **Registry / Discovery** (`registry.py`, `discovery.py`)  
  - Keeps track of venues, adapters, instruments.  

- **Execution**  
  - `router.py`: global arbitrage router  
  - `market_maker.py`: synthetic order book  
  - `pricer.py`: fair value estimates  
  - `cost_model.py`: estimate impact/fees  

- **Pipelines (`pipelines.py`)**  
  - Stage graph builder; compose enrich → signals → eco → agents.  

- **Policy Tools**  
  - `policy_sim.py` + `policy_sim.yaml` for central bank behaviour.  
  - `policy_sim.py` also feeds into scenarios (FOMC, ECB, RBI).  

- **Stress Harness (`stress.py`)**  
  - Load tests + chaos injection.  
  - Latency distribution, error rates, toy PnL drawdowns.  
  - Outputs JSON + CSV.  

---

## 🚀 Extensions & Crazy Features

- **Global Arbitrage Router** — routes trades across venues & synthetic assets.  
- **Synthetic Asset Generator** — builds new tradeables on the fly.  
- **Emotion-Driven Trading Engine** — plugs sentiment scores directly into risk sizing.  
- **Scenario Library** — extreme what-ifs: pandemics, flash crashes, sovereign defaults.  
- **Multi-Agent Swarm** — domain experts negotiate portfolios.  
- **Explainable Narratives** — turn econ state into plain-English reports.  
- **Stress & Chaos Testing** — inject random failures, measure resilience.  

---

## 🗂 File Index (selected)

- `backend/config/` → YAMLs & JSONs (fed, ecb, rbi, scenarios, shocks, sentiment, venues)  
- `backend/sim/` → simulators (`policy_sim.py`, `shock_models.py`, `scenarios.py`, `runner.py`)  
- `backend/macro/` → `economy.py`, gauges & explainers  
- `backend/runtime/` → `manager.py`, `pipelines.py`  
- `backend/tests/` → `stress.py`  

---

## 🔑 Notes
- All features are modular: you can run **just sim**, **just swarm agents**, or wire the **whole stack**.  
- Configs are always external (YAML/JSON) → easy to reproduce or tweak scenarios.  
- The system is both **a trading research platform** and **a Wharton-application-worthy showcase**.

---