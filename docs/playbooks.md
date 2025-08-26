# Playbooks

This guide is a collection of **playbooks**: step-by-step runbooks for common tasks in the arbitrage + macro simulation platform.

---

## 🚀 Running Simulations

### Run a Plain Simulation
```bash
python -m backend.sim.runner --days 120 --dt 1.0 --out runs/sim.json