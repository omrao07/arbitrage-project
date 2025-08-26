# Configuration Guide

This document describes the configuration files used across the arbitrage / macro simulation project.
All configs are **JSON** or **YAML**, and are designed to be human-editable and machine-parseable.

---

## 📊 Policy & Central Bank Configs

These files define starting conditions for monetary authorities.

- **`fed.yaml`**
  ```yaml
  start_rate: 0.05        # 5% upper bound of Fed funds
  balance_sheet: 8.0e12   # USD assets
  neutral_rate: 0.025