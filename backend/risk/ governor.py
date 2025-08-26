# backend/risk/governor.py
"""
Governor (Static Policy Layer)
------------------------------
Defines baseline hard limits & circuit breakers for the hedge fund system.
Acts as the ultimate safety net before orders leave OMS.

Responsibilities
- Hard risk caps: max leverage, max order notional, max participation
- Global kill-switch if equity drawdown breaches threshold
- Max concurrent strategies allowed
- Circuit breakers on symbol/market (volatility spikes, halts)
- Mirrors decisions to Redis so dashboards & dynamic_governor can override

Inputs
- Current portfolio state (PnL, leverage, equity)
- Router/strategy proposals (order payloads)
- External risk signals (market halt flags, volatility index)

Outputs
- Approve/reject orders
- Update global HSET risk:limits:* keys
- Publish governor alerts to "risk.governor" stream

Usage
------
gov = Governor(cfg)
ok, reason = gov.approve_order(order, metrics)
if not ok: reject

Run in background to enforce continuously:
python -m backend.risk.governor --probe
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:
    from backend.bus.streams import publish_stream, hset
except Exception:
    publish_stream = None
    hset = None

# ---------------- Config ----------------

@dataclass
class GovLimits:
    max_leverage: float = 5.0          # firm-level leverage cap
    max_drawdown: float = 0.20         # 20% equity drawdown circuit breaker
    max_order_notional: float = 5e6    # single-order notional cap
    max_participation: float = 0.15    # max ADV participation
    max_strategies: int = 25           # max concurrently enabled strats
    volatility_halt: float = 0.10      # 10% single-bar move triggers halt

@dataclass
class GovConfig:
    limits: GovLimits = field(default_factory=GovLimits)
    region: Optional[str] = None


# ---------------- Governor Core ----------------

class Governor:
    def __init__(self, cfg: Optional[GovConfig] = None):
        self.cfg = cfg or GovConfig()
        self._kill = False
        self._kill_reason: Optional[str] = None

    # ---- approval API ----
    def approve_order(self, order: Dict[str, Any], metrics: Dict[str, Any]) -> tuple[bool,str]:
        """
        Order fields expected: {"symbol","qty","price","strategy",...}
        Metrics expected: {"equity","pnl","drawdown","leverage","adv","bar_return"}
        """
        if self._kill:
            return False, f"kill_switch:{self._kill_reason}"

        eq   = float(metrics.get("equity", 1e6))
        dd   = float(metrics.get("drawdown", 0.0))
        lev  = float(metrics.get("leverage", 1.0))
        adv  = float(metrics.get("adv", 1e7))
        bar  = float(metrics.get("bar_return", 0.0))
        notional = float(order.get("qty",0))*float(order.get("price",0))

        L = self.cfg.limits

        if dd >= L.max_drawdown:
            self._trip("drawdown", dd)
            return False, "drawdown_circuit"

        if lev > L.max_leverage:
            return False, f"leverage_cap:{lev:.2f}>{L.max_leverage}"

        if notional > L.max_order_notional:
            return False, f"notional_cap:{notional}>{L.max_order_notional}"

        part = notional / max(1e-9, adv)
        if part > L.max_participation:
            return False, f"participation_cap:{part:.2f}>{L.max_participation}"

        if abs(bar) >= L.volatility_halt:
            return False, f"vol_halt:{bar:.2f}"

        return True, "ok"

    def approve_strategy_enable(self, enabled_count: int) -> tuple[bool,str]:
        if enabled_count >= self.cfg.limits.max_strategies:
            return False, f"max_strategies:{enabled_count}>{self.cfg.limits.max_strategies}"
        return True, "ok"

    def _trip(self, reason: str, val: Any):
        self._kill = True
        self._kill_reason = f"{reason}:{val}"
        self._emit("kill_switch", {"reason":reason,"val":val})

    def reset_kill(self):
        self._kill = False
        self._kill_reason = None
        self._emit("reset", {})

    # ---- telemetry / bus ----
    def _emit(self, kind: str, payload: Dict[str,Any]):
        if publish_stream:
            try:
                publish_stream("risk.governor", {"ts_ms": int(time.time()*1000),"kind":kind,**payload})
            except Exception:
                pass


# ---------------- CLI ----------------

def _probe():
    cfg = GovConfig()
    gov = Governor(cfg)

    order = {"symbol":"RELIANCE","qty":2000,"price":2500}
    metrics = {"equity":1e7,"drawdown":0.22,"leverage":2.0,"adv":1e6,"bar_return":0.03}

    ok, reason = gov.approve_order(order, metrics)
    print("Approve?", ok, reason)

    if not ok:
        print("Kill switch tripped. Resetting...")
        gov.reset_kill()
        ok, reason = gov.approve_order(order, {"equity":1e7,"drawdown":0.05,"leverage":2,"adv":1e6,"bar_return":0.01})
        print("Approve after reset?", ok, reason)

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Static Governor")
    ap.add_argument("--probe",action="store_true")
    args = ap.parse_args()
    if args.probe:
        _probe()

if __name__=="__main__":
    main()