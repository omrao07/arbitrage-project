# backend/risk/liquidity_spiral.py
"""
Liquidity Spiral Simulator
--------------------------
System-level spiral where falling prices widen spreads, raise haircuts,
force deleveraging/margin calls, trigger fire-sales, which further depress prices.

State (per asset class "bucket", e.g. {IG, HY, EM, EQ, GILT, GSEC})
- price:               normalized (1.0 at t0)
- spread_bps:          trading/liquidity spread
- haircut:             collateral haircut used by lenders
- depth:               market depth proxy (notional absorbable per dt at ~mid)
- inventory:           aggregate leveraged holdings (notional) in the system
- leverage:            avg leverage of those holders
- funding_rate:        funding cost (annualized)
- margin_buffer:       free equity fraction before margin call
- fire_sale_elasticity: price impact coef per 1 unit net sell/ depth

Mechanics (per step):
 1) Shock exogenous vars (spread bump, funding shock, haircut shock)
 2) Mark-to-market -> equity erosion -> margin buffer check
 3) If buffer < 0, compute required deleveraging to restore target leverage
 4) Forced sell = min(required_sells, depth * dt * liquidity_factor)
 5) Price impact from natural + forced sells; update spread & haircuts endogenously
 6) Feedback to funding_rate via spread/liquidity stress
 7) Repeat across buckets with a simple cross-impact (rho matrix)

Outputs
- Per-step trail and system metrics (price drops, total forced sells, VaR-like drawdown)
- Optional Redis publish (risk.liqspiral)

CLI
  python -m backend.risk.liquidity_spiral --probe
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

# Optional bus
try:
    from backend.bus.streams import publish_stream
except Exception:
    publish_stream = None  # type: ignore

_NOW_MS = lambda: int(time.time() * 1000)


# ---------------------------- Models ----------------------------

@dataclass
class Bucket:
    name: str
    price: float = 1.0
    spread_bps: float = 30.0
    haircut: float = 0.15
    depth: float = 5e9                 # notional absorbable per dt (system-wide)
    inventory: float = 20e9            # leveraged holdings (notional)
    leverage: float = 5.0              # A/E (assets over equity)
    funding_rate: float = 0.03
    margin_buffer: float = 0.15        # equity cushion as % of assets
    target_leverage: float = 4.0
    fire_sale_elasticity: float = 0.08 # price drop per (sell/depth) in %
    endog_spread_k: float = 0.6        # spread sensitivity to price drop
    endog_haircut_k: float = 0.5       # haircut sensitivity to spread/liquidity
    min_depth_frac: float = 0.25       # depth floor under stress (as fraction of base)
    notes: List[Dict] = field(default_factory=list)

    def clone(self) -> "Bucket":
        b = Bucket(**{k: v for k, v in asdict(self).items() if k != "notes"})
        b.notes = list(self.notes)
        return b


@dataclass
class Shock:
    # Per-step exogenous shocks (small, can be zero)
    d_spread_bps: float = 0.0
    d_funding: float = 0.0
    d_haircut: float = 0.0
    natural_flow: float = 0.0    # discretionary net flow (sell positive) as notional


@dataclass
class SpiralConfig:
    dt: float = 1/24              # 1 hour step
    max_steps: int = 240          # up to 10 days @ 24 steps/day
    cross_impact: float = 0.15    # fraction of price drop propagated to other buckets
    publish_topic: str = "risk.liqspiral"


# ---------------------------- Engine ----------------------------

class LiquiditySpiral:
    def __init__(self, buckets: Dict[str, Bucket], cfg: Optional[SpiralConfig] = None):
        if not buckets:
            raise ValueError("At least one bucket required")
        self.b = buckets
        self.cfg = cfg or SpiralConfig()
        self.trail: List[Dict] = []

    # ---- one step ----
    def step(self, exo: Dict[str, Shock], publish: bool = False) -> Dict[str, Dict]:
        """
        exo: {bucket_name: Shock(...)} (missing buckets default to zero shock)
        Returns per-bucket snapshot of the step.
        """
        dt = self.cfg.dt
        # placeholders to apply cross-impact after computing own impacts
        price_drop_frac: Dict[str, float] = {}
        forced_sells: Dict[str, float] = {}

        # 1) For each bucket, compute margin state & forced deleveraging
        for k, s in self.b.items():
            sh = exo.get(k, Shock())

            # exogenous nudges
            s.spread_bps = max(0.0, s.spread_bps + sh.d_spread_bps)
            s.funding_rate = max(0.0, s.funding_rate + sh.d_funding)
            s.haircut = min(0.95, max(0.0, s.haircut + sh.d_haircut))

            # mark-to-market equity erosion from any price change last step already applied.
            assets = s.inventory
            equity = assets / max(1e-9, s.leverage)
            buffer_amt = s.margin_buffer * assets
            # funding cost drain over dt (tiny)
            equity -= s.funding_rate * equity * dt

            # Required equity to maintain target leverage after this step:
            req_equity = assets / max(1e-9, s.target_leverage)
            deficit = req_equity - equity - buffer_amt  # if >0 → need to raise equity via selling assets

            # Selling ΔA reduces assets and equity by haircut*ΔA (as equity retained after paying back funding)
            need_sell = 0.0
            if deficit > 0.0:
                # Solve: equity' = equity + haircut * sell  >= req_equity' + buffer
                # with req_equity' = (assets - sell) / target_lev
                # → (equity + h*sell) >= (assets - sell)/L + buffer
                # → sell * (h + 1/L) >= assets/L + buffer - equity
                denom = (s.haircut + 1.0 / max(1e-9, s.target_leverage))
                need_sell = max(0.0, (assets / max(1e-9, s.target_leverage) + buffer_amt - equity) / max(1e-9, denom))

            # execution capacity limited by depth
            cap = max(s.min_depth_frac * s.depth, s.depth) * dt  # depth floor; depth may later shrink
            # include natural discretionary flow
            want_sell = max(0.0, need_sell) + max(0.0, sh.natural_flow)
            sell = min(want_sell, cap)
            forced_sells[k] = sell

            # 2) Price impact from sells: impact % ≈ elasticity * (sell/depth)
            impact_frac = s.fire_sale_elasticity * (sell / max(1e-9, s.depth))
            # cap extreme one-step impact to avoid negative prices
            impact_frac = max(0.0, min(0.5, impact_frac))
            price_drop_frac[k] = impact_frac

        # 3) Cross-impact propagation (very simple linear spillover)
        rho = self.cfg.cross_impact
        for a in list(price_drop_frac.keys()):
            for b in self.b.keys():
                if b == a:
                    continue
                price_drop_frac[b] = price_drop_frac.get(b, 0.0) + rho * price_drop_frac[a] / max(1, len(self.b)-1)

        # 4) Apply price/spread/haircut updates and recompute positions
        snap: Dict[str, Dict] = {}
        for k, s in self.b.items():
            drop = price_drop_frac.get(k, 0.0)
            sell = forced_sells.get(k, 0.0)

            # apply price drop
            old_price = s.price
            s.price = max(0.01, s.price * (1.0 - drop))

            # update assets & equity post trade:
            # - assets reduce by sell
            # - equity increases by haircut*sell (cash retained after paying back funding)
            assets = s.inventory - sell
            equity = (assets / max(1e-9, s.leverage)) + s.haircut * sell

            # but mark-to-market loss also hits equity: Δprice*(inventory_after)
            mtm_loss = (old_price - s.price) * (assets)
            equity -= mtm_loss

            # recompute leverage
            s.inventory = max(0.0, assets)
            s.leverage = max(1.0, s.inventory / max(1e-9, equity))

            # Endogenous spread/haircut/depth adjustments
            s.spread_bps = max(1.0, s.spread_bps * (1.0 + s.endog_spread_k * drop))
            s.haircut = max(s.haircut, min(0.95, s.haircut + s.endog_haircut_k * drop + 0.001 * (s.spread_bps / 10.0)))
            s.depth = max(s.min_depth_frac * s.depth, s.depth * (1.0 - 1.25 * drop))  # depth shrinks when price gaps

            note = {
                "ts_ms": _NOW_MS(),
                "price": s.price,
                "spread_bps": s.spread_bps,
                "haircut": s.haircut,
                "depth": s.depth,
                "sell_forced": sell,
                "price_drop_frac": drop,
                "mtm_loss": mtm_loss,
                "inventory": s.inventory,
                "leverage": s.leverage,
            }
            s.notes.append(note)
            snap[k] = note

        # 5) Publish system snapshot (optional)
        if publish and publish_stream:
            try:
                publish_stream(self.cfg.publish_topic, {"ts_ms": _NOW_MS(), "snapshot": snap})
            except Exception:
                pass

        self.trail.append(snap)
        return snap

    # ---- run multiple steps ----
    def run(self, steps: int, exo_seq: Optional[List[Dict[str, Shock]]] = None, publish: bool = False) -> List[Dict]:
        N = min(steps, self.cfg.max_steps)
        out = []
        for t in range(N):
            exo = exo_seq[t] if (exo_seq and t < len(exo_seq)) else {}
            out.append(self.step(exo, publish=publish))
        return out

    # ---- quick health metrics ----
    def summary(self) -> Dict[str, float]:
        dd = []
        fs = 0.0
        for k, s in self.b.items():
            px0 = 1.0  # normalized start
            dd.append(1.0 - s.price / px0)
            if s.notes:
                fs += sum(max(0.0, n.get("sell_forced", 0.0)) for n in s.notes)
        return {
            "max_drawdown_pct": 100 * max(dd) if dd else 0.0,
            "total_forced_sells": fs,
            "buckets": len(self.b),
        }


# ---------------------------- CLI Probe ----------------------------

def _probe():
    # Two buckets: Investment Grade credit vs Equities
    ig = Bucket(name="IG", price=1.0, spread_bps=70, haircut=0.12, depth=8e9, inventory=30e9, leverage=6.0,
                target_leverage=5.0, fire_sale_elasticity=0.06)
    eq = Bucket(name="EQ", price=1.0, spread_bps=25, haircut=0.15, depth=10e9, inventory=25e9, leverage=3.0,
                target_leverage=2.5, fire_sale_elasticity=0.10)

    sim = LiquiditySpiral({"IG": ig, "EQ": eq}, SpiralConfig(dt=1/24, cross_impact=0.20))

    # Day 1 hour 0: mild exogenous funding + haircut shock to IG, small natural selling in EQ
    exo_seq = []
    for t in range(48):  # 2 days
        exo = {}
        if t == 1:
            exo["IG"] = Shock(d_funding=0.01, d_haircut=0.03)  # lenders hike haircuts and funding
        if t in (2, 3, 4):
            exo.setdefault("IG", Shock())
            exo["IG"].natural_flow = 1.5e9
        if t in (5, 6):
            exo["EQ"] = Shock(natural_flow=1.0e9)
        exo_seq.append(exo)

    sim.run(steps=48, exo_seq=exo_seq, publish=False)
    print("Summary:", sim.summary())
    # Print last snapshot
    print("Last:", sim.trail[-1])

def main():
    import argparse, json
    ap = argparse.ArgumentParser(description="Liquidity Spiral Simulator")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--steps", type=int, default=48)
    args = ap.parse_args()
    if args.probe:
        _probe()
        return

    # Minimal single-bucket run
    b = Bucket(name="EQ", price=1.0, spread_bps=25, haircut=0.15, depth=10e9, inventory=20e9, leverage=3.0)
    sim = LiquiditySpiral({"EQ": b})
    sim.run(args.steps, publish=False)
    print(json.dumps(sim.summary(), indent=2))

if __name__ == "__main__":
    main()