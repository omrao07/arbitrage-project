# backend/agents/competitor_ai.py
"""
Competitor AI
-------------
Simulated competitor funds / prop desks that react to market states, news,
and your own activity. Provides adversarial order flow for stress tests,
liquidity spiral dynamics, and crisis theatre.

Concepts
- Competitor: param set (risk appetite, leverage, style: trend, arb, HF, ETF)
- Brain: simple state machine + heuristics (trend follow, mean reversion, panic sell)
- Input: ticks, news sentiment, volatility/risk metrics
- Output: synthetic orders (market/limit), position updates
- Publish: to "sim.competitors" and optionally into OMS pre-risk stream (orders.incoming)

Usage
------
from backend.agents.competitor_ai import CompetitorAgent, CompetitorWorld

comp = CompetitorAgent("HF_X", style="trend", max_leverage=5.0)
world = CompetitorWorld([comp])
world.step(tick={"symbol":"NIFTY","price":22200,"vol":0.35,"news_sentiment":-0.7})
"""

from __future__ import annotations

import math
import os
import random
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# optional bus
try:
    from backend.bus.streams import publish_stream
except Exception:
    publish_stream = None  # type: ignore

# ------------- Config / State -------------
@dataclass
class CompetitorState:
    equity: float = 1e8
    leverage: float = 1.0
    positions: Dict[str, float] = field(default_factory=dict)
    pnl: float = 0.0
    distress: float = 0.0   # 0..1 panic indicator
    style: str = "trend"    # trend | meanrev | arb | etf

@dataclass
class CompetitorAgent:
    name: str
    style: str = "trend"
    max_leverage: float = 5.0
    capital: float = 1e8
    state: CompetitorState = field(default_factory=CompetitorState)

    def __post_init__(self):
        self.state.style = self.style
        self.state.equity = self.capital

    def react(self, tick: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate an order given a tick + state.
        Returns order payload or None.
        """
        sym = tick.get("symbol") or "NIFTY"
        px  = float(tick.get("price",0))
        vol = float(tick.get("vol",0.2))
        senti = float(tick.get("news_sentiment",0.0))

        # compute distress: high vol + neg pnl + bad sentiment
        dd_factor = max(0.0, -self.state.pnl/self.capital)
        self.state.distress = min(1.0, 0.5*vol + 0.3*dd_factor + 0.2*max(0.0,-senti))

        # baseline size
        base_notional = self.state.equity * 0.01
        qty = base_notional/px

        order = None
        if self.style=="trend":
            if tick.get("ret",0) > 0 or senti>0:
                order = self._order(sym,"buy",qty)
            elif tick.get("ret",0)<0 or senti<0:
                order = self._order(sym,"sell",qty)
        elif self.style=="meanrev":
            if tick.get("ret",0) > 0.01:
                order = self._order(sym,"sell",qty*0.5)
            elif tick.get("ret",0)<-0.01:
                order = self._order(sym,"buy",qty*0.5)
        elif self.style=="arb":
            # placeholder: random cross hedge
            side = random.choice(["buy","sell"])
            order = self._order(sym,side,qty*0.3)
        elif self.style=="etf":
            # replicate benchmark flows: respond to sentiment
            side = "buy" if senti>0 else "sell"
            order = self._order(sym,side,qty*0.8)

        # distress override
        if self.state.distress>0.7:
            # panic liquidation
            order = self._order(sym,"sell",qty*2.0)

        if order:
            # update positions/pnl
            self.state.positions[sym] = self.state.positions.get(sym,0)+order["qty"]*(1 if order["side"]=="buy" else -1)
            # pretend PnL update
            self.state.pnl += (random.random()-0.5)*1000
            return order
        return None

    def _order(self, sym: str, side: str, qty: float) -> Dict[str, Any]:
        return {
            "id": str(uuid.uuid4()),
            "ts_ms": int(time.time()*1000),
            "competitor": self.name,
            "symbol": sym,
            "side": side,
            "qty": float(qty),
            "typ": "market",
            "style": self.style
        }

# ------------- World (multi agents) -------------
class CompetitorWorld:
    def __init__(self, agents: Optional[List[CompetitorAgent]]=None):
        self.agents: List[CompetitorAgent] = agents or []

    def step(self, tick: Dict[str,Any]) -> List[Dict[str,Any]]:
        orders: List[Dict[str,Any]] = []
        for a in self.agents:
            o = a.react(tick)
            if o:
                orders.append(o)
                if publish_stream:
                    try: publish_stream("sim.competitors", o)
                    except Exception: pass
        return orders

# ------------- CLI probe -------------
def _probe():
    agents = [CompetitorAgent("FundA",style="trend"),
              CompetitorAgent("FundB",style="meanrev"),
              CompetitorAgent("FundC",style="etf")]
    world = CompetitorWorld(agents)
    ticks = [
        {"symbol":"NIFTY","price":22000,"ret":0.01,"vol":0.3,"news_sentiment":-0.6},
        {"symbol":"NIFTY","price":21800,"ret":-0.02,"vol":0.4,"news_sentiment":-0.4},
        {"symbol":"NIFTY","price":22100,"ret":0.015,"vol":0.2,"news_sentiment":0.2},
    ]
    for t in ticks:
        out = world.step(t)
        print("Tick",t["price"],"->",out)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe",action="store_true")
    args = ap.parse_args()
    if args.probe: _probe()

if __name__=="__main__":
    main()