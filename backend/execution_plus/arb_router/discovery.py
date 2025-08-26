# backend/execution_plus/arb_router/discovery.py
"""
Venue discovery & readiness probe for the Global Arbitrage Router.

Responsibilities
---------------
- Load venue configs from venues.yaml
- Instantiate/collect adapters (real or mock) using execution_plus.registry
- Build a tradable universe per venue (symbols list)
- Probe quotes (latency, spread, mid) for a sample of symbols
- Emit a compact DiscoveryResult for the router to consume

YAML shape (example):
---------------------
venues:
  - id: "BINANCE"
    name: "Binance"
    type: "crypto"
    region: "SG"
    base_currency: "USDT"
    maker_fee_bps: 0.1
    taker_fee_bps: 0.1
    min_order_size: 10.0
    max_order_size: 10000000.0
    avg_latency_ms: 80
    # adapter: "plugins.exec.ccxt:BinanceAdapter"   # optional dotted path

Usage
-----
from backend.execution_plus.arb_router.discovery import Discovery, DiscoveryResult
res = Discovery("backend/config/venues.yaml").run()
print(res.venues.keys(), res.universe, res.health)
"""

from __future__ import annotations

import os
import time
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Optional YAML
try:
    import yaml  # type: ignore
    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False

# Pull in your registry/adapters
from backend.execution_plus.registry import HUB, load_adapters_from_yaml # type: ignore
from backend.execution_plus.adapters import AdapterBase, Quote # type: ignore


# ----------------------------- data models -----------------------------

@dataclass
class VenueInfo:
    id: str
    type: str
    region: str
    base_currency: str
    fee_taker_bps: float
    fee_maker_bps: float
    latency_ms: int


@dataclass
class QuoteProbe:
    symbol: str
    bid: Optional[float]
    ask: Optional[float]
    mid: Optional[float]
    spread_bps: Optional[float]
    ts: float


@dataclass
class DiscoveryResult:
    venues: Dict[str, VenueInfo]                       # id -> info
    adapters: Dict[str, AdapterBase]                   # id -> adapter instance
    universe: Dict[str, List[str]]                     # id -> symbols
    health: Dict[str, Dict[str, Any]]                  # id -> { status, reason, probes[...] }
    ts: float = field(default_factory=lambda: time.time())


# ----------------------------- helpers --------------------------------

def _spread_bps(q: Quote) -> Optional[float]:
    if q.bid is None or q.ask is None or q.bid <= 0 or q.ask <= q.bid:
        return None
    mid = q.mid or (q.bid + q.ask) / 2.0
    if mid <= 0:
        return None
    return 10_000.0 * (q.ask - q.bid) / mid


def _pick_samples(symbols: List[str], k: int = 3) -> List[str]:
    if not symbols:
        return []
    if len(symbols) <= k:
        return symbols
    return random.sample(symbols, k)


# ----------------------------- discovery ------------------------------

class Discovery:
    """
    Orchestrates venue bootstrapping + probes.
    """

    def __init__(self, venues_yaml: str, *, sample_symbols: int = 3):
        self.venues_yaml = venues_yaml
        self.sample_symbols = int(sample_symbols)
        self._adapters: Dict[str, AdapterBase] = {}

    # ---- config/adapters ----

    def _ensure_adapters(self) -> Dict[str, AdapterBase]:
        """
        Load adapters from YAML (preferred). Falls back to HUB.adapters if YAML missing.
        """
        adapters: Dict[str, AdapterBase] = {}
        if _HAVE_YAML and os.path.isfile(self.venues_yaml):
            adapters = load_adapters_from_yaml(self.venues_yaml)
        else:
            # Use whatever is already registered (mocks from AdapterRegistry)
            adapters = HUB.adapters.all()
            if not adapters:
                raise RuntimeError("No adapters available (venues.yaml missing and HUB empty).")
        self._adapters = adapters
        return adapters

    # ---- summarize venue info ----

    def _venue_info(self, adapter: AdapterBase) -> VenueInfo:
        cfg = adapter.cfg  # from VenueConfig in adapters.py
        return VenueInfo(
            id=cfg.id,
            type=cfg.type,
            region=cfg.region,
            base_currency=cfg.base_currency,
            fee_taker_bps=cfg.taker_fee_bps,
            fee_maker_bps=cfg.maker_fee_bps,
            latency_ms=cfg.avg_latency_ms,
        )

    # ---- probe a venue ----

    def _probe_venue(self, vid: str, adapter: AdapterBase, symbols: List[str]) -> Dict[str, Any]:
        """
        Returns a health dict with status + quote probes.
        """
        probes: List[QuoteProbe] = []
        status = "ok"
        reason = ""

        try:
            sample = _pick_samples(symbols, self.sample_symbols)
            for s in sample:
                q = adapter.get_quote(s)
                probes.append(
                    QuoteProbe(
                        symbol=s,
                        bid=q.bid,
                        ask=q.ask,
                        mid=q.mid,
                        spread_bps=_spread_bps(q),
                        ts=q.ts,
                    )
                )
        except Exception as e:
            status = "error"
            reason = f"quote_failed:{e}"

        # If no quotes or all invalid, mark degraded
        if not probes:
            status = "degraded" if status == "ok" else status
            reason = reason or "no_probes"
        else:
            valid = [p for p in probes if p.mid and p.bid and p.ask and p.spread_bps is not None]
            if not valid and status == "ok":
                status = "degraded"
                reason = "invalid_quotes"

        # Convert dataclasses to dicts for serialization
        probes_out = [p.__dict__ for p in probes]

        return {"status": status, "reason": reason, "probes": probes_out}

    # ---- main entry ----

    def run(self) -> DiscoveryResult:
        adapters = self._ensure_adapters()

        venues: Dict[str, VenueInfo] = {}
        universe: Dict[str, List[str]] = {}
        health: Dict[str, Dict[str, Any]] = {}

        for vid, adapter in adapters.items():
            try:
                venues[vid] = self._venue_info(adapter)
            except Exception:
                continue

            # symbol universe
            syms: List[str] = []
            try:
                syms = adapter.get_symbols() or []
            except Exception:
                syms = []
            universe[vid] = syms

            # health probes
            health[vid] = self._probe_venue(vid, adapter, syms)

        return DiscoveryResult(venues=venues, adapters=adapters, universe=universe, health=health)


# ----------------------------- CLI ------------------------------------

if __name__ == "__main__":
    path = os.getenv("VENUES_YAML", "backend/config/venues.yaml")
    res = Discovery(path).run()

    # Pretty print a tiny summary without adding dependencies
    print("=== Venues ===")
    for vid, info in res.venues.items():
        print(f"{vid} ({info.type}/{info.region}) fees(taker/maker)={info.fee_taker_bps}/{info.fee_maker_bps}bps latency={info.latency_ms}ms")

    print("\n=== Universe sizes ===")
    for vid, syms in res.universe.items():
        print(f"{vid}: {len(syms)} symbols")

    print("\n=== Health ===")
    for vid, h in res.health.items():
        probes = h.get("probes", [])
        spreads = [p.get("spread_bps") for p in probes if p.get("spread_bps") is not None]
        avg_spread = (sum(spreads) / len(spreads)) if spreads else None
        print(f"{vid}: {h.get('status')} ({h.get('reason') or 'ok'}), probes={len(probes)}, avg_spread_bps={avg_spread}")