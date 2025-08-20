# backend/strategies/diversified/commodity_transport_arbitrage.py
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import redis

from backend.engine.strategy_base import Strategy

"""
Commodity Transport (Spatial) Arbitrage
---------------------------------------
Exploits regional price differentials after subtracting all-in transport costs (freight + fees + losses).
Examples:
  - Brent vs WTI (seaborne arbitrage)
  - Henry Hub (US gas) vs TTF (EU gas) via LNG shipping
  - Soybean Brazil vs CBOT delivery parity
  - Iron ore China CFR vs Australia FOB

Mechanics:
1) Normalize regional prices to USD per *physical unit* (bbl, MMBtu, bu, mt, etc.).
2) For each route (ORIGIN -> DEST), compute Netback:
       netback = Price_DEST_USD - Price_ORIG_USD - TransportCost_USD
   Positive netback >> 0 ⇒ profitable to move physical from ORIGIN to DEST.
3) Maintain EWMA mean/variance on netback ⇒ z-score.
4) Trade futures/ETFs proxies:
   - If netback >> threshold & z > ENTRY_Z: LONG origin (cheap), SHORT destination (rich).
   - Exit when |netback| <= EXIT_USD or |z| <= EXIT_Z.

Inputs (you already publish/maintain these):
  - Redis HSET last_price <symbol> -> {"price": <px>}
  - Optional FX HSET fx:spot <pair> -> <spot>  (for non‑USD quotes)
  - Optional route-level dynamic costs stream (HSET/Stream): route_cost:<ORIG>-<DEST> -> {"cost": <usd_per_unit>}

Symbols are your futures proxies (e.g., CL.F1 for WTI, BZ.F1 for Brent, TTF.F1 for Dutch gas).
"""

# ============================== CONFIG ==============================
REDIS_HOST = os.getenv("CTA_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("CTA_REDIS_PORT", "6379"))

# Routes (ORIGIN, DEST) with per-unit transport assumptions (USD/unit) if no dynamic feed available
# Format env: "CL.F1,BZ.F1,2.10;NG.F1,TTF.F1,4.50"
ROUTES_ENV = os.getenv("CTA_ROUTES", "CL.F1,BZ.F1,2.10;NG.F1,TTF.F1,4.50")

# Units per futures contract (for sizing); override per commodity prefix
CONTRACT_UNITS = {
    "CL": float(os.getenv("CTA_CL_UNITS", "1000")),     # bbl
    "BZ": float(os.getenv("CTA_BZ_UNITS", "1000")),     # bbl
    "NG": float(os.getenv("CTA_NG_UNITS", "10000")),    # MMBtu
    "TTF": float(os.getenv("CTA_TTF_UNITS", "10000")),  # MWh or MMBtu equiv (ensure your price normalization matches)
    "C":  float(os.getenv("CTA_C_UNITS", "5000")),      # bushels
    "S":  float(os.getenv("CTA_S_UNITS", "5000")),
    "W":  float(os.getenv("CTA_W_UNITS", "5000")),
    "IO": float(os.getenv("CTA_IO_UNITS", "100")),      # example metric tons
}

# FX mapping (prefix -> USD pair); set USDUSD if already USD quotes
FX_FOR = {
    "CL": os.getenv("CTA_CL_FX", "USDUSD").upper(),
    "BZ": os.getenv("CTA_BZ_FX", "USDUSD").upper(),
    "NG": os.getenv("CTA_NG_FX", "USDUSD").upper(),
    "TTF": os.getenv("CTA_TTF_FX", "EURUSD").upper(),
    "C":  os.getenv("CTA_C_FX",  "USDUSD").upper(),
    "S":  os.getenv("CTA_S_FX",  "USDUSD").upper(),
    "W":  os.getenv("CTA_W_FX",  "USDUSD").upper(),
    "IO": os.getenv("CTA_IO_FX", "USDUSD").upper(),
}

# Thresholds
ENTRY_USD = float(os.getenv("CTA_ENTRY_USD", "0.75"))  # abs USD/unit netback
EXIT_USD  = float(os.getenv("CTA_EXIT_USD",  "0.20"))
ENTRY_Z   = float(os.getenv("CTA_ENTRY_Z",   "2.0"))
EXIT_Z    = float(os.getenv("CTA_EXIT_Z",    "0.7"))

# Sizing
USD_PER_LEG    = float(os.getenv("CTA_USD_PER_LEG", "30000"))
MAX_CONCURRENT = int(os.getenv("CTA_MAX_CONCURRENT", "4"))

# EWMA params (event-based)
EWMA_ALPHA = float(os.getenv("CTA_EWMA_ALPHA", "0.03"))

# Re-check cadence (seconds)
RECHECK_SECS = int(os.getenv("CTA_RECHECK_SECS", "10"))

# Venue hints (optional)
VENUE_HINTS = {
    "CL": "NYMEX", "BZ": "ICE_EU", "NG": "NYMEX", "TTF": "ICE_EU",
    "C": "CBOT", "S": "CBOT", "W": "CBOT", "IO": "SGX"
}

# Redis keys
LAST_PRICE_HKEY = os.getenv("CTA_LAST_PRICE_KEY", "last_price")
FX_SPOT_HKEY    = os.getenv("CTA_FX_SPOT_KEY",    "fx:spot")
ROUTE_COST_KEY  = "route_cost"  # per-route override: HGET route_cost:<ORIG>-<DEST> cost -> usd_per_unit

# ============================== REDIS ==============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================== HELPERS ==============================
def _prefix(sym: str) -> str:
    return sym.split(".", 1)[0].upper()

def _venue(sym: str) -> Optional[str]:
    return VENUE_HINTS.get(_prefix(sym))

def _hget_last(symbol: str) -> Optional[float]:
    raw = r.hget(LAST_PRICE_HKEY, symbol)
    if not raw:
        return None
    try:
        return float(json.loads(raw)["price"])
    except Exception:
        try:
            return float(raw)
        except Exception:
            return None

def _fx(pair: str) -> float:
    if pair == "USDUSD":
        return 1.0
    v = r.hget(FX_SPOT_HKEY, pair)
    if v:
        try:
            return float(v)
        except Exception:
            pass
    # fallback to last_price FX symbol if present
    px = _hget_last(pair)
    return float(px or 1.0)

def _usd_price(sym: str) -> Optional[float]:
    px = _hget_last(sym)
    if px is None:
        return None
    fx = _fx(FX_FOR.get(_prefix(sym), "USDUSD"))
    return float(px) * float(fx)

def _units(sym: str) -> float:
    return CONTRACT_UNITS.get(_prefix(sym), 1.0)

def _route_dyn_cost(origin: str, dest: str) -> Optional[float]:
    # Optional dynamic cost: HGET route_cost:ORIGIN-DEST cost
    key = f"{ROUTE_COST_KEY}:{origin}-{dest}"
    v = r.hget(key, "cost")
    try:
        return float(v) if v is not None else None
    except Exception:
        return None

def _poskey(name: str, origin: str, dest: str) -> str:
    return f"cta:open:{name}:{origin}->{dest}"

def _ewma_key(origin: str, dest: str) -> str:
    return f"cta:ewma:{origin}->{dest}"

# ============================== EWMA MV ==============================
@dataclass
class EwmaMV:
    mean: float
    var: float
    alpha: float
    def update(self, x: float) -> Tuple[float, float]:
        m0 = self.mean
        self.mean = (1 - self.alpha) * self.mean + self.alpha * x
        self.var = max(1e-12, (1 - self.alpha) * (self.var + (x - m0) * (x - self.mean)))
        return self.mean, self.var

def _load_ewma(origin: str, dest: str, alpha: float) -> EwmaMV:
    raw = r.get(_ewma_key(origin, dest))
    if raw:
        try:
            o = json.loads(raw)
            return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", alpha)))
        except Exception:
            pass
    return EwmaMV(mean=0.0, var=1.0, alpha=alpha)

def _save_ewma(origin: str, dest: str, ew: EwmaMV) -> None:
    r.set(_ewma_key(origin, dest), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ============================== ROUTE SPEC ==============================
@dataclass
class Route:
    origin: str
    dest: str
    cost_usd_per_unit_static: float

def _parse_routes(env: str) -> List[Route]:
    routes: List[Route] = []
    for part in env.split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            o, d, c = [x.strip().upper() for x in part.split(",")]
            routes.append(Route(origin=o, dest=d, cost_usd_per_unit_static=float(c)))
        except Exception:
            continue
    return routes

ROUTES = _parse_routes(ROUTES_ENV)

# ============================== STRATEGY ==============================
@dataclass
class OpenState:
    side: str = ""             # "long_origin_short_dest" OR "short_origin_long_dest"
    qty_o: float = 0.0
    qty_d: float = 0.0
    entry_netback: float = 0.0
    entry_z: float = 0.0
    ts_ms: int = 0

class CommodityTransportArbitrage(Strategy):
    """
    Long cheap origin / short rich destination when netback >> threshold; symmetric on negative.
    """

    def __init__(self, name: str = "commodity_transport_arbitrage", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self.last_check = 0.0

    def on_start(self) -> None:
        super().on_start()
        # Advertise routes for UI
        r.hset("strategy:universe", self.ctx.name, json.dumps({
            "routes": [f"{rt.origin}->{rt.dest}" for rt in ROUTES],
            "ts": int(time.time()*1000)
        }))

    def on_tick(self, tick: Dict) -> None:
        now = time.time()
        if now - self.last_check < RECHECK_SECS:
            return
        self.last_check = now
        self._evaluate_all()

    # ---------------- core engine ----------------
    def _evaluate_all(self) -> None:
        open_count = 0
        for rt in ROUTES:
            if r.get(_poskey(self.ctx.name, rt.origin, rt.dest)):
                open_count += 1

        for rt in ROUTES:
            po = _usd_price(rt.origin)
            pd = _usd_price(rt.dest)
            if po is None or pd is None:
                continue

            dyn = _route_dyn_cost(rt.origin, rt.dest)
            cost = float(dyn if dyn is not None else rt.cost_usd_per_unit_static)

            netback = pd - po - cost  # USD per unit
            ew = _load_ewma(rt.origin, rt.dest, EWMA_ALPHA)
            m, v = ew.update(netback)
            _save_ewma(rt.origin, rt.dest, ew)
            z = (netback - m) / math.sqrt(max(v, 1e-12))

            st = self._load_state(rt.origin, rt.dest)

            # -------- exits first --------
            if st:
                if (abs(netback) <= EXIT_USD) or (abs(z) <= EXIT_Z):
                    self._close(rt, st)
                continue

            # -------- entries --------
            if open_count >= MAX_CONCURRENT:
                continue
            if not (abs(netback) >= ENTRY_USD and abs(z) >= ENTRY_Z):
                continue

            # size legs (contracts)
            uo = _units(rt.origin)
            ud = _units(rt.dest)
            if uo <= 0 or ud <= 0:
                continue

            qo = USD_PER_LEG / max(po * uo, 1e-9)
            qd = USD_PER_LEG / max(pd * ud, 1e-9)

            if netback > 0:
                # destination rich vs origin: LONG origin, SHORT destination
                self.order(rt.origin, "buy",  qty=qo, order_type="market", venue=_venue(rt.origin))
                self.order(rt.dest,   "sell", qty=qd, order_type="market", venue=_venue(rt.dest))
                self._save_state(rt.origin, rt.dest, OpenState(
                    side="long_origin_short_dest", qty_o=qo, qty_d=qd,
                    entry_netback=netback, entry_z=z, ts_ms=int(time.time()*1000)
                ))
            else:
                # origin rich vs dest: SHORT origin, LONG destination
                self.order(rt.origin, "sell", qty=qo, order_type="market", venue=_venue(rt.origin))
                self.order(rt.dest,   "buy",  qty=qd, order_type="market", venue=_venue(rt.dest))
                self._save_state(rt.origin, rt.dest, OpenState(
                    side="short_origin_long_dest", qty_o=qo, qty_d=qd,
                    entry_netback=netback, entry_z=z, ts_ms=int(time.time()*1000)
                ))
            open_count += 1

    # ---------------- state/persistence ----------------
    def _load_state(self, origin: str, dest: str) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, origin, dest))
        if not raw:
            return None
        try:
            o = json.loads(raw)
            return OpenState(**o)
        except Exception:
            return None

    def _save_state(self, origin: str, dest: str, st: Optional[OpenState]) -> None:
        k = _poskey(self.ctx.name, origin, dest)
        if st is None:
            r.delete(k)
        else:
            r.set(k, json.dumps(st.__dict__))

    # ---------------- closing ----------------
    def _close(self, rt: Route, st: OpenState) -> None:
        if not st:
            self._save_state(rt.origin, rt.dest, None)
            return
        if st.side == "long_origin_short_dest":
            self.order(rt.origin, "sell", qty=st.qty_o, order_type="market", venue=_venue(rt.origin))
            self.order(rt.dest,   "buy",  qty=st.qty_d, order_type="market", venue=_venue(rt.dest))
        else:
            self.order(rt.origin, "buy",  qty=st.qty_o, order_type="market", venue=_venue(rt.origin))
            self.order(rt.dest,   "sell", qty=st.qty_d, order_type="market", venue=_venue(rt.dest))
        self._save_state(rt.origin, rt.dest, None)