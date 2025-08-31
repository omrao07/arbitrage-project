# backend/strategies/diversified/climate_to_commodity.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import redis
from backend.engine.strategy_base import Strategy

"""
Climate → Commodity — paper
---------------------------
Idea:
  • Map climate anomalies to supply/demand shocks for commodities.
  • Signals from: ENSO (ONI), IOD, regional drought/precip anomalies, HDD/CDD shocks.
  • Per‑commodity weights per region turn climate → expected price impact.
  • Trade outright or as spreads (e.g., Corn vs Wheat) with z‑score gates.

Redis you publish elsewhere (examples):

  # Global indices (monthly or weekly updates)
  HSET climate:global ONI 1.2               # Oceanic Niño Index (°C)
  HSET climate:global IOD 0.5               # Indian Ocean Dipole
  HSET climate:global AMO 0.1               # Atlantic Multidecadal Oscillation (optional)

  # Regional anomalies (rolling 30d vs climatology)
  HSET climate:region "US-MW" '{"precip_z":-1.1,"soil_moist_z":-0.8,"temp_z":0.6}'
  HSET climate:region "BR-CEN" '{"precip_z":-0.9,"soil_moist_z":-0.7,"temp_z":0.2}'
  HSET climate:region "IN-GANG" '{"precip_z":0.4,"soil_moist_z":0.1,"temp_z":-0.1}'
  HSET climate:hddcdd "US-E" '{"hdd_z":-0.5,"cdd_z":1.4}'

  # Commodity mapping (where climate matters + sign)
  # weights sum loosely to 1 by commodity; positive → higher price when anomaly positive
  HSET climate:map "CORN"  '{"regions":{"US-MW":0.6,"BR-CEN":0.4},"drivers":{"precip_z":-0.7,"soil_moist_z":-0.3,"temp_z":0.2},"global":{"ONI":0.3}}'
  HSET climate:map "WHEAT" '{"regions":{"US-MW":0.5},"drivers":{"precip_z":-0.6,"soil_moist_z":-0.4},"global":{"ONI":0.2}}'
  HSET climate:map "SOY"   '{"regions":{"US-MW":0.5,"BR-CEN":0.5},"drivers":{"precip_z":-0.5,"soil_moist_z":-0.5},"global":{"ONI":0.25}}'
  HSET climate:map "COFFEE"'{"regions":{"BR-CEN":1.0},"drivers":{"precip_z":-0.7,"temp_z":0.3},"global":{"IOD":0.2}}'
  HSET climate:map "NG"    '{"regions":{"US-E":1.0},"drivers":{"hdd_z":0.6,"cdd_z":0.6},"global":{}}'

  # Last prices for routing sanity
  HSET last_price "CMD:CORN"   '{"price": 4.85}'
  HSET last_price "CMD:WHEAT"  '{"price": 6.10}'
  HSET last_price "CMD:SOY"    '{"price": 12.7}'
  HSET last_price "CMD:COFFEE" '{"price": 2.10}'
  HSET last_price "CMD:NG"     '{"price": 2.35}'

  # Fees and kill
  HSET fees:cmd EXCH 3     # bps on notional (paper guard)
  SET  risk:halt 0|1

Routing (paper; adapters wire later):
  order("CMD:<SYM>", side, qty, order_type="market", venue="EXCH")
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("C2C_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("C2C_REDIS_PORT", "6379"))

MODE = os.getenv("C2C_MODE", "OUTRIGHT").upper()           # OUTRIGHT | SPREAD
UNIVERSE = (os.getenv("C2C_UNIVERSE", "CORN,WHEAT,SOY,COFFEE,NG")).upper().split(",")

# z-score gates
ENTRY_Z = float(os.getenv("C2C_ENTRY_Z", "0.9"))
EXIT_Z  = float(os.getenv("C2C_EXIT_Z",  "0.3"))

# sizing / risk
USD_NOTIONAL_PER_NAME = float(os.getenv("C2C_USD_NOTIONAL_PER_NAME", "4000"))
MIN_TICKET_USD        = float(os.getenv("C2C_MIN_TICKET_USD", "200"))
MAX_NAMES             = int(os.getenv("C2C_MAX_NAMES", "6"))
LOT                   = float(os.getenv("C2C_LOT", "1"))

# stats / cadence
RECHECK_SECS = float(os.getenv("C2C_RECHECK_SECS", "1.2"))
EWMA_ALPHA   = float(os.getenv("C2C_EWMA_ALPHA", "0.08"))

# spreads (only if MODE=SPREAD); define pairs and signs (+1 long first, -1 short second)
SPREADS = os.getenv("C2C_SPREADS", "CORN-WHEAT,SOY-CORN").upper().split(",")

# Redis keys
HALT_KEY   = os.getenv("C2C_HALT_KEY", "risk:halt")
GLOBAL_HK  = os.getenv("C2C_GLOBAL_HK", "climate:global")
REGION_HK  = os.getenv("C2C_REGION_HK", "climate:region")
HDDCDD_HK  = os.getenv("C2C_HDDCDD_HK", "climate:hddcdd")
MAP_HK     = os.getenv("C2C_MAP_HK", "climate:map")
LAST_HK    = os.getenv("C2C_LAST_HK", "last_price")
FEES_HK    = os.getenv("C2C_FEES_HK", "fees:cmd")

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================ helpers ============================
def _hget_json(hk: str, field: str) -> Optional[dict]:
    raw = r.hget(hk, field)
    if not raw: return None
    try:
        j = json.loads(raw); return j if isinstance(j, dict) else None # type: ignore
    except Exception:
        return None

def _px(sym: str) -> Optional[float]:
    raw = r.hget(LAST_HK, f"CMD:{sym}")
    if not raw: return None
    try:
        j = json.loads(raw); return float(j.get("price", 0.0)) # type: ignore
    except Exception:
        try: return float(raw) # type: ignore
        except Exception: return None

def _fees_bps(venue: str="EXCH") -> float:
    v = r.hget(FEES_HK, venue)
    try: return float(v) if v is not None else 3.0 # type: ignore
    except Exception: return 3.0

def _now_ms() -> int: return int(time.time()*1000)

# ============================ EWMA ============================
@dataclass
class EwmaMV:
    mean: float
    var: float
    alpha: float
    def update(self, x: float) -> Tuple[float,float]:
        m0 = self.mean
        self.mean = (1 - self.alpha)*self.mean + self.alpha*x
        self.var  = max(1e-12, (1 - self.alpha)*(self.var + (x - m0)*(x - self.mean)))
        return self.mean, self.var

def _ewma_key(tag: str) -> str: return f"c2c:ewma:{tag}"
def _load_ewma(tag: str) -> EwmaMV:
    raw = r.get(_ewma_key(tag))
    if raw:
        try:
            o = json.loads(raw); return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", EWMA_ALPHA))) # type: ignore
        except Exception: pass
    return EwmaMV(mean=0.0, var=1.0, alpha=EWMA_ALPHA)
def _save_ewma(tag: str, ew: EwmaMV) -> None:
    r.set(_ewma_key(tag), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ============================ state ============================
@dataclass
class OpenState:
    side: str     # "long" | "short" | "long_spread" | "short_spread"
    qty1: float
    qty2: float
    z_at_entry: float
    ts_ms: int

def _poskey(name: str, tag: str) -> str: return f"c2c:open:{name}:{tag}"

# ============================ core scoring ============================
def _score_commodity(sym: str) -> Optional[float]:
    """
    Build a climate score for a commodity from: global indices + regional drivers.
    Returns a raw score (unbounded). Cross-sectional z is applied later.
    """
    cfg = _hget_json(MAP_HK, sym)
    if not cfg: return None

    global_idx = r.hgetall(GLOBAL_HK) or {}    # strings -> numeric
    gsum = 0.0
    for k, w in (cfg.get("global") or {}).items():
        try:
            val = float(global_idx.get(k, 0.0)); gsum += float(w) * val # type: ignore
        except Exception:
            continue

    rsum = 0.0
    for region, rw in (cfg.get("regions") or {}).items():
        reg = _hget_json(REGION_HK, region) or _hget_json(HDDCDD_HK, region) or {}
        drivers: Dict[str, float] = (cfg.get("drivers") or {})
        acc = 0.0
        for dname, dw in drivers.items():
            try:
                dv = float(reg.get(dname, 0.0))
                acc += float(dw) * dv
            except Exception:
                continue
        rsum += float(rw) * acc

    return gsum + rsum

# ============================ Strategy ============================
class ClimateToCommodity(Strategy):
    """
    Converts climate anomalies to commodity trades (outright or spreads) with z-gates.
    """
    def __init__(self, name: str = "climate_to_commodity", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_tick(self, tick: Dict) -> None:
        if (r.get(HALT_KEY) or "0") == "1": return
        now = time.time()
        if now - self._last < RECHECK_SECS: return
        self._last = now

        if MODE == "SPREAD":
            self._run_spreads()
        else:
            self._run_outright()

    # --------------- OUTRIGHT ---------------
    def _run_outright(self) -> None:
        scores: Dict[str, float] = {}
        for s in UNIVERSE:
            sc = _score_commodity(s)
            if sc is not None:
                scores[s] = sc
        if not scores: 
            self.emit_signal(0.0); 
            return

        # cross-sectional z
        vals = list(scores.values())
        mu = sum(vals)/len(vals)
        var = sum((x-mu)*(x-mu) for x in vals)/max(1.0, len(vals)-1.0)
        sd = math.sqrt(max(1e-9, var))
        zmap = {s: (scores[s]-mu)/sd for s in scores}

        # manage existing
        open_tags: List[str] = []
        for s in UNIVERSE:
            tag = f"CMD:{s}"
            st = self._load_state(tag)
            if st:
                open_tags.append(tag)
                z = zmap.get(s, 0.0)
                if abs(z) <= EXIT_Z:
                    self._close_out(tag, st, s)

        # entries
        fee = _fees_bps("EXCH") * 1e-4
        n_open = len(open_tags)
        # sort by |z| desc
        for s, z in sorted(zmap.items(), key=lambda kv: abs(kv[1]), reverse=True):
            if n_open >= MAX_NAMES: break
            tag = f"CMD:{s}"
            if r.get(_poskey(self.ctx.name, tag)) is not None: continue
            if abs(z) < ENTRY_Z: continue
            px = _px(s)
            if not px or px <= 0: continue
            qty = math.floor((USD_NOTIONAL_PER_NAME / px) / max(1.0, LOT)) * LOT
            if qty <= 0 or qty*px < MIN_TICKET_USD: continue

            side = "buy" if z > 0 else "sell"
            self.order(f"CMD:{s}", side, qty=qty, order_type="market", venue="EXCH")
            self._save_state(tag, OpenState(side=("long" if z>0 else "short"),
                                            qty1=qty, qty2=0.0, z_at_entry=z, ts_ms=_now_ms()))
            n_open += 1

        # dashboard = average |z|
        avgabs = sum(abs(z) for z in zmap.values())/len(zmap)
        self.emit_signal(max(-1.0, min(1.0, avgabs/3.0)))

    def _close_out(self, tag: str, st: OpenState, sym: str) -> None:
        if st.side == "long":
            self.order(f"CMD:{sym}", "sell", qty=st.qty1, order_type="market", venue="EXCH")
        else:
            self.order(f"CMD:{sym}", "buy",  qty=st.qty1, order_type="market", venue="EXCH")
        r.delete(_poskey(self.ctx.name, tag))

    # --------------- SPREAD ---------------
    def _run_spreads(self) -> None:
        # Build per‑name z first
        scores = {s: _score_commodity(s) for s in UNIVERSE}
        scores = {k:v for k,v in scores.items() if v is not None}
        if not scores: self.emit_signal(0.0); return
        vals = list(scores.values()); mu = sum(vals)/len(vals)
        var = sum((x-mu)*(x-mu) for x in vals)/max(1.0, len(vals)-1.0)
        sd = math.sqrt(max(1e-9, var)); zmap = {s:(scores[s]-mu)/sd for s in scores}

        # manage exits & entries per defined pair
        for pair in SPREADS:
            try:
                a,b = pair.split("-")
                a=a.strip(); b=b.strip()
            except Exception:
                continue
            tag = f"SPR:{a}-{b}"
            st = self._load_state(tag)
            z_a = zmap.get(a, 0.0); z_b = zmap.get(b, 0.0)
            edge = z_a - z_b  # positive ⇒ a richer climate tailwind than b
            ew = _load_ewma(tag); m,v = ew.update(edge); _save_ewma(tag, ew)
            z_edge = (edge - m)/math.sqrt(max(v,1e-12))

            # exits
            if st and (abs(z_edge) <= EXIT_Z):
                if st.side == "long_spread":
                    self.order(f"CMD:{a}", "sell", qty=st.qty1, order_type="market", venue="EXCH")
                    self.order(f"CMD:{b}", "buy",  qty=st.qty2, order_type="market", venue="EXCH")
                else:
                    self.order(f"CMD:{a}", "buy",  qty=st.qty1, order_type="market", venue="EXCH")
                    self.order(f"CMD:{b}", "sell", qty=st.qty2, order_type="market", venue="EXCH")
                r.delete(_poskey(self.ctx.name, tag))
                continue

            # entries
            if r.get(_poskey(self.ctx.name, tag)) is not None: continue
            if abs(z_edge) < ENTRY_Z: continue

            pxa = _px(a); pxb = _px(b)
            if not pxa or not pxb or pxa<=0 or pxb<=0: continue
            qa = max(1.0, math.floor((USD_NOTIONAL_PER_NAME/pxa)/max(1.0, LOT)))*LOT
            qb = max(1.0, math.floor((USD_NOTIONAL_PER_NAME/pxb)/max(1.0, LOT)))*LOT
            if qa*pxa < MIN_TICKET_USD or qb*pxb < MIN_TICKET_USD: continue

            if z_edge > 0:
                # long A / short B
                self.order(f"CMD:{a}", "buy",  qty=qa, order_type="market", venue="EXCH")
                self.order(f"CMD:{b}", "sell", qty=qb, order_type="market", venue="EXCH")
                side = "long_spread"
            else:
                self.order(f"CMD:{a}", "sell", qty=qa, order_type="market", venue="EXCH")
                self.order(f"CMD:{b}", "buy",  qty=qb, order_type="market", venue="EXCH")
                side = "short_spread"

            self._save_state(tag, OpenState(side=side, qty1=qa, qty2=qb, z_at_entry=z_edge, ts_ms=_now_ms()))
            # emit live signal per pair
            self.emit_signal(max(-1.0, min(1.0, z_edge/3.0)))

    # --------------- state I/O ---------------
    def _load_state(self, tag: str) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, tag))
        if not raw: return None
        try:
            o = json.loads(raw) # type: ignore
            return OpenState(side=str(o["side"]), qty1=float(o["qty1"]), qty2=float(o.get("qty2",0.0)),
                             z_at_entry=float(o["z_at_entry"]), ts_ms=int(o["ts_ms"]))
        except Exception:
            return None

    def _save_state(self, tag: str, st: Optional[OpenState]) -> None:
        if st is None: return
        r.set(_poskey(self.ctx.name, tag), json.dumps({
            "side": st.side, "qty1": st.qty1, "qty2": st.qty2,
            "z_at_entry": st.z_at_entry, "ts_ms": st.ts_ms
        }))