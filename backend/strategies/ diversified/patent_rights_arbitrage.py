# backend/strategies/diversified/patent_rights_arbitrage.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import redis
from backend.engine.strategy_base import Strategy

"""
Patent Rights Arbitrage (paper)
-------------------------------
Buy a patent/licensing claim if its *ask* is below the modeled NPV of expected royalties net
of legal costs, then hedge sector/market beta with a short equity basket.

NPV model (annual buckets):
  CF_t = prob_settle * royalty_rate * (addressable_revenue_t)
  addressable_revenue_t = A0 * (1 + g)^t    (or apply an explicit decay if g<0)
  NPV = Σ_{t=1..TERM} CF_t / (1 + disc)^t  - legal_costs
Edge:
  edge_usd = NPV - ask_price

Entry when: edge_usd >= ENTRY_USD and z-score(edge_usd) >= ENTRY_Z
Exit when:  edge_usd <= EXIT_USD, z <= EXIT_Z, or close/cancel flag is set.

Paper symbols you will map in adapters:
  • Patent leg: "PAT:<ID>"
  • Hedge legs: "EQ:<SYM>" (weights from Redis)

Redis feeds (examples after code):
  HSET patent:ask           <ID> <usd>
  HSET patent:royalty_rate  <ID> <decimal>           # e.g., 0.02 = 2% royalty
  HSET patent:A0            <ID> <usd_per_year>      # current addressable revenue
  HSET patent:g             <ID> <decimal>           # annual growth (neg = decay), optional
  HSET patent:term          <ID> <years>             # integer years
  HSET patent:disc          <ID> <decimal>           # discount rate (annual)
  HSET patent:prob          <ID> <decimal>           # probability of settle/license success
  HSET patent:legal_cost    <ID> <usd_present_value> # expected legal cost PV
  HSET patent:basket:<ID>   weights '{"EQ:SYM": w, "EQ:SYM2": w2, ...}'  # hedge beta weights (sum≈1)
  HSET last_price           "EQ:<SYM>" '{"price": <px>}'                  # for hedge pricing

Lifecycle flags:
  SET  patent:closed:<ID> 1      # realized completion (license signed, payout)
  SET  patent:cancel:<ID> 1      # deal scrapped / adverse ruling

Notes:
  • This is a *paper* asset; adapters decide how to simulate execution (e.g., marketplace bid/ask).
  • Hedge basket is optional but recommended to reduce general market drift while waiting.
"""

# ============================ CONFIG (env) ============================
REDIS_HOST = os.getenv("PR_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("PR_REDIS_PORT", "6379"))

PATENT_ID  = os.getenv("PR_ID", "PR-ACME-001").upper()
VENUE_PAT  = os.getenv("PR_VENUE_PAT", "OTC").upper()
VENUE_EQ   = os.getenv("PR_VENUE_EQ", "EXCH").upper()

# Entry/exit thresholds
ENTRY_USD = float(os.getenv("PR_ENTRY_USD", "20000"))  # minimum modeled edge to act
EXIT_USD  = float(os.getenv("PR_EXIT_USD",  "3000"))
ENTRY_Z   = float(os.getenv("PR_ENTRY_Z",   "1.2"))
EXIT_Z    = float(os.getenv("PR_EXIT_Z",    "0.5"))

# Sizing / risk
USD_NOTIONAL_PATENT = float(os.getenv("PR_USD_NOTIONAL_PAT", "50000"))  # purchase size
HEDGE_MULT          = float(os.getenv("PR_HEDGE_MULT", "1.0"))          # scale hedge DV01/beta ~1
MIN_TICKET_USD      = float(os.getenv("PR_MIN_TICKET_USD", "2000"))
MAX_CONCURRENT      = int(os.getenv("PR_MAX_CONCURRENT", "1"))

# Cadence & stats
RECHECK_SECS = int(os.getenv("PR_RECHECK_SECS", "10"))
EWMA_ALPHA   = float(os.getenv("PR_EWMA_ALPHA", "0.06"))

# Redis keys
ASK_HK     = os.getenv("PR_ASK_HK",          "patent:ask")
RATE_HK    = os.getenv("PR_RATE_HK",         "patent:royalty_rate")
A0_HK      = os.getenv("PR_A0_HK",           "patent:A0")
G_HK       = os.getenv("PR_G_HK",            "patent:g")
TERM_HK    = os.getenv("PR_TERM_HK",         "patent:term")
DISC_HK    = os.getenv("PR_DISC_HK",         "patent:disc")
PROB_HK    = os.getenv("PR_PROB_HK",         "patent:prob")
LEGAL_HK   = os.getenv("PR_LEGAL_HK",        "patent:legal_cost")
BASKET_HK  = os.getenv("PR_BASKET_HK",       "patent:basket:{id}")
LAST_PRICE = os.getenv("PR_LAST_PRICE_HK",   "last_price")

CLOSED_KEY = os.getenv("PR_CLOSED_KEY",      "patent:closed:{id}")
CANCEL_KEY = os.getenv("PR_CANCEL_KEY",      "patent:cancel:{id}")

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================ helpers ============================
def _hgetf(hk: str, field: str) -> Optional[float]:
    v = r.hget(hk, field)
    if v is None: return None
    try: return float(v)
    except Exception:
        try:
            j = json.loads(v); 
            # accept {"v": number} style too
            if isinstance(j, dict) and "v" in j: return float(j["v"])
        except Exception: pass
    return None

def _hget_json(hk: str, field: str) -> Optional[dict]:
    raw = r.hget(hk, field)
    if not raw: return None
    try: 
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def _price(sym: str) -> Optional[float]:
    raw = r.hget(LAST_PRICE, sym)
    if not raw: return None
    try:
        j = json.loads(raw)
        return float(j.get("price", 0))
    except Exception:
        try: return float(raw)
        except Exception: return None

def _now_ms() -> int:
    return int(time.time() * 1000)

def _npv(ask: float, rate: float, A0: float, g: float, term: int, disc: float, prob: float, legal_cost: float) -> float:
    """
    Simple annual NPV of royalty stream net legal costs.
    Uses prob as a success probability applied to all cash flows (can refine by phase).
    """
    term = max(0, int(term))
    disc = max(1e-6, float(disc))
    npv_stream = 0.0
    At = A0
    for t in range(1, term + 1):
        cf = prob * rate * At
        npv_stream += cf / ((1.0 + disc) ** t)
        At *= (1.0 + g)
    return npv_stream - legal_cost

def _basket_for(id_: str) -> Dict[str, float]:
    j = _hget_json(BASKET_HK.format(id=id_), "weights")
    if not j: return {}
    # normalize weights to sum to 1 (keep signs)
    s = sum(abs(float(w)) for w in j.values()) or 1.0
    return {k: float(w)/s for k, w in j.items()}

# ============================ EWMA ============================
@dataclass
class EwmaMV:
    mean: float
    var: float
    alpha: float
    def update(self, x: float) -> Tuple[float,float]:
        m0 = self.mean
        self.mean = (1 - self.alpha) * self.mean + self.alpha * x
        self.var  = max(1e-12, (1 - self.alpha) * (self.var + (x - m0) * (x - self.mean)))
        return self.mean, self.var

def _ewma_key(id_: str) -> str:
    return f"pr:ewma:{id_}"

def _load_ewma(id_: str) -> EwmaMV:
    raw = r.get(_ewma_key(id_))
    if raw:
        try:
            o = json.loads(raw)
            return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", EWMA_ALPHA)))
        except Exception: pass
    return EwmaMV(mean=0.0, var=1.0, alpha=EWMA_ALPHA)

def _save_ewma(id_: str, ew: EwmaMV) -> None:
    r.set(_ewma_key(id_), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ============================ state ============================
@dataclass
class OpenState:
    qty_patent: float
    hedge: Dict[str, float]   # sym → qty (signed, negative = short)
    entry_edge_usd: float
    entry_z: float
    ts_ms: int

def _poskey(name: str, id_: str) -> str:
    return f"pr:open:{name}:{id_}"

# ============================ strategy ============================
class PatentRightsArbitrage(Strategy):
    """
    Buy underpriced patent right vs modeled NPV; short equity basket to neutralize beta.
    """
    def __init__(self, name: str = "patent_rights_arbitrage", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_start(self) -> None:
        super().on_start()
        r.hset("strategy:universe", self.ctx.name, json.dumps({
            "patent": f"PAT:{PATENT_ID}", "ts": _now_ms()
        }))

    def on_tick(self, tick: Dict) -> None:
        now = time.time()
        if now - self._last < RECHECK_SECS:
            return
        self._last = now
        self._evaluate()

    # --------------- core ---------------
    def _inputs(self) -> Optional[Tuple[float,float,float,float,int,float,float]]:
        ask   = _hgetf(ASK_HK,   PATENT_ID)
        rate  = _hgetf(RATE_HK,  PATENT_ID)
        A0    = _hgetf(A0_HK,    PATENT_ID)
        g     = _hgetf(G_HK,     PATENT_ID) or 0.0
        term  = int(_hgetf(TERM_HK,  PATENT_ID) or 0)
        disc  = _hgetf(DISC_HK,  PATENT_ID)
        prob  = _hgetf(PROB_HK,  PATENT_ID)
        legal = _hgetf(LEGAL_HK, PATENT_ID) or 0.0
        if None in (ask, rate, A0, disc, prob) or term <= 0 or ask <= 0 or rate < 0 or A0 < 0 or disc <= 0 or prob < 0:
            return None
        return ask, rate, A0, g, term, disc, prob, legal

    def _evaluate(self) -> None:
        vals = self._inputs()
        if not vals: return
        ask, rate, A0, g, term, disc, prob, legal = vals

        npv = _npv(ask, rate, A0, g, term, disc, prob, legal)
        edge = npv - ask

        ew = _load_ewma(PATENT_ID)
        m, v = ew.update(edge)
        _save_ewma(PATENT_ID, ew)
        z = (edge - m) / math.sqrt(max(v, 1e-12))

        # Emit monitor signal (scaled to entry)
        self.emit_signal(max(-1.0, min(1.0, edge / max(1.0, ENTRY_USD))))

        st = self._load_state()

        # Event flags
        if str(r.get(CLOSED_KEY.format(id=PATENT_ID)) or "0") == "1":
            if st: self._close(st)
            return
        if str(r.get(CANCEL_KEY.format(id=PATENT_ID)) or "0") == "1":
            if st: self._close(st)
            return

        # Exits
        if st:
            if (edge <= EXIT_USD) or (abs(z) <= EXIT_Z):
                self._close(st)
            return

        # Entries
        if r.get(_poskey(self.ctx.name, PATENT_ID)) is not None:
            return
        if not (edge >= ENTRY_USD and abs(z) >= ENTRY_Z):
            return

        # Size patent purchase
        qty_pat = max(0.0, USD_NOTIONAL_PATENT / ask)
        if qty_pat * ask < MIN_TICKET_USD:
            return

        # Build hedge basket (optional)
        hedge_qty: Dict[str, float] = {}
        basket = _basket_for(PATENT_ID)
        notional = qty_pat * ask
        for sym, w in basket.items():
            px = _price(sym)
            if px is None or px <= 0: 
                continue
            # Short the basket (negative qty) to hedge; scale by notional * weight
            usd_leg = HEDGE_MULT * notional * w
            qty = (usd_leg / px) * -1.0
            if abs(qty * px) >= MIN_TICKET_USD * 0.3:  # tiny leg guard
                hedge_qty[sym] = qty

        # Place orders
        self.order(f"PAT:{PATENT_ID}", "buy", qty=qty_pat, order_type="market", venue=VENUE_PAT)
        for sym, q in hedge_qty.items():
            side = "sell" if q < 0 else "buy"
            self.order(sym, side, qty=abs(q), order_type="market", venue=VENUE_EQ)

        self._save_state(OpenState(
            qty_patent=qty_pat, hedge=hedge_qty,
            entry_edge_usd=edge, entry_z=z, ts_ms=_now_ms()
        ))

    # --------------- state I/O ---------------
    def _load_state(self) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, PATENT_ID))
        if not raw: return None
        try:
            o = json.loads(raw)
            return OpenState(qty_patent=float(o["qty_patent"]),
                             hedge={k: float(v) for k, v in (o.get("hedge") or {}).items()},
                             entry_edge_usd=float(o["entry_edge_usd"]),
                             entry_z=float(o["entry_z"]),
                             ts_ms=int(o["ts_ms"]))
        except Exception:
            return None

    def _save_state(self, st: Optional[OpenState]) -> None:
        if st is None: return
        r.set(_poskey(self.ctx.name, PATENT_ID), json.dumps({
            "qty_patent": st.qty_patent,
            "hedge": st.hedge,
            "entry_edge_usd": st.entry_edge_usd,
            "entry_z": st.entry_z,
            "ts_ms": st.ts_ms
        }))

    # --------------- close ---------------
    def _close(self, st: OpenState) -> None:
        # unwind hedge first
        for sym, q in (st.hedge or {}).items():
            side = "buy" if q < 0 else "sell"  # reverse
            self.order(sym, side, qty=abs(q), order_type="market", venue=VENUE_EQ)
        # sell patent right
        self.order(f"PAT:{PATENT_ID}", "sell", qty=st.qty_patent, order_type="market", venue=VENUE_PAT)
        r.delete(_poskey(self.ctx.name, PATENT_ID))