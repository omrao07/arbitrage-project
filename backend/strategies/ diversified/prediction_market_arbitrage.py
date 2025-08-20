# backend/strategies/diversified/prediction_market_arbitrage.py
from __future__ import annotations

import json, math, os, time, uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import redis
from backend.engine.strategy_base import Strategy

"""
Prediction Market Arbitrage (paper)
-----------------------------------
Two modes:

1) CROSS_VENUE (same market/outcome across venues)
   • Quote model: prices in [0,1] = probability-style "yes-shares"
   • If p_rich - p_cheap - fees > ENTRY_BPScalc → BUY cheap venue, SELL rich venue
   • Settlement: positions are 0/1 payoff; close on resolution or if spread collapses

2) DUTCH_BOOK (single venue, multi-outcome)
   • If sum(best_ask_outcomes) + fees < 1 → buy all outcomes proportionally
   • Locks in payoff ≥ 1 at resolution, minus fees

All orders are PAPER and routed to adapters via synthetic symbols:
  PM:<VENUE>:<MARKET_ID>:<OUTCOME_ID>   (side=buy/sell; qty = shares; price=limit in [0,1])

Redis feeds (examples below):
  HSET pm:ob:<VENUE>:<MID>:<OID> '{"bid":0.44,"ask":0.46,"bid_sz":800,"ask_sz":900,"fee_bps":80}'
  HSET pm:map:<MID> '{"venueA":"VENUE1","venueB":"VENUE2","outcomes":["YES","NO"]}'  # meta
  HSET pm:resolve <MID> '{"resolved":0|1,"winning":"YES|NO|<OID>"}'
  HSET pm:fees:<VENUE> <bps_default>              # fallback trading fee bps
  SET  risk:halt 0|1

Notes:
 • For venues with "NO" shares instead of shorting, adapters should convert SELL(YES) to BUY(NO) mechanically.
 • Fees are treated as taker bps on notional; you can enhance with maker rebates via adapters.
"""

# -------------------- ENV / CONFIG --------------------
REDIS_HOST = os.getenv("PM_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("PM_REDIS_PORT", "6379"))

MODE        = os.getenv("PM_MODE", "CROSS_VENUE").upper()  # "CROSS_VENUE" | "DUTCH_BOOK"

# Cross-venue params
VENUE_CHEAP = os.getenv("PM_VENUE_CHEAP", "VENUE1").upper()
VENUE_RICH  = os.getenv("PM_VENUE_RICH",  "VENUE2").upper()
MARKET_ID   = os.getenv("PM_MARKET_ID", "M-ABC123").upper()
OUTCOME_ID  = os.getenv("PM_OUTCOME_ID", "YES").upper()

# Dutch-book params
DB_VENUE    = os.getenv("PM_DB_VENUE", "VENUE1").upper()
DB_MARKET   = os.getenv("PM_DB_MARKET", "M-ABC123").upper()

# Thresholds / risk
ENTRY_BPS        = float(os.getenv("PM_ENTRY_BPS", "100"))  # min net bps spread (1% = 100 bps)
EXIT_BPS         = float(os.getenv("PM_EXIT_BPS",  "40"))
ENTRY_Z          = float(os.getenv("PM_ENTRY_Z",   "1.2"))
EXIT_Z           = float(os.getenv("PM_EXIT_Z",    "0.5"))
USD_NOTIONAL     = float(os.getenv("PM_USD_NOTIONAL", "1000"))
MIN_TICKET_USD   = float(os.getenv("PM_MIN_TICKET_USD", "50"))
MAX_CONCURRENT   = int(os.getenv("PM_MAX_CONCURRENT", "1"))
RECHECK_SECS     = float(os.getenv("PM_RECHECK_SECS", "1.5"))
EWMA_ALPHA       = float(os.getenv("PM_EWMA_ALPHA", "0.08"))

# Keys
HALT_KEY         = os.getenv("PM_HALT_KEY", "risk:halt")
OB_KEY_FMT       = os.getenv("PM_OB_KEY_FMT", "pm:ob:{venue}:{mid}:{oid}")
FEES_VENUE_HK    = os.getenv("PM_FEES_VENUE_HK", "pm:fees")
RESOLVE_HK       = os.getenv("PM_RESOLVE_HK", "pm:resolve")
META_HK          = os.getenv("PM_META_HK", "pm:map")

# -------------------- Redis --------------------
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# -------------------- utils --------------------
def _hget_json(hk: str, field: str) -> Optional[dict]:
    raw = r.hget(hk, field)
    if not raw: return None
    try:
        j = json.loads(raw)
        return j if isinstance(j, dict) else None
    except Exception:
        return None

def _fees_bps(venue: str, ob: Optional[dict]) -> float:
    if ob and "fee_bps" in ob:
        try: return float(ob["fee_bps"])
        except Exception: pass
    v = r.hget(FEES_VENUE_HK, venue)
    try: return float(v) if v is not None else 100.0
    except Exception: return 100.0

def _now_ms() -> int: return int(time.time()*1000)

# -------------------- EWMA --------------------
from dataclasses import dataclass
@dataclass
class EwmaMV:
    mean: float
    var: float
    alpha: float
    def update(self, x: float) -> Tuple[float,float]:
        m0 = self.mean
        self.mean = (1 - self.alpha) * self.mean + self.alpha * x
        self.var  = max(1e-12, (1 - self.alpha) * (self.var + (x - m0)*(x - self.mean)))
        return self.mean, self.var

def _ewma_key(tag: str) -> str:
    return f"pm:ewma:{tag}"

def _load_ewma(tag: str) -> EwmaMV:
    raw = r.get(_ewma_key(tag))
    if raw:
        try:
            o = json.loads(raw)
            return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", EWMA_ALPHA)))
        except Exception: pass
    return EwmaMV(mean=0.0, var=1.0, alpha=EWMA_ALPHA)

def _save_ewma(tag: str, ew: EwmaMV) -> None:
    r.set(_ewma_key(tag), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# -------------------- state --------------------
@dataclass
class OpenState:
    mode: str
    legs: List[dict]      # [{"sym":..., "side":"buy/sell", "qty":..., "price":...}, ...]
    edge_bps: float
    entry_z: float
    ts_ms: int

def _poskey(name: str, tag: str) -> str:
    return f"pm:open:{name}:{tag}"

# -------------------- Strategy --------------------
class PredictionMarketArbitrage(Strategy):
    """
    CROSS_VENUE: buy cheap venue, sell rich venue (same outcome)
    DUTCH_BOOK : buy all outcomes at sum(ask)<1 to lock payout
    """
    def __init__(self, name: str = "prediction_market_arbitrage", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_tick(self, tick: Dict) -> None:
        if (r.get(HALT_KEY) or "0") == "1": return
        now = time.time()
        if now - self._last < RECHECK_SECS: return
        self._last = now

        if MODE == "CROSS_VENUE":
            self._eval_cross()
        else:
            self._eval_dutch()

    # -------- CROSS_VENUE --------
    def _eval_cross(self) -> None:
        tag = f"{VENUE_CHEAP}:{VENUE_RICH}:{MARKET_ID}:{OUTCOME_ID}"
        st = self._load_state(tag)
        if self._resolved(MARKET_ID):
            if st: self._close(tag, st)
            return

        ob_c = _hget_json(OB_KEY_FMT.format(venue=VENUE_CHEAP, mid=MARKET_ID, oid=OUTCOME_ID), OUTCOME_ID) or \
               _hget_json(OB_KEY_FMT.format(venue=VENUE_CHEAP, mid=MARKET_ID, oid=OUTCOME_ID), "book")
        ob_r = _hget_json(OB_KEY_FMT.format(venue=VENUE_RICH,  mid=MARKET_ID, oid=OUTCOME_ID), OUTCOME_ID) or \
               _hget_json(OB_KEY_FMT.format(venue=VENUE_RICH,  mid=MARKET_ID, oid=OUTCOME_ID), "book")
        if not (ob_c and ob_r): return

        ask_c  = float(ob_c.get("ask", 0) or 0)
        bid_r  = float(ob_r.get("bid", 0) or 0)
        if ask_c <= 0 or bid_r <= 0: return

        f_c = _fees_bps(VENUE_CHEAP, ob_c) * 1e-4
        f_r = _fees_bps(VENUE_RICH,  ob_r) * 1e-4

        # Net executable spread (sell rich, buy cheap), in probability points
        # Profit per share at resolution (if YES wins) is 1 - buy - sell_fee - buy_fee, but cross-venue carry cancels in expectation.
        # Here we approximate edge by immediate carry: bid_r*(1 - f_r) - ask_c*(1 + f_c)
        edge = bid_r*(1 - f_r) - ask_c*(1 + f_c)
        edge_bps = 1e4 * edge  # since prices are 0..1

        ew = _load_ewma(tag); m,v = ew.update(edge_bps); _save_ewma(tag, ew)
        z = (edge_bps - m)/math.sqrt(max(v,1e-12))

        self.emit_signal(max(-1.0, min(1.0, edge_bps / max(1.0, ENTRY_BPS))))

        # exits
        if st:
            if (edge_bps <= EXIT_BPS) or (abs(z) <= EXIT_Z):
                self._close(tag, st)
            return

        if r.get(_poskey(self.ctx.name, tag)) is not None:
            return
        if not (edge_bps >= ENTRY_BPS and abs(z) >= ENTRY_Z):
            return

        # size (shares are USD‑like since payoff=1 per share if YES occurs)
        qty = max(0.0, USD_NOTIONAL)  # each share notionally = $1 payoff
        if qty < (MIN_TICKET_USD / 1.0):  # $1 per share
            return

        # place legs: BUY cheap, SELL rich (limitable around quotes)
        sym_c = f"PM:{VENUE_CHEAP}:{MARKET_ID}:{OUTCOME_ID}"
        sym_r = f"PM:{VENUE_RICH}:{MARKET_ID}:{OUTCOME_ID}"
        self.order(sym_c, "buy",  qty=qty, price=ask_c, order_type="limit", venue=VENUE_CHEAP)
        self.order(sym_r, "sell", qty=qty, price=bid_r, order_type="limit", venue=VENUE_RICH)

        legs = [
            {"sym": sym_c, "side": "buy",  "qty": qty, "price": ask_c},
            {"sym": sym_r, "side": "sell", "qty": qty, "price": bid_r},
        ]
        self._save_state(tag, OpenState(mode="CROSS_VENUE", legs=legs, edge_bps=edge_bps, entry_z=z, ts_ms=_now_ms()))

    # -------- DUTCH_BOOK --------
    def _eval_dutch(self) -> None:
        tag = f"DUTCH:{DB_VENUE}:{DB_MARKET}"
        st = self._load_state(tag)
        if self._resolved(DB_MARKET):
            if st: self._close(tag, st)
            return

        meta = _hget_json(META_HK, DB_MARKET) or {}
        outcomes: List[str] = list(meta.get("outcomes") or [])
        if not outcomes:
            # Try to discover outcomes by scanning small list? (keep simple: require meta)
            return

        asks: Dict[str, Tuple[float, float]] = {}  # outcome -> (ask, ask_sz)
        total_ask = 0.0
        for oid in outcomes:
            ob = _hget_json(OB_KEY_FMT.format(venue=DB_VENUE, mid=DB_MARKET, oid=oid), oid)
            if not ob: return
            a = float(ob.get("ask", 0) or 0); asz = float(ob.get("ask_sz", 0) or 0)
            if a <= 0 or asz <= 0: return
            asks[oid] = (a, asz)
            total_ask += a

        f = _fees_bps(DB_VENUE, None) * 1e-4
        net_sum = total_ask * (1 + f)
        edge = 1.0 - net_sum  # positive = profit margin per $1 payout
        edge_bps = 1e4 * edge

        ew = _load_ewma(tag); m,v = ew.update(edge_bps); _save_ewma(tag, ew)
        z = (edge_bps - m)/math.sqrt(max(v,1e-12))

        self.emit_signal(max(-1.0, min(1.0, edge_bps / max(1.0, ENTRY_BPS))))

        # exits
        if st:
            if (edge_bps <= EXIT_BPS) or (abs(z) <= EXIT_Z):
                self._close(tag, st)
            return

        if r.get(_poskey(self.ctx.name, tag)) is not None:
            return
        if not (edge_bps >= ENTRY_BPS and abs(z) >= ENTRY_Z and edge > 0):
            return

        # Allocate notional across outcomes proportional to (1 / ask) to equalize payoff
        # Each outcome share pays 1 if it wins, 0 otherwise. To guarantee ≥1 payoff, buy weights w_i s.t. sum(w_i * ask_i) = B and sum(w_i) >= 1.
        # Simple construction: target per-outcome shares = k / ask_i ; choose k so total cost = USD_NOTIONAL.
        denom = sum((1.0 / asks[o][0]) for o in outcomes)
        if denom <= 0: return
        k = USD_NOTIONAL / denom
        legs = []
        for oid in outcomes:
            ask_i, asz = asks[oid]
            shares = min(k / ask_i, asz)  # cap by available size
            if shares * ask_i < MIN_TICKET_USD * 0.2:  # tiny guard
                continue
            sym = f"PM:{DB_VENUE}:{DB_MARKET}:{oid}"
            self.order(sym, "buy", qty=shares, price=ask_i, order_type="limit", venue=DB_VENUE)
            legs.append({"sym": sym, "side": "buy", "qty": shares, "price": ask_i})

        if not legs: return
        self._save_state(tag, OpenState(mode="DUTCH_BOOK", legs=legs, edge_bps=edge_bps, entry_z=z, ts_ms=_now_ms()))

    # -------- resolution & close --------
    def _resolved(self, mid: str) -> bool:
        j = _hget_json(RESOLVE_HK, mid)
        return bool(j and int(j.get("resolved", 0)) == 1)

    def _close(self, tag: str, st: OpenState) -> None:
        # In practice you cannot "close" after resolution; positions cash-settle automatically.
        # Here we just clear state. If you want pre‑resolution unwind, add SELL orders for bought legs, BUY back sells, etc.
        r.delete(_poskey(self.ctx.name, tag))

    # -------- state io --------
    def _load_state(self, tag: str) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, tag))
        if not raw: return None
        try:
            o = json.loads(raw)
            return OpenState(mode=str(o["mode"]),
                             legs=list(o.get("legs") or []),
                             edge_bps=float(o["edge_bps"]),
                             entry_z=float(o["entry_z"]),
                             ts_ms=int(o["ts_ms"]))
        except Exception:
            return None

    def _save_state(self, tag: str, st: OpenState) -> None:
        r.set(_poskey(self.ctx.name, tag), json.dumps({
            "mode": st.mode, "legs": st.legs, "edge_bps": st.edge_bps,
            "entry_z": st.entry_z, "ts_ms": st.ts_ms
        }))