# backend/strategies/diversified/synthetic_forward_arbitrage.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional

import redis
from backend.engine.strategy_base import Strategy

"""
Synthetic Forward Arbitrage — paper
-----------------------------------
Modes:
  1) CARRY_BASIS : Spot vs Futures (equity index / commodity)
     F_theo = S * exp( (r - q + c - y) * T )
     Edge_bps = 1e4 * (F_mkt - F_theo)/F_theo  (positive ⇒ future rich)
     Trade DV01‑less notionally: SELL rich leg / BUY cheap leg, hedge with spot.

  2) FX_CIP : Covered Interest Parity (spot vs forward)
     F_theo = S * exp( (r_dom - r_for) * T )
     Same gating and trade logic using SPOT:CCYPAIR and FWD:CCYPAIR:<TENOR>.

Redis feeds you publish elsewhere:
  # Marks
  HSET last_price "<SPOT_SYM>"  '{"price": <S>}'          # e.g., "SPOT:ES" or "SPOT:EURUSD"
  HSET last_price "<FWD_SYM>"   '{"price": <F>}'          # e.g., "FUT:ESZ5" or "FWD:EURUSD:1M"

  # Meta (tenor & carries)
  HSET fwd:meta <TAG> '{
    "mode": "CARRY_BASIS",                 # or "FX_CIP"
    "spot_sym": "SPOT:ES",
    "fwd_sym":  "FUT:ESZ5",
    "start_ms": 1764547200000,             # period start (delivery window start)
    "end_ms":   1767139200000,             # period end (or maturity)
    "r": 0.05,                              # funding APR (domestic for FX)
    "q": 0.012,                             # dividend yield / lease (equity/index) OR foreign rate (FX mode)
    "c": 0.00,                              # storage/borrow APR (optional)
    "y": 0.00,                              # convenience yield APR (optional)
    "mult": 50.0,                           # contract $ per index point (futures)
    "venue_spot": "EXCH",
    "venue_fwd":  "FUT"
  }'

  # Fees (bps on notional)
  HSET fees:carry EXCH  5
  HSET fees:carry FUT   5
  HSET fees:carry FWD   5

  # Ops
  SET risk:halt 0|1

Paper routing (map in adapters later):
  • Spot leg : "SPOT:<TAG>"
  • Futures  : "FUT:<TAG>"
  • FX Fwds  : "FWD:<PAIR>:<TENOR>"
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("SFA_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("SFA_REDIS_PORT", "6379"))

TAG       = os.getenv("SFA_TAG", "ES_DEC25").upper()     # key into fwd:meta
MODE      = os.getenv("SFA_MODE", "CARRY_BASIS").upper() # CARRY_BASIS | FX_CIP

# Thresholds
ENTRY_BPS = float(os.getenv("SFA_ENTRY_BPS", "50"))  # edge needed to enter (bps of fair)
EXIT_BPS  = float(os.getenv("SFA_EXIT_BPS",  "20"))
ENTRY_Z   = float(os.getenv("SFA_ENTRY_Z",   "1.1"))
EXIT_Z    = float(os.getenv("SFA_EXIT_Z",    "0.5"))

# Sizing / risk
USD_NOTIONAL   = float(os.getenv("SFA_USD_NOTIONAL", "50000"))
MIN_TICKET_USD = float(os.getenv("SFA_MIN_TKT", "500"))
MAX_CONCURRENT = int(os.getenv("SFA_MAX_CONC", "1"))

# Cadence / stats
RECHECK_SECS = float(os.getenv("SFA_RECHECK_SECS", "1.0"))
EWMA_ALPHA   = float(os.getenv("SFA_EWMA_ALPHA", "0.06"))

# Redis keys
HALT_KEY   = os.getenv("SFA_HALT_KEY", "risk:halt")
LAST_HK    = os.getenv("SFA_LAST_HK",  "last_price")
META_HK    = os.getenv("SFA_META_HK",  "fwd:meta")
FEES_HK    = os.getenv("SFA_FEES_HK",  "fees:carry")

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================ helpers ============================
def _hget_json(hk: str, field: str) -> Optional[dict]:
    raw = r.hget(hk, field)
    if not raw: return None
    try:
        j = json.loads(raw) # type: ignore
        return j if isinstance(j, dict) else None
    except Exception:
        return None

def _px(sym: str) -> Optional[float]:
    raw = r.hget(LAST_HK, sym)
    if not raw: return None
    try:
        j = json.loads(raw); return float(j.get("price", 0.0)) # type: ignore
    except Exception:
        try: return float(raw) # type: ignore
        except Exception: return None

def _fees_bps(venue: str) -> float:
    v = r.hget(FEES_HK, venue)
    try: return float(v) if v is not None else 5.0 # type: ignore
    except Exception: return 5.0

def _now_ms() -> int: return int(time.time() * 1000)

def _T_years(start_ms: int, end_ms: int) -> float:
    if not (start_ms and end_ms): return 0.25
    days = max(1.0, (end_ms - start_ms) / 86400000.0)
    return days / 365.0

# ============================ EWMA ============================
@dataclass
class EwmaMV:
    mean: float
    var: float
    alpha: float
    def update(self, x: float):
        m0 = self.mean
        self.mean = (1 - self.alpha)*self.mean + self.alpha*x
        self.var  = max(1e-12, (1 - self.alpha)*(self.var + (x - m0)*(x - self.mean)))
        return self.mean, self.var

def _ewma_key(tag: str) -> str:
    return f"sfa:ewma:{tag}"

def _load_ewma(tag: str) -> EwmaMV:
    raw = r.get(_ewma_key(tag))
    if raw:
        try:
            o = json.loads(raw) # type: ignore
            return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", EWMA_ALPHA)))
        except Exception: pass
    return EwmaMV(mean=0.0, var=1.0, alpha=EWMA_ALPHA)

def _save_ewma(tag: str, ew: EwmaMV) -> None:
    r.set(_ewma_key(tag), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ============================ state ============================
@dataclass
class OpenState:
    mode: str
    side: str    # "sell_fwd_buy_spot" or "buy_fwd_sell_spot"
    qty_spot: float
    qty_fwd: float
    entry_bps: float
    entry_z: float
    ts_ms: int

def _poskey(name: str, tag: str) -> str:
    return f"sfa:open:{name}:{tag}"

# ============================ Strategy ============================
class SyntheticForwardArbitrage(Strategy):
    """
    Spot vs forward/futures parity arbitrage (carry & FX CIP).
    """
    def __init__(self, name: str = "synthetic_forward_arbitrage", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_tick(self, tick: Dict) -> None:
        if (r.get(HALT_KEY) or "0") == "1": return
        now = time.time()
        if now - self._last < RECHECK_SECS: return
        self._last = now

        meta = _hget_json(META_HK, TAG) or {}
        if not meta: return

        mode = (meta.get("mode") or MODE).upper()
        spot_sym = str(meta.get("spot_sym", "SPOT:UNKNOWN"))
        fwd_sym  = str(meta.get("fwd_sym",  "FUT:UNKNOWN"))
        venue_spot = str(meta.get("venue_spot", "EXCH"))
        venue_fwd  = str(meta.get("venue_fwd",  "FUT"))
        S = _px(spot_sym); Fmkt = _px(fwd_sym)
        if S is None or Fmkt is None or S <= 0: return

        start_ms = int(meta.get("start_ms", 0) or 0)
        end_ms   = int(meta.get("end_ms", 0) or 0)
        T = _T_years(start_ms, end_ms)

        if mode == "FX_CIP":
            r_d = float(meta.get("r", 0.0))
            r_f = float(meta.get("q", 0.0))   # reuse 'q' field as foreign rate
            Ftheo = S * math.exp((r_d - r_f) * max(1e-6, T))
            contract_mult = float(meta.get("mult", 100000.0))  # FX notional per contract (e.g., 100k base)
        else:
            r_carry = float(meta.get("r", 0.0))
            q_div   = float(meta.get("q", 0.0))
            c_store = float(meta.get("c", 0.0))
            y_conv  = float(meta.get("y", 0.0))
            Ftheo = S * math.exp((r_carry - q_div + c_store - y_conv) * max(1e-6, T))
            contract_mult = float(meta.get("mult", 50.0))      # e.g., ES = $50 per point

        # Edge in bps of fair
        edge_bps = 1e4 * ((Fmkt - Ftheo) / max(1e-9, Ftheo))

        tag = f"{mode}:{spot_sym}|{fwd_sym}"
        ew = _load_ewma(tag); m,v = ew.update(edge_bps); _save_ewma(tag, ew)
        z = (edge_bps - m)/math.sqrt(max(v, 1e-12))

        # dashboard signal
        self.emit_signal(max(-1.0, min(1.0, edge_bps / max(1.0, ENTRY_BPS))))

        st = self._load_state(tag)
        if st:
            if (abs(edge_bps) <= EXIT_BPS) or (abs(z) <= EXIT_Z):
                self._close(tag, st, spot_sym, fwd_sym, venue_spot, venue_fwd)
            return

        if r.get(_poskey(self.ctx.name, tag)) is not None: return
        if not (abs(edge_bps) >= ENTRY_BPS and abs(z) >= ENTRY_Z): return

        # Fees (bps guard)
        fee_spot = _fees_bps(venue_spot) * 1e-4
        fee_fwd  = _fees_bps(venue_fwd)  * 1e-4

        # Size: use USD_NOTIONAL; futures/fwd qty by contract notional, spot qty by dollar
        # Approx futures contract notional ≈ Fmkt * contract_mult
        fut_notional = max(1e-6, Fmkt * contract_mult)
        qty_fwd = max(1.0, math.floor(USD_NOTIONAL / fut_notional))
        if qty_fwd * fut_notional < MIN_TICKET_USD: return
        # Spot hedge in USD ≈ qty_fwd * contract_notional; convert to shares/units ~ (qty_fwd * fut_notional) / S
        qty_spot = (qty_fwd * fut_notional) / S

        if edge_bps > 0:
            # Forward/Future rich → SELL forward, BUY spot (cash-and-carry)
            self.order(fwd_sym,  "sell", qty=qty_fwd,  order_type="market", venue=venue_fwd)
            self.order(spot_sym, "buy",  qty=qty_spot, order_type="market", venue=venue_spot)
            side = "sell_fwd_buy_spot"
        else:
            # Forward/Future cheap → BUY forward, SELL spot (reverse cash-and-carry)
            self.order(fwd_sym,  "buy",  qty=qty_fwd,  order_type="market", venue=venue_fwd)
            self.order(spot_sym, "sell", qty=qty_spot, order_type="market", venue=venue_spot)
            side = "buy_fwd_sell_spot"

        self._save_state(tag, OpenState(mode=mode, side=side, qty_spot=qty_spot, qty_fwd=qty_fwd,
                                        entry_bps=edge_bps, entry_z=z, ts_ms=_now_ms()))

    # ---------------- close / unwind ----------------
    def _close(self, tag: str, st: OpenState, spot_sym: str, fwd_sym: str, venue_spot: str, venue_fwd: str) -> None:
        if st.side == "sell_fwd_buy_spot":
            self.order(fwd_sym,  "buy",  qty=st.qty_fwd,  order_type="market", venue=venue_fwd)
            self.order(spot_sym, "sell", qty=st.qty_spot, order_type="market", venue=venue_spot)
        else:
            self.order(fwd_sym,  "sell", qty=st.qty_fwd,  order_type="market", venue=venue_fwd)
            self.order(spot_sym, "buy",  qty=st.qty_spot, order_type="market", venue=venue_spot)
        r.delete(_poskey(self.ctx.name, tag))

    # ---------------- state I/O ----------------
    def _load_state(self, tag: str) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, tag))
        if not raw: return None
        try:
            o = json.loads(raw) # type: ignore
            return OpenState(mode=str(o["mode"]), side=str(o["side"]),
                             qty_spot=float(o["qty_spot"]), qty_fwd=float(o["qty_fwd"]),
                             entry_bps=float(o["entry_bps"]), entry_z=float(o["entry_z"]),
                             ts_ms=int(o["ts_ms"]))
        except Exception:
            return None

    def _save_state(self, tag: str, st: Optional[OpenState]) -> None:
        if st is None: return
        r.set(_poskey(self.ctx.name, tag), json.dumps({
            "mode": st.mode, "side": st.side,
            "qty_spot": st.qty_spot, "qty_fwd": st.qty_fwd,
            "entry_bps": st.entry_bps, "entry_z": st.entry_z, "ts_ms": st.ts_ms
        }))