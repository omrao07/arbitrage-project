# backend/strategies/diversified/spin_off_arbitrage.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import redis
from backend.engine.strategy_base import Strategy

"""
Spin-off Arbitrage — paper
--------------------------
Assume a spin-off where each parent share receives r = (ratio_num/ratio_den) shares of SpinCo.
When-issued (WI) tickers often trade before record date/when-issued period.

Identities (ignoring frictions/withholding):
  P_REG  ≈  P_WI + r * S_WI                      (triangle identity)
  STUB   :=  P_REG - r * S_WI
  Fair check: P_WI ≈ STUB

Two modes:

1) TRIANGLE_WI:
   Edge_unit = P_REG_exec_sell - (P_WI_exec_buy + r*S_WI_exec_buy)   (positive ⇒ P_REG rich)
   Trade legs accordingly:
     • If Edge > ENTRY → SELL P_REG; BUY P_WI and BUY r*S_WI
     • If Edge < -ENTRY → BUY P_REG; SELL P_WI and SELL r*S_WI
   Borrow guards for SELL legs are checked.

2) STUB_PAIR:
   Stub_exec = P_REG_exec_buy - r*S_WI_exec_sell  (to buy the stub)
   Edge_unit = P_WI_exec_sell - Stub_exec         (positive ⇒ P_WI rich vs stub)
   Trade:
     • Edge > ENTRY → SELL P_WI ; BUY P_REG ; SELL r*S_WI
     • Edge < -ENTRY → BUY P_WI ; SELL P_REG ; BUY r*S_WI

All orders are PAPER and routed via synthetic symbols:
  EQ:REG:<PARENT>     parent regular line
  EQ:WI_P:<PARENT>    parent when-issued
  EQ:WI_S:<SPIN>      spinco when-issued

Redis feeds (publish via adapters):
  # Last prices (can be mid or tradable mid)
  HSET last_price "EQ:REG:<P>"  '{"price": <px>}'
  HSET last_price "EQ:WI_P:<P>" '{"price": <px>}'
  HSET last_price "EQ:WI_S:<S>" '{"price": <px>}'

  # Meta
  HSET spinoff:meta <TAG> '{"parent":"PARENT","spin":"SPIN","ratio_num":<a>,"ratio_den":<b>,"record_ms":<ms>,"wi_start_ms":<ms>,"dist_ms":<ms>}'

  # Fees/borrow/funding (bps or APR)
  HSET fees:eq <VENUE> <bps>                         # taker guard
  HSET borrow:ok "EQ:REG:<P>" 0|1
  HSET borrow:ok "EQ:WI_P:<P>" 0|1
  HSET borrow:ok "EQ:WI_S:<S>" 0|1
  HSET borrow:fee "<SYM>" <apr_decimal>              # optional carry for shorts over horizon
  HSET funding:cash <CCY> <apr_decimal>              # optional if you want carry guards
  SET  risk:halt 0|1
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("SPIN_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("SPIN_REDIS_PORT", "6379"))

MODE        = os.getenv("SPIN_MODE", "TRIANGLE_WI").upper()   # TRIANGLE_WI | STUB_PAIR
TAG         = os.getenv("SPIN_TAG", "ACME-2025").upper()      # key for meta
VENUE_EQ    = os.getenv("SPIN_VENUE_EQ", "EXCH").upper()
CCY         = os.getenv("SPIN_CCY", "USD").upper()

# Thresholds
ENTRY_BPS   = float(os.getenv("SPIN_ENTRY_BPS", "60"))   # bps of parent price
EXIT_BPS    = float(os.getenv("SPIN_EXIT_BPS",  "25"))
ENTRY_Z     = float(os.getenv("SPIN_ENTRY_Z",   "1.1"))
EXIT_Z      = float(os.getenv("SPIN_EXIT_Z",    "0.5"))

# Sizing / risk
USD_NOTIONAL   = float(os.getenv("SPIN_USD_NOTIONAL", "30000"))
MIN_TICKET_USD = float(os.getenv("SPIN_MIN_TICKET_USD", "200"))
MAX_CONCURRENT = int(os.getenv("SPIN_MAX_CONCURRENT", "1"))

# Cadence
RECHECK_SECS   = float(os.getenv("SPIN_RECHECK_SECS", "1.0"))
EWMA_ALPHA     = float(os.getenv("SPIN_EWMA_ALPHA", "0.06"))

# Redis keys
HALT_KEY    = os.getenv("SPIN_HALT_KEY", "risk:halt")
LAST_HK     = os.getenv("SPIN_LAST_HK", "last_price")
META_HK     = os.getenv("SPIN_META_HK", "spinoff:meta")
FEES_HK     = os.getenv("SPIN_FEES_HK", "fees:eq")
BORROW_OK_HK= os.getenv("SPIN_BORROW_OK_HK", "borrow:ok")
BORROW_FEE_HK=os.getenv("SPIN_BORROW_FEE_HK","borrow:fee")
FUND_HK     = os.getenv("SPIN_FUND_HK", "funding:cash")

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================ helpers ============================
def _hget_json(hk: str, field: str) -> Optional[dict]:
    raw = r.hget(hk, field)
    if not raw: return None
    try: j = json.loads(raw);  return j if isinstance(j, dict) else None # type: ignore
    except Exception: return None

def _px(sym: str) -> Optional[float]:
    raw = r.hget(LAST_HK, sym)
    if not raw: return None
    try:
        j = json.loads(raw); return float(j.get("price", 0)) # type: ignore
    except Exception:
        try: return float(raw) # type: ignore
        except Exception: return None

def _fees_bps() -> float:
    v = r.hget(FEES_HK, VENUE_EQ)
    try: return float(v) if v is not None else 10.0 # type: ignore
    except Exception: return 10.0

def _borrow_ok(sym: str) -> bool:
    v = r.hget(BORROW_OK_HK, sym)
    return False if v is not None and str(v) == "0" else True

def _borrow_apr(sym: str) -> float:
    v = r.hget(BORROW_FEE_HK, sym)
    try: return float(v) if v is not None else 0.0 # type: ignore
    except Exception: return 0.0

def _fund_apr() -> float:
    v = r.hget(FUND_HK, CCY)
    try: return float(v) if v is not None else 0.0 # type: ignore
    except Exception: return 0.0

def _now_ms() -> int: return int(time.time() * 1000)

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

def _ewma_key(tag: str) -> str:
    return f"spin:ewma:{tag}"

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
    side: str        # labels like "sell_reg_buy_combo" or "sell_wi_buy_stub"
    qty_reg: float
    qty_wip: float
    qty_wis: float
    entry_bps: float
    entry_z: float
    ts_ms: int

def _poskey(name: str, tag: str) -> str:
    return f"spin:open:{name}:{tag}"

# ============================ strategy ============================
class SpinOffArbitrage(Strategy):
    """
    Triangle WI arbitrage and Parent‑stub pair, with borrow/funding guards.
    """
    def __init__(self, name: str = "spin_off_arbitrage", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_tick(self, tick: Dict) -> None:
        if (r.get(HALT_KEY) or "0") == "1": return
        now = time.time()
        if now - self._last < RECHECK_SECS: return
        self._last = now

        meta = _hget_json(META_HK, TAG) or {}
        P  = (meta.get("parent") or "PARENT").upper()
        S  = (meta.get("spin") or "SPIN").upper()
        ratio_num = float(meta.get("ratio_num", 1.0))
        ratio_den = float(meta.get("ratio_den", 5.0))
        r_ratio   = ratio_num / max(1.0, ratio_den)

        sym_reg = f"EQ:REG:{P}"
        sym_wip = f"EQ:WI_P:{P}"
        sym_wis = f"EQ:WI_S:{S}"

        p_reg = _px(sym_reg)
        p_wip = _px(sym_wip)
        p_wis = _px(sym_wis)
        if None in (p_reg, p_wip, p_wis): return

        if MODE == "TRIANGLE_WI":
            self._eval_triangle(P, S, r_ratio, sym_reg, sym_wip, sym_wis, p_reg, p_wip, p_wis) # type: ignore
        else:
            self._eval_stubpair(P, S, r_ratio, sym_reg, sym_wip, sym_wis, p_reg, p_wip, p_wis) # type: ignore

    # --------------- TRIANGLE (P_REG vs P_WI + r*S_WI) ---------------
    def _eval_triangle(self, P: str, S: str, r_ratio: float, sym_reg: str, sym_wip: str, sym_wis: str,
                       p_reg: float, p_wip: float, p_wis: float) -> None:
        tag = f"TRI:{P}|{S}"

        fees = _fees_bps() * 1e-4
        # Executable combo cost to buy synthetic REG: buy P_WI and r*S_WI
        combo_buy = p_wip * (1 + fees) + (r_ratio * p_wis) * (1 + fees)
        combo_sell= p_wip * (1 - fees) + (r_ratio * p_wis) * (1 - fees)

        # Edge quoted in bps of p_reg
        edge_unit = (p_reg * (1 - fees)) - combo_buy
        edge_bps  = 1e4 * (edge_unit / max(1e-6, p_reg))

        ew = _load_ewma(tag); m,v = ew.update(edge_bps); _save_ewma(tag, ew)
        z = (edge_bps - m) / math.sqrt(max(v,1e-12))
        self.emit_signal(max(-1.0, min(1.0, abs(edge_bps)/max(1.0, ENTRY_BPS))))

        st = self._load_state(tag)
        if st:
            if (abs(edge_bps) <= EXIT_BPS) or (abs(z) <= EXIT_Z):
                self._close(tag, st, sym_reg, sym_wip, sym_wis)
            return

        if r.get(_poskey(self.ctx.name, tag)) is not None: return
        if not (abs(edge_bps) >= ENTRY_BPS and abs(z) >= ENTRY_Z): return

        # Borrow guards for any SELL legs
        need_short_reg = edge_bps > 0
        need_short_wip = edge_bps < 0
        need_short_wis = edge_bps < 0

        if (need_short_reg and not _borrow_ok(sym_reg)) or \
           (need_short_wip and not _borrow_ok(sym_wip)) or \
           (need_short_wis and not _borrow_ok(sym_wis)):
            return

        # Size by USD_NOTIONAL on parent leg
        usd = USD_NOTIONAL
        if usd < MIN_TICKET_USD: return
        qty_reg = usd / max(1e-6, p_reg)
        qty_wip = qty_reg
        qty_wis = qty_reg * r_ratio

        if edge_bps > 0:
            # Parent REG rich: SELL REG, BUY WIP, BUY r*S_WI
            self.order(sym_reg, "sell", qty=qty_reg, order_type="market", venue=VENUE_EQ)
            self.order(sym_wip, "buy",  qty=qty_wip, order_type="market", venue=VENUE_EQ)
            self.order(sym_wis, "buy",  qty=qty_wis, order_type="market", venue=VENUE_EQ)
            side = "sell_reg_buy_combo"
        else:
            # Combo rich: BUY REG, SELL WIP, SELL r*S_WI
            self.order(sym_reg, "buy",  qty=qty_reg, order_type="market", venue=VENUE_EQ)
            self.order(sym_wip, "sell", qty=qty_wip, order_type="market", venue=VENUE_EQ)
            self.order(sym_wis, "sell", qty=qty_wis, order_type="market", venue=VENUE_EQ)
            side = "buy_reg_sell_combo"

        self._save_state(tag, OpenState(mode="TRIANGLE_WI", side=side,
                                        qty_reg=qty_reg, qty_wip=qty_wip, qty_wis=qty_wis,
                                        entry_bps=edge_bps, entry_z=z, ts_ms=_now_ms()))

    # --------------- STUB vs P_WI ---------------
    def _eval_stubpair(self, P: str, S: str, r_ratio: float, sym_reg: str, sym_wip: str, sym_wis: str,
                       p_reg: float, p_wip: float, p_wis: float) -> None:
        tag = f"STUB:{P}|{S}"

        fees = _fees_bps() * 1e-4
        # To BUY stub synthetically, you BUY P_REG and SELL r*S_WI
        stub_buy  = p_reg * (1 + fees) - (r_ratio * p_wis) * (1 - fees)
        stub_sell = p_reg * (1 - fees) - (r_ratio * p_wis) * (1 + fees)

        # Compare P_WI vs stub
        edge_unit = (p_wip * (1 - fees)) - stub_buy    # positive ⇒ P_WI rich vs stub
        edge_bps  = 1e4 * (edge_unit / max(1e-6, p_wip))

        ew = _load_ewma(tag); m,v = ew.update(edge_bps); _save_ewma(tag, ew)
        z = (edge_bps - m)/math.sqrt(max(v,1e-12))
        self.emit_signal(max(-1.0, min(1.0, abs(edge_bps)/max(1.0, ENTRY_BPS))))

        st = self._load_state(tag)
        if st:
            if (abs(edge_bps) <= EXIT_BPS) or (abs(z) <= EXIT_Z):
                self._close(tag, st, sym_reg, sym_wip, sym_wis)
            return

        if r.get(_poskey(self.ctx.name, tag)) is not None: return
        if not (abs(edge_bps) >= ENTRY_BPS and abs(z) >= ENTRY_Z): return

        # Borrow guards for SELL legs
        short_wip = edge_bps > 0
        short_reg = edge_bps < 0
        short_wis = edge_bps < 0  # because we'd SELL r*S_WI when buying P_WI cheap
        if (short_wip and not _borrow_ok(sym_wip)) or \
           (short_reg and not _borrow_ok(sym_reg)) or \
           (short_wis and not _borrow_ok(sym_wis)):
            return

        usd = USD_NOTIONAL
        if usd < MIN_TICKET_USD: return
        # Base on P_WI line
        qty_wip = usd / max(1e-6, p_wip)
        qty_reg = qty_wip
        qty_wis = qty_wip * r_ratio

        if edge_bps > 0:
            # P_WI rich → SELL P_WI ; BUY P_REG ; SELL r*S_WI
            self.order(sym_wip, "sell", qty=qty_wip, order_type="market", venue=VENUE_EQ)
            self.order(sym_reg, "buy",  qty=qty_reg, order_type="market", venue=VENUE_EQ)
            self.order(sym_wis, "sell", qty=qty_wis, order_type="market", venue=VENUE_EQ)
            side = "sell_wi_buy_stub"
        else:
            # P_WI cheap → BUY P_WI ; SELL P_REG ; BUY r*S_WI
            self.order(sym_wip, "buy",  qty=qty_wip, order_type="market", venue=VENUE_EQ)
            self.order(sym_reg, "sell", qty=qty_reg, order_type="market", venue=VENUE_EQ)
            self.order(sym_wis, "buy",  qty=qty_wis, order_type="market", venue=VENUE_EQ)
            side = "buy_wi_sell_stub"

        self._save_state(tag, OpenState(mode="STUB_PAIR", side=side,
                                        qty_reg=qty_reg, qty_wip=qty_wip, qty_wis=qty_wis,
                                        entry_bps=edge_bps, entry_z=z, ts_ms=_now_ms()))

    # --------------- close / unwind ---------------
    def _close(self, tag: str, st: OpenState, sym_reg: str, sym_wip: str, sym_wis: str) -> None:
        # Reverse legs
        def rev(side: str) -> str: return "buy" if side == "sell" else "sell"
        # We don't store per-leg sides; reconstruct from st.side semantic
        if st.side in ("sell_reg_buy_combo", "buy_wi_sell_stub"):
            # positions: short REG, long WIP, long WIS  OR short REG, long WIS, long WIP
            self.order(sym_reg, "buy",  qty=st.qty_reg, order_type="market", venue=VENUE_EQ)
            self.order(sym_wip, "sell", qty=st.qty_wip, order_type="market", venue=VENUE_EQ)
            self.order(sym_wis, "sell", qty=st.qty_wis, order_type="market", venue=VENUE_EQ)
        elif st.side in ("buy_reg_sell_combo", "sell_wi_buy_stub"):
            self.order(sym_reg, "sell", qty=st.qty_reg, order_type="market", venue=VENUE_EQ)
            self.order(sym_wip, "buy",  qty=st.qty_wip, order_type="market", venue=VENUE_EQ)
            self.order(sym_wis, "buy",  qty=st.qty_wis, order_type="market", venue=VENUE_EQ)
        r.delete(_poskey(self.ctx.name, tag))

    # --------------- state I/O ---------------
    def _load_state(self, tag: str) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, tag))
        if not raw: return None
        try:
            o = json.loads(raw) # type: ignore
            return OpenState(mode=str(o["mode"]), side=str(o["side"]),
                             qty_reg=float(o["qty_reg"]), qty_wip=float(o["qty_wip"]), qty_wis=float(o["qty_wis"]),
                             entry_bps=float(o["entry_bps"]), entry_z=float(o["entry_z"]), ts_ms=int(o["ts_ms"]))
        except Exception:
            return None

    def _save_state(self, tag: str, st: Optional[OpenState]) -> None:
        if st is None: return
        r.set(_poskey(self.ctx.name, tag), json.dumps({
            "mode": st.mode, "side": st.side,
            "qty_reg": st.qty_reg, "qty_wip": st.qty_wip, "qty_wis": st.qty_wis,
            "entry_bps": st.entry_bps, "entry_z": st.entry_z, "ts_ms": st.ts_ms
        }))