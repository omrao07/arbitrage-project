# backend/strategies/diversified/rights_issue_arbitrage.py
from __future__ import annotations

import json, math, os, time, uuid
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import redis
from backend.engine.strategy_base import Strategy

"""
Rights Issue Arbitrage — paper
------------------------------
Two modes:

1) CUM_EX_SYNTH:
   Compare:
     Price(cum)    ↔  Price(ex) + value(right)
   with:
     TERP = (P_cum + r * SubPrice) / (1 + r)
     Right_theo = P_cum - TERP = (P_cum - SubPrice)/(1+r)
   r = new_shares_per_old (e.g., 1-for-5 → r = 1/5).

   Trade long cheap, short rich (package vs single), hedge with the other leg.
   Works during the brief window when cum/ex/rights are all trading.

2) RIGHTS_SUBSCRIBE:
   If rights (nil-paid) are **undervalued** vs intrinsic:
     Right_intrinsic ≈ max(0, P_ex - SubPrice)
   Adjust for fees, borrow, and funding over days until allotment.
   Buy rights, SHORT ex-shares for the future allotment quantity,
   subscribe, receive new shares, then **deliver** to cover the short.

Redis feeds (publish via adapters):
  # Marks
  HSET last_price "EQ:CUM:<SYM>" '{"price": <px>}'
  HSET last_price "EQ:EX:<SYM>"  '{"price": <px>}'
  HSET ob:rights:<RID> <RID>     '{"bid": <px>, "ask": <px>, "fee_bps": <bps>}'  # nil-paid rights orderbook
  HSET rights:meta <RID>         '{"sym":"<SYM>","ratio_num":<a>,"ratio_den":<b>,"sub_price":<px>,"record_ms":<ms>,"ex_ms":<ms>,"subs_end_ms":<ms>,"allot_ms":<ms>}'
  # Frictions (annualized decimals unless noted)
  HSET fees:trading <VENUE> <bps>                   # fallback taker bps
  HSET borrow:fee "EQ:EX:<SYM>" <apr_decimal>       # borrow fee for shorting ex-shares
  HSET borrow:ok  "EQ:EX:<SYM>" 0|1
  HSET funding:cash <CCY> <apr_decimal>             # cash carry for subscription capital
  # Ops
  SET  risk:halt 0|1

Paper symbols for router/OMS:
  EQ:CUM:<SYM>, EQ:EX:<SYM>, RIGHTS:<RID>
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("RIA_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("RIA_REDIS_PORT", "6379"))

MODE      = os.getenv("RIA_MODE", "CUM_EX_SYNTH").upper()  # "CUM_EX_SYNTH" | "RIGHTS_SUBSCRIBE"
RID       = os.getenv("RIA_RIGHTS_ID", "R-ACME-2025").upper()
VENUE_EQ  = os.getenv("RIA_VENUE_EQ", "EXCH").upper()
VENUE_RTS = os.getenv("RIA_VENUE_RTS", "RIGHTS").upper()
CCY       = os.getenv("RIA_CCY", "USD").upper()

# Thresholds
ENTRY_BPS = float(os.getenv("RIA_ENTRY_BPS", "80"))   # net edge in bps of package value
EXIT_BPS  = float(os.getenv("RIA_EXIT_BPS",  "30"))
ENTRY_Z   = float(os.getenv("RIA_ENTRY_Z",   "1.2"))
EXIT_Z    = float(os.getenv("RIA_EXIT_Z",    "0.5"))

# Sizing / risk
USD_NOTIONAL     = float(os.getenv("RIA_USD_NOTIONAL", "30000"))
MIN_TICKET_USD   = float(os.getenv("RIA_MIN_TICKET_USD", "200"))
MAX_CONCURRENT   = int(os.getenv("RIA_MAX_CONCURRENT", "1"))

# Cadence
RECHECK_SECS = float(os.getenv("RIA_RECHECK_SECS", "1.0"))
EWMA_ALPHA   = float(os.getenv("RIA_EWMA_ALPHA", "0.06"))

# Redis keys
LAST_HK      = os.getenv("RIA_LAST_HK", "last_price")
OB_RTS_HK    = os.getenv("RIA_OB_RTS_HK", "ob:rights:{rid}")
META_HK      = os.getenv("RIA_META_HK",   "rights:meta")
FEES_EQ_HK   = os.getenv("RIA_FEES_EQ_HK","fees:trading")
BORROW_FEE_HK= os.getenv("RIA_BORROW_FEE_HK","borrow:fee")
BORROW_OK_HK = os.getenv("RIA_BORROW_OK_HK", "borrow:ok")
FUNDING_HK   = os.getenv("RIA_FUNDING_HK","funding:cash")
HALT_KEY     = os.getenv("RIA_HALT_KEY",  "risk:halt")

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================ helpers ============================
def _hget_json(hk: str, field: str) -> Optional[dict]:
    raw = r.hget(hk, field)
    if not raw: return None
    try:
        j = json.loads(raw);  return j if isinstance(j, dict) else None
    except Exception:
        return None

def _hgetf(hk: str, field: str) -> Optional[float]:
    v = r.hget(hk, field)
    if v is None: return None
    try: return float(v)
    except Exception:
        try: return float(json.loads(v))
        except Exception: return None

def _px(sym: str) -> Optional[float]:
    raw = r.hget(LAST_HK, sym)
    if not raw: return None
    try:
        j = json.loads(raw); return float(j.get("price", 0))
    except Exception:
        try: return float(raw)
        except Exception: return None

def _ob_rights(rid: str) -> Optional[Tuple[float, float, float]]:
    j = _hget_json(OB_RTS_HK.format(rid=rid), rid) or _hget_json(OB_RTS_HK.format(rid=rid), "book")
    if not j: return None
    try:
        bid = float(j.get("bid", 0)); ask = float(j.get("ask", 0))
        fee_bps = float(j.get("fee_bps", _hgetf(FEES_EQ_HK, VENUE_RTS) or 80.0))
        if bid <= 0 or ask <= 0: return None
        return bid, ask, fee_bps
    except Exception:
        return None

def _now_ms() -> int: return int(time.time() * 1000)

def _terp(p_cum: float, r: float, sub_px: float) -> float:
    return (p_cum + r * sub_px) / (1.0 + r)

def _right_theo(p_cum: float, r: float, sub_px: float) -> float:
    return (p_cum - sub_px) / (1.0 + r)

def _fees_bps_eq() -> float:
    v = _hgetf(FEES_EQ_HK, VENUE_EQ)
    return float(v) if v is not None else 10.0

def _borrow_apr(sym_ex: str) -> float:
    ok = r.hget(BORROW_OK_HK, sym_ex)
    if ok is not None and str(ok) == "0":
        return 1.0  # block
    return _hgetf(BORROW_FEE_HK, sym_ex) or 0.0

def _funding_apr() -> float:
    return _hgetf(FUNDING_HK, CCY) or 0.0

# ============================ EWMA ============================
@dataclass
class EwmaMV:
    mean: float
    var: float
    alpha: float
    def update(self, x: float) -> Tuple[float, float]:
        m0 = self.mean
        self.mean = (1 - self.alpha) * self.mean + self.alpha * x
        self.var  = max(1e-12, (1 - self.alpha) * (self.var + (x - m0) * (x - self.mean)))
        return self.mean, self.var

def _ewma_key(tag: str) -> str:
    return f"ria:ewma:{tag}"

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

# ============================ state ============================
@dataclass
class OpenState:
    mode: str
    tag: str
    qty_old: float         # old shares notional-equivalent (for ratio math)
    qty_rights: float
    qty_ex: float
    entry_bps: float
    entry_z: float
    ts_ms: int
    txid: str = ""

def _poskey(name: str, tag: str) -> str:
    return f"ria:open:{name}:{tag}"

# ============================ strategy ============================
class RightsIssueArbitrage(Strategy):
    """
    CUM vs (EX+RIGHTS) or buy-undervalued-rights & subscribe with hedge.
    """
    def __init__(self, name: str = "rights_issue_arbitrage", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_tick(self, tick: Dict) -> None:
        if (r.get(HALT_KEY) or "0") == "1": return
        now = time.time()
        if now - self._last < RECHECK_SECS: return
        self._last = now

        meta = _hget_json(META_HK, RID) or {}
        sym = (meta.get("sym") or "ACME").upper()
        ratio_num = float(meta.get("ratio_num", 1.0))
        ratio_den = float(meta.get("ratio_den", 5.0))
        sub_px    = float(meta.get("sub_price", 0.0))
        ex_ms     = int(meta.get("ex_ms", 0) or 0)
        allot_ms  = int(meta.get("allot_ms", 0) or 0)

        if ratio_den <= 0 or sub_px <= 0: return
        r_ratio = ratio_num / ratio_den

        p_cum = _px(f"EQ:CUM:{sym}")
        p_ex  = _px(f"EQ:EX:{sym}")
        rights = _ob_rights(RID)

        if MODE == "CUM_EX_SYNTH":
            if None in (p_cum, p_ex) or rights is None: return
            self._eval_cum_ex(sym, r_ratio, sub_px, p_cum, p_ex, rights)
        else:
            if p_ex is None or rights is None: return
            self._eval_subscribe(sym, r_ratio, sub_px, p_ex, rights, allot_ms)

    # --------------- CUM vs (EX + RIGHTS) ---------------
    def _eval_cum_ex(self, sym: str, r_ratio: float, sub_px: float, p_cum: float, p_ex: float,
                     rights: Tuple[float,float,float]) -> None:
        tag = f"CUMEX:{RID}:{sym}"
        r_bid, r_ask, r_fee_bps = rights

        # Theoretical values
        terp = _terp(p_cum, r_ratio, sub_px)
        right_theo = _right_theo(p_cum, r_ratio, sub_px)

        # Executable package value (sell/buy at quotes with fees)
        eq_fee = _fees_bps_eq() * 1e-4
        r_fee  = (r_fee_bps * 1e-4)

        # We compare: cum vs (ex + right)
        package_buy  = p_ex * (1 + eq_fee) + r_ask * (1 + r_fee)  # cost to buy synthetic cum
        package_sell = p_ex * (1 - eq_fee) + r_bid * (1 - r_fee)

        # Net edge (bps of cum) when **cum is rich vs package** (so we sell cum, buy package)
        edge_unit = (p_cum * (1 - eq_fee)) - package_buy
        edge_bps  = 1e4 * (edge_unit / max(1e-6, p_cum))

        # EWMA gate
        ew = _load_ewma(tag); m,v = ew.update(edge_bps); _save_ewma(tag, ew)
        z = (edge_bps - m) / math.sqrt(max(v, 1e-12))

        # monitor signal
        self.emit_signal(max(-1.0, min(1.0, edge_bps / max(1.0, ENTRY_BPS))))

        st = self._load_state(tag)
        if st:
            if (abs(edge_bps) <= EXIT_BPS) or (abs(z) <= EXIT_Z):
                self._close(tag, st, sym)
            return

        if r.get(_poskey(self.ctx.name, tag)) is not None: return
        if not (edge_bps >= ENTRY_BPS and abs(z) >= ENTRY_Z): return

        # Size: target notional in cum leg
        usd = USD_NOTIONAL
        if usd < MIN_TICKET_USD: return
        qty_old = usd / max(1e-6, p_cum)
        qty_ex  = qty_old                               # 1 ex share per cum share
        qty_rts = qty_old * r_ratio * ratio_den_safe()  # but a right usually represents "the right to buy X new for Y old"
        # In many markets, ONE right is defined as the entitlement unit; assume here 1 right per old share * (ratio_num/ratio_den).
        qty_rts = qty_old * r_ratio

        # Place legs: SELL cum, BUY ex + rights
        self.order(f"EQ:CUM:{sym}", "sell", qty=qty_old, order_type="market", venue=VENUE_EQ)
        self.order(f"EQ:EX:{sym}",  "buy",  qty=qty_ex,  order_type="market", venue=VENUE_EQ)
        self.order(f"RIGHTS:{RID}", "buy",  qty=qty_rts, order_type="limit", price=r_ask, venue=VENUE_RTS)

        self._save_state(tag, OpenState(mode="CUM_EX_SYNTH", tag=tag,
                                        qty_old=qty_old, qty_ex=qty_ex, qty_rights=qty_rts,
                                        entry_bps=edge_bps, entry_z=z, ts_ms=_now_ms()))

    # --------------- RIGHTS_SUBSCRIBE ---------------
    def _eval_subscribe(self, sym: str, r_ratio: float, sub_px: float, p_ex: float,
                        rights: Tuple[float,float,float], allot_ms: int) -> None:
        tag = f"SUBS:{RID}:{sym}"
        r_bid, r_ask, r_fee_bps = rights

        # Intrinsic and expected value of right
        intrinsic = max(0.0, p_ex - sub_px)
        fees_r = (r_fee_bps * 1e-4)
        fees_e = _fees_bps_eq() * 1e-4

        # Carry/borrow over settlement (hours → years approximation)
        hours = max(24.0, (allot_ms - _now_ms()) / 3600000.0) if allot_ms else 72.0
        years = hours / (24.0 * 365.0)
        borrow_apr = _borrow_apr(f"EQ:EX:{sym}")      # to short ex-shares
        funding_apr= _funding_apr()                   # cash for subscription

        # Net edge per right if we buy rights and short ex for the coming new shares:
        # Strategy payoff approx: (deliver new at sub_px to cover short @ p_ex) ⇒ lock intrinsic,
        # minus trading fees and carry.
        edge_unit = intrinsic - (r_ask * (1 + fees_r)) \
                    - (p_ex * fees_e) \
                    - (p_ex * borrow_apr * years) \
                    - (sub_px * funding_apr * years)
        # Quote in bps of p_ex
        edge_bps = 1e4 * (edge_unit / max(1e-6, p_ex))

        ew = _load_ewma(tag); m,v = ew.update(edge_bps); _save_ewma(tag, ew)
        z = (edge_bps - m) / math.sqrt(max(v, 1e-12))
        self.emit_signal(max(-1.0, min(1.0, edge_bps / max(1.0, ENTRY_BPS))))

        st = self._load_state(tag)
        if st:
            if (edge_bps <= EXIT_BPS) or (abs(z) <= EXIT_Z):
                self._close(tag, st, sym)
            return

        # Borrow availability guard
        if borrow_apr >= 0.99:  # blocked/unavailable
            return

        if r.get(_poskey(self.ctx.name, tag)) is not None: return
        if not (edge_bps >= ENTRY_BPS and abs(z) >= ENTRY_Z and edge_unit > 0):
            return

        usd = USD_NOTIONAL
        if usd < MIN_TICKET_USD: return

        # Sizing:
        # For every "old share", entitlement = r_ratio * old_shares new @ sub_px.
        # We buy N_rights that entitle us to N_new = N_rights (assuming 1 right ↔ 1 new share per r_ratio; here we keep it simple).
        # Target new shares notionally ~ USD_NOTIONAL / p_ex
        target_new = usd / max(1e-6, p_ex)
        qty_rights = target_new  # simple 1:1 mapping; adapt per local right unit if needed
        qty_short  = target_new  # short ex now, will deliver new on allotment

        # Place legs: BUY rights, SHORT ex-shares
        self.order(f"RIGHTS:{RID}", "buy",  qty=qty_rights, price=r_ask, order_type="limit", venue=VENUE_RTS)
        self.order(f"EQ:EX:{sym}",  "sell", qty=qty_short, order_type="market", venue=VENUE_EQ)

        # Subscription instruction (paper): OMS/adapter should commit funds at sub_px * qty_rights and deliver at allotment
        txid = f"sub-{uuid.uuid4().hex[:10]}"
        self.order(f"RIGHTS:{RID}", "subscribe", qty=qty_rights, order_type="subscription",
                   venue=VENUE_RTS, flags={"sub_price": sub_px, "txid": txid, "allot_ms": allot_ms})

        self._save_state(tag, OpenState(mode="RIGHTS_SUBSCRIBE", tag=tag,
                                        qty_old=0.0, qty_ex=qty_short, qty_rights=qty_rights,
                                        entry_bps=edge_bps, entry_z=z, ts_ms=_now_ms(), txid=txid))

    # --------------- close / unwind ---------------
    def _close(self, tag: str, st: OpenState, sym: str) -> None:
        if st.mode == "CUM_EX_SYNTH":
            # Reverse: buy back cum, sell ex and rights
            self.order(f"EQ:CUM:{sym}", "buy",  qty=st.qty_old, order_type="market", venue=VENUE_EQ)
            self.order(f"EQ:EX:{sym}",  "sell", qty=st.qty_ex,  order_type="market", venue=VENUE_EQ)
            self.order(f"RIGHTS:{RID}", "sell", qty=st.qty_rights, order_type="market", venue=VENUE_RTS)
        else:
            # Pre‑allotment unwind: buy back short ex, sell the rights (if still trading)
            self.order(f"EQ:EX:{sym}",  "buy",  qty=st.qty_ex, order_type="market", venue=VENUE_EQ)
            self.order(f"RIGHTS:{RID}", "sell", qty=st.qty_rights, order_type="market", venue=VENUE_RTS)
        r.delete(_poskey(self.ctx.name, tag))

    # --------------- state I/O ---------------
    def _load_state(self, tag: str) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, tag))
        if not raw: return None
        try:
            o = json.loads(raw)
            return OpenState(mode=str(o["mode"]), tag=str(o["tag"]),
                             qty_old=float(o["qty_old"]), qty_rights=float(o["qty_rights"]),
                             qty_ex=float(o["qty_ex"]), entry_bps=float(o["entry_bps"]),
                             entry_z=float(o["entry_z"]), ts_ms=int(o["ts_ms"]),
                             txid=str(o.get("txid","")))
        except Exception:
            return None

    def _save_state(self, tag: str, st: Optional[OpenState]) -> None:
        if st is None: return
        r.set(_poskey(self.ctx.name, tag), json.dumps({
            "mode": st.mode, "tag": st.tag,
            "qty_old": st.qty_old, "qty_rights": st.qty_rights, "qty_ex": st.qty_ex,
            "entry_bps": st.entry_bps, "entry_z": st.entry_z, "ts_ms": st.ts_ms,
            "txid": st.txid
        }))

# ---------- small helper for robust ratio (kept simple here) ----------
def ratio_den_safe() -> float:
    return 1.0