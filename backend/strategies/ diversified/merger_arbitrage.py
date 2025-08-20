# backend/strategies/diversified/merger_arbitrage.py
from __future__ import annotations

import json, math, os, time, datetime as dt
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import redis
from backend.engine.strategy_base import Strategy

"""
Merger Arbitrage (paper)
------------------------
Supports:
  • CASH  : Target receives fixed cash.
  • STOCK : Target receives acquirer shares (exchange ratio), with optional price collar.
  • MIXED : Cash + stock combo (with optional collar on the stock leg).

Core idea:
  spread = (Consideration_implied - P_target)
  annualized_spread = spread / P_target * (365 / days_to_close)
Entry when annualized_spread minus frictions > ENTRY_APR and z >= ENTRY_Z.
For STOCK/MIXED, hedge: LONG target, SHORT acquirer * hedge_shares.
Exit on close/break or when spread reverts below EXIT_APR or z small.

Paper symbols (map later via adapters):
  • Target   : "EQ:<TGT>"
  • Acquirer : "EQ:<ACQ>"

Redis feeds (examples at bottom):
  HSET last_price "EQ:<SYM>" '{"price": <px>}'
  HSET deal:type     <PAIR> CASH|STOCK|MIXED             # e.g., "ABC_XYZ"
  HSET deal:cash     <PAIR> <cash_per_target_share>      # for CASH/MIXED
  HSET deal:ratio    <PAIR> <shares_of_acq_per_target>   # for STOCK/MIXED
  HSET deal:collar   <PAIR> '{"low_px":..,"high_px":..,"low_ratio":..,"high_ratio":..}'
  HSET deal:deadline <PAIR> <ms_epoch_expected_close>    # close date estimate
  HSET deal:rf       <PAIR> <risk_free_decimal>          # optional (for sanity)
  HSET deal:prob     <PAIR> <0..1>                       # subjective close prob (optional monitor)
  HSET deal:break_px <PAIR> <standalone_target_px>       # break scenario guide (optional)
  HSET borrow:fee    "EQ:<SYM>" <decimal_per_year>       # borrow fees (short side)
  HSET borrow:ok     "EQ:<SYM>" 0|1                      # availability flag for short
  (optional) HSET fees:trading <VENUE> <decimal_per_turnover_year_equiv>  # crude guard
  SET  deal:closed:<PAIR> 1   |  SET deal:broken:<PAIR> 1                # event flags
"""

# ============================ CONFIG (env) ============================
REDIS_HOST = os.getenv("MA_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("MA_REDIS_PORT", "6379"))

TGT = os.getenv("MA_TARGET", "ABC").upper()
ACQ = os.getenv("MA_ACQUIRER", "XYZ").upper()
PAIR = os.getenv("MA_PAIR", f"{TGT}_{ACQ}").upper()

VENUE_EQ = os.getenv("MA_VENUE_EQ", "EXCH").upper()

# Entry/Exit thresholds
ENTRY_APR = float(os.getenv("MA_ENTRY_APR", "0.20"))  # need ≥ 20% annualized gross edge (before break adj)
EXIT_APR  = float(os.getenv("MA_EXIT_APR",  "0.05"))
ENTRY_Z   = float(os.getenv("MA_ENTRY_Z",   "1.2"))
EXIT_Z    = float(os.getenv("MA_EXIT_Z",    "0.5"))

# Sizing / risk
USD_NOTIONAL_TARGET = float(os.getenv("MA_USD_NOTIONAL_TGT", "50000"))  # target leg notional
MIN_TICKET_USD      = float(os.getenv("MA_MIN_TICKET_USD", "200"))
MAX_CONCURRENT      = int(os.getenv("MA_MAX_CONCURRENT", "1"))

# Frictions / guards
TURNOVER_FEE_APR_GUARD = float(os.getenv("MA_FEE_APR", "0.01"))   # 1%/yr rough trading friction
BORROW_MAX_APR         = float(os.getenv("MA_BORROW_MAX_APR", "0.50"))  # don't short if borrow too expensive

# Cadence & stats
RECHECK_SECS = int(os.getenv("MA_RECHECK_SECS", "3"))
EWMA_ALPHA   = float(os.getenv("MA_EWMA_ALPHA", "0.06"))

# Redis keys
LAST_PRICE_HKEY = os.getenv("MA_LAST_PRICE_KEY", "last_price")
DEAL_TYPE_HKEY  = os.getenv("MA_DEAL_TYPE_KEY",  "deal:type")
DEAL_CASH_HKEY  = os.getenv("MA_DEAL_CASH_KEY",  "deal:cash")
DEAL_RATIO_HKEY = os.getenv("MA_DEAL_RATIO_KEY", "deal:ratio")
DEAL_COLLAR_HK  = os.getenv("MA_DEAL_COLLAR_KEY","deal:collar")   # json per PAIR
DEAL_DEADLINE_HK= os.getenv("MA_DEAL_DEADLINE_KEY", "deal:deadline")
DEAL_RF_HKEY    = os.getenv("MA_DEAL_RF_KEY", "deal:rf")
DEAL_PROB_HK    = os.getenv("MA_DEAL_PROB_KEY", "deal:prob")
DEAL_BREAK_HK   = os.getenv("MA_DEAL_BREAKPX_KEY","deal:break_px")

BORROW_FEE_HK   = os.getenv("MA_BORROW_FEE_KEY", "borrow:fee")    # field = "EQ:<SYM>"
BORROW_OK_HK    = os.getenv("MA_BORROW_OK_KEY",  "borrow:ok")     # 1/0 availability
FEES_TURNOVER_HK= os.getenv("MA_FEES_TURNOVER_KEY", "fees:trading")

# Event flags
CLOSED_KEY_FMT  = os.getenv("MA_CLOSED_KEY_FMT", "deal:closed:{pair}")
BROKEN_KEY_FMT  = os.getenv("MA_BROKEN_KEY_FMT", "deal:broken:{pair}")

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================ helpers ============================
def _hget_price(sym: str) -> Optional[float]:
    raw = r.hget(LAST_PRICE_HKEY, sym)
    if not raw: return None
    try: return float(json.loads(raw)["price"])
    except Exception:
        try: return float(raw)
        except Exception: return None

def _hgetf(hk: str, field: str) -> Optional[float]:
    v = r.hget(hk, field)
    if v is None: return None
    try: return float(v)
    except Exception:
        try: return float(json.loads(v))
        except Exception: return None

def _hget_json(hk: str, field: str) -> Optional[dict]:
    raw = r.hget(hk, field)
    if not raw: return None
    try: return json.loads(raw)
    except Exception: return None

def _now_ms() -> int:
    return int(time.time() * 1000)

def _days_to(ms_deadline: Optional[int]) -> float:
    if not ms_deadline: return 30.0  # fallback 30d
    now = _now_ms()
    days = max(1.0, (ms_deadline - now) / (1000.0 * 3600.0 * 24.0))
    return days

def _borrow_apr(sym: str) -> float:
    apr = _hgetf(BORROW_FEE_HK, sym) or 0.0
    ok  = r.hget(BORROW_OK_HK, sym)
    if ok is not None and str(ok) == "0":
        return max(apr, BORROW_MAX_APR + 1.0)  # force block
    return apr

def _fees_apr(venue: str) -> float:
    v = _hgetf(FEES_TURNOVER_HK, venue)
    return v if v is not None else TURNOVER_FEE_APR_GUARD

# --- collar handling (simple clamp between (low_px,high_px) mapping to (low_ratio,high_ratio)) ---
def _collared_ratio(acq_px: float, base_ratio: float, collar: Optional[dict]) -> float:
    if not collar: return base_ratio
    lp = float(collar.get("low_px", 0) or 0)
    hp = float(collar.get("high_px", 0) or 0)
    lr = float(collar.get("low_ratio", base_ratio))
    hr = float(collar.get("high_ratio", base_ratio))
    if lp > 0 and acq_px <= lp:  return lr
    if hp > 0 and acq_px >= hp:  return hr
    # linear interpolate inside collar if both bands defined
    if lp > 0 and hp > lp:
        t = (acq_px - lp) / (hp - lp)
        return lr + t * (hr - lr)
    return base_ratio

def _consideration(acq_px: float, deal_type: str, cash_per_sh: float, ratio: float, collar: Optional[dict]) -> float:
    deal_type = (deal_type or "CASH").upper()
    if deal_type == "CASH":
        return cash_per_sh
    if deal_type == "STOCK":
        er = _collared_ratio(acq_px, ratio, collar)
        return er * acq_px
    # MIXED
    er = _collared_ratio(acq_px, ratio, collar)
    return cash_per_sh + er * acq_px

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

def _ewma_key() -> str:
    return f"ma:ewma:{PAIR}"

def _load_ewma() -> EwmaMV:
    raw = r.get(_ewma_key())
    if raw:
        try:
            o = json.loads(raw)
            return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", EWMA_ALPHA)))
        except Exception: pass
    return EwmaMV(mean=0.0, var=1.0, alpha=EWMA_ALPHA)

def _save_ewma(ew: EwmaMV) -> None:
    r.set(_ewma_key(), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ============================ state ============================
@dataclass
class OpenState:
    side: str  # "long_tgt_short_acq" or "long_tgt"
    qty_tgt: float
    qty_acq: float
    entry_apr: float
    entry_z: float
    ts_ms: int

def _poskey(name: str) -> str:
    return f"ma:open:{name}:{PAIR}"

# ============================ strategy ============================
class MergerArbitrage(Strategy):
    """
    Long target vs cash/stock consideration, hedge acquirer for stock/mixed deals.
    """
    def __init__(self, name: str = "merger_arbitrage", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_start(self) -> None:
        super().on_start()
        r.hset("strategy:universe", self.ctx.name, json.dumps({
            "pair": PAIR, "tgt": f"EQ:{TGT}", "acq": f"EQ:{ACQ}", "ts": _now_ms()
        }))

    def on_tick(self, tick: Dict) -> None:
        now = time.time()
        if now - self._last < RECHECK_SECS:
            return
        self._last = now
        self._evaluate()

    # --------------- core ---------------
    def _evaluate(self) -> None:
        pt = _hget_price(f"EQ:{TGT}")
        pa = _hget_price(f"EQ:{ACQ}") or 0.0
        if pt is None or pt <= 0:
            return

        deal_type = r.hget(DEAL_TYPE_HKEY, PAIR) or "CASH"
        cash_per  = _hgetf(DEAL_CASH_HKEY, PAIR) or 0.0
        ratio     = _hgetf(DEAL_RATIO_HKEY, PAIR) or 0.0
        collar    = _hget_json(DEAL_COLLAR_HK, PAIR)
        deadline  = r.hget(DEAL_DEADLINE_HK, PAIR)
        deadline  = int(deadline) if deadline and str(deadline).isdigit() else None

        cons = _consideration(pa, deal_type, cash_per, ratio, collar)
        if cons <= 0:
            return

        days = _days_to(deadline)
        raw_spread = cons - pt
        gross_apr  = (raw_spread / pt) * (365.0 / days)

        # friction guards (borrow on acquirer if we short; turnover fee roughness)
        fee_apr = _fees_apr(VENUE_EQ)
        borrow_apr_acq = 0.0
        require_short = deal_type.upper() in ("STOCK", "MIXED") and (ratio > 0 or (collar is not None))
        if require_short:
            borrow_apr_acq = _borrow_apr(f"EQ:{ACQ}")
            if borrow_apr_acq > BORROW_MAX_APR:
                return  # borrow unavailable/too expensive

        net_apr = gross_apr - fee_apr - (borrow_apr_acq if require_short else 0.0)

        # z-score gate on net_apr
        ew = _load_ewma()
        m, v = ew.update(net_apr)
        _save_ewma(ew)
        z = (net_apr - m) / math.sqrt(max(v, 1e-12))

        # monitor signal: scaled to ENTRY_APR
        self.emit_signal(max(-1.0, min(1.0, net_apr / max(1e-6, ENTRY_APR))))

        st = self._load_state()

        # ----- handle events -----
        if str(r.get(CLOSED_KEY_FMT.format(pair=PAIR)) or "0") == "1":
            if st: self._close(st)
            return
        if str(r.get(BROKEN_KEY_FMT.format(pair=PAIR)) or "0") == "1":
            # On break, close hedge and target (let PnL realize to break px in OMS layer if you simulate)
            if st: self._close(st)
            return

        # ----- exits -----
        if st:
            if (net_apr <= EXIT_APR) or (abs(z) <= EXIT_Z):
                self._close(st)
            return

        # ----- entries -----
        if r.get(_poskey(self.ctx.name)) is not None:
            return
        if not (net_apr >= ENTRY_APR and abs(z) >= ENTRY_Z and raw_spread > 0):
            return

        # Size legs
        qty_tgt = USD_NOTIONAL_TARGET / pt
        if qty_tgt * pt < MIN_TICKET_USD:
            return

        if require_short and pa > 0:
            # hedge shares = exchange ratio (collar-adjusted at current price) * qty_tgt
            er_now = _collared_ratio(pa, ratio, collar)
            qty_acq = er_now * qty_tgt
            if qty_acq * pa < MIN_TICKET_USD:
                return
            # place orders: buy target / short acquirer
            self.order(f"EQ:{TGT}", "buy",  qty=qty_tgt, order_type="market", venue=VENUE_EQ)
            self.order(f"EQ:{ACQ}", "sell", qty=qty_acq, order_type="market", venue=VENUE_EQ)
            side = "long_tgt_short_acq"
        else:
            # pure cash deal: just long target
            self.order(f"EQ:{TGT}", "buy", qty=qty_tgt, order_type="market", venue=VENUE_EQ)
            qty_acq = 0.0
            side = "long_tgt"

        self._save_state(OpenState(
            side=side, qty_tgt=qty_tgt, qty_acq=qty_acq,
            entry_apr=net_apr, entry_z=z, ts_ms=_now_ms()
        ))

    # --------------- state io ---------------
    def _load_state(self) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name))
        if not raw: return None
        try:
            return OpenState(**json.loads(raw))
        except Exception:
            return None

    def _save_state(self, st: Optional[OpenState]) -> None:
        if st is None: return
        r.set(_poskey(self.ctx.name), json.dumps(st.__dict__))

    # --------------- close ---------------
    def _close(self, st: OpenState) -> None:
        # unwind in reverse
        if st.side == "long_tgt_short_acq":
            self.order(f"EQ:{ACQ}", "buy",  qty=st.qty_acq, order_type="market", venue=VENUE_EQ)
            self.order(f"EQ:{TGT}", "sell", qty=st.qty_tgt, order_type="market", venue=VENUE_EQ)
        else:
            self.order(f"EQ:{TGT}", "sell", qty=st.qty_tgt, order_type="market", venue=VENUE_EQ)
        r.delete(_poskey(self.ctx.name))