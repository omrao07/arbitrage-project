# backend/strategies/diversified/tail_risk_hedge.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import redis
from backend.engine.strategy_base import Strategy

"""
Tail-Risk Hedge — paper
-----------------------
Modes:
  1) PUT_SPREAD: buy SPY (or index) put spread when composite crash score breaches entry;
                 exit on profit target, score decay, or near-expiry; auto-roll monthly.
  2) VIX_CALLS : buy VIX (or vol proxy) out-of-the-money calls on the same trigger.

Crash score (0..1) combines:
  • Drawdown / momentum shock          (fast vs slow EWMA of returns)
  • VIX term-structure inversion       (front/backwardation)
  • Credit spread stress (optional)    (e.g., HY-IG or CDX HY OAS jump)

Redis feeds you publish elsewhere:

  # Spot & VIX futures (or any vol proxy)
  HSET last_price "EQ:SPY"           '{"price": 500.00}'
  HSET last_price "VIX:FRONT"        '{"price": 16.2}'
  HSET last_price "VIX:BACK"         '{"price": 18.8}'

  # Optional: option mids (if you have them). Otherwise module estimates via IV.
  HSET opt:mid "OPT_PUT:SPY:450:30"  '{"price": 3.25}'
  HSET opt:mid "OPT_PUT:SPY:420:30"  '{"price": 1.10}'
  HSET opt:mid "OPT_CALL:VIX:30:30"  '{"price": 0.80}'

  # IV inputs (if no mid): ATM IV for tenor (decimals)
  HSET iv:atm "SPY:30" 0.19
  HSET iv:atm "VIX:30" 1.10  # VIX option IV is naturally high; treat as direct vol

  # Credit stress (optional; bps)
  SET credit:hy_ig_oas_bps 350

  # Fees (bps guard)
  HSET fees:opt OPT 15

  # Ops
  SET risk:halt 0|1

Paper routing (adapters map later):
  • Index options: "OPT_PUT:<SYM>:<K>:<EXP_DAYS>"
  • VIX options:   "OPT_CALL:VIX:<K>:<EXP_DAYS>"
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("TRH_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("TRH_REDIS_PORT", "6379"))

MODE        = os.getenv("TRH_MODE", "PUT_SPREAD").upper()      # PUT_SPREAD | VIX_CALLS
SYM         = os.getenv("TRH_SYM", "SPY").upper()              # Underlying for PUT_SPREAD
TENOR_DAYS  = int(os.getenv("TRH_TENOR_DAYS", "30"))           # target expiry
PUT_MONEY_L = float(os.getenv("TRH_PUT_L_MNY", "0.90"))        # lower strike ≈ 90% spot
PUT_MONEY_H = float(os.getenv("TRH_PUT_H_MNY", "0.84"))        # higher protection (further OTM)
VIX_K_MULT  = float(os.getenv("TRH_VIX_K_MULT", "1.2"))        # VIX call strike ≈ 1.2 * front

# Budget & exits
PORTFOLIO_NOTIONAL_USD = float(os.getenv("TRH_PORTFOLIO_USD", "100000"))
BUDGET_BPS             = float(os.getenv("TRH_BUDGET_BPS", "50"))   # spend up to 50 bps per deployment
TP_PCT                 = float(os.getenv("TRH_TP_PCT", "80"))       # take profit if hedge value +80%
MAX_DECAY_PCT          = float(os.getenv("TRH_MAX_DECAY_PCT", "60"))# cut if premium decays >60%
MIN_DTE_EXIT           = int(os.getenv("TRH_MIN_DTE_EXIT", "5"))    # exit/roll when <= this many days

# Signals & gates
ENTRY_SCORE  = float(os.getenv("TRH_ENTRY_SCORE", "0.65"))
EXIT_SCORE   = float(os.getenv("TRH_EXIT_SCORE",  "0.30"))
EWMA_FAST    = float(os.getenv("TRH_EWMA_FAST", "0.2"))
EWMA_SLOW    = float(os.getenv("TRH_EWMA_SLOW", "0.02"))
RECHECK_SECS = float(os.getenv("TRH_RECHECK_SECS", "1.0"))

# Redis keys
HALT_KEY = os.getenv("TRH_HALT_KEY", "risk:halt")
LAST_HK  = os.getenv("TRH_LAST_HK",  "last_price")
IV_ATM_HK= os.getenv("TRH_IV_ATM_HK","iv:atm")
OPT_MID  = os.getenv("TRH_OPT_MID",  "opt:mid")
FEES_HK  = os.getenv("TRH_FEES_HK",  "fees:opt")

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

def _iv_atm(sym: str, dte: int) -> Optional[float]:
    v = r.hget(IV_ATM_HK, f"{sym}:{dte}")
    try: return float(v) if v is not None else None # type: ignore
    except Exception: return None

def _opt_mid(sym: str, right: str, k: float, dte: int) -> Optional[float]:
    j = _hget_json(OPT_MID, f"OPT_{right}:{sym}:{int(round(k))}:{int(dte)}")
    if j: 
        try: return float(j.get("price", 0.0))
        except Exception: return None
    return None

def _fees_bps(venue: str="OPT") -> float:
    v = r.hget(FEES_HK, venue)
    try: return float(v) if v is not None else 15.0 # type: ignore
    except Exception: return 15.0

def _now_ms() -> int: return int(time.time() * 1000)

def _bsm_put_price(S: float, K: float, T: float, iv: float, r: float=0.0, q: float=0.0) -> float:
    if T<=0 or S<=0 or K<=0 or iv<=0: return max(0.0, K-S)
    import math
    f = S*math.exp((r-q)*T)
    sig = iv*math.sqrt(T)
    d1 = (math.log(f/K) + 0.5*iv*iv*T) / max(1e-12, sig)
    d2 = d1 - sig
    N = 0.5*(1+math.erf(-d1/math.sqrt(2)))
    N2= 0.5*(1+math.erf(-d2/math.sqrt(2)))
    return math.exp(-r*T)*K*N2 - math.exp(-q*T)*S*(1-N)

def _bsm_call_price(S: float, K: float, T: float, iv: float, r: float=0.0, q: float=0.0) -> float:
    if T<=0 or S<=0 or K<=0 or iv<=0: return max(0.0, S-K)
    import math
    f = S*math.exp((r-q)*T)
    sig = iv*math.sqrt(T)
    d1 = (math.log(f/K) + 0.5*iv*iv*T) / max(1e-12, sig)
    d2 = d1 - sig
    N = 0.5*(1+math.erf(d1/math.sqrt(2)))
    N2= 0.5*(1+math.erf(d2/math.sqrt(2)))
    return math.exp(-q*T)*S*N - math.exp(-r*T)*K*N2

# ============================ stats (fast/slow) ============================
@dataclass
class Ema:
    val: float
    a: float
    def update(self, x: float) -> float:
        self.val = (1-self.a)*self.val + self.a*x
        return self.val

def _ema_key(tag: str) -> str: return f"trh:ema:{tag}"

def _load_ema(tag: str, a: float, default: float=0.0) -> Ema:
    raw = r.get(_ema_key(tag))
    if raw:
        try:
            o = json.loads(raw); return Ema(val=float(o["v"]), a=float(o.get("a", a))) # type: ignore
        except Exception: pass
    return Ema(val=default, a=a)

def _save_ema(tag: str, e: Ema) -> None:
    r.set(_ema_key(tag), json.dumps({"v": e.val, "a": e.a}))

# ============================ state ============================
@dataclass
class OpenState:
    mode: str
    tag: str
    qty_long: float
    qty_short: float
    k_long: float
    k_short: float
    dte: int
    cost_usd: float
    entry_score: float
    ts_ms: int

def _poskey(name: str, tag: str) -> str:
    return f"trh:open:{name}:{tag}"

# ============================ Strategy ============================
class TailRiskHedge(Strategy):
    """
    Crash-score triggered tail hedge via put spreads or VIX calls (paper).
    """
    def __init__(self, name: str = "tail_risk_hedge", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0
        self._spot_prev = None

    def on_tick(self, tick: Dict) -> None:
        if (r.get(HALT_KEY) or "0") == "1": return
        now = time.time()
        if now - self._last < RECHECK_SECS: return
        self._last = now

        if MODE == "PUT_SPREAD":
            self._run_put_spread()
        else:
            self._run_vix_calls()

    # ---------------- signal & score ----------------
    def _crash_score(self, spot_sym: str="EQ:SPY") -> float:
        S = _px(spot_sym); f1 = _px("VIX:FRONT"); f2 = _px("VIX:BACK")
        if S is None: return 0.0

        # 1) Drawdown/momo shock: fast vs slow EMA of log returns
        if self._spot_prev is None:
            self._spot_prev = S
        ret = math.log(S / max(1e-12, self._spot_prev))
        self._spot_prev = S

        ema_f = _load_ema("ret_fast", EWMA_FAST, 0.0)
        ema_s = _load_ema("ret_slow", EWMA_SLOW, 0.0)
        rf = ema_f.update(ret);  rs = ema_s.update(ret)
        _save_ema("ret_fast", ema_f); _save_ema("ret_slow", ema_s)
        shock = min(1.0, max(0.0, -(rf - rs) * 50.0))  # scale negative momentum into 0..1

        # 2) VIX term structure (backwardation → stress)
        term = 0.0
        if f1 is not None and f2 is not None and f2 > 0:
            contango = (f2 - f1) / f2
            term = min(1.0, max(0.0, -contango * 10.0))  # negative contango (backwardation) → up to 1.0

        # 3) Credit stress (bps; >350 → stress)
        credit = 0.0
        raw = r.get("credit:hy_ig_oas_bps")
        if raw:
            try:
                oas = float(raw) # type: ignore
                credit = min(1.0, max(0.0, (oas - 250.0) / 200.0))
            except Exception:
                pass

        # Composite (weights can be tuned)
        score = 0.5*shock + 0.3*term + 0.2*credit
        return max(0.0, min(1.0, score))

    # ---------------- PUT_SPREAD ----------------
    def _run_put_spread(self) -> None:
        spot_sym = f"EQ:{SYM}"
        S = _px(spot_sym)
        if S is None or S <= 0: return

        # strikes & DTE
        k_hi = round(PUT_MONEY_L * S)   # closer to ATM
        k_lo = round(PUT_MONEY_H * S)   # further OTM (higher protection)
        dte = TENOR_DAYS
        tag = f"PS:{SYM}:{k_lo}-{k_hi}:{dte}"

        # Pricing: prefer mid from Redis, else BSM with ATM IV
        p_hi = _opt_mid(SYM, "PUT", k_hi, dte)
        p_lo = _opt_mid(SYM, "PUT", k_lo, dte)
        if p_hi is None or p_lo is None:
            iv = _iv_atm(SYM, dte) or 0.20
            T = max(1.0/365.0, dte/365.0)
            p_hi = p_hi or _bsm_put_price(S, k_hi, T, iv)
            p_lo = p_lo or _bsm_put_price(S, k_lo, T, iv)
        spread_px = max(0.0, p_hi - p_lo)

        # Signal
        score = self._crash_score(spot_sym)
        self.emit_signal(score)

        st = self._load_state(tag)
        if st:
            # Check exits
            # Reprice spread (same way)
            curr_val = max(0.0, (p_hi - p_lo))
            pnl_pct = 100.0 * (curr_val - st.cost_usd / max(1e-9, st.qty_long)) / max(1e-9, st.cost_usd / max(1e-9, st.qty_long))
            # Time decay / roll
            if (score <= EXIT_SCORE) or (pnl_pct >= TP_PCT) or (pnl_pct <= -MAX_DECAY_PCT) or (dte <= MIN_DTE_EXIT):
                self._close_put_spread(tag, st, k_hi, k_lo, dte)
            return

        # Gate new entry
        if r.get(_poskey(self.ctx.name, tag)) is not None: return
        if score < ENTRY_SCORE: return

        # Budget & sizing (equity options multiplier = 100)
        budget_usd = PORTFOLIO_NOTIONAL_USD * (BUDGET_BPS * 1e-4)
        mult = 100.0
        if spread_px * mult <= 0: return
        contracts = max(1.0, math.floor(budget_usd / (spread_px * mult)))
        if contracts < 1.0: return

        fee = _fees_bps("OPT") * 1e-4
        qty = contracts
        # Buy spread: +PUT(k_hi) / -PUT(k_lo)
        self.order(f"OPT_PUT:{SYM}:{int(k_hi)}:{dte}", "buy",  qty=qty, order_type="market", venue="OPT")
        self.order(f"OPT_PUT:{SYM}:{int(k_lo)}:{dte}", "sell", qty=qty, order_type="market", venue="OPT")

        self._save_state(tag, OpenState(
            mode="PUT_SPREAD", tag=tag, qty_long=qty, qty_short=qty, k_long=k_hi, k_short=k_lo,
            dte=dte, cost_usd=spread_px*mult*qty*(1+fee), entry_score=score, ts_ms=_now_ms()
        ))

    def _close_put_spread(self, tag: str, st: OpenState, k_hi: float, k_lo: float, dte: int) -> None:
        # Reverse legs
        self.order(f"OPT_PUT:{SYM}:{int(k_hi)}:{dte}", "sell", qty=st.qty_long, order_type="market", venue="OPT")
        self.order(f"OPT_PUT:{SYM}:{int(k_lo)}:{dte}", "buy",  qty=st.qty_short, order_type="market", venue="OPT")
        r.delete(_poskey(self.ctx.name, tag))

    # ---------------- VIX_CALLS ----------------
    def _run_vix_calls(self) -> None:
        f1 = _px("VIX:FRONT")
        if f1 is None or f1 <= 0: return
        k = round(VIX_K_MULT * f1)
        dte = TENOR_DAYS
        tag = f"VXC:{k}:{dte}"

        # Price: mid if available, else BSM with huge IV as proxy
        c_mid = _opt_mid("VIX", "CALL", k, dte)
        if c_mid is None:
            iv = _iv_atm("VIX", dte) or 1.0
            T = max(1.0/365.0, dte/365.0)
            # treat S ≈ f1 for option pricing
            c_mid = _bsm_call_price(f1, k, T, iv)

        score = self._crash_score(f"EQ:{SYM}")
        self.emit_signal(score)

        st = self._load_state(tag)
        if st:
            curr_val = max(0.0, c_mid) * 100.0
            entry_unit = st.cost_usd / max(1e-9, st.qty_long)
            pnl_pct = 100.0 * (curr_val - entry_unit) / max(1e-9, entry_unit)
            if (score <= EXIT_SCORE) or (pnl_pct >= TP_PCT) or (pnl_pct <= -MAX_DECAY_PCT) or (dte <= MIN_DTE_EXIT):
                self.order(f"OPT_CALL:VIX:{int(k)}:{dte}", "sell", qty=st.qty_long, order_type="market", venue="OPT")
                r.delete(_poskey(self.ctx.name, tag))
            return

        if r.get(_poskey(self.ctx.name, tag)) is not None: return
        if score < ENTRY_SCORE: return

        budget_usd = PORTFOLIO_NOTIONAL_USD * (BUDGET_BPS * 1e-4)
        mult = 100.0
        if c_mid * mult <= 0: return
        qty = max(1.0, math.floor(budget_usd / (c_mid * mult)))
        if qty < 1.0: return

        fee = _fees_bps("OPT") * 1e-4
        self.order(f"OPT_CALL:VIX:{int(k)}:{dte}", "buy", qty=qty, order_type="market", venue="OPT")
        self._save_state(tag, OpenState(
            mode="VIX_CALLS", tag=tag, qty_long=qty, qty_short=0.0, k_long=k, k_short=0.0,
            dte=dte, cost_usd=c_mid*mult*qty*(1+fee), entry_score=score, ts_ms=_now_ms()
        ))

    # ---------------- state I/O ----------------
    def _load_state(self, tag: str) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, tag))
        if not raw: return None
        try:
            o = json.loads(raw) # type: ignore
            return OpenState(mode=str(o["mode"]), tag=str(o["tag"]),
                             qty_long=float(o["qty_long"]), qty_short=float(o["qty_short"]),
                             k_long=float(o["k_long"]), k_short=float(o["k_short"]),
                             dte=int(o["dte"]), cost_usd=float(o["cost_usd"]),
                             entry_score=float(o["entry_score"]), ts_ms=int(o["ts_ms"]))
        except Exception:
            return None

    def _save_state(self, tag: str, st: Optional[OpenState]) -> None:
        if st is None: return
        r.set(_poskey(self.ctx.name, tag), json.dumps({
            "mode": st.mode, "tag": st.tag,
            "qty_long": st.qty_long, "qty_short": st.qty_short,
            "k_long": st.k_long, "k_short": st.k_short,
            "dte": st.dte, "cost_usd": st.cost_usd,
            "entry_score": st.entry_score, "ts_ms": st.ts_ms
        }))