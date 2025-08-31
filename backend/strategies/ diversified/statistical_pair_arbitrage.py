# backend/strategies/diversified/statistical_pair_arbitrage.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import redis
from backend.engine.strategy_base import Strategy

"""
Statistical Pairs Arbitrage — paper
-----------------------------------
Modes:
  1) PAIR_ZSCORE   : fixed beta (from Redis) and rolling spread EWMA → z-score gates
  2) BETA_KALMAN   : time-varying beta via simple Kalman filter; same z gates on residual

Paper routing (adapters map to your broker later):
  • "EQ:<SYM>" with side in {"buy","sell"}, qty in shares

Redis feeds you publish elsewhere:
  # Last prices (mid or trade)
  HSET last_price "EQ:SYM1" '{"price": <px1>}'
  HSET last_price "EQ:SYM2" '{"price": <px2>}'

  # Meta & params
  HSET pair:meta <TAG> '{"sym1":"SYM1","sym2":"SYM2","ccy":"USD",
                         "beta":0.8,"hedge_on_log":true,"half_life_days":5,
                         "tick_size":0.01,"lot":1}'
  # Optional fees per venue (bps)
  HSET fees:eq EXCH 10

  # Ops
  SET  risk:halt 0|1
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("PAIR_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("PAIR_REDIS_PORT", "6379"))

MODE       = os.getenv("PAIR_MODE", "PAIR_ZSCORE").upper()    # PAIR_ZSCORE | BETA_KALMAN
TAG        = os.getenv("PAIR_TAG", "SPREAD-SPY-QQQ").upper()
VENUE_EQ   = os.getenv("PAIR_VENUE_EQ", "EXCH").upper()

# Thresholds
ENTRY_Z    = float(os.getenv("PAIR_ENTRY_Z", "2.0"))
EXIT_Z     = float(os.getenv("PAIR_EXIT_Z",  "0.5"))
STOP_Z     = float(os.getenv("PAIR_STOP_Z",  "4.0"))          # emergency exit if |z| explodes
TAKE_Z     = float(os.getenv("PAIR_TAKE_Z",  "0.0"))          # optional take-profit (<= EXIT_Z to disable)
MAX_CONCURRENT = int(os.getenv("PAIR_MAX_CONCURRENT", "1"))

# Sizing / risk
USD_NOTIONAL   = float(os.getenv("PAIR_USD_NOTIONAL", "30000"))
MIN_TICKET_USD = float(os.getenv("PAIR_MIN_TICKET_USD", "200"))
MAX_LEVER      = float(os.getenv("PAIR_MAX_LEVER", "1.5"))    # cap gross exposure / USD_NOTIONAL

# Stats
EWMA_ALPHA_S   = float(os.getenv("PAIR_EWMA_ALPHA_S", "0.06"))  # spread mean/var smoothing
RECHECK_SECS   = float(os.getenv("PAIR_RECHECK_SECS", "0.6"))

# Redis keys
HALT_KEY   = os.getenv("PAIR_HALT_KEY", "risk:halt")
LAST_HK    = os.getenv("PAIR_LAST_HK",  "last_price")
META_HK    = os.getenv("PAIR_META_HK",  "pair:meta")
FEES_HK    = os.getenv("PAIR_FEES_HK",  "fees:eq")

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

def _price(sym: str) -> Optional[float]:
    raw = r.hget(LAST_HK, sym)
    if not raw: return None
    try:
        j = json.loads(raw); return float(j.get("price", 0.0)) # type: ignore
    except Exception:
        try: return float(raw) # type: ignore
        except Exception: return None

def _fees_bps(venue: str) -> float:
    v = r.hget(FEES_HK, venue)
    try: return float(v) if v is not None else 10.0 # type: ignore
    except Exception: return 10.0

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
    return f"pair:ewma:{tag}"

def _load_ewma(tag: str) -> EwmaMV:
    raw = r.get(_ewma_key(tag))
    if raw:
        try:
            o = json.loads(raw) # type: ignore
            return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", EWMA_ALPHA_S)))
        except Exception: pass
    return EwmaMV(mean=0.0, var=1.0, alpha=EWMA_ALPHA_S)

def _save_ewma(tag: str, ew: EwmaMV) -> None:
    r.set(_ewma_key(tag), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ============================ Kalman (simple) ============================
@dataclass
class KalmanBeta:
    beta: float
    P: float       # variance
    q: float       # process noise
    r: float       # obs noise
    def step(self, x: float, y: float) -> float:
        # model: y = beta*x + eps, beta_{t}=beta_{t-1}+w, w~N(0,q), eps~N(0,r)
        # Predict
        beta_pred = self.beta
        P_pred = self.P + self.q
        # Update
        # H = x; K = P_pred*H / (H*P_pred*H + r) -> scalar
        S = P_pred * (x*x) + self.r
        K = (P_pred * x) / max(1e-12, S)
        self.beta = beta_pred + K * (y - x * beta_pred)
        self.P = (1.0 - K * x) * P_pred
        return self.beta

def _kalman_key(tag: str) -> str:
    return f"pair:kalman:{tag}"

def _load_kalman(tag: str) -> KalmanBeta:
    raw = r.get(_kalman_key(tag))
    if raw:
        try:
            o = json.loads(raw) # type: ignore
            return KalmanBeta(beta=float(o["b"]), P=float(o["P"]), q=float(o["q"]), r=float(o["r"]))
        except Exception: pass
    # defaults
    return KalmanBeta(beta=1.0, P=1.0, q=1e-4, r=1e-2)

def _save_kalman(tag: str, kb: KalmanBeta) -> None:
    r.set(_kalman_key(tag), json.dumps({"b": kb.beta, "P": kb.P, "q": kb.q, "r": kb.r}))

# ============================ state ============================
@dataclass
class OpenState:
    side: str        # "long1_short2" or "short1_long2"
    qty1: float
    qty2: float
    beta: float
    entry_z: float
    ts_ms: int

def _poskey(name: str, tag: str) -> str:
    return f"pair:open:{name}:{tag}"

# ============================ Strategy ============================
class StatisticalPairArbitrage(Strategy):
    """
    Z-score mean-reversion pairs with optional Kalman beta; paper orders.
    """
    def __init__(self, name: str = "statistical_pair_arbitrage", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_tick(self, tick: Dict) -> None:
        if (r.get(HALT_KEY) or "0") == "1": return
        now = time.time()
        if now - self._last < RECHECK_SECS: return
        self._last = now

        meta = _hget_json(META_HK, TAG) or {}
        s1 = (meta.get("sym1") or "SYM1").upper()
        s2 = (meta.get("sym2") or "SYM2").upper()
        hedge_on_log = bool(meta.get("hedge_on_log", True))
        beta_fixed = float(meta.get("beta", 1.0))
        lot = float(meta.get("lot", 1.0))
        tick_size = float(meta.get("tick_size", 0.01))

        p1 = _price(f"EQ:{s1}"); p2 = _price(f"EQ:{s2}")
        if p1 is None or p2 is None or p1 <= 0 or p2 <= 0: return

        # Build regressor/response for beta & spread
        x = math.log(p2) if hedge_on_log else p2
        y = math.log(p1) if hedge_on_log else p1

        if MODE == "BETA_KALMAN":
            kb = _load_kalman(TAG)
            beta = max(1e-6, kb.step(x, y))
            _save_kalman(TAG, kb)
        else:
            beta = max(1e-6, beta_fixed)

        spread = y - beta * x
        ew = _load_ewma(TAG); m,v = ew.update(spread); _save_ewma(TAG, ew)
        z = (spread - m) / math.sqrt(max(v, 1e-12))

        # Emit normalized signal for dashboards
        self.emit_signal(max(-1.0, min(1.0, z / max(1e-6, ENTRY_Z))))

        # Manage open state
        st = self._load_state(TAG)
        if st:
            # Exit conditions: z back to EXIT_Z, or stop, or take-profit (optional)
            if (abs(z) <= EXIT_Z) or (abs(z) >= STOP_Z) or (TAKE_Z > 0 and abs(z) <= TAKE_Z):
                self._close_positions(TAG, st, s1, s2)
            return

        # Gate for new entries
        if r.get(_poskey(self.ctx.name, TAG)) is not None: return
        if abs(z) < ENTRY_Z: return

        # Sizing: target dollar‑neutral on logs: qty2 ≈ beta * qty1 * (p1/p2)
        usd = USD_NOTIONAL
        if usd < MIN_TICKET_USD: return
        # Entry on z>0: spread high ⇒ y too high vs x → short 1, long 2; z<0 inverse
        # Choose qty1 such that gross ≈ USD_NOTIONAL, respect leverage cap
        qty1 = max(lot, math.floor((usd / (p1 + beta * p2)) / lot) * lot)
        qty2 = max(lot, math.floor((beta * qty1 * p1 / p2) / lot) * lot)

        gross = qty1 * p1 + qty2 * p2
        if gross > USD_NOTIONAL * MAX_LEVER or gross < MIN_TICKET_USD: return

        fee = _fees_bps(VENUE_EQ) * 1e-4  # bps guard

        if z > 0:
            # Short SYM1, Long SYM2
            self.order(f"EQ:{s1}", "sell", qty=qty1, order_type="market", venue=VENUE_EQ)
            self.order(f"EQ:{s2}", "buy",  qty=qty2, order_type="market", venue=VENUE_EQ)
            side = "short1_long2"
        else:
            # Long SYM1, Short SYM2
            self.order(f"EQ:{s1}", "buy",  qty=qty1, order_type="market", venue=VENUE_EQ)
            self.order(f"EQ:{s2}", "sell", qty=qty2, order_type="market", venue=VENUE_EQ)
            side = "long1_short2"

        self._save_state(TAG, OpenState(side=side, qty1=qty1, qty2=qty2, beta=beta, entry_z=z, ts_ms=_now_ms()))

    # ---------------- close / unwind ----------------
    def _close_positions(self, tag: str, st: OpenState, s1: str, s2: str) -> None:
        if st.side == "short1_long2":
            self.order(f"EQ:{s1}", "buy",  qty=st.qty1, order_type="market", venue=VENUE_EQ)
            self.order(f"EQ:{s2}", "sell", qty=st.qty2, order_type="market", venue=VENUE_EQ)
        else:
            self.order(f"EQ:{s1}", "sell", qty=st.qty1, order_type="market", venue=VENUE_EQ)
            self.order(f"EQ:{s2}", "buy",  qty=st.qty2, order_type="market", venue=VENUE_EQ)
        r.delete(_poskey(self.ctx.name, tag))

    # ---------------- state I/O ----------------
    def _load_state(self, tag: str) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, tag))
        if not raw: return None
        try:
            o = json.loads(raw) # type: ignore
            return OpenState(side=str(o["side"]), qty1=float(o["qty1"]), qty2=float(o["qty2"]),
                             beta=float(o["beta"]), entry_z=float(o["entry_z"]), ts_ms=int(o["ts_ms"]))
        except Exception:
            return None

    def _save_state(self, tag: str, st: Optional[OpenState]) -> None:
        if st is None: return
        r.set(_poskey(self.ctx.name, tag), json.dumps({
            "side": st.side, "qty1": st.qty1, "qty2": st.qty2,
            "beta": st.beta, "entry_z": st.entry_z, "ts_ms": st.ts_ms
        }))