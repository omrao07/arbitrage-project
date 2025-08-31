# backend/strategies/diversified/recession_indicator_alpha.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional

import redis
from backend.engine.strategy_base import Strategy

"""
Recession Indicator Alpha — paper
---------------------------------
Inputs (publish elsewhere to Redis; daily/weekly is fine):

HSET macro:factors "YC_10Y_3M"     -1.35   # 10y-3m (pct points)
HSET macro:factors "YC_10Y_2Y"     -0.60
HSET macro:factors "UNEMP"          0.040  # unemployment rate
HSET macro:factors "UNEMP_3M_CHG"   0.40   # Sahm-like rule (%-pts *100) e.g., +0.50 → 0.50
HSET macro:factors "PMI_COMP"      49.3    # composite PMI
HSET macro:factors "CREDIT_IG_OAS"  0.013  # 130 bps
HSET macro:factors "CREDIT_HY_OAS"  0.040  # 400 bps
HSET macro:factors "LEI_6M_ANN"    -0.035  # LEI 6m annualized
HSET macro:factors "EARN_REV_Z"    -0.6    # analyst revisions z (neg worse)
HSET macro:factors "VIX"           18.0

# Prices for proxies (for sizing)
HSET last_price "ETF:SPY" '{"price":530.0}'
HSET last_price "ETF:TLT" '{"price":95.2}'
HSET last_price "ETF:IEF" '{"price":96.5}'
HSET last_price "ETF:LQD" '{"price":112.3}'
HSET last_price "ETF:HYG" '{"price":78.1}'
HSET last_price "ETF:GLD" '{"price":215.0}'
HSET last_price "ETF:UUP" '{"price":31.2}'

# Fees (bps) & ops
HSET fees:etf EXCH 2
SET  risk:halt 0|1

Routing (paper; adapters later):
  order("ETF:<TICKER>", side, qty, order_type="market", venue="EXCH")
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("RIA_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("RIA_REDIS_PORT", "6379"))

FACT_HK   = os.getenv("RIA_FACT_HK", "macro:factors")
LAST_HK   = os.getenv("RIA_LAST_HK", "last_price")
FEES_HK   = os.getenv("RIA_FEES_HK", "fees:etf")
HALT_KEY  = os.getenv("RIA_HALT_KEY", "risk:halt")
STATE_KEY = os.getenv("RIA_STATE_KEY", "ria:state")

RECHECK_SECS   = float(os.getenv("RIA_RECHECK_SECS", "10.0"))
ENTRY_P        = float(os.getenv("RIA_ENTRY_P", "0.55"))  # act if p_rec >= 55%
EXIT_P         = float(os.getenv("RIA_EXIT_P",  "0.48"))  # hysteresis
COOLDOWN_SECS  = float(os.getenv("RIA_COOLDOWN_SECS", "1800"))  # 30 min

MAX_GROSS_USD  = float(os.getenv("RIA_MAX_GROSS_USD", "60000"))
MIN_TICKET_USD = float(os.getenv("RIA_MIN_TICKET_USD", "200"))
LOT            = float(os.getenv("RIA_LOT", "1"))

# Universe proxies
PROXIES = {
    "EQUITIES": "ETF:SPY",
    "UST_20Y":  "ETF:TLT",
    "UST_7_10": "ETF:IEF",
    "IG":       "ETF:LQD",
    "HY":       "ETF:HYG",
    "GOLD":     "ETF:GLD",
    "USD":      "ETF:UUP",
}

# Defensive weights when recession risk is high (gross scaled by MAX_GROSS_USD)
DEF_WEIGHTS = {
    "EQUITIES": -0.40,
    "HY":       -0.15,
    "IG":       +0.15,
    "UST_7_10": +0.20,
    "UST_20Y":  +0.20,
    "GOLD":     +0.10,
    "USD":      +0.10,
}

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================ helpers ============================
def _hgetf(hk: str, field: str) -> Optional[float]:
    raw = r.hget(hk, field)
    if raw is None: return None
    try: return float(raw) # type: ignore
    except Exception:
        try:
            j = json.loads(raw) # type: ignore
            if isinstance(j, dict) and "price" in j: return float(j["price"])
            if isinstance(j, (int,float)): return float(j)
        except Exception:
            return None
    return None

def _px(sym: str) -> Optional[float]:
    return _hgetf(LAST_HK, sym)

def _fees_bps(venue: str="EXCH") -> float:
    v = r.hget(FEES_HK, venue)
    try: return float(v) if v is not None else 2.0 # type: ignore
    except Exception: return 2.0

def _now() -> float: return time.time()

# ============================ probability model ============================
def _sigmoid(x: float) -> float:
    return 1.0/(1.0+math.exp(-x))

def _std(x: float, m: float, s: float) -> float:
    return (x - m)/max(1e-6, s)

def _recession_probability(f: Dict[str,float]) -> float:
    """
    Lightweight logistic blend of standardized indicators.
    Tunables are rough priors; calibrate later with history if desired.
    """
    pri = dict(
        YC_10Y_3M=(0.75, 1.0),     # average curve slope (pp)
        YC_10Y_2Y=(0.20, 0.6),
        UNEMP=(0.045, 0.01),
        UNEMP_3M_CHG=(0.00, 0.25), # Sahm-rule like
        PMI_COMP=(52.0, 3.0),
        CREDIT_IG_OAS=(0.012, 0.004),
        CREDIT_HY_OAS=(0.045, 0.015),
        LEI_6M_ANN=(0.00, 0.02),
        EARN_REV_Z=(0.00, 1.0),
        VIX=(18.0, 6.0),
    )
    z = {k: _std(float(f.get(k, pri[k][0])), *pri[k]) for k in pri.keys()}

    # Signs: inverted curve (neg YC) ↑ risk; rising UNEMP / PMI<50 ↑ risk; wider spreads ↑ risk; LEI<0 ↑ risk
    s = (-1.2*z["YC_10Y_3M"]) + (-0.6*z["YC_10Y_2Y"]) \
        + (0.9*z["UNEMP"]) + (1.2*z["UNEMP_3M_CHG"]) \
        + (-0.9*z["PMI_COMP"]) \
        + (0.8*z["CREDIT_IG_OAS"]) + (1.1*z["CREDIT_HY_OAS"]) \
        + (1.0*z["LEI_6M_ANN"]) \
        + (0.5*(-z["EARN_REV_Z"])) \
        + (0.3*z["VIX"])

    p = _sigmoid(s)
    return float(max(0.0, min(1.0, p)))

# ============================ state ============================
@dataclass
class RIAState:
    active: bool
    p_rec: float
    last_rebalance_s: float

def _state_key(ctx: str) -> str: return f"{STATE_KEY}:{ctx}"

def _load_state(ctx: str) -> Optional[RIAState]:
    raw = r.get(_state_key(ctx))
    if not raw: return None
    try:
        o = json.loads(raw) # type: ignore
        return RIAState(active=bool(o.get("active", False)),
                        p_rec=float(o.get("p_rec", 0.0)),
                        last_rebalance_s=float(o.get("last_rebalance_s", 0.0)))
    except Exception:
        return None

def _save_state(ctx: str, st: RIAState) -> None:
    r.set(_state_key(ctx), json.dumps({"active": st.active, "p_rec": st.p_rec, "last_rebalance_s": st.last_rebalance_s}))

# ============================ Strategy ============================
class RecessionIndicatorAlpha(Strategy):
    """
    Converts macro indicators into recession probability → defensive overlay trades (paper).
    """
    def __init__(self, name: str = "recession_indicator_alpha", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_tick(self, tick: Dict) -> None:
        if (r.get(HALT_KEY) or "0") == "1": return
        now = _now()
        if now - self._last < RECHECK_SECS: return
        self._last = now

        # read factors
        f = {k: float(v) for k,v in (r.hgetall(FACT_HK) or {}).items()} # type: ignore
        if not f:
            self.emit_signal(0.0); return

        p = _recession_probability(f)

        # UI heartbeat: signed tilt (risk-off positive)
        self.emit_signal(max(-1.0, min(1.0, (p - 0.5)*2.0)))

        st = _load_state(self.ctx.name) or RIAState(active=False, p_rec=p, last_rebalance_s=0.0)

        # hysteresis
        want_active = st.active
        if st.active:
            if p <= EXIT_P:
                want_active = False
        else:
            if p >= ENTRY_P:
                want_active = True

        # cooldown
        if now - st.last_rebalance_s < COOLDOWN_SECS:
            _save_state(self.ctx.name, RIAState(active=want_active, p_rec=p, last_rebalance_s=st.last_rebalance_s))
            return

        # rebalance if state flip OR periodic refresh while active
        if (want_active != st.active) or (want_active and (now - st.last_rebalance_s >= COOLDOWN_SECS)):
            self._rebalance(active=want_active)
            _save_state(self.ctx.name, RIAState(active=want_active, p_rec=p, last_rebalance_s=now))

    # --------------- rebalance helper ---------------
    def _rebalance(self, active: bool) -> None:
        # For paper simplicity we just place target legs (delta-only could be added later).
        tgt = DEF_WEIGHTS if active else {k: 0.0 for k in DEF_WEIGHTS.keys()}

        budget = MAX_GROSS_USD
        for sleeve, w in tgt.items():
            sym = PROXIES.get(sleeve)
            if not sym: continue
            px = _px(sym)
            if not px or px <= 0: continue
            notional = abs(w) * budget
            if notional < MIN_TICKET_USD: continue
            qty = math.floor((notional / px) / max(1.0, LOT)) * LOT
            if qty <= 0: continue

            side = "buy" if w > 0 else "sell"
            if w == 0:
                # flatten both sides conservatively (send both directions once for paper)
                self.order(sym, "buy",  qty=qty, order_type="market", venue="EXCH")
                self.order(sym, "sell", qty=qty, order_type="market", venue="EXCH")
            else:
                self.order(sym, side, qty=qty, order_type="market", venue="EXCH")