# backend/strategies/diversified/gamma_scalping.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import redis
from backend.engine.strategy_base import Strategy

"""
Gamma Scalping (ATM Straddle, paper)
------------------------------------
Mode:
  1) BUILD_AND_HEDGE (default): if IV is "cheap" vs realized (or a floor), BUY ATM straddle
     and keep delta ≈ 0 by trading the underlier as price moves (scalp gamma).
  2) HEDGE_ONLY: assume you already hold the straddle; we only run the delta hedger.

Redis you already publish elsewhere in this repo:
  HSET last_price <SYM> '{"price": <spot>}'               # underlier
  HSET iv:imp:<TENOR> <SYM> <iv_decimal>                  # implied vol (annualized, e.g. 0.32)
  (optional) HSET iv:real:<TENOR> <SYM> <rv_decimal>      # realized vol est (annualized)
  (optional) HSET opt:mid:<TENOR> "<SYM>:<K>:C" <price>   # mid prices if you want sanity checks
  (optional) HSET opt:mid:<TENOR> "<SYM>:<K>:P" <price>

Paper symbols (map later in your adapter):
  • Underlier (spot): <SYM>  (equities/ETF) or "SPOT:<SYM>" if you prefer; see VENUE_EQ.
  • Options        : OPT:<SYM>:<K>:<TENOR>:C / :P

Core:
  • Long ATM straddle (qty N). Compute Black‑Scholes greeks (γ, Δ) with your IV.
  • If |Δ_port| > DELTA_BAND per straddle * N → trade underlier to re‑center delta≈0.
  • Optional IV edge gate to (re)build only when IV ≤ min(RV_EST − EDGE_VOL, IV_CEIL).

Notes:
  • This is *intraday* style hedging. We don’t theta/PnL‑attribute; your recorder handles that.
  • Size in USD notional per straddle; strike snapped to STRIKE_STEP around spot.
"""

# ========================= CONFIG (env) =========================
REDIS_HOST = os.getenv("GSCALP_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("GSCALP_REDIS_PORT", "6379"))

SYM   = os.getenv("GSCALP_SYMBOL", "SPY").upper()
TENOR = os.getenv("GSCALP_TENOR", "30D").upper()

MODE = os.getenv("GSCALP_MODE", "BUILD_AND_HEDGE").upper()  # "BUILD_AND_HEDGE" | "HEDGE_ONLY"
LONG_ONLY = os.getenv("GSCALP_LONG_ONLY", "true").lower() in ("1","true","yes")

# Strike snap
STRIKE_STEP = float(os.getenv("GSCALP_STRIKE_STEP", "1.0"))  # e.g., 1 for equities, 5 for index future

# Entry (IV edge)
USE_RV_EDGE  = os.getenv("GSCALP_USE_RV_EDGE", "true").lower() in ("1","true","yes")
EDGE_VOL     = float(os.getenv("GSCALP_EDGE_VOL", "0.03"))   # need RV_EST - IV_IMP >= EDGE_VOL (3 vol pts)
IV_CEIL      = float(os.getenv("GSCALP_IV_CEIL", "0.80"))    # never buy over this IV
MIN_IV       = float(os.getenv("GSCALP_MIN_IV", "0.05"))     # ignore quotes below this

# Sizing
USD_PER_STRADDLE = float(os.getenv("GSCALP_USD_PER_STRADDLE", "20000"))
MIN_TICKET_USD   = float(os.getenv("GSCALP_MIN_TICKET_USD", "200"))
MAX_CONCURRENT   = int(os.getenv("GSCALP_MAX_CONCURRENT", "1"))

# Hedging
DELTA_BAND_PER_STRADDLE = float(os.getenv("GSCALP_DELTA_BAND", "0.05"))  # 0.05 = 5 deltas per straddle
REHEDGE_COOLDOWN_SECS   = int(os.getenv("GSCALP_COOLDOWN", "1"))

# Cadence
RECHECK_SECS = int(os.getenv("GSCALP_RECHECK_SECS", "1"))

# Venues (advisory)
VENUE_EQ  = os.getenv("GSCALP_VENUE_EQ", "ARCA").upper()
VENUE_OPT = os.getenv("GSCALP_VENUE_OPT", "CBOE").upper()

# Redis keys
LAST_PRICE_HKEY = os.getenv("GSCALP_LAST_PRICE_KEY", "last_price")
IV_IMP_HKEY     = f"iv:imp:{TENOR}"
IV_REAL_HKEY    = f"iv:real:{TENOR}"
OPT_MID_HKEY    = os.getenv("GSCALP_OPT_MID_HKEY", f"opt:mid:{TENOR}")

# ========================= Redis =========================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ========================= Utils / BS =========================
def _hget_price(sym: str) -> Optional[float]:
    raw = r.hget(LAST_PRICE_HKEY, sym)
    if not raw: return None
    try: return float(json.loads(raw)["price"])
    except Exception:
        try: return float(raw)
        except Exception: return None

def _hgetf(hkey: str, field: str) -> Optional[float]:
    v = r.hget(hkey, field)
    if v is None: return None
    try: return float(v)
    except Exception:
        try: return float(json.loads(v))
        except Exception: return None

def _round_strike(S: float, step: float) -> float:
    return round(S / step) * step if step > 0 else S

def _tenor_years(t: str) -> float:
    t = t.upper().strip()
    if t.endswith("D"):  return max(1e-9, float(t[:-1] or 1.0) / 365.0)
    if t.endswith("M"):  return max(1e-9, 30.0 * float(t[:-1] or 1.0) / 365.0)
    if t.endswith("Y"):  return max(1e-9, float(t[:-1] or 1.0))
    return 30.0 / 365.0

T_YEARS = _tenor_years(TENOR)

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5*x*x)/math.sqrt(2.0*math.pi)

def _bs_d1(S: float, K: float, sigma: float, T: float) -> float:
    if S<=0 or K<=0 or sigma<=0 or T<=0: return 0.0
    return (math.log(S/K) + 0.5*sigma*sigma*T) / (sigma*math.sqrt(T))

def _call_delta(S: float, K: float, sigma: float, T: float) -> float:
    d1 = _bs_d1(S,K,sigma,T); return 0.5*(1.0+math.erf(d1/math.sqrt(2.0)))

def _gamma(S: float, K: float, sigma: float, T: float) -> float:
    if S<=0 or sigma<=0 or T<=0: return 0.0
    d1 = _bs_d1(S,K,sigma,T)
    return _norm_pdf(d1)/(S*sigma*math.sqrt(T))

def _vega(S: float, K: float, sigma: float, T: float) -> float:
    if S<=0 or sigma<=0 or T<=0: return 0.0
    d1 = _bs_d1(S,K,sigma,T); return S*math.sqrt(T)*_norm_pdf(d1)  # per 1.00 vol

def _opt_sym(sym: str, K: float, cp: str) -> str:
    return f"OPT:{sym}:{K:.4f}:{TENOR}:{cp}"

# ========================= State =========================
@dataclass
class OpenState:
    K: float
    n_straddles: float
    delta_hedge_qty: float  # signed underlier qty (positive = long)
    last_hedge_ms: int
    entry_iv: float
    entry_spot: float
    ts_ms: int

def _poskey(name: str, sym: str, tenor: str) -> str:
    return f"gscalp:open:{name}:{sym}:{tenor}"

# ========================= Strategy =========================
class GammaScalping(Strategy):
    """
    Buy ATM straddle when IV is cheap (optional) and delta‑hedge around a band.
    """
    def __init__(self, name: str = "gamma_scalping", region: Optional[str] = "US", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    # ---- lifecycle ----
    def on_start(self) -> None:
        super().on_start()
        r.hset("strategy:universe", self.ctx.name, json.dumps({
            "symbol": SYM, "tenor": TENOR, "mode": MODE, "ts": int(time.time()*1000)
        }))

    def on_tick(self, tick: Dict) -> None:
        now = time.time()
        if now - self._last < RECHECK_SECS:
            return
        self._last = now
        # ensure position
        st = self._load_state()
        if not st and MODE == "BUILD_AND_HEDGE":
            self._maybe_build()
            st = self._load_state()
        if st:
            self._maybe_hedge(st)

    # ---- build logic ----
    def _maybe_build(self) -> None:
        S = _hget_price(SYM)
        iv = _hgetf(IV_IMP_HKEY, SYM)
        if S is None or S <= 0 or iv is None or iv < MIN_IV or iv > IV_CEIL:
            return
        if USE_RV_EDGE:
            rv = _hgetf(IV_REAL_HKEY, SYM)
            if rv is None or (rv - iv) < EDGE_VOL:
                return

        # size → shares per leg from USD_PER_STRADDLE
        K = _round_strike(S, STRIKE_STEP)
        # quick vega sanity to avoid dust sizing on very low vol
        vega = _vega(S, K, iv, T_YEARS)  # per 1.00 vol per 1 straddle ~ 2*call vega, but close enough
        qty_approx = max(1.0, USD_PER_STRADDLE / max(1e-9, S))  # shares per leg proxy
        if qty_approx * S < MIN_TICKET_USD:
            return

        # Place: buy 1*call + 1*put per "straddle unit" (we aggregate into single order qty)
        # Paper: we just buy 'qty_approx' of each option.
        self.order(_opt_sym(SYM, K, "C"), "buy",  qty=qty_approx, order_type="market", venue=VENUE_OPT)
        self.order(_opt_sym(SYM, K, "P"), "buy",  qty=qty_approx, order_type="market", venue=VENUE_OPT)

        st = OpenState(
            K=K, n_straddles=qty_approx, delta_hedge_qty=0.0,
            last_hedge_ms=0, entry_iv=iv, entry_spot=S, ts_ms=int(time.time()*1000)
        )
        r.set(_poskey(self.ctx.name, SYM, TENOR), json.dumps(st.__dict__))

    # ---- hedge logic ----
    def _maybe_hedge(self, st: OpenState) -> None:
        S = _hget_price(SYM)
        iv = _hgetf(IV_IMP_HKEY, SYM) or st.entry_iv  # fall back to entry iv
        if S is None or S <= 0 or iv < MIN_IV:
            return

        # Straddle delta ≈ 2*N(d1) - 1 at K ≈ ATM. Multiply by number of straddles.
        Dc = _call_delta(S, st.K, iv, T_YEARS)
        straddle_delta = (2.0 * Dc - 1.0)  # per 1 straddle
        port_delta = straddle_delta * st.n_straddles - st.delta_hedge_qty  # hedge qty offsets
        band = DELTA_BAND_PER_STRADDLE * st.n_straddles

        now_ms = int(time.time()*1000)
        if abs(port_delta) >= band and (now_ms - st.last_hedge_ms) >= REHEDGE_COOLDOWN_SECS * 1000:
            # trade underlier to neutralize delta: buy if port_delta > 0 (need more long underlier)
            qty = port_delta  # shares to buy (positive) or sell (negative)
            side = "buy" if qty > 0 else "sell"
            self.order(SYM, side, qty=abs(qty), order_type="market", venue=VENUE_EQ)
            st.delta_hedge_qty += qty
            st.last_hedge_ms = now_ms
            self._save_state(st)

        # Emit a monitoring signal: scaled delta in bands
        sig = max(-1.0, min(1.0, port_delta / max(1e-9, band)))
        self.emit_signal(sig)

    # ---- state io ----
    def _load_state(self) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, SYM, TENOR))
        if not raw: return None
        try:
            o = json.loads(raw)
            return OpenState(**o)
        except Exception:
            return None

    def _save_state(self, st: OpenState) -> None:
        r.set(_poskey(self.ctx.name, SYM, TENOR), json.dumps(st.__dict__))

    # ---- manual close helper (optional) ----
    def close_all(self) -> None:
        st = self._load_state()
        if not st: return
        # unwind delta hedge
        if abs(st.delta_hedge_qty) > 0:
            side = "sell" if st.delta_hedge_qty > 0 else "buy"
            self.order(SYM, side, qty=abs(st.delta_hedge_qty), order_type="market", venue=VENUE_EQ)
        # NOTE: Options unwind is left to your OMS/adapter (or add symmetric sell orders here).
        r.delete(_poskey(self.ctx.name, SYM, TENOR))