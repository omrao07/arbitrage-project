# backend/strategies/diversified/convertible_bond_arbitrage.py
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import redis

from backend.engine.strategy_base import Strategy

# ============================ CONFIG (env) ============================
REDIS_HOST = os.getenv("CBARB_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("CBARB_REDIS_PORT", "6379"))

# Identifiers (must match your feed/OMS/last_price keys)
EQ_SYMBOL     = os.getenv("CBARB_EQ_SYMBOL", "ACME").upper()           # underlying equity symbol in your feed
CB_SYMBOL     = os.getenv("CBARB_CB_SYMBOL", "ACME.28CB").upper()      # convertible bond symbol in your feed/OMS
VENUE_EQ      = os.getenv("CBARB_VENUE_EQ", "ARCA").upper()
VENUE_CB      = os.getenv("CBARB_VENUE_CB", "BONDS").upper()           # advisory (your router/OMS can map)

# Convertible terms
FACE_VALUE        = float(os.getenv("CBARB_FACE", "1000"))             # face per bond (e.g., $1000)
COUPON_RATE       = float(os.getenv("CBARB_COUPON", "0.02"))           # annual coupon (decimals)
COUPON_FREQ       = int(os.getenv("CBARB_COUPON_FREQ", "2"))           # payments per year (2 = semiannual)
CONV_RATIO        = float(os.getenv("CBARB_CONV_RATIO", "20.0"))       # shares per bond when converted
MATURITY_DATE     = os.getenv("CBARB_MATURITY", "2028-12-31")          # YYYY-MM-DD
CALL_STRIKE_PRICE = FACE_VALUE / max(CONV_RATIO, 1e-9)                 # implied conversion price (S strike)

# Market/Model inputs (Redis keys)
LAST_PRICE_HKEY = os.getenv("CBARB_LAST_PRICE_KEY", "last_price")      # HSET symbol -> {"price": ...}

# Optional Redis overrides for model inputs (if not provided per tick):
#   HGET iv:eq:<EQ_SYMBOL>         -> equity IV (annualized decimal)
#   HGET rate:risk_free            -> annualized rf
#   HGET spread:credit:<CB_SYMBOL> -> issuer credit spread (decimal)
IV_KEY           = f"iv:eq:{EQ_SYMBOL}"
RATE_KEY         = "rate:risk_free"
CREDIT_SPREAD_KEY= f"spread:credit:{CB_SYMBOL}"

# Trading thresholds
ENTRY_USD      = float(os.getenv("CBARB_ENTRY_USD", "2.0"))            # theo - mkt absolute (USD per bond)
EXIT_USD       = float(os.getenv("CBARB_EXIT_USD",  "0.75"))
DELTA_REHEDGE  = float(os.getenv("CBARB_DELTA_REHEDGE", "0.05"))       # re-hedge when delta drift > 5%
MAX_CONCURRENT = int(os.getenv("CBARB_MAX_CONCURRENT", "1"))
ALLOW_SHORT_CB = os.getenv("CBARB_ALLOW_SHORT_CB", "false").lower() in ("1","true","yes")

# Sizing
USD_PER_TRADE  = float(os.getenv("CBARB_USD_PER_TRADE", "25000"))      # target notional in CB (market value)
MIN_ORDER_USD  = float(os.getenv("CBARB_MIN_ORDER_USD", "200"))        # skip dust

# Heartbeat / cadence
RECHECK_SECS   = int(os.getenv("CBARB_RECHECK_SECS", "5"))

# ============================ REDIS ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def _now_ms() -> int:
    return int(time.time() * 1000)

def _hget_price(symbol: str) -> Optional[float]:
    raw = r.hget(LAST_PRICE_HKEY, symbol.upper())
    if not raw: return None
    try: return float(json.loads(raw)["price"]) # type: ignore
    except Exception:
        try: return float(raw) # type: ignore
        except Exception: return None

def _getf(key: str) -> Optional[float]:
    v = r.get(key)
    if v is None: return None
    try: return float(v) # type: ignore
    except Exception:
        try: return float(json.loads(v)) # type: ignore
        except Exception: return None

def _hgetf(key: str, field: str) -> Optional[float]:
    v = r.hget(key, field)
    if v is None: return None
    try: return float(v) # type: ignore
    except Exception:
        try: return float(json.loads(v)) # type: ignore
        except Exception: return None

# ============================ PRICING ============================
def _time_to_maturity_yr(today: datetime, maturity: str) -> float:
    try:
        mat = datetime.fromisoformat(maturity)
    except Exception:
        return 0.0
    dt = (mat - today).days
    return max(0.0, dt) / 365.0

def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def _bs_call(S: float, K: float, T: float, r: float, q: float, sigma: float) -> Tuple[float, float, float]:
    """
    Black–Scholes call with continuous dividend yield q.
    Returns (price, delta, gamma). (Theta/Vega not used here.)
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        # intrinsic as fallback (q ignored)
        price = max(0.0, S - K)
        delta = 1.0 if S > K else 0.0
        return price, delta, 0.0
    srt = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / srt
    d2 = d1 - srt
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    price = S * disc_q * _norm_cdf(d1) - K * disc_r * _norm_cdf(d2)
    delta = disc_q * _norm_cdf(d1)
    gamma = disc_q * _norm_pdf(d1) / (S * srt)
    return price, delta, gamma

def _pv_straight_bond(face: float, cpn_rate: float, freq: int, T: float, y: float) -> float:
    """
    Present value of a bullet coupon bond (no optionality), yield = r + credit_spread.
    """
    if T <= 0:
        return face
    n = max(1, int(round(T * freq)))
    dt = 1.0 / freq
    cpn = face * cpn_rate / freq
    pv = 0.0
    for i in range(1, n + 1):
        t = i * dt
        if t > T: t = T  # tiny clamp at the end
        pv += cpn / ((1 + y / freq) ** (i))
    pv += face / ((1 + y / freq) ** n)
    return pv

@dataclass
class CBInputs:
    S: float          # equity price
    CB: float         # CB market price (per bond)
    r: float          # risk-free rate (annualized decimal)
    spr: float        # credit spread (annualized decimal)
    sigma: float      # equity IV (annualized decimal)
    q: float          # dividend yield (decimal)
    T: float          # time to maturity in years

@dataclass
class CBVal:
    theo: float
    floor: float
    opt: float
    delta_cb_shares_per_bond: float
    gamma_cb: float

def _price_cb(inp: CBInputs) -> CBVal:
    # equity option leg
    call_px, call_delta, call_gamma = _bs_call(
        S=inp.S, K=CALL_STRIKE_PRICE, T=inp.T, r=inp.r, q=inp.q, sigma=inp.sigma
    )
    # convertible option value per bond = call * conversion ratio
    opt_val = call_px * CONV_RATIO
    # bond floor discounted at (r + spread)
    y = max(0.0, inp.r + max(inp.spr, 0.0))
    floor = _pv_straight_bond(FACE_VALUE, COUPON_RATE, COUPON_FREQ, inp.T, y)
    theo = floor + opt_val
    # CB equity delta (shares per bond) = call delta * conversion ratio
    delta_cb = call_delta * CONV_RATIO
    gamma_cb = call_gamma * (CONV_RATIO ** 2)  # approx shares^2 sensitivity per bond
    return CBVal(theo=theo, floor=floor, opt=opt_val, delta_cb_shares_per_bond=delta_cb, gamma_cb=gamma_cb)

# ============================ STATE ============================
def _poskey(name: str) -> str:
    return f"cbarb:open:{name}:{CB_SYMBOL}"

@dataclass
class OpenState:
    side: str               # "long_cb_short_eq" or "short_cb_long_eq"
    nbonds: float
    hedge_shares: float
    entry_edge: float       # theo - mkt at entry
    entry_ts_ms: int
    last_delta: float       # shares per bond at last hedge

# ============================ STRATEGY ============================
class ConvertibleBondArbitrage(Strategy):
    """
    Long cheap convertibles vs equity (delta‑hedged). Option leg via Black–Scholes, bond floor via (r+spread).
    Entry when |theo - market| >= ENTRY_USD; exit when |edge| <= EXIT_USD.
    Re‑hedges when CB delta drifts by > DELTA_REHEDGE.
    """

    def __init__(self, name: str = "convertible_bond_arbitrage", region: Optional[str] = "US", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last_check = 0.0

    # ---- lifecycle ----
    def on_start(self) -> None:
        super().on_start()
        r.hset("strategy:universe", self.ctx.name, json.dumps({"eq": EQ_SYMBOL, "cb": CB_SYMBOL, "ts": _now_ms()}))

    # ---- ticks ----
    def on_tick(self, tick: Dict) -> None:
        now = time.time()
        if now - self._last_check < RECHECK_SECS:
            return
        self._last_check = now
        self._evaluate()

    # ---- engine ----
    def _inputs(self) -> Optional[CBInputs]:
        S = _hget_price(EQ_SYMBOL)
        CBp = _hget_price(CB_SYMBOL)
        if S is None or CBp is None or S <= 0 or CBp <= 0:
            return None

        # pull model inputs (gracefully default)
        sigma = _getf(IV_KEY) or 0.35          # fallback IV 35%
        r = _getf(RATE_KEY) or 0.03            # fallback 3%
        spr = _getf(CREDIT_SPREAD_KEY) or 0.02 # fallback 200 bps
        q = 0.0                                # dividend yield; extend with HGET if you store it
        T = _time_to_maturity_yr(datetime.utcnow(), MATURITY_DATE)

        return CBInputs(S=S, CB=CBp, r=r, spr=spr, sigma=sigma, q=q, T=T)

    def _evaluate(self) -> None:
        inp = self._inputs()
        if not inp or inp.T <= 0:
            return

        val = _price_cb(inp)
        edge = val.theo - inp.CB  # >0 => cheap CB (buy CB / short equity)

        # emit a normalized signal for allocator/monitoring
        self.emit_signal(max(-1.0, min(1.0, math.tanh(edge / 5.0))))

        st = self._load_state()

        # ---------- manage open position ----------
        if st:
            # Exit condition
            if abs(edge) <= EXIT_USD:
                self._close(st)
                return

            # Re‑hedge if delta drifted
            if abs(val.delta_cb_shares_per_bond - st.last_delta) >= DELTA_REHEDGE:
                target_shares = st.nbonds * val.delta_cb_shares_per_bond
                delta_shares  = target_shares - st.hedge_shares
                if abs(delta_shares) * inp.S >= MIN_ORDER_USD:
                    side = "sell" if delta_shares > 0 else "buy"   # need more short if positive delta gap
                    self.order(EQ_SYMBOL, side, qty=abs(delta_shares), order_type="market", venue=VENUE_EQ)
                    st.hedge_shares += delta_shares
                    st.last_delta = val.delta_cb_shares_per_bond
                    self._save_state(st)
            return

        # ---------- consider new entry ----------
        # do we already have concurrency elsewhere? (one CB per strategy instance)
        open_any = r.get(_poskey(self.ctx.name)) is not None
        if open_any:
            return

        if abs(edge) >= ENTRY_USD:
            # size: nbonds = USD_PER_TRADE / CB_price
            nbonds = USD_PER_TRADE / max(inp.CB, 1e-9)
            if nbonds * inp.CB < MIN_ORDER_USD:
                return

            hedge_shares = nbonds * val.delta_cb_shares_per_bond

            if edge > 0:
                # CB cheap -> BUY CB, SHORT equity delta
                self.order(CB_SYMBOL, "buy",  qty=nbonds,       order_type="market", venue=VENUE_CB)
                if hedge_shares > 0:
                    self.order(EQ_SYMBOL, "sell", qty=hedge_shares, order_type="market", venue=VENUE_EQ)
                state = OpenState(
                    side="long_cb_short_eq",
                    nbonds=nbonds,
                    hedge_shares=hedge_shares,
                    entry_edge=edge,
                    entry_ts_ms=_now_ms(),
                    last_delta=val.delta_cb_shares_per_bond
                )
                self._save_state(state)
            else:
                # CB rich -> SHORT CB, LONG equity delta (only if allowed)
                if not ALLOW_SHORT_CB:
                    return
                self.order(CB_SYMBOL, "sell", qty=nbonds,       order_type="market", venue=VENUE_CB)
                if hedge_shares > 0:
                    self.order(EQ_SYMBOL, "buy",  qty=hedge_shares, order_type="market", venue=VENUE_EQ)
                state = OpenState(
                    side="short_cb_long_eq",
                    nbonds=nbonds,
                    hedge_shares=hedge_shares,
                    entry_edge=edge,
                    entry_ts_ms=_now_ms(),
                    last_delta=val.delta_cb_shares_per_bond
                )
                self._save_state(state)

    # ---- state helpers ----
    def _load_state(self) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name))
        if not raw:
            return None
        try:
            o = json.loads(raw) # type: ignore
            return OpenState(**o)
        except Exception:
            return None

    def _save_state(self, st: Optional[OpenState]) -> None:
        k = _poskey(self.ctx.name)
        if st is None:
            r.delete(k)
        else:
            r.set(k, json.dumps(st.__dict__))

    # ---- closing ----
    def _close(self, st: OpenState) -> None:
        if st.side == "long_cb_short_eq":
            self.order(CB_SYMBOL, "sell", qty=st.nbonds,        order_type="market", venue=VENUE_CB)
            if st.hedge_shares > 0:
                self.order(EQ_SYMBOL, "buy", qty=st.hedge_shares, order_type="market", venue=VENUE_EQ)
        else:
            self.order(CB_SYMBOL, "buy",  qty=st.nbonds,        order_type="market", venue=VENUE_CB)
            if st.hedge_shares > 0:
                self.order(EQ_SYMBOL, "sell", qty=st.hedge_shares, order_type="market", venue=VENUE_EQ)
        self._save_state(None)