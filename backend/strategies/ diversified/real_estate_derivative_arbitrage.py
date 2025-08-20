# backend/strategies/diversified/real_estate_derivative_arbitrage.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import redis
from backend.engine.strategy_base import Strategy

"""
Real Estate Derivative Arbitrage — paper
----------------------------------------
Two modes:
  1) FUTURES_BASKET: Long/short real-estate **futures** vs. a **REIT basket** proxy for spot.
     Basis ~ F - S * exp((rf - div - carry)*T)
     • If basis >> threshold → SELL future / BUY basket
     • If basis << -threshold → BUY future / SELL basket

  2) SWAP_REIT: Long/short **TRS/Index** (on-chain or OTC) vs. **REIT basket**
     Net edge ~ (Index_fair - Basket) after carry (rf, borrow, dividends).

Redis feeds you publish elsewhere:
  # Core marks
  HSET last_price <SYMBOL> '{"price": <px>}'                        # futures, index, and each basket leg
  HSET basket:weights <BASKET_ID> '{"EQ:SYM1": w1, "EQ:SYM2": w2}'  # weights sum≈1 (signed allowed)

  # Carry inputs (annualized decimals)
  HSET carry:rf <CCY> <rf_decimal>                      # risk-free rate
  HSET carry:div <BASKET_ID> <div_yield_decimal>        # basket dividend yield
  HSET carry:costs <TAG> <decimal>                      # misc carry/borrowing/friction

  # Futures / swap meta
  HSET fut:meta <FUT> '{"expiry_ms": <ms_epoch>, "ccy":"USD"}'
  HSET swap:meta <IDX> '{"maturity_ms": <ms_epoch>, "ccy":"USD"}'

  # Risk / ops
  SET  risk:halt 0|1
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("READ_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("READ_REDIS_PORT", "6379"))

MODE = os.getenv("READ_MODE", "FUTURES_BASKET").upper()   # FUTURES_BASKET | SWAP_REIT
CCY  = os.getenv("READ_CCY", "USD").upper()

# Instruments (pick one set depending on MODE)
FUT_SYM   = os.getenv("READ_FUT", "FUT:CSXR_NOV25").upper()       # e.g., "FUT:CSXR_NOV25" (Case-Shiller comp)
IDX_SYM   = os.getenv("READ_INDEX", "IDX:CASESHILLER").upper()    # for SWAP_REIT mode (total-return index)
BASKET_ID = os.getenv("READ_BASKET_ID", "REIT_US").upper()        # your REIT basket id

# Thresholds
ENTRY_BPS = float(os.getenv("READ_ENTRY_BPS", "60"))  # net basis in bps of fair
EXIT_BPS  = float(os.getenv("READ_EXIT_BPS",  "25"))
ENTRY_Z   = float(os.getenv("READ_ENTRY_Z",   "1.2"))
EXIT_Z    = float(os.getenv("READ_EXIT_Z",    "0.5"))

# Sizing / risk
USD_NOTIONAL     = float(os.getenv("READ_USD_NOTIONAL", "50000"))
MIN_TICKET_USD   = float(os.getenv("READ_MIN_TICKET_USD", "200"))
HEDGE_BETA       = float(os.getenv("READ_HEDGE_BETA", "1.0"))   # beta of basket vs derivative (≈1 by default)
MAX_CONCURRENT   = int(os.getenv("READ_MAX_CONCURRENT", "1"))

# Cadence & stats
RECHECK_SECS = float(os.getenv("READ_RECHECK_SECS", "2.0"))
EWMA_ALPHA   = float(os.getenv("READ_EWMA_ALPHA", "0.06"))

# Redis keys
LAST_PRICE_HK = os.getenv("READ_LAST_PRICE_HK", "last_price")
BKT_WTS_HK    = os.getenv("READ_BKT_WTS_HK",    "basket:weights")
RF_HK         = os.getenv("READ_RF_HK",         "carry:rf")
DIV_HK        = os.getenv("READ_DIV_HK",        "carry:div")
COSTS_HK      = os.getenv("READ_COSTS_HK",      "carry:costs")
FUT_META_HK   = os.getenv("READ_FUT_META_HK",   "fut:meta")
SWAP_META_HK  = os.getenv("READ_SWAP_META_HK",  "swap:meta")
HALT_KEY      = os.getenv("READ_HALT_KEY",      "risk:halt")

# Venues (advisory)
VENUE_FUT   = os.getenv("READ_VENUE_FUT", "FUT").upper()
VENUE_IDX   = os.getenv("READ_VENUE_IDX", "SWAP").upper()
VENUE_EQ    = os.getenv("READ_VENUE_EQ",  "EXCH").upper()

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================ helpers ============================
def _hget_json(hk: str, field: str) -> Optional[dict]:
    raw = r.hget(hk, field)
    if not raw: return None
    try:
        j = json.loads(raw)
        return j if isinstance(j, dict) else None
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
    raw = r.hget(LAST_PRICE_HK, sym)
    if not raw: return None
    try:
        j = json.loads(raw); return float(j.get("price", 0))
    except Exception:
        try: return float(raw)
        except Exception: return None

def _now_ms() -> int: return int(time.time() * 1000)

def _tenor_years(expiry_ms: Optional[int]) -> float:
    if not expiry_ms: return 0.5
    dt_days = max(1.0, (expiry_ms - _now_ms()) / 86400000.0)
    return dt_days / 365.0

def _basket_price(basket_id: str) -> Optional[float]:
    wts = _hget_json(BKT_WTS_HK, basket_id) or {}
    if not wts: return None
    # simple NAV proxy = sum(w_i * price_i); assumes weights sum to 1 in absolute terms
    s = 0.0
    for sym, w in wts.items():
        px = _px(sym)
        if px is None or px <= 0: return None
        s += float(w) * px
    return s

def _basket_legs(basket_id: str) -> Dict[str, float]:
    wts = _hget_json(BKT_WTS_HK, basket_id) or {}
    # normalize absolute weights to 1 (keep signs)
    total = sum(abs(float(w)) for w in wts.values()) or 1.0
    return {sym: float(w) / total for sym, w in wts.items()}

def _carry_inputs(tag: str) -> Tuple[float, float, float]:
    rf = _hgetf(RF_HK, CCY) or 0.03
    div = _hgetf(DIV_HK, BASKET_ID) or 0.03
    misc = _hgetf(COSTS_HK, tag) or 0.0
    return rf, div, misc

# ============================ EWMA ============================
@dataclass
class EwmaMV:
    mean: float
    var: float
    alpha: float
    def update(self, x: float) -> Tuple[float, float]:
        m0 = self.mean
        self.mean = (1 - self.alpha)*self.mean + self.alpha*x
        self.var  = max(1e-12, (1 - self.alpha)*(self.var + (x - m0)*(x - self.mean)))
        return self.mean, self.var

def _ewma_key(tag: str) -> str:
    return f"read:ewma:{tag}"

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
    side: str   # "sell_deriv_buy_basket" | "buy_deriv_sell_basket"
    qty_deriv: float
    basket_qtys: Dict[str, float]
    entry_bps: float
    entry_z: float
    ts_ms: int

def _poskey(name: str, tag: str) -> str:
    return f"read:open:{name}:{tag}"

# ============================ strategy ============================
class RealEstateDerivativeArbitrage(Strategy):
    """
    Futures/Swap vs REIT-basket basis arb with carry-aware fair value and beta-scaled hedge.
    """
    def __init__(self, name: str = "real_estate_derivative_arbitrage", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_tick(self, tick: Dict) -> None:
        if (r.get(HALT_KEY) or "0") == "1": return
        now = time.time()
        if now - self._last < RECHECK_SECS: return
        self._last = now

        if MODE == "FUTURES_BASKET":
            self._eval_fut_basket()
        else:
            self._eval_swap_reit()

    # ---------------- FUTURES vs BASKET ----------------
    def _eval_fut_basket(self) -> None:
        tag = f"FUTBKT:{FUT_SYM}:{BASKET_ID}"
        fut_px = _px(FUT_SYM)
        bkt_px = _basket_price(BASKET_ID)
        meta   = _hget_json(FUT_META_HK, FUT_SYM)
        if None in (fut_px, bkt_px) or not meta: return

        T = _tenor_years(int(meta.get("expiry_ms") or 0))
        rf, div, misc = _carry_inputs(tag)
        fair = bkt_px * math.exp((rf - div - misc) * max(T, 1e-6))
        basis = fut_px - fair
        basis_bps = 1e4 * (basis / max(1e-6, fair))

        ew = _load_ewma(tag); m,v = ew.update(basis_bps); _save_ewma(tag, ew)
        z = (basis_bps - m) / math.sqrt(max(v, 1e-12))

        # monitoring
        self.emit_signal(max(-1.0, min(1.0, basis_bps / max(1.0, ENTRY_BPS))))

        st = self._load_state(tag)

        # exits
        if st:
            if (abs(basis_bps) <= EXIT_BPS) or (abs(z) <= EXIT_Z):
                self._close(tag, st)
            return

        if r.get(_poskey(self.ctx.name, tag)) is not None: return
        if not (abs(basis_bps) >= ENTRY_BPS and abs(z) >= ENTRY_Z):
            return

        # sizing (derivative notional against basket NAV proxy)
        qty_deriv = max(0.0, USD_NOTIONAL / max(1e-6, fut_px))
        if qty_deriv * fut_px < MIN_TICKET_USD: return

        # build hedge: basket quantities scaled by beta and notional
        legs = _basket_legs(BASKET_ID)
        basket_qtys: Dict[str, float] = {}
        usd_hedge = HEDGE_BETA * qty_deriv * fut_px
        for sym, w in legs.items():
            px = _px(sym)
            if px is None or px <= 0: return
            q = (usd_hedge * abs(w)) / px
            # sign: if we SELL future (rich), we BUY basket → positive qty; inverse otherwise
            basket_qtys[sym] = q

        if basis_bps > 0:
            # Future rich → SELL future / BUY basket
            self.order(FUT_SYM, "sell", qty=qty_deriv, order_type="market", venue=VENUE_FUT)
            for sym, q in basket_qtys.items():
                self.order(sym, "buy", qty=q, order_type="market", venue=VENUE_EQ)
            side = "sell_deriv_buy_basket"
        else:
            # Future cheap → BUY future / SELL basket
            self.order(FUT_SYM, "buy", qty=qty_deriv, order_type="market", venue=VENUE_FUT)
            for sym, q in basket_qtys.items():
                self.order(sym, "sell", qty=q, order_type="market", venue=VENUE_EQ)
            side = "buy_deriv_sell_basket"

        self._save_state(tag, OpenState(mode="FUTURES_BASKET", side=side,
                                        qty_deriv=qty_deriv, basket_qtys=basket_qtys,
                                        entry_bps=basis_bps, entry_z=z, ts_ms=_now_ms()))

    # ---------------- SWAP/INDEX vs REIT ----------------
    def _eval_swap_reit(self) -> None:
        tag = f"SWAPBKT:{IDX_SYM}:{BASKET_ID}"
        idx_px = _px(IDX_SYM)
        bkt_px = _basket_price(BASKET_ID)
        meta   = _hget_json(SWAP_META_HK, IDX_SYM)
        if None in (idx_px, bkt_px) or not meta: return

        T = _tenor_years(int(meta.get("maturity_ms") or 0))
        rf, div, misc = _carry_inputs(tag)
        fair_idx = bkt_px * math.exp((rf - div - misc) * max(T, 1e-6))
        basis = idx_px - fair_idx
        basis_bps = 1e4 * (basis / max(1e-6, fair_idx))

        ew = _load_ewma(tag); m,v = ew.update(basis_bps); _save_ewma(tag, ew)
        z = (basis_bps - m) / math.sqrt(max(v, 1e-12))
        self.emit_signal(max(-1.0, min(1.0, basis_bps / max(1.0, ENTRY_BPS))))

        st = self._load_state(tag)
        if st:
            if (abs(basis_bps) <= EXIT_BPS) or (abs(z) <= EXIT_Z):
                self._close(tag, st)
            return

        if r.get(_poskey(self.ctx.name, tag)) is not None: return
        if not (abs(basis_bps) >= ENTRY_BPS and abs(z) >= ENTRY_Z):
            return

        qty_deriv = max(0.0, USD_NOTIONAL / max(1e-6, idx_px))
        if qty_deriv * idx_px < MIN_TICKET_USD: return

        legs = _basket_legs(BASKET_ID)
        basket_qtys: Dict[str, float] = {}
        usd_hedge = HEDGE_BETA * qty_deriv * idx_px
        for sym, w in legs.items():
            px = _px(sym)
            if px is None or px <= 0: return
            q = (usd_hedge * abs(w)) / px
            basket_qtys[sym] = q

        if basis_bps > 0:
            # Index rich → SELL index (receive TRS) / BUY basket
            self.order(IDX_SYM, "sell", qty=qty_deriv, order_type="market", venue=VENUE_IDX)
            for sym, q in basket_qtys.items():
                self.order(sym, "buy", qty=q, order_type="market", venue=VENUE_EQ)
            side = "sell_deriv_buy_basket"
        else:
            # Index cheap → BUY index (pay TRS) / SELL basket
            self.order(IDX_SYM, "buy", qty=qty_deriv, order_type="market", venue=VENUE_IDX)
            for sym, q in basket_qtys.items():
                self.order(sym, "sell", qty=q, order_type="market", venue=VENUE_EQ)
            side = "buy_deriv_sell_basket"

        self._save_state(tag, OpenState(mode="SWAP_REIT", side=side,
                                        qty_deriv=qty_deriv, basket_qtys=basket_qtys,
                                        entry_bps=basis_bps, entry_z=z, ts_ms=_now_ms()))

    # ---------------- state I/O & close ----------------
    def _load_state(self, tag: str) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, tag))
        if not raw: return None
        try:
            o = json.loads(raw)
            return OpenState(mode=str(o["mode"]),
                             side=str(o["side"]),
                             qty_deriv=float(o["qty_deriv"]),
                             basket_qtys={k: float(v) for k, v in (o.get("basket_qtys") or {}).items()},
                             entry_bps=float(o["entry_bps"]),
                             entry_z=float(o["entry_z"]),
                             ts_ms=int(o["ts_ms"]))
        except Exception:
            return None

    def _save_state(self, tag: str, st: Optional[OpenState]) -> None:
        if st is None: return
        r.set(_poskey(self.ctx.name, tag), json.dumps({
            "mode": st.mode, "side": st.side, "qty_deriv": st.qty_deriv,
            "basket_qtys": st.basket_qtys, "entry_bps": st.entry_bps,
            "entry_z": st.entry_z, "ts_ms": st.ts_ms
        }))

    def _close(self, tag: str, st: OpenState) -> None:
        # unwind in reverse
        if st.side == "sell_deriv_buy_basket":
            # buy back derivative, sell basket
            if st.mode == "FUTURES_BASKET":
                self.order(FUT_SYM, "buy", qty=st.qty_deriv, order_type="market", venue=VENUE_FUT)
            else:
                self.order(IDX_SYM, "buy", qty=st.qty_deriv, order_type="market", venue=VENUE_IDX)
            for sym, q in st.basket_qtys.items():
                self.order(sym, "sell", qty=q, order_type="market", venue=VENUE_EQ)
        else:
            # sell derivative, buy basket
            if st.mode == "FUTURES_BASKET":
                self.order(FUT_SYM, "sell", qty=st.qty_deriv, order_type="market", venue=VENUE_FUT)
            else:
                self.order(IDX_SYM, "sell", qty=st.qty_deriv, order_type="market", venue=VENUE_IDX)
            for sym, q in st.basket_qtys.items():
                self.order(sym, "buy", qty=q, order_type="market", venue=VENUE_EQ)
        r.delete(_poskey(self.ctx.name, tag))