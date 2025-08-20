# backend/strategies/diversified/liquidity_mismatch.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import redis
from backend.engine.strategy_base import Strategy

"""
Liquidity Mismatch Arbitrage (paper)
------------------------------------
Trade idea:
  When the *slow/illiquid* market L deviates from the *fast/liquid* market F beyond a
  depth-aware threshold, fade the move on L and hedge on F. Expectation: L mean-reverts
  to F as liquidity returns.

Core signal (bps over fast mid):
  spr_bps = 1e4 * (P_L - P_F) / P_F
Gate by:
  - Absolute |spr_bps| >= ENTRY_BPS
  - Z-score on EWMA(spr_bps) >= ENTRY_Z
  - Optional imbalance filter: if lifting offers caused rich print, require ask depth thin, etc.

Execution:
  If spr_bps > 0 (L rich): SELL L (impact-aware size), BUY F (hedge)
  If spr_bps < 0 (L cheap): BUY  L, SELL F

Sizing:
  USD_NOTIONAL_BASE scaled down by an *impact budget* based on top-of-book depth on L:
    impact_scale = min(1.0, IMPACT_BUDGET_USD / max(1, best_depth_usd(L)))
  Also cap by MAX_INSTR_BPS_MOVE assumed slippage.

Restart-safe via Redis state.
Paper symbols (map later in adapters):
  • L_SYMBOL (slow): e.g., "EQ:MICRO@NSE" or "ETF:XYZ"
  • F_SYMBOL (fast): e.g., "FUT:XYZ@CME" or "INDEX:XYZ"
"""

# ============================ CONFIG (env) ============================
REDIS_HOST = os.getenv("LM_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("LM_REDIS_PORT", "6379"))

L_SYMBOL = os.getenv("LM_L_SYMBOL", "EQ:XYZ@SLOW").upper()  # slow / illiquid
F_SYMBOL = os.getenv("LM_F_SYMBOL", "FUT:XYZ@FAST").upper()  # fast / liquid

# Thresholds
ENTRY_BPS = float(os.getenv("LM_ENTRY_BPS", "8.0"))
EXIT_BPS  = float(os.getenv("LM_EXIT_BPS",  "3.0"))
ENTRY_Z   = float(os.getenv("LM_ENTRY_Z",   "1.2"))
EXIT_Z    = float(os.getenv("LM_EXIT_Z",    "0.5"))

# Sizing / risk
USD_NOTIONAL_BASE  = float(os.getenv("LM_USD_NOTIONAL_BASE", "30000"))
IMPACT_BUDGET_USD  = float(os.getenv("LM_IMPACT_BUDGET_USD", "5000"))   # don't take more than this top-of-book USD
MAX_INSTR_BPS_MOVE = float(os.getenv("LM_MAX_INSTR_BPS_MOVE", "20.0"))  # assume worst slippage on L
MIN_TICKET_USD     = float(os.getenv("LM_MIN_TICKET_USD", "200"))
MAX_CONCURRENT     = int(os.getenv("LM_MAX_CONCURRENT", "1"))

# Cadence & stats
RECHECK_SECS = int(os.getenv("LM_RECHECK_SECS", "1"))
EWMA_ALPHA   = float(os.getenv("LM_EWMA_ALPHA", "0.06"))

# Venues (advisory)
VENUE_L = os.getenv("LM_VENUE_L", "SLOW").upper()
VENUE_F = os.getenv("LM_VENUE_F", "FAST").upper()

# Redis keys
LAST_PRICE_HKEY = os.getenv("LM_LAST_PRICE_KEY", "last_price")        # HSET last_price <SYM> '{"price": ...}'
DEPTH_HKEY      = os.getenv("LM_DEPTH_KEY", "orderbook:best")         # HSET orderbook:best <SYM> '{"bid":px,"ask":px,"bid_sz":qty,"ask_sz":qty}'
VOL_HKEY        = os.getenv("LM_VOL_KEY",   "vol:ann")                 # optional: HSET vol:ann <SYM> <ann_vol_decimal> for dynamic bps

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

def _depth(sym: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    raw = r.hget(DEPTH_HKEY, sym)
    if not raw: return (None, None, None, None)
    try:
        o = json.loads(raw)
        return (float(o.get("bid") or 0), float(o.get("ask") or 0),
                float(o.get("bid_sz") or 0), float(o.get("ask_sz") or 0))
    except Exception:
        return (None, None, None, None)

def _fast_mid() -> Optional[float]:
    # prefer fast book mid; fallback to last price
    fb, fa, _, _ = _depth(F_SYMBOL)
    if fb and fa and fb > 0 and fa > 0:
        return 0.5 * (fb + fa)
    return _hget_price(F_SYMBOL)

def _slow_mid() -> Optional[float]:
    lb, la, _, _ = _depth(L_SYMBOL)
    if lb and la and lb > 0 and la > 0:
        return 0.5 * (lb + la)
    return _hget_price(L_SYMBOL)

def _best_depth_usd(sym: str, ref_px: float) -> float:
    b, a, bs, asz = _depth(sym)
    # pick the side we'd hit when trading the slow leg
    # if we SELL L (rich), we hit bid (depth = bid_sz); if BUY L (cheap), we lift ask (ask_sz)
    # We return *max* side depth to be conservative; we compute both and our caller chooses.
    usd_bid = (bs or 0.0) * (b or ref_px or 0.0)
    usd_ask = (asz or 0.0) * (a or ref_px or 0.0)
    return max(usd_bid, usd_ask)

def _now_ms() -> int:
    return int(time.time() * 1000)

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
    return f"lm:ewma:{L_SYMBOL}:{F_SYMBOL}"

def _load_ewma() -> EwmaMV:
    raw = r.get(_ewma_key())
    if raw:
        try:
            o = json.loads(raw)
            return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", EWMA_ALPHA)))
        except Exception:
            pass
    return EwmaMV(mean=0.0, var=1.0, alpha=EWMA_ALPHA)

def _save_ewma(ew: EwmaMV) -> None:
    r.set(_ewma_key(), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ============================ state ============================
@dataclass
class OpenState:
    side: str  # "sell_L_buy_F" or "buy_L_sell_F"
    qty_L: float
    qty_F: float
    entry_bps: float
    entry_z: float
    ts_ms: int

def _poskey(name: str) -> str:
    return f"lm:open:{name}:{L_SYMBOL}:{F_SYMBOL}"

# ============================ strategy ============================
class LiquidityMismatchArbitrage(Strategy):
    """
    Fade slow/illiquid price dislocations vs a fast hedge with depth-aware sizing.
    """
    def __init__(self, name: str = "liquidity_mismatch", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_start(self) -> None:
        super().on_start()
        r.hset("strategy:universe", self.ctx.name, json.dumps({
            "slow": L_SYMBOL, "fast": F_SYMBOL, "ts": _now_ms()
        }))

    def on_tick(self, tick: Dict) -> None:
        now = time.time()
        if now - self._last < RECHECK_SECS:
            return
        self._last = now
        self._evaluate()

    # --------------- core ---------------
    def _evaluate(self) -> None:
        pF = _fast_mid()
        pL = _slow_mid()
        if pF is None or pL is None or pF <= 0 or pL <= 0:
            return

        spr_bps = 1e4 * (pL - pF) / pF

        # EWMA + z
        ew = _load_ewma()
        m, v = ew.update(spr_bps)
        _save_ewma(ew)
        z = (spr_bps - m) / math.sqrt(max(v, 1e-12))

        # monitoring signal (positive if L rich vs F)
        self.emit_signal(max(-1.0, min(1.0, spr_bps / max(1.0, ENTRY_BPS))))

        st = self._load_state()

        # ----- exit logic -----
        if st:
            if (abs(spr_bps) <= EXIT_BPS) or (abs(z) <= EXIT_Z):
                self._close(st)
            return

        # ----- entry gates -----
        if r.get(_poskey(self.ctx.name)) is not None:
            return
        if not (abs(spr_bps) >= ENTRY_BPS and abs(z) >= ENTRY_Z):
            return

        # Depth-aware sizing on slow leg
        depth_usd = _best_depth_usd(L_SYMBOL, pL)
        if depth_usd <= 0:
            return
        impact_scale = min(1.0, IMPACT_BUDGET_USD / max(1.0, depth_usd))
        usd_size = max(0.0, USD_NOTIONAL_BASE * impact_scale)

        # Slippage guard: assume MAX_INSTR_BPS_MOVE on L; require potential edge > slippage
        if abs(spr_bps) <= (MAX_INSTR_BPS_MOVE + EXIT_BPS):
            return

        qty_L = usd_size / pL
        qty_F = usd_size / pF
        if pL * qty_L < MIN_TICKET_USD or pF * qty_F < MIN_TICKET_USD:
            return

        if spr_bps > 0:
            # L rich → SELL L / BUY F
            self.order(L_SYMBOL, "sell", qty=qty_L, order_type="market", venue=VENUE_L)
            self.order(F_SYMBOL, "buy",  qty=qty_F, order_type="market", venue=VENUE_F)
            side = "sell_L_buy_F"
        else:
            # L cheap → BUY L / SELL F
            self.order(L_SYMBOL, "buy",  qty=qty_L, order_type="market", venue=VENUE_L)
            self.order(F_SYMBOL, "sell", qty=qty_F, order_type="market", venue=VENUE_F)
            side = "buy_L_sell_F"

        self._save_state(OpenState(side=side, qty_L=qty_L, qty_F=qty_F,
                                   entry_bps=spr_bps, entry_z=z, ts_ms=_now_ms()))

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
        if st.side == "sell_L_buy_F":
            self.order(L_SYMBOL, "buy",  qty=st.qty_L, order_type="market", venue=VENUE_L)
            self.order(F_SYMBOL, "sell", qty=st.qty_F, order_type="market", venue=VENUE_F)
        else:
            self.order(L_SYMBOL, "sell", qty=st.qty_L, order_type="market", venue=VENUE_L)
            self.order(F_SYMBOL, "buy",  qty=st.qty_F, order_type="market", venue=VENUE_F)
        r.delete(_poskey(self.ctx.name))