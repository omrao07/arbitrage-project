# backend/strategies/diversified/volatility_term_structure.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import redis
from backend.engine.strategy_base import Strategy

"""
Volatility Term Structure — paper
---------------------------------
Modes:
  1) IV_CALENDAR
       Compare IV at a chosen moneyness m (e.g., ATM or 25d put) between TENOR_SHORT and TENOR_LONG.
       Edge_vol_bps = 1e4 * (IV_long - IV_short - fair_slope), where fair_slope can be 0 or a small
       carry adjustment you publish. Trade:
         • If long too cheap (edge << -ENTRY) → BUY long-dated option / SELL short-dated (calendar)
         • If long too rich  (edge >>  ENTRY) → SELL long-dated / BUY short-dated (reverse calendar)

       Data (Redis):
         HSET iv:point "<SYM>:<DTE>:<POINT>" <iv_decimal>
           e.g., iv:point "SPX:7:ATM" 0.17 ; iv:point "SPX:30:ATM" 0.19
         HSET iv:fair_slope "<SYM>:<POINT>" <vol_decimal>   # optional expected term slope (long - short)

       Orders (paper):
         OPT_*:<SYM>:<K_PCT>:<DTE> (we trade by % moneyness to stay scale-stable)
         Side: "buy" / "sell", qty in contracts (mult 100 assumed by adapter)

  2) VIX_CURVE
       Term structure on VIX futures (front vs back). Use contango/backwardation z-score.
       Edge_bps = 1e4 * ((F1 - F2) / max(F2,1e-9))  (positive ⇒ backwardation)
       Trade:
         • Backwardation deep (edge >> ENTRY) → BUY F1 / SELL F2 (long curve flare)
         • Contango deep      (edge << -ENTRY) → SELL F1 / BUY F2

       Data (Redis):
         HSET last_price "FUT:VIX:M1" '{"price": 15.8}'
         HSET last_price "FUT:VIX:M2" '{"price": 17.2}'

Shared:
  HSET fees:vol CAL 15     # bps guard on option notional
  HSET fees:vix FUT  2     # bps guard on futures notional
  SET  risk:halt 0|1
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("VTS_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("VTS_REDIS_PORT", "6379"))

MODE         = os.getenv("VTS_MODE", "IV_CALENDAR").upper()   # IV_CALENDAR | VIX_CURVE
SYM          = os.getenv("VTS_SYM", "SPX").upper()

# IV_CALENDAR params
POINT        = os.getenv("VTS_POINT", "ATM").upper()          # ATM | PUT25 | CALL25, etc (you publish it)
TENOR_SHORT  = int(os.getenv("VTS_TENOR_SHORT", "7"))         # in days
TENOR_LONG   = int(os.getenv("VTS_TENOR_LONG",  "30"))        # in days
K_PCT        = float(os.getenv("VTS_K_PCT", "1.00"))          # strike as fraction of spot for both expiries (e.g., 1.00=ATM)

# VIX curve params
VIX_F1_SYM   = os.getenv("VTS_VIX_F1", "FUT:VIX:M1").upper()
VIX_F2_SYM   = os.getenv("VTS_VIX_F2", "FUT:VIX:M2").upper()

# Thresholds / gates
ENTRY_VOL_BPS  = float(os.getenv("VTS_ENTRY_VOL_BPS", "60"))  # 60 vol-bps (~0.6 vol pt)
EXIT_VOL_BPS   = float(os.getenv("VTS_EXIT_VOL_BPS",  "25"))
ENTRY_Z        = float(os.getenv("VTS_ENTRY_Z", "1.1"))
EXIT_Z         = float(os.getenv("VTS_EXIT_Z",  "0.5"))

# Sizing / risk
USD_BUDGET     = float(os.getenv("VTS_USD_BUDGET", "30000"))
MIN_TICKET_USD = float(os.getenv("VTS_MIN_TICKET_USD", "300"))
OPT_MULT       = float(os.getenv("VTS_OPT_MULT", "100"))      # per contract $ multiplier (equities)
FUT_MULT       = float(os.getenv("VTS_FUT_MULT", "1000"))     # per point for VIX futures (illustrative)

# Cadence / stats
RECHECK_SECS   = float(os.getenv("VTS_RECHECK_SECS", "0.8"))
EWMA_ALPHA     = float(os.getenv("VTS_EWMA_ALPHA", "0.08"))

# Redis keys
HALT_KEY    = os.getenv("VTS_HALT_KEY", "risk:halt")
IV_POINT_HK = os.getenv("VTS_IV_POINT_HK", "iv:point")
IV_SLOPE_HK = os.getenv("VTS_IV_SLOPE_HK", "iv:fair_slope")
LAST_HK     = os.getenv("VTS_LAST_HK", "last_price")
FEES_OPT_HK = os.getenv("VTS_FEES_OPT_HK", "fees:vol")
FEES_FUT_HK = os.getenv("VTS_FEES_FUT_HK", "fees:vix")

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================ helpers ============================
def _hgetf(hk: str, field: str) -> Optional[float]:
    v = r.hget(hk, field)
    if v is None: return None
    try: return float(v) # type: ignore
    except Exception:
        try:
            j = json.loads(v) # type: ignore
            return float(j) if isinstance(j, (int,float)) else None
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

def _fees_bps(hk: str, venue: str) -> float:
    v = r.hget(hk, venue)
    try: return float(v) if v is not None else 5.0 # type: ignore
    except Exception: return 5.0

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

def _ewma_key(tag: str) -> str: return f"vts:ewma:{tag}"

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
    side: str     # calendar: "buy_long_sell_short" or "sell_long_buy_short"; vix: "long_f1_short_f2" / reverse
    qty1: float
    qty2: float
    entry_bps: float
    entry_z: float
    ts_ms: int

def _poskey(name: str, tag: str) -> str:
    return f"vts:open:{name}:{tag}"

# ============================ Strategy ============================
class VolatilityTermStructure(Strategy):
    """
    IV calendar (same-moneyness) and VIX futures curve trades with z-gates.
    """
    def __init__(self, name: str = "volatility_term_structure", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_tick(self, tick: Dict) -> None:
        if (r.get(HALT_KEY) or "0") == "1": return
        now = time.time()
        if now - self._last < RECHECK_SECS: return
        self._last = now

        if MODE == "VIX_CURVE":
            self._eval_vix_curve()
        else:
            self._eval_iv_calendar()

    # --------------- IV_CALENDAR ---------------
    def _iv_point(self, sym: str, dte: int, point: str) -> Optional[float]:
        return _hgetf(IV_POINT_HK, f"{sym}:{int(dte)}:{point}")

    def _eval_iv_calendar(self) -> None:
        tag = f"IVCAL:{SYM}:{POINT}:{TENOR_SHORT}->{TENOR_LONG}"

        iv_s = self._iv_point(SYM, TENOR_SHORT, POINT)
        iv_l = self._iv_point(SYM, TENOR_LONG,  POINT)
        if iv_s is None or iv_l is None: return

        # optional fair slope (long - short) to compare against
        fair_slope = _hgetf(IV_SLOPE_HK, f"{SYM}:{POINT}") or 0.0
        edge_vol_bps = 1e4 * ((iv_l - iv_s) - fair_slope)

        ew = _load_ewma(tag); m,v = ew.update(edge_vol_bps); _save_ewma(tag, ew)
        z = (edge_vol_bps - m) / math.sqrt(max(v, 1e-12))
        self.emit_signal(max(-1.0, min(1.0, edge_vol_bps / max(1.0, ENTRY_VOL_BPS))))

        st = self._load_state(tag)
        if st:
            if (abs(edge_vol_bps) <= EXIT_VOL_BPS) or (abs(z) <= EXIT_Z):
                self._close_calendar(tag, st)
            return

        if r.get(_poskey(self.ctx.name, tag)) is not None: return
        if not (abs(edge_vol_bps) >= ENTRY_VOL_BPS and abs(z) >= ENTRY_Z): return

        # --- Sizing: spend USD_BUDGET notionally on the long leg premium proxy ---
        # We trade % moneyness strikes so we don't need spot; adapter translates K_PCT to real strike.
        # Use "OPT_CALL" for POINT not containing "PUT", else "OPT_PUT" (simplification).
        right = "OPT_PUT" if "PUT" in POINT else "OPT_CALL"
        k_pct = K_PCT
        fee_bps = _fees_bps(FEES_OPT_HK, "CAL") * 1e-4

        # rough premium proxy (iv * sqrt(T) * spot * d_approx). In paper mode, just set contracts by budget/1000.
        contracts = max(1.0, math.floor(USD_BUDGET / 1000.0))
        if contracts * OPT_MULT * 1.0 < MIN_TICKET_USD: return

        sym_long  = f"{right}:{SYM}:{k_pct:.4f}:{TENOR_LONG}"
        sym_short = f"{right}:{SYM}:{k_pct:.4f}:{TENOR_SHORT}"

        if edge_vol_bps < 0:
            # long-dated cheap → BUY long / SELL short
            self.order(sym_long,  "buy",  qty=contracts, order_type="market", venue="CAL")
            self.order(sym_short, "sell", qty=contracts, order_type="market", venue="CAL")
            side = "buy_long_sell_short"
        else:
            # long-dated rich → SELL long / BUY short
            self.order(sym_long,  "sell", qty=contracts, order_type="market", venue="CAL")
            self.order(sym_short, "buy",  qty=contracts, order_type="market", venue="CAL")
            side = "sell_long_buy_short"

        self._save_state(tag, OpenState(mode="IV_CALENDAR", side=side, qty1=contracts, qty2=contracts,
                                        entry_bps=edge_vol_bps, entry_z=z, ts_ms=_now_ms()))

    def _close_calendar(self, tag: str, st: OpenState) -> None:
        right = "OPT_PUT" if "PUT" in POINT else "OPT_CALL"
        sym_long  = f"{right}:{SYM}:{K_PCT:.4f}:{TENOR_LONG}"
        sym_short = f"{right}:{SYM}:{K_PCT:.4f}:{TENOR_SHORT}"
        if st.side == "buy_long_sell_short":
            self.order(sym_long,  "sell", qty=st.qty1, order_type="market", venue="CAL")
            self.order(sym_short, "buy",  qty=st.qty2, order_type="market", venue="CAL")
        else:
            self.order(sym_long,  "buy",  qty=st.qty1, order_type="market", venue="CAL")
            self.order(sym_short, "sell", qty=st.qty2, order_type="market", venue="CAL")
        r.delete(_poskey(self.ctx.name, tag))

    # --------------- VIX_CURVE ---------------
    def _eval_vix_curve(self) -> None:
        tag = f"VIX:{VIX_F1_SYM}|{VIX_F2_SYM}"

        f1 = _px(VIX_F1_SYM); f2 = _px(VIX_F2_SYM)
        if f1 is None or f2 is None or f1 <= 0 or f2 <= 0: return

        # Edge: % backwardation/contango expressed in bps
        edge_bps = 1e4 * ((f1 - f2) / max(1e-9, f2))

        ew = _load_ewma(tag); m,v = ew.update(edge_bps); _save_ewma(tag, ew)
        z = (edge_bps - m) / math.sqrt(max(v, 1e-12))
        self.emit_signal(max(-1.0, min(1.0, edge_bps / max(1.0, ENTRY_VOL_BPS))))

        st = self._load_state(tag)
        if st:
            if (abs(edge_bps) <= EXIT_VOL_BPS) or (abs(z) <= EXIT_Z):
                self._close_vix(tag, st)
            return

        if r.get(_poskey(self.ctx.name, tag)) is not None: return
        if not (abs(edge_bps) >= ENTRY_VOL_BPS and abs(z) >= ENTRY_Z): return

        fee_bps = _fees_bps(FEES_FUT_HK, "FUT") * 1e-4
        # contract notional ≈ price * FUT_MULT
        notional = f1 * FUT_MULT
        qty = max(1.0, math.floor(USD_BUDGET / max(1.0, notional)))
        if qty * notional < MIN_TICKET_USD: return

        if edge_bps > 0:
            # backwardation (front > back) → BUY F1 / SELL F2
            self.order(VIX_F1_SYM, "buy",  qty=qty, order_type="market", venue="FUT")
            self.order(VIX_F2_SYM, "sell", qty=qty, order_type="market", venue="FUT")
            side = "long_f1_short_f2"
        else:
            # contango (front < back) → SELL F1 / BUY F2
            self.order(VIX_F1_SYM, "sell", qty=qty, order_type="market", venue="FUT")
            self.order(VIX_F2_SYM, "buy",  qty=qty, order_type="market", venue="FUT")
            side = "short_f1_long_f2"

        self._save_state(tag, OpenState(mode="VIX_CURVE", side=side, qty1=qty, qty2=qty,
                                        entry_bps=edge_bps, entry_z=z, ts_ms=_now_ms()))

    def _close_vix(self, tag: str, st: OpenState) -> None:
        if st.side == "long_f1_short_f2":
            self.order(VIX_F1_SYM, "sell", qty=st.qty1, order_type="market", venue="FUT")
            self.order(VIX_F2_SYM, "buy",  qty=st.qty2, order_type="market", venue="FUT")
        else:
            self.order(VIX_F1_SYM, "buy",  qty=st.qty1, order_type="market", venue="FUT")
            self.order(VIX_F2_SYM, "sell", qty=st.qty2, order_type="market", venue="FUT")
        r.delete(_poskey(self.ctx.name, tag))

    # --------------- state I/O ---------------
    def _load_state(self, tag: str) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, tag))
        if not raw: return None
        try:
            o = json.loads(raw) # type: ignore
            return OpenState(mode=str(o["mode"]), side=str(o["side"]),
                             qty1=float(o["qty1"]), qty2=float(o["qty2"]),
                             entry_bps=float(o["entry_bps"]), entry_z=float(o["entry_z"]),
                             ts_ms=int(o["ts_ms"]))
        except Exception:
            return None

    def _save_state(self, tag: str, st: Optional[OpenState]) -> None:
        if st is None: return
        r.set(_poskey(self.ctx.name, tag), json.dumps({
            "mode": st.mode, "side": st.side,
            "qty1": st.qty1, "qty2": st.qty2,
            "entry_bps": st.entry_bps, "entry_z": st.entry_z, "ts_ms": st.ts_ms
        }))