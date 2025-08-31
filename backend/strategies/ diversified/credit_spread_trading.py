# backend/strategies/diversified/credit_spread_trading.py
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import redis

from backend.engine.strategy_base import Strategy

"""
Credit Spread Trading (mean-reversion)
-------------------------------------
Goes LONG credit when spreads are abnormally WIDE (expect tightening),
and SHORT credit when spreads are abnormally TIGHT (expect widening).

Universe supports two kinds of legs:
  • CDS indices (synthetic paper symbol):  "CDS:CDX_IG:5Y", "CDS:CDX_HY:5Y", "CDS:iTraxx_EU:5Y"
     - You publish spreads (bps) under:  HSET cds:spread:<TENOR> <INDEX> <bps>
  • Bond ETFs as credit proxies: "LQD", "HYG", "JNK", "EMB" (or your tickers)
     - You publish option-adjusted spread (bps) under:  HSET credit:oas <ETF> <bps>
     - We trade the ETF equity symbol for exposure (paper)

Optional rates hedge (duration neutral):
  • Publish DV01 (USD per 1 unit) for each tradable symbol:
       HSET dv01 <SYMBOL> <usd_per_unit>     (ETF dv01 per 1 share, CDS dv01 per $1 notional)
  • Treasury hedge instrument (futures or ETF), publish its DV01:
       HSET dv01 <HEDGE_SYMBOL> <usd_per_unit>

Other inputs your stack already keeps:
  • HSET last_price <SYMBOL> '{"price": <px>}'   (for ETFs/treasury only; CDS legs ignore price)

State & stats:
  • Per-name EWMA mean/var on spread (bps) → z-score gating.
  • Restart-safe in Redis.
"""

# ============================== CONFIG ==============================
REDIS_HOST = os.getenv("CRED_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("CRED_REDIS_PORT", "6379"))

# Universe (comma-separated). Use either "CDS:<INDEX>:<TENOR>" or plain ETF symbol.
# Examples: "CDS:CDX_IG:5Y,CDS:CDX_HY:5Y,LQD,HYG"
UNIVERSE = [s.strip().upper() for s in os.getenv(
    "CRED_UNIVERSE",
    "CDS:CDX_IG:5Y,CDS:CDX_HY:5Y,LQD,HYG"
).split(",") if s.strip()]

# Hedge settings
USE_DURATION_HEDGE = os.getenv("CRED_USE_HEDGE", "true").lower() in ("1", "true", "yes")
HEDGE_SYMBOL       = os.getenv("CRED_HEDGE_SYMBOL", "IEF").upper()   # e.g., 10Y UST ETF or future proxy
HEDGE_VENUE        = os.getenv("CRED_HEDGE_VENUE", "ARCA").upper()

# Sizing
USD_PER_PACKAGE    = float(os.getenv("CRED_USD_PER_PKG", "50000"))   # target credit exposure per leg
MAX_CONCURRENT     = int(os.getenv("CRED_MAX_CONCURRENT", "4"))

# Entry/exit gates (bps & z-score)
ENTRY_BPS = float(os.getenv("CRED_ENTRY_BPS", "20"))  # |spread - mean| >= 20 bps
EXIT_BPS  = float(os.getenv("CRED_EXIT_BPS",  "7"))
ENTRY_Z   = float(os.getenv("CRED_ENTRY_Z",   "1.5"))
EXIT_Z    = float(os.getenv("CRED_EXIT_Z",    "0.5"))

# Cadence and EWMA
RECHECK_SECS = int(os.getenv("CRED_RECHECK_SECS", "10"))
EWMA_ALPHA   = float(os.getenv("CRED_EWMA_ALPHA", "0.05"))

# Venues (advisory; your OMS can map)
VENUE_ETF = os.getenv("CRED_VENUE_ETF", "ARCA").upper()
VENUE_CDS = os.getenv("CRED_VENUE_CDS", "SWAPS").upper()

# Redis keys
LAST_PRICE_HKEY = os.getenv("CRED_LAST_PRICE_KEY", "last_price")   # HSET symbol -> {"price": ...}
OAS_HKEY        = os.getenv("CRED_OAS_KEY",        "credit:oas")   # HSET credit:oas <ETF> <bps>
CDS_FMT         = os.getenv("CRED_CDS_FMT",        "cds:spread:{tenor}")  # HSET cds:spread:5Y <INDEX> <bps>
DV01_HKEY       = os.getenv("CRED_DV01_KEY",       "dv01")         # HSET dv01 <SYMBOL> <usd_per_unit>

# ============================== REDIS ==============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================== HELPERS ==============================
def _is_cds(sym: str) -> bool:
    return sym.startswith("CDS:")

def _parse_cds(sym: str) -> Tuple[str, str]:
    # "CDS:CDX_IG:5Y" -> ("CDX_IG", "5Y")
    parts = sym.split(":")
    if len(parts) >= 3:
        return parts[1], parts[2]
    return sym, "5Y"

def _cds_key(tenor: str) -> str:
    return CDS_FMT.format(tenor=tenor.upper())

def _hgetf(key: str, field: str) -> Optional[float]:
    v = r.hget(key, field)
    if v is None: return None
    try:
        return float(v) # type: ignore
    except Exception:
        try:
            return float(json.loads(v)) # type: ignore
        except Exception:
            return None

def _px(sym: str) -> Optional[float]:
    raw = r.hget(LAST_PRICE_HKEY, sym)
    if not raw: return None
    try: return float(json.loads(raw)["price"])  # type: ignore
    except Exception:
        try: return float(raw) # type: ignore
        except Exception: return None

def _dv01(sym: str) -> Optional[float]:
    return _hgetf(DV01_HKEY, sym)

# ============================== EWMA ==============================
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

def _ewma_key(name: str) -> str:
    return f"cred:ewma:{name}"

def _load_ewma(name: str, alpha: float) -> EwmaMV:
    raw = r.get(_ewma_key(name))
    if raw:
        try:
            o = json.loads(raw) # type: ignore
            return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", alpha)))
        except Exception:
            pass
    return EwmaMV(mean=0.0, var=1.0, alpha=alpha)

def _save_ewma(name: str, ew: EwmaMV) -> None:
    r.set(_ewma_key(name), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ============================== STATE ==============================
@dataclass
class OpenState:
    side: str            # "long_credit" or "short_credit"
    credit_sym: str
    qty_credit: float    # ETF shares or CDS notional USD
    qty_hedge: float     # hedge units (ETF shares / futures units)
    entry_bps_dev: float # (spread - mean) at entry, bps
    entry_z: float
    ts_ms: int

def _poskey(name: str, sym: str) -> str:
    return f"cred:open:{name}:{sym}"

# ============================== STRATEGY ==============================
class CreditSpreadTrading(Strategy):
    """
    Mean-reversion on credit spreads with optional duration hedge.
    """
    def __init__(self, name: str = "credit_spread_trading", region: Optional[str] = "US", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self.last_check = 0.0
        self._open_count = 0

    def on_start(self) -> None:
        super().on_start()
        r.hset("strategy:universe", self.ctx.name, json.dumps({
            "universe": UNIVERSE, "hedge": HEDGE_SYMBOL, "use_hedge": USE_DURATION_HEDGE,
            "ts": int(time.time()*1000)
        }))

    def on_tick(self, tick: Dict) -> None:
        now = time.time()
        if now - self.last_check < RECHECK_SECS:
            return
        self.last_check = now
        self._evaluate_all()

    # ------------- engine -------------
    def _spread_bps(self, sym: str) -> Optional[float]:
        """
        Fetch current credit spread in bps for the symbol (CDS index or ETF OAS).
        """
        if _is_cds(sym):
            idx, tenor = _parse_cds(sym)
            return _hgetf(_cds_key(tenor), idx)
        # ETF proxy
        return _hgetf(OAS_HKEY, sym)

    def _evaluate_all(self) -> None:
        # recompute open count
        self._open_count = sum(1 for s in UNIVERSE if r.get(_poskey(self.ctx.name, s)))
        for sym in UNIVERSE:
            sp = self._spread_bps(sym)
            if sp is None:
                continue

            ew = _load_ewma(sym, EWMA_ALPHA)
            m, v = ew.update(sp)
            _save_ewma(sym, ew)
            z = (sp - m) / math.sqrt(max(v, 1e-12))
            dev = sp - m

            # monitoring signal (wider spreads -> negative; tighter -> positive)
            self.emit_signal(max(-1.0, min(1.0, -math.tanh(dev / 25.0))))

            st = self._load_state(sym)

            # exits first
            if st:
                if (abs(dev) <= EXIT_BPS) or (abs(z) <= EXIT_Z):
                    self._close(st)
                continue

            # entries
            if self._open_count >= MAX_CONCURRENT:
                continue
            if not (abs(dev) >= ENTRY_BPS and abs(z) >= ENTRY_Z):
                continue

            # Long credit when wide (dev << 0): buy ETF or sell CDS protection?
            if dev <= -ENTRY_BPS and z <= -ENTRY_Z:
                self._enter(sym, side="long_credit")
                self._open_count += 1
                continue

            # Short credit when tight (dev >> 0): sell ETF or buy CDS protection
            if dev >= +ENTRY_BPS and z >= +ENTRY_Z:
                self._enter(sym, side="short_credit")
                self._open_count += 1
                continue

    # ------------- entries/hedge -------------
    def _size_credit_units(self, sym: str) -> Tuple[float, Optional[float]]:
        """
        Returns (qty_credit_units, px) where:
          - for ETF: qty = USD_PER_PACKAGE / price, px = price
          - for CDS: qty = USD_PER_PACKAGE notional, px = None
        """
        if _is_cds(sym):
            return USD_PER_PACKAGE, None
        px = _px(sym)
        if px is None or px <= 0:
            return 0.0, None
        return USD_PER_PACKAGE / px, px

    def _hedge_units(self, credit_sym: str, qty_credit: float, side: str) -> float:
        """
        Compute hedge quantity to approx duration-neutral using DV01s.
        For ETFs: DV01 is per 1 share; for CDS: per $1 notional of index.
        Hedge sign:
          - long_credit (risk-on): SHORT rates (sell hedge)
          - short_credit (risk-off): LONG rates (buy hedge)
        """
        if not USE_DURATION_HEDGE:
            return 0.0
        dv01_c = _dv01(credit_sym)
        dv01_h = _dv01(HEDGE_SYMBOL)
        if dv01_c is None or dv01_h is None or dv01_h == 0:
            return 0.0
        # Target equal-and-opposite DV01
        hedge_qty = (dv01_c * qty_credit) / dv01_h
        # Direction handled at order time
        return hedge_qty

    def _enter(self, sym: str, side: str) -> None:
        qty_credit, px = self._size_credit_units(sym)
        if qty_credit <= 0:
            return

        # Hedge qty
        hqty = self._hedge_units(sym, qty_credit, side)

        # Orders for credit leg
        if _is_cds(sym):
            # CDS synthetic notional; buy=buy protection (short credit), sell=sell protection (long credit)
            if side == "long_credit":
                self.order(sym, "sell", qty=qty_credit, order_type="market", venue=VENUE_CDS)  # sell protection
            else:
                self.order(sym, "buy",  qty=qty_credit, order_type="market", venue=VENUE_CDS)  # buy protection
        else:
            # ETF shares
            venue = VENUE_ETF
            if side == "long_credit":
                self.order(sym, "buy",  qty=qty_credit, order_type="market", venue=venue)
            else:
                self.order(sym, "sell", qty=qty_credit, order_type="market", venue=venue)

        # Hedge orders
        if USE_DURATION_HEDGE and hqty != 0.0:
            if side == "long_credit":
                # short rates
                self.order(HEDGE_SYMBOL, "sell", qty=abs(hqty), order_type="market", venue=HEDGE_VENUE)
            else:
                # long rates
                self.order(HEDGE_SYMBOL, "buy",  qty=abs(hqty), order_type="market", venue=HEDGE_VENUE)

        # Save state (store entry dev/z for diagnostics)
        st = OpenState(
            side=side, credit_sym=sym, qty_credit=qty_credit, qty_hedge=hqty,
            entry_bps_dev=0.0, entry_z=0.0, ts_ms=int(time.time()*1000)
        )
        self._save_state(st)

    # ------------- state/persistence -------------
    def _load_state(self, sym: str) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, sym))
        if not raw:
            return None
        try:
            o = json.loads(raw) # type: ignore
            return OpenState(**o)
        except Exception:
            return None

    def _save_state(self, st: Optional[OpenState]) -> None:
        if st is None:
            return
        k = _poskey(self.ctx.name, st.credit_sym)
        r.set(k, json.dumps(st.__dict__))

    # ------------- closing -------------
    def _close(self, st: OpenState) -> None:
        sym = st.credit_sym
        # close credit leg
        if _is_cds(sym):
            if st.side == "long_credit":
                self.order(sym, "buy",  qty=st.qty_credit, order_type="market", venue=VENUE_CDS)  # buy back protection sold
            else:
                self.order(sym, "sell", qty=st.qty_credit, order_type="market", venue=VENUE_CDS)  # sell back protection bought
        else:
            venue = VENUE_ETF
            if st.side == "long_credit":
                self.order(sym, "sell", qty=st.qty_credit, order_type="market", venue=venue)
            else:
                self.order(sym, "buy",  qty=st.qty_credit, order_type="market", venue=venue)

        # close hedge
        if USE_DURATION_HEDGE and st.qty_hedge != 0.0:
            if st.side == "long_credit":
                self.order(HEDGE_SYMBOL, "buy",  qty=abs(st.qty_hedge), order_type="market", venue=HEDGE_VENUE)
            else:
                self.order(HEDGE_SYMBOL, "sell", qty=abs(st.qty_hedge), order_type="market", venue=HEDGE_VENUE)

        # clear state
        r.delete(_poskey(self.ctx.name, sym))