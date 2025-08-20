# backend/strategies/diversified/dividend_future_arbitrage.py
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import redis

from backend.engine.strategy_base import Strategy

"""
Dividend Futures Arbitrage
--------------------------
Arb between listed **index dividend futures** and your forecast of realized dividends.

Notation
  • Contract price (points) ~= total cash dividends per index unit for that calendar year.
  • Fair value = ForecastDiv (optionally PV‑discounted).

Signal
  basis_pts = F_mkt_pts - F_fair_pts
    > +ENTRY_PTS  → "rich" → SHORT dividend future
    < -ENTRY_PTS  → "cheap" → LONG dividend future

Optional beta hedge
  • Some dividend futures still correlate with the equity index. You can neutralize a fraction
    of that with an index future (hedge ratio via ENV or Redis).

Inputs you already publish in Redis:
  HSET divfut:price:<YEAR> <INDEX> <price_pts>             # market dividend future price
  HSET div:forecast:<YEAR> <INDEX> <forecast_pts>          # your forecast (per index unit)
  (optional) HSET rate:risk_free:<CCY> <CCY> <r_decimal>   # if you want to auto‑discount
  (optional) HSET beta:div:<YEAR> <INDEX> <beta>           # hedge beta vs index future
  (optional) HSET last_price <INDEX> '{"price": <spot>}'   # for monitoring only
  (optional alt) components: HSET div:forecast:<YEAR> <TICKER> <per‑index‑unit pts> and
                   store weights under HSET index:weight:<INDEX> <TICKER> <w>, then set
                   DSP_USE_COMPONENTS=true to aggregate.

Paper OMS symbols:
  • Dividend future:  DIVFUT:<INDEX>:<YEAR>   (points)
  • Index future   :  IFUT:<INDEX>:<NEAR>     (for beta hedge; optional)
"""

# ============================ CONFIG (env) ============================
REDIS_HOST = os.getenv("DVF_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("DVF_REDIS_PORT", "6379"))

INDEX = os.getenv("DVF_INDEX", "SX5E").upper()       # e.g., SX5E, SPX, NIFTY50
CONTRACTS = [s.strip().upper() for s in os.getenv(
    "DVF_CONTRACTS", "2025;2026"
).split(";") if s.strip()]

CCY = os.getenv("DVF_CCY", "EUR").upper()            # currency of the dividend future
TENOR_YEARS = 1.0                                    # calendar‑year contracts (fixed)

# Use component‑level forecasts instead of top‑down?
USE_COMPONENTS = os.getenv("DVF_USE_COMPONENTS", "false").lower() in ("1","true","yes")

# Thresholds
ENTRY_PTS = float(os.getenv("DVF_ENTRY_PTS", "2.0"))   # enter when |basis| >= 2 index points
EXIT_PTS  = float(os.getenv("DVF_EXIT_PTS",  "0.7"))
ENTRY_Z   = float(os.getenv("DVF_ENTRY_Z",   "1.2"))
EXIT_Z    = float(os.getenv("DVF_EXIT_Z",    "0.4"))

# Discounting
DISCOUNT_PV = os.getenv("DVF_DISCOUNT_PV", "false").lower() in ("1","true","yes")
RATE_FALLBACK = float(os.getenv("DVF_RATE_FALLBACK", "0.02"))   # when rate not in Redis

# Sizing
USD_PER_CONTRACT = float(os.getenv("DVF_USD_PER_CONTRACT", "2000"))  # $ risk per 1 point (notional proxy)
CONTRACTS_PER_TRADE = float(os.getenv("DVF_CONTRACTS_PER_TRADE", "5"))
MIN_TICKET_USD = float(os.getenv("DVF_MIN_TICKET_USD", "200"))
MAX_CONCURRENT = int(os.getenv("DVF_MAX_CONCURRENT", "3"))

# Hedge (optional)
USE_BETA_HEDGE  = os.getenv("DVF_USE_BETA_HEDGE", "true").lower() in ("1","true","yes")
INDEX_FUT_SYM   = os.getenv("DVF_INDEX_FUTURE", "IFUT:SX5E:NEAR").upper()
HEDGE_RATIO_ENV = os.getenv("DVF_HEDGE_RATIO", "")  # e.g., "2025:0.15;2026:0.10" overrides beta store

# Cadence & stats
RECHECK_SECS = int(os.getenv("DVF_RECHECK_SECS", "10"))
EWMA_ALPHA   = float(os.getenv("DVF_EWMA_ALPHA", "0.06"))

# Venues (advisory)
VENUE_DVF = os.getenv("DVF_VENUE_DVF", "EUREX").upper()
VENUE_IFU = os.getenv("DVF_VENUE_IFU", "EUREX").upper()

# Redis keys
LAST_PRICE_HKEY   = os.getenv("DVF_LAST_PRICE_KEY", "last_price")
DIVFUT_PRICE_FMT  = os.getenv("DVF_PRICE_FMT", "divfut:price:{year}")      # HSET divfut:price:2026 <INDEX> <px>
DIV_FORECAST_FMT  = os.getenv("DVF_FORE_FMT",  "div:forecast:{year}")      # HSET div:forecast:2026 <INDEX> <pts>
RATE_HKEY_FMT     = os.getenv("DVF_RATE_FMT",  "rate:risk_free:{ccy}")     # HSET rate:risk_free:EUR EUR 0.03
BETA_HKEY_FMT     = os.getenv("DVF_BETA_FMT",  "beta:div:{year}")          # HSET beta:div:2026 <INDEX> 0.12
WGT_HKEY_FMT      = os.getenv("DVF_WGT_FMT",   "index:weight:{index}")     # HSET index:weight:SX5E <TICKER> <w>

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================ Helpers ============================
def _hgetf(key: str, field: str) -> Optional[float]:
    v = r.hget(key, field)
    if v is None: return None
    try: return float(v)
    except Exception:
        try:
            return float(json.loads(v))
        except Exception:
            return None

def _rate(ccy: str) -> float:
    v = _hgetf(RATE_HKEY_FMT.format(ccy=ccy), ccy)
    return float(v) if v is not None else RATE_FALLBACK

def _price(sym: str) -> Optional[float]:
    raw = r.hget(LAST_PRICE_HKEY, sym)
    if not raw: return None
    try: return float(json.loads(raw)["price"])
    except Exception:
        try: return float(raw)
        except Exception:
            return None

def _fut_sym(year: str) -> str:
    return f"DIVFUT:{INDEX}:{year}"

def _hedge_ratio_from_env() -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not HEDGE_RATIO_ENV.strip():
        return out
    for part in HEDGE_RATIO_ENV.split(";"):
        if ":" not in part: continue
        y, v = part.split(":", 1)
        try:
            out[y.strip().upper()] = float(v)
        except Exception:
            pass
    return out

HEDGE_RATIO_OVERRIDE = _hedge_ratio_from_env()

def _forecast_points(year: str) -> Optional[float]:
    if USE_COMPONENTS:
        # sum over components: HGETALL weights; div forecasts are stored per ticker in same div:forecast:<year>
        wmap = r.hgetall(WGT_HKEY_FMT.format(index=INDEX))
        if not wmap: return None
        tot = 0.0
        for ticker, w in wmap.items():
            dv = _hgetf(DIV_FORECAST_FMT.format(year=year), ticker.upper())
            if dv is None: return None
            try:
                tot += float(w) * float(dv)
            except Exception:
                return None
        return tot
    # top‑down per index
    return _hgetf(DIV_FORECAST_FMT.format(year=year), INDEX)

def _beta(year: str) -> float:
    if year in HEDGE_RATIO_OVERRIDE:
        return HEDGE_RATIO_OVERRIDE[year]
    v = _hgetf(BETA_HKEY_FMT.format(year=year), INDEX)
    return float(v) if v is not None else 0.0

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

def _ewma_key(year: str) -> str:
    return f"dvf:ewma:{INDEX}:{year}"

def _load_ewma(year: str) -> EwmaMV:
    raw = r.get(_ewma_key(year))
    if raw:
        try:
            o = json.loads(raw)
            return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", EWMA_ALPHA)))
        except Exception:
            pass
    return EwmaMV(mean=0.0, var=1.0, alpha=EWMA_ALPHA)

def _save_ewma(year: str, ew: EwmaMV) -> None:
    r.set(_ewma_key(year), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ============================ State ============================
@dataclass
class OpenState:
    year: str
    side: str       # "short_dvf" or "long_dvf"
    n_contracts: float
    hedge_qty: float
    entry_basis_pts: float
    entry_z: float
    ts_ms: int

def _poskey(name: str, year: str) -> str:
    return f"dvf:open:{name}:{INDEX}:{year}"

# ============================ Strategy ============================
class DividendFutureArbitrage(Strategy):
    """
    Long/short dividend future vs forecast with optional beta hedge.
    """
    def __init__(self, name: str = "dividend_future_arbitrage", region: Optional[str] = "EU", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_start(self) -> None:
        super().on_start()
        r.hset("strategy:universe", self.ctx.name, json.dumps({
            "index": INDEX, "contracts": CONTRACTS, "tenor_years": TENOR_YEARS,
            "use_components": USE_COMPONENTS, "beta_hedge": USE_BETA_HEDGE,
            "ts": int(time.time()*1000)
        }))

    def on_tick(self, tick: Dict) -> None:
        now = time.time()
        if now - self._last < RECHECK_SECS:
            return
        self._last = now
        self._evaluate_all()

    # ------------- engine -------------
    def _evaluate_all(self) -> None:
        open_count = sum(1 for y in CONTRACTS if r.get(_poskey(self.ctx.name, y)))
        for year in CONTRACTS:
            px = _hgetf(DIVFUT_PRICE_FMT.format(year=year), INDEX)
            fc = _forecast_points(year)
            if px is None or fc is None:
                continue

            if DISCOUNT_PV:
                r_rf = _rate(CCY)
                fc = fc / math.exp(r_rf * TENOR_YEARS)

            basis_pts = float(px) - float(fc)

            ew = _load_ewma(year)
            m, v = ew.update(basis_pts)
            _save_ewma(year, ew)
            z = (basis_pts - m) / math.sqrt(max(v, 1e-12))

            # emit monitor signal: positive when contract rich
            self.emit_signal(max(-1.0, min(1.0, (basis_pts - m) / 2.0)))

            st = self._load_state(year)

            # ----- exits -----
            if st:
                if (abs(basis_pts) <= EXIT_PTS) or (abs(z) <= EXIT_Z):
                    self._close(st)
                continue

            # ----- entries -----
            if open_count >= MAX_CONCURRENT:
                continue
            if not (abs(basis_pts) >= ENTRY_PTS and abs(z) >= ENTRY_Z):
                continue

            n_ctrs = max(1.0, CONTRACTS_PER_TRADE)
            hedge_qty = 0.0
            if USE_BETA_HEDGE:
                beta = _beta(year)  # dvf vs index future beta
                # hedge notional roughly: hedge_qty ≈ beta * n_ctrs
                hedge_qty = beta * n_ctrs

            if basis_pts > 0:
                # rich → short dividend future (+ optional long index fut to hedge)
                self.order(_fut_sym(year), "sell", qty=n_ctrs, order_type="market", venue=VENUE_DVF)
                if USE_BETA_HEDGE and hedge_qty != 0.0:
                    self.order(INDEX_FUT_SYM, "buy", qty=abs(hedge_qty), order_type="market", venue=VENUE_IFU)
                side = "short_dvf"
            else:
                # cheap → long dividend future (+ optional short index fut)
                self.order(_fut_sym(year), "buy", qty=n_ctrs, order_type="market", venue=VENUE_DVF)
                if USE_BETA_HEDGE and hedge_qty != 0.0:
                    self.order(INDEX_FUT_SYM, "sell", qty=abs(hedge_qty), order_type="market", venue=VENUE_IFU)
                side = "long_dvf"

            self._save_state(OpenState(
                year=year, side=side, n_contracts=n_ctrs, hedge_qty=hedge_qty,
                entry_basis_pts=basis_pts, entry_z=z, ts_ms=int(time.time()*1000)
            ))
            open_count += 1

    # ------------- state -------------
    def _load_state(self, year: str) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, year))
        if not raw: return None
        try:
            return OpenState(**json.loads(raw))
        except Exception:
            return None

    def _save_state(self, st: Optional[OpenState]) -> None:
        if st is None: return
        r.set(_poskey(self.ctx.name, st.year), json.dumps(st.__dict__))

    # ------------- close -------------
    def _close(self, st: OpenState) -> None:
        if st.side == "short_dvf":
            self.order(_fut_sym(st.year), "buy", qty=st.n_contracts, order_type="market", venue=VENUE_DVF)
            if USE_BETA_HEDGE and st.hedge_qty != 0.0:
                self.order(INDEX_FUT_SYM, "sell", qty=abs(st.hedge_qty), order_type="market", venue=VENUE_IFU)
        else:
            self.order(_fut_sym(st.year), "sell", qty=st.n_contracts, order_type="market", venue=VENUE_DVF)
            if USE_BETA_HEDGE and st.hedge_qty != 0.0:
                self.order(INDEX_FUT_SYM, "buy", qty=abs(st.hedge_qty), order_type="market", venue=VENUE_IFU)
        r.delete(_poskey(self.ctx.name, st.year))