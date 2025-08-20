# backend/strategies/diversified/carbon_credit_arbitrage.py
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import redis

from backend.engine.strategy_base import Strategy

# ====================== CONFIG (env overrides) ======================
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# Region hint (helps router/policy)
REGION_HINT = os.getenv("CARBON_REGION", "EU").upper()   # EU/US/UK

# Trade universe (you can extend freely)
# Use outright futures (front liquid delivery) or active Dec contract per program.
# Symbols should match your market data feed + OMS symbols.
EUA_SYMBOL = os.getenv("CARBON_EUA", "EUA.DEC25").upper()    # EU ETS (ICE EU)
UKA_SYMBOL = os.getenv("CARBON_UKA", "UKA.DEC25").upper()    # UK ETS (ICE EU)
CCA_SYMBOL = os.getenv("CARBON_CCA", "CCA.DEC25").upper()    # California Cap-and-Trade (ICE US)
RGGI_SYMBOL= os.getenv("CARBON_RGGI","RGGI.DEC25").upper()   # RGGI (ICE US) – optional

# Pairs to monitor (left minus right). Order matters for sign conventions.
# You can list 2-leg x-market and calendar (same program, different maturities).
PAIR_LIST = [                        # (rich_leg, cheap_leg) defined by spread = L - R
    (EUA_SYMBOL, UKA_SYMBOL),
    (EUA_SYMBOL, CCA_SYMBOL),
    (UKA_SYMBOL, CCA_SYMBOL),
    # add calendar spreads as well, e.g. EUA.DEC25 vs EUA.DEC26:
    # ("EUA.DEC26", "EUA.DEC25"),
]

# Per-ton contract size (tons per futures contract) – typical ICE EUA/UKA/CCA are 1000 t
TONS_PER_CONTRACT = float(os.getenv("CARBON_TONS_PER_CONTRACT", "1000"))

# Target notional per trade (USD). Position sizing converts to contracts per leg.
USD_PER_TRADE = float(os.getenv("CARBON_USD_PER_TRADE", "25000"))

# Entry/exit thresholds
ENTRY_USD  = float(os.getenv("CARBON_ENTRY_USD", "1.50"))  # absolute spread in USD/ton
EXIT_USD   = float(os.getenv("CARBON_EXIT_USD",  "0.50"))
ENTRY_Z    = float(os.getenv("CARBON_ENTRY_Z",   "2.0"))   # z-score entry (EWMA mean/var)
EXIT_Z     = float(os.getenv("CARBON_EXIT_Z",    "0.7"))

# Optional carry/cost‑of‑carry guard (financing, holding, borrow)
ANNUAL_CARRY_CAP_USD = float(os.getenv("CARBON_CARRY_CAP_USD", "3.0"))  # block if theoretical carry > this

# Cooldown & safety
COOLDOWN_S = int(os.getenv("CARBON_COOLDOWN_S", "60"))
MAX_CONCURRENT = int(os.getenv("CARBON_MAX_CONCURRENT", "3"))

# Venue hints for OMS (purely advisory)
VENUE_HINTS = {
    "EUA": "ICE_EU",
    "UKA": "ICE_EU",
    "CCA": "ICE_US",
    "RGGI": "ICE_US",
}

# FX pairs for normalization (mark all in USD/ton).
# If your futures already tick in USD, the conversion returns 1.0.
FX_FOR = {               # symbol prefix -> FX quote symbol (USD per 1 unit of quote)
    "EUA": "EURUSD",     # EUA trades in EUR typically
    "UKA": "GBPUSD",     # UKA trades in GBP typically
    "CCA": "USDUSD",     # already USD
    "RGGI": "USDUSD",
}

# Stream/channel keys your bus uses
LAST_PRICE_KEY = "last_price"       # HSET symbol -> {"price": ...}
RATES_FX_HKEY  = "fx:spot"          # HSET "EURUSD" -> 1.0945 etc (optional)

# ====================== Redis / helpers ======================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def _now_ms() -> int:
    return int(time.time() * 1000)

def _get_last(symbol: str) -> Optional[float]:
    raw = r.hget(LAST_PRICE_KEY, symbol)
    if not raw: return None
    try: return float(json.loads(raw)["price"])
    except Exception: return None

def _fx(symbol_prefix: str) -> float:
    """Return USD conversion for a given product prefix (e.g., 'EUA'->EURUSD spot)."""
    pair = FX_FOR.get(symbol_prefix, "USDUSD").upper()
    if pair == "USDUSD": return 1.0
    # Try dedicated FX key then fallback to last_price of FX symbol in your feed
    v = r.hget(RATES_FX_HKEY, pair)
    if v:
        try: return float(v)
        except Exception: pass
    return float(_get_last(pair) or 1.0)

def _prefix(sym: str) -> str:
    """EUA.DEC25 -> EUA ; CCA.DEC25 -> CCA"""
    return sym.split(".", 1)[0].upper()

def _venue_for(sym: str) -> Optional[str]:
    return VENUE_HINTS.get(_prefix(sym))

# ============ EWMA mean/var tracker for spreads (restart-safe) ============
@dataclass
class EwmaMV:
    mean: float
    var: float
    alpha: float

    def update(self, x: float) -> Tuple[float, float]:
        # classic EWMA variance update
        m_prev = self.mean
        self.mean = (1 - self.alpha) * self.mean + self.alpha * x
        self.var  = max(1e-12, (1 - self.alpha) * (self.var + (x - m_prev) * (x - self.mean)))
        return self.mean, self.var

def _ewma_key(pair: Tuple[str,str]) -> str:
    a,b = pair
    return f"carbon:ewma:{a}-{b}"

def _load_ewma(pair: Tuple[str,str], alpha: float = 0.02) -> EwmaMV:
    raw = r.get(_ewma_key(pair))
    if raw:
        try:
            obj = json.loads(raw)
            return EwmaMV(mean=float(obj["m"]), var=float(obj["v"]), alpha=float(obj.get("a", alpha)))
        except Exception:
            pass
    return EwmaMV(mean=0.0, var=1.0, alpha=alpha)

def _save_ewma(pair: Tuple[str,str], ew: EwmaMV) -> None:
    r.set(_ewma_key(pair), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ============ Carry/rational-value check (very coarse) ============
def _carry_guard_usd_per_ton(days: int, r_quote_ccy: float = 0.04, storage: float = 0.0) -> float:
    """
    Very rough guard: cost of financing + storage for the richer leg's currency.
    If absolute spread < this theoretical carry, skip entry (prevents false signals near delivery rolls).
    """
    return (r_quote_ccy + storage) * (days / 365.0) * 1.0  # USD/ton since we normalize later

# ============ Strategy ============

class CarbonCreditArbitrage(Strategy):
    """
    Cross‑market & calendar relative value in compliance carbon programs (EUA/UKA/CCA/RGGI).

    Mechanics
    ---------
    1) Normalize all contracts to USD/ton via FX.
    2) For each (L, R) pair in PAIR_LIST, compute spread = price(L) - price(R) in USD/ton.
    3) Maintain EWMA mean/variance; compute z-score.
    4) Enter when |spread| >= ENTRY_USD AND |z| >= ENTRY_Z (both must fire).
       - If spread > 0 (L rich): short L, long R
       - If spread < 0 (R rich): long L, short R
    5) Exit when |spread| <= EXIT_USD OR |z| <= EXIT_Z.

    Sizing
    ------
    Target USD notionals per leg equal to USD_PER_TRADE; converts to #contracts using TONS_PER_CONTRACT.
    """

    def __init__(self, name: str = "carbon_credit_arbitrage", region: Optional[str] = REGION_HINT, default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self.cooldown_until = 0.0

    # ---- persistence ----
    def _poskey(self, a: str, b: str) -> str:
        return f"carbon:open:{self.ctx.name}:{a}-{b}"

    def _load_state(self, a: str, b: str) -> Optional[Dict]:
        raw = r.get(self._poskey(a,b))
        try: return json.loads(raw) if raw else None
        except Exception: return None

    def _save_state(self, a: str, b: str, state: Optional[Dict]) -> None:
        k = self._poskey(a,b)
        if state is None: r.delete(k)
        else: r.set(k, json.dumps(state))

    def _cooldown_ok(self) -> bool:
        return time.time() >= self.cooldown_until

    def _bump_cooldown(self):
        self.cooldown_until = time.time() + COOLDOWN_S

    # ---- core tick ----
    def on_tick(self, tick: Dict) -> None:
        """
        Subscribe the router to any stream that feeds last prices for EUA/UKA/CCA/RGGI
        (e.g., 'trades.eu' or a dedicated 'trades.carbon').
        We recompute decisions whenever any carbon price (or FX) updates.
        """
        # only act on relevant symbols or FX updates
        sym = str(tick.get("symbol") or "").upper()
        if not sym:
            return

        # evaluate all pairs when any component moves
        self._process_pairs()

    # ---- decision engine ----
    def _usd_per_ton(self, sym: str) -> Optional[float]:
        px = _get_last(sym)
        if px is None: return None
        fx = _fx(_prefix(sym))
        return float(px) * float(fx)

    def _contracts_for_usd(self, usd_notional: float, usd_per_ton: float) -> float:
        if usd_per_ton <= 0: return 0.0
        tons = usd_notional / usd_per_ton
        return tons / TONS_PER_CONTRACT

    def _process_pairs(self) -> None:
        open_count = sum(1 for a,b in PAIR_LIST if self._load_state(a,b))
        for a, b in PAIR_LIST:
            pa = self._usd_per_ton(a)
            pb = self._usd_per_ton(b)
            if pa is None or pb is None:
                continue

            spread = pa - pb  # USD/ton
            ew = _load_ewma((a,b))
            m, v = ew.update(spread)
            _save_ewma((a,b), ew)
            z = (spread - m) / math.sqrt(max(v, 1e-12))

            state = self._load_state(a,b)

            # exit logic first
            if state:
                if (abs(spread) <= EXIT_USD) or (abs(z) <= EXIT_Z):
                    self._close_pair(a, b, state)
                    continue
                # still open; skip new entries for this pair
                continue

            # entry checks
            if open_count >= MAX_CONCURRENT or not self._cooldown_ok():
                continue

            if abs(spread) >= ENTRY_USD and abs(z) >= ENTRY_Z:
                # coarse carry guard: skip if within carry bounds (near delivery)
                # (If you track exact tenor days per contract, read them and compute carry precisely.)
                carry_cap = _carry_guard_usd_per_ton(days=180)  # very rough default 6 months
                if ANNUAL_CARRY_CAP_USD > 0 and abs(spread) < min(ANNUAL_CARRY_CAP_USD, carry_cap):
                    continue

                # sizing
                qa = self._contracts_for_usd(USD_PER_TRADE, pa)
                qb = self._contracts_for_usd(USD_PER_TRADE, pb)
                if qa <= 0 or qb <= 0:
                    continue

                # If spread > 0 (A rich): short A, long B. If spread < 0: long A, short B
                if spread > 0:
                    self._leg(a, "sell", qa)
                    self._leg(b, "buy",  qb)
                    side = "shortA_longB"
                else:
                    self._leg(a, "buy",  qa)
                    self._leg(b, "sell", qb)
                    side = "longA_shortB"

                self._save_state(a, b, {
                    "ts_ms": _now_ms(),
                    "a": a, "b": b,
                    "entry_spread": spread, "entry_z": z,
                    "qa": qa, "qb": qb,
                    "side": side
                })
                self._bump_cooldown()
                open_count += 1

    # ---- order helpers ----
    def _leg(self, symbol: str, side: str, qty_contracts: float) -> None:
        self.order(symbol, side, qty=qty_contracts, order_type="market", venue=_venue_for(symbol))

    def _close_pair(self, a: str, b: str, state: Dict) -> None:
        qa = float(state.get("qa", 0.0))
        qb = float(state.get("qb", 0.0))
        if qa <= 0 or qb <= 0:
            self._save_state(a,b,None)
            return
        if state.get("side") == "shortA_longB":
            self._leg(a, "buy",  qa)
            self._leg(b, "sell", qb)
        else:
            self._leg(a, "sell", qa)
            self._leg(b, "buy",  qb)
        self._save_state(a, b, None)