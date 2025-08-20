# backend/strategies/diversified/emissions_credit_arbitrage.py
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import redis

from backend.engine.strategy_base import Strategy

"""
Emissions Credit Arbitrage (EUA vs UKA)
---------------------------------------
Compares EUR-normalized EUA vs UKA futures (or spot proxies) and trades the spread.

Definitions (per metric ton, in EUR):
  px_EUA_eur = last_price(EUA)   [already in EUR]
  px_UKA_eur = last_price(UKA) * FX(GBP->EUR)

Spread (EUR/ton):
  S = px_EUA_eur - beta * px_UKA_eur

Logic (mean-reverting by default):
  • If S << mean and z low  -> LONG spread:   BUY EUA,  SELL UKA
  • If S >> mean and z high -> SHORT spread:  SELL EUA, BUY  UKA

Sizing:
  • Target a fixed EUR notional per leg; convert to contracts using contract units (tons/contract).
  • UKA leg is multiplied by the hedge beta.

Inputs you already publish to Redis:
  HSET last_price "<SYM>"         '{"price": <number>}'      # EUA, UKA symbols
  HSET fx:spot "GBPEUR"           <rate>                     # GBP->EUR (e.g., 1.175)
  (optional) HSET carbon:basis_adj "EUA_UKA" <eur_per_ton>   # guard/transport basis (adds to S)
  (optional) HSET dv01 "<SYM>"    <usd_per_ctrt>             # not required here; future use

Paper symbols / venues (map later in your adapters):
  • EUA front future:  "EUA.F1@ICE_EU"
  • UKA front future:  "UKA.F1@ICE_UK"
"""

# ============================= CONFIG (env) =============================
REDIS_HOST = os.getenv("CARB_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("CARB_REDIS_PORT", "6379"))

EUA_SYMBOL = os.getenv("CARB_EUA_SYMBOL", "EUA.F1@ICE_EU").upper()
UKA_SYMBOL = os.getenv("CARB_UKA_SYMBOL", "UKA.F1@ICE_UK").upper()

# Contract units (metric tons per 1 futures contract)
EUA_TONS_PER_CONTRACT = float(os.getenv("CARB_EUA_TONS_PER_CONTRACT", "1000"))
UKA_TONS_PER_CONTRACT = float(os.getenv("CARB_UKA_TONS_PER_CONTRACT", "1000"))

# FX keys
FX_HASH      = os.getenv("CARB_FX_HASH", "fx:spot")
FX_PAIR_GBPEUR = os.getenv("CARB_FX_PAIR", "GBPEUR").upper()  # HSET fx:spot GBPEUR 1.175

# Hedge beta (EUA vs UKA). 1.0 = notional-neutral; you can store beta in Redis to override.
HEDGE_BETA = float(os.getenv("CARB_HEDGE_BETA", "1.0"))
BETA_REDIS_HASH = os.getenv("CARB_BETA_HASH", "beta:carb")  # HSET beta:carb EUA_UKA 0.95 (optional)

# Entry/exit thresholds (EUR/ton & z-score)
ENTRY_EUR = float(os.getenv("CARB_ENTRY_EUR", "1.50"))
EXIT_EUR  = float(os.getenv("CARB_EXIT_EUR",  "0.50"))
ENTRY_Z   = float(os.getenv("CARB_ENTRY_Z",   "1.5"))
EXIT_Z    = float(os.getenv("CARB_EXIT_Z",    "0.6"))

# Sizing
EUR_NOTIONAL_PER_LEG = float(os.getenv("CARB_EUR_PER_LEG", "25000"))  # EUR per side
MAX_CONCURRENT       = int(os.getenv("CARB_MAX_CONCURRENT", "2"))
MIN_TICKET_EUR       = float(os.getenv("CARB_MIN_TICKET_EUR", "200"))

# Cadence & stats
RECHECK_SECS = int(os.getenv("CARB_RECHECK_SECS", "5"))
EWMA_ALPHA   = float(os.getenv("CARB_EWMA_ALPHA", "0.05"))

# Venues (advisory)
VENUE_EUA = os.getenv("CARB_VENUE_EUA", "ICE_EU").upper()
VENUE_UKA = os.getenv("CARB_VENUE_UKA", "ICE_UK").upper()

# Redis keys
LAST_PRICE_HKEY   = os.getenv("CARB_LAST_PRICE_KEY", "last_price")
BASIS_ADJ_HASH    = os.getenv("CARB_BASIS_ADJ_HASH", "carbon:basis_adj")  # HGET ... EUA_UKA
BASIS_ADJ_FIELD   = os.getenv("CARB_BASIS_ADJ_FIELD", "EUA_UKA")

# ============================= Redis =============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================= Helpers =============================
def _hget_price(sym: str) -> Optional[float]:
    raw = r.hget(LAST_PRICE_HKEY, sym)
    if not raw:
        return None
    try:
        return float(json.loads(raw)["price"])
    except Exception:
        try:
            return float(raw)
        except Exception:
            return None

def _hgetf(key: str, field: str) -> Optional[float]:
    v = r.hget(key, field)
    if v is None: return None
    try:
        return float(v)
    except Exception:
        try:
            return float(json.loads(v))
        except Exception:
            return None

def _gbp_to_eur(px_gbp: float) -> Optional[float]:
    fx = _hgetf(FX_HASH, FX_PAIR_GBPEUR)
    if fx is None or fx <= 0:
        return None
    return px_gbp * fx

def _beta() -> float:
    v = _hgetf(BETA_REDIS_HASH, "EUA_UKA")
    if v is None:
        return HEDGE_BETA
    return float(v)

def _basis_guard() -> float:
    v = _hgetf(BASIS_ADJ_HASH, BASIS_ADJ_FIELD)
    return float(v) if v is not None else 0.0

def _now_ms() -> int:
    return int(time.time() * 1000)

# ============================= EWMA =============================
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
    return f"carb:ewma:EUA_UKA"

def _load_ewma(alpha: float) -> EwmaMV:
    raw = r.get(_ewma_key())
    if raw:
        try:
            o = json.loads(raw)
            return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", alpha)))
        except Exception:
            pass
    return EwmaMV(mean=0.0, var=1.0, alpha=alpha)

def _save_ewma(ew: EwmaMV) -> None:
    r.set(_ewma_key(), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ============================= State =============================
@dataclass
class OpenState:
    side: str        # "long_spread" (BUY EUA / SELL UKA) or "short_spread"
    q_eua: float
    q_uka: float
    entry_spread_eur: float
    entry_z: float
    ts_ms: int

def _poskey(name: str) -> str:
    return f"carb:open:{name}:EUA_UKA"

# ============================= Strategy =============================
class EmissionsCreditArbitrage(Strategy):
    """
    Mean-reverting EUA–UKA spread arb, EUR-normalized and beta-hedged.
    """
    def __init__(self, name: str = "emissions_credit_arbitrage", region: Optional[str] = "EU", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_start(self) -> None:
        super().on_start()
        r.hset("strategy:universe", self.ctx.name, json.dumps({
            "pair": {"eua": EUA_SYMBOL, "uka": UKA_SYMBOL, "fx": FX_PAIR_GBPEUR},
            "tons_per_contract": {"eua": EUA_TONS_PER_CONTRACT, "uka": UKA_TONS_PER_CONTRACT},
            "ts": _now_ms()
        }))

    def on_tick(self, tick: Dict) -> None:
        now = time.time()
        if now - self._last < RECHECK_SECS:
            return
        self._last = now
        self._evaluate()

    # --------------- core ---------------
    def _evaluate(self) -> None:
        eua_px_eur = _hget_price(EUA_SYMBOL)
        uka_px_gbp = _hget_price(UKA_SYMBOL)
        if eua_px_eur is None or uka_px_gbp is None or eua_px_eur <= 0 or uka_px_gbp <= 0:
            return
        uka_px_eur = _gbp_to_eur(uka_px_gbp)
        if uka_px_eur is None or uka_px_eur <= 0:
            return

        beta = _beta()
        guard = _basis_guard()  # adds to spread as a conservative adjustment (e.g., transport/registry costs)

        spread = (eua_px_eur - beta * uka_px_eur) + guard  # EUR/ton

        ew = _load_ewma(EWMA_ALPHA)
        m, v = ew.update(spread)
        _save_ewma(ew)
        z = (spread - m) / math.sqrt(max(v, 1e-12))

        # dashboard signal (positive when EUA rich vs UKA)
        self.emit_signal(max(-1.0, min(1.0, (spread - m) / 1.0)))

        st = self._load_state()

        # ---------- exits ----------
        if st:
            if (abs(spread - m) <= EXIT_EUR) or (abs(z) <= EXIT_Z):
                self._close(st)
            return

        # ---------- entries ----------
        if r.get(_poskey(self.ctx.name)) is not None:
            return
        dev = spread - m
        if not (abs(dev) >= ENTRY_EUR and abs(z) >= ENTRY_Z):
            return

        q_eua, q_uka = self._ratio_contracts(eua_px_eur, uka_px_eur, beta)
        if q_eua <= 0 or q_uka <= 0:
            return

        if dev < 0:
            # EUA cheap vs UKA -> LONG spread: BUY EUA / SELL UKA
            self.order(EUA_SYMBOL, "buy",  qty=q_eua, order_type="market", venue=VENUE_EUA)
            self.order(UKA_SYMBOL, "sell", qty=q_uka, order_type="market", venue=VENUE_UKA)
            self._save_state(OpenState(
                side="long_spread", q_eua=q_eua, q_uka=q_uka,
                entry_spread_eur=spread, entry_z=z, ts_ms=_now_ms()
            ))
        else:
            # EUA rich vs UKA -> SHORT spread: SELL EUA / BUY UKA
            self.order(EUA_SYMBOL, "sell", qty=q_eua, order_type="market", venue=VENUE_EUA)
            self.order(UKA_SYMBOL, "buy",  qty=q_uka, order_type="market", venue=VENUE_UKA)
            self._save_state(OpenState(
                side="short_spread", q_eua=q_eua, q_uka=q_uka,
                entry_spread_eur=spread, entry_z=z, ts_ms=_now_ms()
            ))

    # --------------- sizing ---------------
    def _ratio_contracts(self, eua_px_eur: float, uka_px_eur: float, beta: float) -> Tuple[float, float]:
        """
        Target EUR_NOTIONAL_PER_LEG on each side; convert to contracts.
        """
        # Contracts = notional / (price * tons_per_contract)
        denom_eua = max(1e-9, eua_px_eur * EUA_TONS_PER_CONTRACT)
        denom_uka = max(1e-9, uka_px_eur * UKA_TONS_PER_CONTRACT)

        q_eua = EUR_NOTIONAL_PER_LEG / denom_eua
        q_uka = beta * (EUR_NOTIONAL_PER_LEG / denom_uka)

        # minimum ticket check (in EUR notionals)
        min_ok = (q_eua * denom_eua >= MIN_TICKET_EUR) and (q_uka * denom_uka >= MIN_TICKET_EUR)
        if not min_ok:
            return 0.0, 0.0
        return q_eua, q_uka

    # --------------- state io ---------------
    def _load_state(self) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name))
        if not raw:
            return None
        try:
            return OpenState(**json.loads(raw))
        except Exception:
            return None

    def _save_state(self, st: Optional[OpenState]) -> None:
        k = _poskey(self.ctx.name)
        if st is None:
            r.delete(k)
        else:
            r.set(k, json.dumps(st.__dict__))

    # --------------- close ---------------
    def _close(self, st: OpenState) -> None:
        if st.side == "long_spread":
            # unwind: SELL EUA / BUY UKA
            self.order(EUA_SYMBOL, "sell", qty=st.q_eua, order_type="market", venue=VENUE_EUA)
            self.order(UKA_SYMBOL, "buy",  qty=st.q_uka, order_type="market", venue=VENUE_UKA)
        else:
            # unwind: BUY EUA / SELL UKA
            self.order(EUA_SYMBOL, "buy",  qty=st.q_eua, order_type="market", venue=VENUE_EUA)
            self.order(UKA_SYMBOL, "sell", qty=st.q_uka, order_type="market", venue=VENUE_UKA)
        self._save_state(None)