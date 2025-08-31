# backend/strategies/diversified/sovereign_yield_spread.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import redis
from backend.engine.strategy_base import Strategy

"""
Sovereign Yield Spread (paper)
------------------------------
Modes:

1) CROSS_SPREAD (Country A vs B, DV01‑neutral)
   Let yA, yB be 10y yields (or chosen tenor). Fair spread may be:
     fair = alpha + beta * (yB - yB_ref) + ois_diff + risk_premium
   Here we keep it simple: fair = yB + adj_bps, or publish a fair in Redis.
   Edge_bps = (yA - yB) - fair_spread
   If edge_bps > ENTRY_BPS → SHORT A (sell DV01_A) / LONG B (buy DV01_B)
   If edge_bps < -ENTRY_BPS → LONG A / SHORT B
   Positioning DV01‑neutral using bond/future DV01s.

2) CASH_FUT_BASIS (single country)
   Basis_bps ≈ (y_cash - y_ctd_implied) minus carry/roll inputs.
   Trade bond vs future DV01‑neutral when basis deviates beyond gates.

Redis feeds you publish elsewhere:

  # Yields (in decimals, e.g. 0.0325 for 3.25%)
  HSET ylds:<TENOR> <CCY:CTRY> <yield>
    e.g., HSET ylds:10Y EUR:DE 0.0255 ; HSET ylds:10Y EUR:IT 0.0402

  # Optional: explicit fair spread in bps for A vs B (else module uses simple adj)
  HSET spr_fair:<TENOR> "<A>|<B>" <bps>
    e.g., HSET spr_fair:10Y "EUR:IT|EUR:DE" 150

  # Bond/Future meta (per 100 par unless noted)
  HSET bond:meta:<CCY:CTRY>:<TENOR> '{"price":<px>, "dv01":<usd_per_1bp_per_100>, "sym":"BOND:IT10"}'
  HSET fut:meta:<CTRY_TENOR>        '{"sym":"FUT:BTPZ5","dv01":<usd_per_ctrt_per_1bp>,"conv_fac":<cf>}'
  HSET ccy:usd_px <CCY> <usd_per_ccy>   # FX for DV01 to USD (if multi‑ccy books). If missing, assume USD.

  # Carry inputs / basis fair for CASH_FUT_BASIS (annualized decimals unless noted)
  HSET carry:repo <CCY:CTRY> <apr>              # repo (reverse) for bond
  HSET carry:fund <CCY:CTRY> <apr>              # funding for cash position
  HSET basis:fair "<CTRY_TENOR>" <bps>          # publish a fair basis in bps vs CTD implied

  # Ops / fees
  HSET fees:rates <VENUE> <bps_guard>
  SET  risk:halt 0|1

Paper routing (adapters map to venues):
  • Bonds:  "BOND:<TAG>"      qty in par
  • Futures:"FUT:<TAG>"       qty in contracts
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("SYS_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("SYS_REDIS_PORT", "6379"))

MODE   = os.getenv("SYS_MODE", "CROSS_SPREAD").upper()   # CROSS_SPREAD | CASH_FUT_BASIS
TENOR  = os.getenv("SYS_TENOR", "10Y").upper()

# CROSS_SPREAD instruments
A_TAG  = os.getenv("SYS_A", "EUR:IT").upper()            # CCY:CTRY for A (e.g., EUR:IT)
B_TAG  = os.getenv("SYS_B", "EUR:DE").upper()            # CCY:CTRY for B (e.g., EUR:DE)
A_BOND = os.getenv("SYS_A_BOND", "BOND:IT10").upper()
B_BOND = os.getenv("SYS_B_BOND", "BOND:DE10").upper()

# CASH_FUT_BASIS instruments
CF_CTRY   = os.getenv("SYS_CF_CTRY", "EUR:DE").upper()
CF_BOND   = os.getenv("SYS_CF_BOND", "BOND:DE10").upper()
CF_FUTKEY = os.getenv("SYS_CF_FUTKEY", "DE10").upper()    # key suffix to find fut:meta; e.g., "DE10"
CF_VENUE  = os.getenv("SYS_VENUE_FUT", "FUT").upper()

VENUE_BOND = os.getenv("SYS_VENUE_BOND", "EXCH").upper()
VENUE_RATES= os.getenv("SYS_VENUE_RATES", "X").upper()

# Thresholds (bps)
ENTRY_BPS = float(os.getenv("SYS_ENTRY_BPS", "12"))   # e.g., 12 bps trigger
EXIT_BPS  = float(os.getenv("SYS_EXIT_BPS",  "5"))
ENTRY_Z   = float(os.getenv("SYS_ENTRY_Z",   "1.2"))
EXIT_Z    = float(os.getenv("SYS_EXIT_Z",    "0.5"))

# Risk / sizing
USD_NOTIONAL   = float(os.getenv("SYS_USD_NOTIONAL", "100000"))
MIN_TICKET_USD = float(os.getenv("SYS_MIN_TICKET_USD", "2000"))
MAX_CONCURRENT = int(os.getenv("SYS_MAX_CONCURRENT", "1"))

# Cadence / stats
RECHECK_SECS = float(os.getenv("SYS_RECHECK_SECS", "1.0"))
EWMA_ALPHA   = float(os.getenv("SYS_EWMA_ALPHA", "0.06"))

# Redis keys
HALT_KEY    = os.getenv("SYS_HALT_KEY", "risk:halt")
YLDS_HK     = os.getenv("SYS_YLDS_HK",  "ylds:{tenor}")
FAIR_HK     = os.getenv("SYS_FAIR_HK",  "spr_fair:{tenor}")
BMETA_HK    = os.getenv("SYS_BMETA_HK", "bond:meta:{tag}:{tenor}")
FMETA_HK    = os.getenv("SYS_FMETA_HK", "fut:meta:{key}")
FX_HK       = os.getenv("SYS_FX_HK",    "ccy:usd_px")
CARRY_REPO  = os.getenv("SYS_CARRY_REPO","carry:repo")
CARRY_FUND  = os.getenv("SYS_CARRY_FUND","carry:fund")
BASIS_FAIR  = os.getenv("SYS_BASIS_FAIR","basis:fair")
FEES_HK     = os.getenv("SYS_FEES_HK",   "fees:rates")

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================ helpers ============================
def _hgetf(hk: str, field: str) -> Optional[float]:
    v = r.hget(hk, field)
    if v is None: return None
    try: return float(v) # type: ignore
    except Exception:
        try: return float(json.loads(v)) # type: ignore
        except Exception: return None

def _hget_json(hk: str, field: str) -> Optional[dict]:
    raw = r.hget(hk, field)
    if not raw: return None
    try:
        j = json.loads(raw) # type: ignore
        return j if isinstance(j, dict) else None
    except Exception:
        return None

def _fx_usd(ccy: str) -> float:
    if ccy == "USD": return 1.0
    v = _hgetf(FX_HK, ccy)
    return v if v is not None else 1.0

def _fees_bps(venue: str) -> float:
    v = _hgetf(FEES_HK, venue)
    return float(v) if v is not None else 1.0

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

def _ewma_key(tag: str) -> str:
    return f"sys:ewma:{tag}"

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
    side: str
    qty_A: float
    qty_B: float
    entry_bps: float
    entry_z: float
    ts_ms: int

def _poskey(name: str, tag: str) -> str:
    return f"sys:open:{name}:{tag}"

# ============================ Strategy ============================
class SovereignYieldSpread(Strategy):
    """
    DV01‑neutral sovereign A vs B spread, and cash‑future basis.
    """
    def __init__(self, name: str = "sovereign_yield_spread", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_tick(self, tick: Dict) -> None:
        if (r.get(HALT_KEY) or "0") == "1": return
        now = time.time()
        if now - self._last < RECHECK_SECS: return
        self._last = now

        if MODE == "CROSS_SPREAD":
            self._eval_cross()
        else:
            self._eval_cash_fut()

    # --------------- CROSS_SPREAD ---------------
    def _eval_cross(self) -> None:
        tag = f"XS:{TENOR}:{A_TAG}|{B_TAG}"

        yA = _hgetf(YLDS_HK.format(tenor=TENOR), A_TAG)
        yB = _hgetf(YLDS_HK.format(tenor=TENOR), B_TAG)
        if yA is None or yB is None: return

        # Fair spread
        fair_bps = _hgetf(FAIR_HK.format(tenor=TENOR), f"{A_TAG}|{B_TAG}")
        if fair_bps is None:
            fair_bps = 0.0  # simple "A vs B raw" if you don't publish a fair
        edge_bps = ( (yA - yB) * 1e4 ) - fair_bps  # positive ⇒ A rich vs B

        ew = _load_ewma(tag); m,v = ew.update(edge_bps); _save_ewma(tag, ew)
        z = (edge_bps - m) / math.sqrt(max(v, 1e-12))
        self.emit_signal(max(-1.0, min(1.0, abs(edge_bps) / max(1.0, ENTRY_BPS))))

        st = self._load_state(tag)
        if st:
            if (abs(edge_bps) <= EXIT_BPS) or (abs(z) <= EXIT_Z):
                self._close_cross(tag, st)
            return

        if r.get(_poskey(self.ctx.name, tag)) is not None: return
        if not (abs(edge_bps) >= ENTRY_BPS and abs(z) >= ENTRY_Z): return

        # DV01s (USD) per 100 par (bond meta) and FX to USD
        metaA = _hget_json(BMETA_HK.format(tag=A_TAG, tenor=TENOR), f"{A_TAG}:{TENOR}")
        metaB = _hget_json(BMETA_HK.format(tag=B_TAG, tenor=TENOR), f"{B_TAG}:{TENOR}")
        if not (metaA and metaB): return

        dv01A = float(metaA.get("dv01", 0.0)); symA = str(metaA.get("sym", A_BOND))
        dv01B = float(metaB.get("dv01", 0.0)); symB = str(metaB.get("sym", B_BOND))
        if dv01A <= 0 or dv01B <= 0: return

        ccyA, ccyB = A_TAG.split(":")[0], B_TAG.split(":")[0]
        fxA, fxB   = _fx_usd(ccyA), _fx_usd(ccyB)
        dv01A_usd = dv01A * fxA
        dv01B_usd = dv01B * fxB

        # Target USD notionals: size par such that USD_NOTIONAL ≈ |DV01_A| + |DV01_B|
        # Solve for par sizes making net DV01 ≈ 0:
        # qtyA * dv01A_usd  ≈ qtyB * dv01B_usd
        # and scale by total notional budget.
        # Start with base qtyA in 100s:
        base_par_100 = max(1.0, USD_NOTIONAL / (dv01A_usd + dv01B_usd + 1e-6))
        qtyA_100 = base_par_100
        qtyB_100 = (qtyA_100 * dv01A_usd) / max(1e-9, dv01B_usd)

        # Sanity min ticket in USD notionals (approx)
        usd_guess = (qtyA_100 * dv01A_usd + qtyB_100 * dv01B_usd) * 100.0  # per 100bp? Rough guard
        if usd_guess < MIN_TICKET_USD: return

        # Sides: edge>0 ⇒ A rich → SHORT A / LONG B; edge<0 ⇒ LONG A / SHORT B
        if edge_bps > 0:
            self.order(symA, "sell", qty=qtyA_100, order_type="market", venue=VENUE_BOND)
            self.order(symB, "buy",  qty=qtyB_100, order_type="market", venue=VENUE_BOND)
            side = "short_A_long_B"
        else:
            self.order(symA, "buy",  qty=qtyA_100, order_type="market", venue=VENUE_BOND)
            self.order(symB, "sell", qty=qtyB_100, order_type="market", venue=VENUE_BOND)
            side = "long_A_short_B"

        self._save_state(tag, OpenState(mode="CROSS_SPREAD", side=side,
                                        qty_A=qtyA_100, qty_B=qtyB_100,
                                        entry_bps=edge_bps, entry_z=z, ts_ms=_now_ms()))

    def _close_cross(self, tag: str, st: OpenState) -> None:
        # Reverse legs
        metaA = _hget_json(BMETA_HK.format(tag=A_TAG, tenor=TENOR), f"{A_TAG}:{TENOR}") or {}
        metaB = _hget_json(BMETA_HK.format(tag=B_TAG, tenor=TENOR), f"{B_TAG}:{TENOR}") or {}
        symA = str(metaA.get("sym", A_BOND)); symB = str(metaB.get("sym", B_BOND))
        if st.side == "short_A_long_B":
            self.order(symA, "buy",  qty=st.qty_A, order_type="market", venue=VENUE_BOND)
            self.order(symB, "sell", qty=st.qty_B, order_type="market", venue=VENUE_BOND)
        else:
            self.order(symA, "sell", qty=st.qty_A, order_type="market", venue=VENUE_BOND)
            self.order(symB, "buy",  qty=st.qty_B, order_type="market", venue=VENUE_BOND)
        r.delete(_poskey(self.ctx.name, tag))

    # --------------- CASH_FUT_BASIS ---------------
    def _eval_cash_fut(self) -> None:
        tag = f"CF:{TENOR}:{CF_CTRY}"

        # Yields (cash and CTD implied): publish y_cash and y_ctd_implied as "ylds:{TENOR}" fields
        y_cash = _hgetf(YLDS_HK.format(tenor=TENOR), CF_CTRY)
        # If you don’t have y_ctd_implied, just publish a fair basis (bps) via BASIS_FAIR and module will
        # compute deviation using cash only.
        fair_bps = _hgetf(BASIS_FAIR, CF_FUTKEY)  # fair basis bps vs CTD
        if y_cash is None or fair_bps is None: return

        # Basis here = observed minus fair. If you also stream y_ctd_implied, replace 0 with that.
        basis_bps = -fair_bps  # treat as deviation (0 - fair) unless you publish an observed basis

        ew = _load_ewma(tag); m,v = ew.update(basis_bps); _save_ewma(tag, ew)
        z = (basis_bps - m) / math.sqrt(max(v, 1e-12))
        self.emit_signal(max(-1.0, min(1.0, abs(basis_bps) / max(1.0, ENTRY_BPS))))

        st = self._load_state(tag)
        if st:
            if (abs(basis_bps) <= EXIT_BPS) or (abs(z) <= EXIT_Z):
                self._close_cf(tag, st)
            return

        if r.get(_poskey(self.ctx.name, tag)) is not None: return
        if not (abs(basis_bps) >= ENTRY_BPS and abs(z) >= ENTRY_Z): return

        # DV01s
        metaB = _hget_json(BMETA_HK.format(tag=CF_CTRY, tenor=TENOR), f"{CF_CTRY}:{TENOR}")
        metaF = _hget_json(FMETA_HK.format(key=CF_FUTKEY), "meta") or _hget_json(FMETA_HK.format(key=CF_FUTKEY), CF_FUTKEY)
        if not (metaB and metaF): return

        dv01_bond = float(metaB.get("dv01", 0.0))
        sym_bond  = str(metaB.get("sym", CF_BOND))
        dv01_fut  = float(metaF.get("dv01", 0.0))
        sym_fut   = str(metaF.get("sym", "FUT:UNKNOWN"))
        conv_fac  = float(metaF.get("conv_fac", 1.0))
        if dv01_bond <= 0 or dv01_fut <= 0: return

        # Size: DV01‑neutral in USD
        ccy = CF_CTRY.split(":")[0]
        fx  = _fx_usd(ccy)
        dv01_bond_usd = dv01_bond * fx
        dv01_fut_usd  = dv01_fut  # already USD per contract per bp

        # Base par size (per 100 par)
        par_100 = max(1.0, USD_NOTIONAL / (dv01_bond_usd + dv01_fut_usd + 1e-6))
        # Futures contracts to neutralize DV01 of bond: q_fut ≈ (par_100 * dv01_bond_usd) / (dv01_fut_usd * conv_fac)
        q_fut = (par_100 * dv01_bond_usd) / max(1e-9, dv01_fut_usd * max(0.1, conv_fac))

        if (par_100 * dv01_bond_usd + abs(q_fut) * dv01_fut_usd) * 100.0 < MIN_TICKET_USD:
            return

        # basis_bps > 0 → cash rich vs CTD: SELL bond / BUY future
        if basis_bps > 0:
            self.order(sym_bond, "sell", qty=par_100, order_type="market", venue=VENUE_BOND)
            self.order(sym_fut,  "buy",  qty=q_fut,   order_type="market", venue=CF_VENUE)
            side = "sell_bond_buy_fut"
        else:
            self.order(sym_bond, "buy",  qty=par_100, order_type="market", venue=VENUE_BOND)
            self.order(sym_fut,  "sell", qty=q_fut,   order_type="market", venue=CF_VENUE)
            side = "buy_bond_sell_fut"

        self._save_state(tag, OpenState(mode="CASH_FUT_BASIS", side=side,
                                        qty_A=par_100, qty_B=q_fut,
                                        entry_bps=basis_bps, entry_z=z, ts_ms=_now_ms()))

    def _close_cf(self, tag: str, st: OpenState) -> None:
        metaB = _hget_json(BMETA_HK.format(tag=CF_CTRY, tenor=TENOR), f"{CF_CTRY}:{TENOR}") or {}
        metaF = _hget_json(FMETA_HK.format(key=CF_FUTKEY), "meta") or {}
        sym_bond = str(metaB.get("sym", CF_BOND)); sym_fut = str(metaF.get("sym", "FUT:UNKNOWN"))
        if st.side == "sell_bond_buy_fut":
            self.order(sym_bond, "buy",  qty=st.qty_A, order_type="market", venue=VENUE_BOND)
            self.order(sym_fut,  "sell", qty=st.qty_B, order_type="market", venue=CF_VENUE)
        else:
            self.order(sym_bond, "sell", qty=st.qty_A, order_type="market", venue=VENUE_BOND)
            self.order(sym_fut,  "buy",  qty=st.qty_B, order_type="market", venue=CF_VENUE)
        r.delete(_poskey(self.ctx.name, tag))

    # --------------- state I/O ---------------
    def _load_state(self, tag: str) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, tag))
        if not raw: return None
        try:
            o = json.loads(raw) # type: ignore
            return OpenState(mode=str(o["mode"]), side=str(o["side"]),
                             qty_A=float(o["qty_A"]), qty_B=float(o["qty_B"]),
                             entry_bps=float(o["entry_bps"]), entry_z=float(o["entry_z"]),
                             ts_ms=int(o["ts_ms"]))
        except Exception:
            return None

    def _save_state(self, tag: str, st: Optional[OpenState]) -> None:
        if st is None: return
        r.set(_poskey(self.ctx.name, tag), json.dumps({
            "mode": st.mode, "side": st.side,
            "qty_A": st.qty_A, "qty_B": st.qty_B,
            "entry_bps": st.entry_bps, "entry_z": st.entry_z, "ts_ms": st.ts_ms
        }))