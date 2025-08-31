# backend/strategies/diversified/credit_default_swap_arbitrage.py
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import redis

from backend.engine.strategy_base import Strategy

"""
CDS–Bond Basis Arbitrage (paper)
--------------------------------
Basis (in bps) = z_spread_bps(bond) - cds_spread_bps(tenor)

Entry logic
  • If basis <= -ENTRY_BPS and z <= -ENTRY_Z:  LONG bond, BUY CDS (negative basis)
  • If basis >= +ENTRY_BPS and z >= +ENTRY_Z:  SHORT bond, SELL CDS (positive basis; optional)

Exit logic
  • Close when |basis| <= EXIT_BPS or |z| <= EXIT_Z

Instruments (paper synthetics supported by your OMS):
  • <BOND>.BOND         -> price per 100 face (clean or dirty; treat as per-100 notional)
  • CDS:<NAME>:<TENOR>  -> CDS premium leg notionals (quotes in bps per year)
You can map these later to a real adapter.

Redis inputs your ETL should maintain (examples):
  HSET bond:zspread         "ACME_28"  210      # bps
  HSET cds:spread:5Y        "ACME"     185      # bps
  HSET last_price           "ACME_28.BOND" '{"price": 98.40}'   # per 100 face
  HSET rate:funding         "ACME"  0.045       # optional, decimal
  HSET repo:bond            "ACME_28" 0.02      # optional carry
  HSET recovery             "ACME" 0.40         # optional LGD calc

Everything degrades gracefully if some optional inputs are missing.
"""

# ========================= CONFIG (env) =========================
REDIS_HOST = os.getenv("CDSB_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("CDSB_REDIS_PORT", "6379"))

# Universe: "issuer|bond_id|tenor", e.g. "ACME|ACME_28|5Y;FOO|FOO_27|5Y"
UNIVERSE_ENV = os.getenv("CDSB_UNIVERSE", "ACME|ACME_28|5Y")
# Pricing tenor key suffix (matches cds:spread:<TENOR>)
DEFAULT_TENOR = os.getenv("CDSB_TENOR", "5Y").upper()

# Thresholds (in bps and z-score)
ENTRY_BPS = float(os.getenv("CDSB_ENTRY_BPS", "25"))   # enter if |basis| >= 25 bps
EXIT_BPS  = float(os.getenv("CDSB_EXIT_BPS",  "8"))    # exit if |basis| <= 8 bps
ENTRY_Z   = float(os.getenv("CDSB_ENTRY_Z",   "1.5"))
EXIT_Z    = float(os.getenv("CDSB_EXIT_Z",    "0.5"))

# Permissions
ALLOW_POSITIVE_BASIS_TRADE = os.getenv("CDSB_ALLOW_POSITIVE", "false").lower() in ("1","true","yes")

# Sizing
USD_FACE_PER_TRADE = float(os.getenv("CDSB_USD_FACE_PER_TRADE", "50000"))  # per package face value
MIN_TICKET_USD     = float(os.getenv("CDSB_MIN_TICKET_USD", "500"))        # skip dust

# Cadence & stats
RECHECK_SECS = int(os.getenv("CDSB_RECHECK_SECS", "7"))
EWMA_ALPHA   = float(os.getenv("CDSB_EWMA_ALPHA", "0.06"))  # event-based EWMA of basis

# Venue hints (advisory)
VENUE_BOND = os.getenv("CDSB_VENUE_BOND", "BONDS").upper()
VENUE_CDS  = os.getenv("CDSB_VENUE_CDS",  "SWAPS").upper()

# Redis key formats
ZSPREAD_HKEY   = "bond:zspread"             # HGET bond:zspread <BOND_ID> -> bps
CDS_SPREAD_HKP = "cds:spread:{tenor}"       # HGET cds:spread:<TENOR> <ISSUER> -> bps
LAST_PRICE_HKEY= "last_price"               # HGET last_price <BOND_ID>.BOND -> {"price": 98.4}
FUNDING_HKEY   = "rate:funding"             # HGET rate:funding <ISSUER> -> decimal
REPO_HKEY      = "repo:bond"                # HGET repo:bond <BOND_ID> -> decimal
RECOV_HKEY     = "recovery"                 # HGET recovery <ISSUER> -> decimal (e.g., 0.40)

# ========================= REDIS =========================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ========================= Helpers =========================
@dataclass
class NameSpec:
    issuer: str
    bond_id: str
    tenor: str

def _parse_universe(env: str) -> List[NameSpec]:
    out: List[NameSpec] = []
    for part in env.split(";"):
        part = part.strip()
        if not part: continue
        try:
            issuer, bond_id, tenor = [p.strip().upper() for p in part.split("|")]
        except Exception:
            # fallback to issuer|bond_id using DEFAULT_TENOR
            try:
                issuer, bond_id = [p.strip().upper() for p in part.split("|")]
                tenor = DEFAULT_TENOR
            except Exception:
                continue
        out.append(NameSpec(issuer=issuer, bond_id=bond_id, tenor=tenor))
    return out

UNIVERSE = _parse_universe(UNIVERSE_ENV)

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

def _hget_price(symbol: str) -> Optional[float]:
    raw = r.hget(LAST_PRICE_HKEY, symbol)
    if not raw: return None
    try:
        return float(json.loads(raw)["price"]) # type: ignore
    except Exception:
        try:
            return float(raw) # type: ignore
        except Exception:
            return None

def _cds_key(tenor: str) -> str:
    return CDS_SPREAD_HKP.format(tenor=tenor.upper())

def _bond_symbol(bond_id: str) -> str:
    return f"{bond_id}.BOND"

def _cds_symbol(issuer: str, tenor: str) -> str:
    return f"CDS:{issuer}:{tenor.upper()}"

# ---- EWMA MV for basis ----
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

def _ewma_key(issuer: str, bond_id: str, tenor: str) -> str:
    return f"cdsb:ewma:{issuer}:{bond_id}:{tenor}"

def _load_ewma(issuer: str, bond_id: str, tenor: str, alpha: float) -> EwmaMV:
    raw = r.get(_ewma_key(issuer, bond_id, tenor))
    if raw:
        try:
            o = json.loads(raw) # type: ignore
            return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", alpha)))
        except Exception:
            pass
    return EwmaMV(mean=0.0, var=1.0, alpha=alpha)

def _save_ewma(issuer: str, bond_id: str, tenor: str, ew: EwmaMV) -> None:
    r.set(_ewma_key(issuer, bond_id, tenor), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ---- Guard adjustments (very coarse) ----
def _carry_penalty_bps(issuer: str, bond_id: str) -> float:
    """
    Rough basis guard to avoid false signals:
    funding/repo carry (bps) shrinks negative basis edge, increases positive basis edge.
    """
    rf = _hgetf(FUNDING_HKEY, issuer) or 0.0
    repo = _hgetf(REPO_HKEY, bond_id) or 0.0
    # Convert difference to bps annualized
    return (rf - repo) * 1e4  # decimal -> bps

def _recovery_floor(issuer: str) -> float:
    rec = _hgetf(RECOV_HKEY, issuer)
    if rec is None:
        return 0.40
    return max(0.0, min(0.8, rec))

# ========================= State =========================
@dataclass
class OpenState:
    side: str               # "neg_basis" or "pos_basis"
    issuer: str
    bond_id: str
    tenor: str
    face_usd: float
    cds_notional_usd: float
    entry_basis_bps: float
    entry_z: float
    ts_ms: int

def _poskey(name: str, ns: NameSpec) -> str:
    return f"cdsb:open:{name}:{ns.issuer}:{ns.bond_id}:{ns.tenor}"

# ========================= Strategy =========================
class CreditDefaultSwapArbitrage(Strategy):
    """
    CDS–bond basis arb with paper CDS and bond legs.
    """
    def __init__(self, name: str = "credit_default_swap_arbitrage", region: Optional[str] = "US", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self.last_check = 0.0

    def on_start(self) -> None:
        super().on_start()
        # Advertise universe for UI
        r.hset("strategy:universe", self.ctx.name, json.dumps({
            "names": [{"issuer": n.issuer, "bond": n.bond_id, "tenor": n.tenor} for n in UNIVERSE],
            "ts": int(time.time()*1000)
        }))

    def on_tick(self, tick: Dict) -> None:
        now = time.time()
        if now - self.last_check < RECHECK_SECS:
            return
        self.last_check = now
        self._evaluate_all()

    # ------------- core engine -------------
    def _evaluate_all(self) -> None:
        for ns in UNIVERSE:
            bond_px = _hget_price(_bond_symbol(ns.bond_id))  # per 100 face
            z_bps   = _hgetf(ZSPREAD_HKEY, ns.bond_id)
            cds_bps = _hgetf(_cds_key(ns.tenor), ns.issuer)

            if bond_px is None or z_bps is None or cds_bps is None:
                continue

            basis = float(z_bps) - float(cds_bps)  # bps
            # carry guard: negative basis edge reduced by funding-repo; positive adjusted similarly
            basis_adj = basis - _carry_penalty_bps(ns.issuer, ns.bond_id)

            ew = _load_ewma(ns.issuer, ns.bond_id, ns.tenor, EWMA_ALPHA)
            m, v = ew.update(basis_adj)
            _save_ewma(ns.issuer, ns.bond_id, ns.tenor, ew)
            z = (basis_adj - m) / math.sqrt(max(v, 1e-12))

            # monitoring signal (squash bps to [-1,1])
            self.emit_signal(max(-1.0, min(1.0, math.tanh(basis_adj / 40.0))))

            st = self._load_state(ns)

            # ---- exits first ----
            if st:
                if (abs(basis_adj) <= EXIT_BPS) or (abs(z) <= EXIT_Z):
                    self._close(ns, st)
                continue

            # ---- entries ----
            # Negative basis: zspread << cds ⇒ buy bond, buy protection
            if (basis_adj <= -ENTRY_BPS) and (z <= -ENTRY_Z):
                self._enter_negative_basis(ns, bond_px, basis_adj, z)
                continue

            # Positive basis (optional): zspread >> cds ⇒ short bond, sell protection
            if ALLOW_POSITIVE_BASIS_TRADE and (basis_adj >= +ENTRY_BPS) and (z >= +ENTRY_Z):
                self._enter_positive_basis(ns, bond_px, basis_adj, z)
                continue

    # ------------- entries -------------
    def _enter_negative_basis(self, ns: NameSpec, bond_px_per_100: float, basis_bps: float, z: float) -> None:
        face = USD_FACE_PER_TRADE  # buy this much face value
        # qty in "bonds" = face / 100
        qty_bonds = face / 100.0
        if face < MIN_TICKET_USD:
            return

        cds_notional = face  # 1:1 hedge notional
        # Orders
        self.order(_bond_symbol(ns.bond_id), "buy", qty=qty_bonds, order_type="market", venue=VENUE_BOND)
        self.order(_cds_symbol(ns.issuer, ns.tenor), "buy", qty=cds_notional, order_type="market", venue=VENUE_CDS)

        self._save_state(ns, OpenState(
            side="neg_basis", issuer=ns.issuer, bond_id=ns.bond_id, tenor=ns.tenor,
            face_usd=face, cds_notional_usd=cds_notional, entry_basis_bps=basis_bps, entry_z=z,
            ts_ms=int(time.time()*1000)
        ))

    def _enter_positive_basis(self, ns: NameSpec, bond_px_per_100: float, basis_bps: float, z: float) -> None:
        face = USD_FACE_PER_TRADE
        qty_bonds = face / 100.0
        if face < MIN_TICKET_USD:
            return
        cds_notional = face
        # Orders
        self.order(_bond_symbol(ns.bond_id), "sell", qty=qty_bonds, order_type="market", venue=VENUE_BOND)
        self.order(_cds_symbol(ns.issuer, ns.tenor), "sell", qty=cds_notional, order_type="market", venue=VENUE_CDS)

        self._save_state(ns, OpenState(
            side="pos_basis", issuer=ns.issuer, bond_id=ns.bond_id, tenor=ns.tenor,
            face_usd=face, cds_notional_usd=cds_notional, entry_basis_bps=basis_bps, entry_z=z,
            ts_ms=int(time.time()*1000)
        ))

    # ------------- state/persistence -------------
    def _load_state(self, ns: NameSpec) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, ns))
        if not raw:
            return None
        try:
            o = json.loads(raw) # type: ignore
            return OpenState(**o)
        except Exception:
            return None

    def _save_state(self, ns: NameSpec, st: Optional[OpenState]) -> None:
        k = _poskey(self.ctx.name, ns)
        if st is None:
            r.delete(k)
        else:
            r.set(k, json.dumps(st.__dict__))

    # ------------- closing -------------
    def _close(self, ns: NameSpec, st: OpenState) -> None:
        qty_bonds = st.face_usd / 100.0
        if st.side == "neg_basis":
            # unwind: sell bond, sell CDS (close protection)
            self.order(_bond_symbol(ns.bond_id), "sell", qty=qty_bonds, order_type="market", venue=VENUE_BOND)
            self.order(_cds_symbol(ns.issuer, ns.tenor), "sell", qty=st.cds_notional_usd, order_type="market", venue=VENUE_CDS)
        else:
            # unwind: buy back bond, buy back CDS short
            self.order(_bond_symbol(ns.bond_id), "buy", qty=qty_bonds, order_type="market", venue=VENUE_BOND)
            self.order(_cds_symbol(ns.issuer, ns.tenor), "buy", qty=st.cds_notional_usd, order_type="market", venue=VENUE_CDS)
        self._save_state(ns, None)