# backend/strategies/diversified/interest_rate_swap_arbitrage.py
from __future__ import annotations
import os, time, json, math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import redis
from backend.engine.strategy_base import Strategy

"""
Interest Rate Swap Arbitrage (paper)
------------------------------------
Modes:

A) OIS_BASIS (default)
   edge_bps = 1e4 * (IRS_par - OIS_par)
   • edge>+ENTRY_BPS → PAY IRS fixed (sell IRS) / RECEIVE OIS (buy OIS)
   • edge<-ENTRY_BPS → RECEIVE IRS fixed (buy IRS) / PAY OIS (sell OIS)

B) SWAP_SPREAD
   edge_bps = 1e4 * (IRS_par - TSY_yield)
   • edge>+ENTRY_BPS → PAY IRS fixed / LONG Tsy DV01
   • edge<-ENTRY_BPS → RECEIVE IRS fixed / SHORT Tsy DV01

DV01‑neutral sizing via Redis DV01s (per 1mm) or rough approximations.
Paper symbols:
  IRS:<CCY>:<TENOR>, OIS:<CCY>:<TENOR>, TSY:<CCY>:<TENOR>
Routing semantics (paper):
  buy IRS/OIS = receive fixed; sell = pay fixed.
"""

# ---------- CONFIG ----------
REDIS_HOST = os.getenv("IRS_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("IRS_REDIS_PORT", "6379"))

MODE  = os.getenv("IRS_MODE", "OIS_BASIS").upper()  # "OIS_BASIS" | "SWAP_SPREAD"
TENOR = os.getenv("IRS_TENOR", "5Y").upper()
CCY   = os.getenv("IRS_CCY", "USD").upper()

ENTRY_BPS = float(os.getenv("IRS_ENTRY_BPS", "5.0"))
EXIT_BPS  = float(os.getenv("IRS_EXIT_BPS",  "2.0"))
ENTRY_Z   = float(os.getenv("IRS_ENTRY_Z",   "1.3"))
EXIT_Z    = float(os.getenv("IRS_EXIT_Z",    "0.5"))

USD_DV01_TARGET = float(os.getenv("IRS_USD_DV01_TARGET", "1000"))
MIN_TICKET_USD  = float(os.getenv("IRS_MIN_TICKET_USD", "200"))
MAX_CONCURRENT  = int(os.getenv("IRS_MAX_CONCURRENT", "1"))

RECHECK_SECS = int(os.getenv("IRS_RECHECK_SECS", "5"))
EWMA_ALPHA   = float(os.getenv("IRS_EWMA_ALPHA", "0.06"))

VENUE_IRS = os.getenv("IRS_VENUE_IRS", "SWAP").upper()
VENUE_OIS = os.getenv("IRS_VENUE_OIS", "SWAP").upper()
VENUE_TSY = os.getenv("IRS_VENUE_TSY", "FUT").upper()

IRS_PAR_HKEY = os.getenv("IRS_PAR_HKEY", "irs:par")                 # HSET irs:par 5Y 0.0325
OIS_PAR_HKEY = os.getenv("OIS_PAR_HKEY", "ois:par")                 # HSET ois:par 5Y 0.0300
TSY_YLD_FMT  = os.getenv("TSY_YLD_FMT", "tsy:yield:{ccy}")          # HSET tsy:yield:USD 5Y 0.029
DV01_IRS_HK  = os.getenv("DV01_IRS_HKEY", "dv01:irs")
DV01_OIS_HK  = os.getenv("DV01_OIS_HKEY", "dv01:ois")
DV01_TSY_HK  = os.getenv("DV01_TSY_HKEY", "dv01:tsy")

# ---------- Redis ----------
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def _hgetf(hk: str, field: str) -> Optional[float]:
    v = r.hget(hk, field)
    if v is None: return None
    try: return float(v) # type: ignore
    except Exception:
        try: return float(json.loads(v)) # type: ignore
        except Exception: return None

def _tenor_years(t: str) -> float:
    t = t.strip().upper()
    if t.endswith("Y"): return max(0.25, float(t[:-1] or 1.0))
    if t.endswith("M"): return max(1/12.0, float(t[:-1] or 1.0) / 12.0)
    return 5.0

def _dv01_approx(hk: str, tenor: str, kind: str) -> float:
    v = _hgetf(hk, tenor)
    if v is not None and v > 0: return v  # USD per bp per 1mm
    y = _tenor_years(tenor)
    if kind == "IRS": dur = 0.75 * y
    elif kind == "OIS": dur = 0.7 * y
    else: dur = 0.9 * y   # TSY
    return 1_000_000 * 0.0001 * max(0.25, dur)

@dataclass
class EwmaMV:
    mean: float
    var: float
    alpha: float
    def update(self, x: float):
        m0 = self.mean
        self.mean = (1 - self.alpha) * self.mean + self.alpha * x
        self.var  = max(1e-12, (1 - self.alpha) * (self.var + (x - m0) * (x - self.mean)))
        return self.mean, self.var

def _ewma_key() -> str:
    return f"irsarb:ewma:{MODE}:{CCY}:{TENOR}"

def _load_ewma() -> EwmaMV:
    raw = r.get(_ewma_key())
    if raw:
        try:
            o = json.loads(raw) # type: ignore
            return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", EWMA_ALPHA)))
        except Exception: pass
    return EwmaMV(mean=0.0, var=1.0, alpha=EWMA_ALPHA)

def _save_ewma(ew: EwmaMV) -> None:
    r.set(_ewma_key(), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

@dataclass
class OpenState:
    side: str  # "pay_irs_receive_ois" | "receive_irs_pay_ois" | "pay_irs_long_tsy" | "receive_irs_short_tsy"
    notional_irs_mm: float
    hedge_mm: float
    entry_bps: float
    entry_z: float
    ts_ms: int

def _poskey(name: str) -> str:
    return f"irsarb:open:{name}:{MODE}:{CCY}:{TENOR}"

class InterestRateSwapArbitrage(Strategy):
    def __init__(self, name: str = "interest_rate_swap_arbitrage", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_start(self) -> None:
        super().on_start()
        r.hset("strategy:universe", self.ctx.name, json.dumps({
            "mode": MODE, "tenor": TENOR, "ccy": CCY, "ts": int(time.time()*1000)
        }))

    def on_tick(self, tick: Dict) -> None:
        now = time.time()
        if now - self._last < RECHECK_SECS: return
        self._last = now
        (self._eval_swap_spread() if MODE == "SWAP_SPREAD" else self._eval_ois_basis())

    # ----- OIS basis -----
    def _eval_ois_basis(self) -> None:
        irs = _hgetf(IRS_PAR_HKEY, TENOR)
        ois = _hgetf(OIS_PAR_HKEY, TENOR)
        if irs is None or ois is None: return
        edge_bps = 1e4 * (irs - ois)

        ew = _load_ewma(); m, v = ew.update(edge_bps); _save_ewma(ew)
        z = (edge_bps - m) / math.sqrt(max(v, 1e-12))
        self.emit_signal(max(-1.0, min(1.0, edge_bps / max(1.0, ENTRY_BPS))))

        st = self._load_state()
        if st:
            if (abs(edge_bps) <= EXIT_BPS) or (abs(z) <= EXIT_Z): self._close(st)
            return

        if r.get(_poskey(self.ctx.name)) is not None: return
        if not (abs(edge_bps) >= ENTRY_BPS and abs(z) >= ENTRY_Z): return

        dv01_irs = _dv01_approx(DV01_IRS_HK, TENOR, "IRS")
        dv01_ois = _dv01_approx(DV01_OIS_HK, TENOR, "OIS")
        if dv01_irs <= 0 or dv01_ois <= 0 or USD_DV01_TARGET < MIN_TICKET_USD: return

        irs_mm = USD_DV01_TARGET / dv01_irs
        ois_mm = irs_mm * (dv01_irs / dv01_ois)

        if edge_bps > 0:
            self.order(f"IRS:{CCY}:{TENOR}", "sell", qty=irs_mm, order_type="market", venue=VENUE_IRS)
            self.order(f"OIS:{CCY}:{TENOR}", "buy",  qty=ois_mm, order_type="market", venue=VENUE_OIS)
            side = "pay_irs_receive_ois"
        else:
            self.order(f"IRS:{CCY}:{TENOR}", "buy",  qty=irs_mm, order_type="market", venue=VENUE_IRS)
            self.order(f"OIS:{CCY}:{TENOR}", "sell", qty=ois_mm, order_type="market", venue=VENUE_OIS)
            side = "receive_irs_pay_ois"

        self._save_state(OpenState(side, irs_mm, ois_mm, edge_bps, z, int(time.time()*1000)))

    # ----- Swap spread -----
    def _eval_swap_spread(self) -> None:
        irs = _hgetf(IRS_PAR_HKEY, TENOR)
        tsy = _hgetf(TSY_YLD_FMT.format(ccy=CCY), TENOR)
        if irs is None or tsy is None: return
        edge_bps = 1e4 * (irs - tsy)

        ew = _load_ewma(); m, v = ew.update(edge_bps); _save_ewma(ew)
        z = (edge_bps - m) / math.sqrt(max(v, 1e-12))
        self.emit_signal(max(-1.0, min(1.0, edge_bps / max(1.0, ENTRY_BPS))))

        st = self._load_state()
        if st:
            if (abs(edge_bps) <= EXIT_BPS) or (abs(z) <= EXIT_Z): self._close(st)
            return

        if r.get(_poskey(self.ctx.name)) is not None: return
        if not (abs(edge_bps) >= ENTRY_BPS and abs(z) >= ENTRY_Z): return

        dv01_irs = _dv01_approx(DV01_IRS_HK, TENOR, "IRS")
        dv01_tsy = _dv01_approx(DV01_TSY_HK, TENOR, "TSY")
        if dv01_irs <= 0 or dv01_tsy <= 0 or USD_DV01_TARGET < MIN_TICKET_USD: return

        irs_mm = USD_DV01_TARGET / dv01_irs
        tsy_mm = irs_mm * (dv01_irs / dv01_tsy)

        if edge_bps > 0:
            self.order(f"IRS:{CCY}:{TENOR}", "sell", qty=irs_mm, order_type="market", venue=VENUE_IRS)
            self.order(f"TSY:{CCY}:{TENOR}", "buy",  qty=tsy_mm, order_type="market", venue=VENUE_TSY)
            side = "pay_irs_long_tsy"
        else:
            self.order(f"IRS:{CCY}:{TENOR}", "buy",  qty=irs_mm, order_type="market", venue=VENUE_IRS)
            self.order(f"TSY:{CCY}:{TENOR}", "sell", qty=tsy_mm, order_type="market", venue=VENUE_TSY)
            side = "receive_irs_short_tsy"

        self._save_state(OpenState(side, irs_mm, tsy_mm, edge_bps, z, int(time.time()*1000)))

    # ----- state & close -----
    def _load_state(self) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name))
        if not raw: return None
        try: return OpenState(**json.loads(raw)) # type: ignore
        except Exception: return None

    def _save_state(self, st: Optional[OpenState]) -> None:
        if st is None: return
        r.set(_poskey(self.ctx.name), json.dumps(st.__dict__))

    def _close(self, st: OpenState) -> None:
        if st.side == "pay_irs_receive_ois":
            self.order(f"IRS:{CCY}:{TENOR}", "buy",  qty=st.notional_irs_mm, order_type="market", venue=VENUE_IRS)
            self.order(f"OIS:{CCY}:{TENOR}", "sell", qty=st.hedge_mm,       order_type="market", venue=VENUE_OIS)
        elif st.side == "receive_irs_pay_ois":
            self.order(f"IRS:{CCY}:{TENOR}", "sell", qty=st.notional_irs_mm, order_type="market", venue=VENUE_IRS)
            self.order(f"OIS:{CCY}:{TENOR}", "buy",  qty=st.hedge_mm,       order_type="market", venue=VENUE_OIS)
        elif st.side == "pay_irs_long_tsy":
            self.order(f"IRS:{CCY}:{TENOR}", "buy",  qty=st.notional_irs_mm, order_type="market", venue=VENUE_IRS)
            self.order(f"TSY:{CCY}:{TENOR}", "sell", qty=st.hedge_mm,       order_type="market", venue=VENUE_TSY)
        else:  # receive_irs_short_tsy
            self.order(f"IRS:{CCY}:{TENOR}", "sell", qty=st.notional_irs_mm, order_type="market", venue=VENUE_IRS)
            self.order(f"TSY:{CCY}:{TENOR}", "buy",  qty=st.hedge_mm,       order_type="market", venue=VENUE_TSY)
        r.delete(_poskey(self.ctx.name))