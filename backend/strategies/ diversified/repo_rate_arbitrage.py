# backend/strategies/diversified/repo_rate_arbitrage.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import redis
from backend.engine.strategy_base import Strategy

"""
Repo Rate Arbitrage — paper
---------------------------
Two modes:

1) GC_SPREAD
   Edge (annualized, decimals) ≈ repo_gc_bid  - cash_fund_ask  - fees_guard
   If edge ≥ ENTRY_APR → LEND cash in reverse repo (GC) for tenor T, funded by borrowing cash.

   Paper orders:
     • "REPO:REV:GC:<ISO>" side=lend, qty=<USD notional>, rate=<repo%>, tenor_days=T
     • "CASH:<CCY>"        side=borrow, qty=<USD>, rate=<fund%>, tenor_days=T

2) SPECIALS
   Specialness ≈ repo_gc_bid  - repo_special_bid(cusip)
   Edge ≈ specialness - fees_guard - hedge_cost_apr
   Trade:
     • BUY bond (collateral), LEND it in specials repo (earn specialness)
     • HEDGE duration with a CTD future to neutralize rate risk (paper)

   Paper orders:
     • "BOND:<CUSIP>"                 side=buy, qty=<par_notional>
     • "REPO:REV:SPECIAL:<CUSIP>"     side=lend, qty=<par>, rate=<repo_special%>, tenor_days=T
     • "FUT:<CTD_FUT>"                side=sell/buy (to hedge DV01), qty=<contracts>

Redis feeds (examples below):
  # Rates (annualized decimals)
  HSET repo:gc:<CCY>   bid <d>     # lend (reverse) rate you can hit
  HSET cash:fund:<CCY> ask <d>     # your funding cost (SOFR/RRP/FF proxy)
  HSET repo:special:<CUSIP> bid <d>
  HSET repo:fees guard_apr <d>     # platform fees/friction guard

  # Specials metadata
  HSET bond:meta:<CUSIP> '{"price": <clean>, "dv01": <usd/100bp per 1 par>, "ctd_fut": "FUT:TYZ5", "conv_fac": 0.9}'
  HSET fut:dv01 FUT:TYZ5 <usd per 1 contract per 1bp>

  # Ops
  SET  risk:halt 0|1
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("REPO_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REPO_REDIS_PORT", "6379"))

MODE   = os.getenv("REPO_MODE", "GC_SPREAD").upper()     # GC_SPREAD | SPECIALS
CCY    = os.getenv("REPO_CCY", "USD").upper()
ISO    = os.getenv("REPO_ISO", "US").upper()             # tag for GC book (e.g., US, EU, JP)
CUSIP  = os.getenv("REPO_CUSIP", "US91282CJK28").upper() # for SPECIALS mode
CTD_FUT= os.getenv("REPO_CTD_FUT", "FUT:TYZ5").upper()

TENOR_DAYS = int(os.getenv("REPO_TENOR_DAYS", "2"))      # overnight=1, term as needed

# Thresholds
ENTRY_APR = float(os.getenv("REPO_ENTRY_APR", "0.004"))  # 40 bps/yr required net
EXIT_APR  = float(os.getenv("REPO_EXIT_APR",  "0.0015")) # 15 bps/yr
ENTRY_Z   = float(os.getenv("REPO_ENTRY_Z",   "1.2"))
EXIT_Z    = float(os.getenv("REPO_EXIT_Z",    "0.5"))

# Sizing / risk
USD_NOTIONAL     = float(os.getenv("REPO_USD_NOTIONAL", "100000"))  # per ticket
MIN_TICKET_USD   = float(os.getenv("REPO_MIN_TICKET_USD", "5000"))
MAX_CONCURRENT   = int(os.getenv("REPO_MAX_CONCURRENT", "1"))

# Cadence & stats
RECHECK_SECS = float(os.getenv("REPO_RECHECK_SECS", "1.5"))
EWMA_ALPHA   = float(os.getenv("REPO_EWMA_ALPHA", "0.06"))

# Redis keys
HALT_KEY     = os.getenv("REPO_HALT_KEY", "risk:halt")
GC_HKEY      = os.getenv("REPO_GC_HKEY",  "repo:gc:{ccy}")
FUND_HKEY    = os.getenv("REPO_FUND_HKEY","cash:fund:{ccy}")
SPC_HKEY     = os.getenv("REPO_SPC_HKEY", "repo:special:{cusip}")
FEES_HKEY    = os.getenv("REPO_FEES_HKEY","repo:fees")

BOND_META_HK = os.getenv("REPO_BOND_META_HK", "bond:meta:{cusip}")
FUT_DV01_HK  = os.getenv("REPO_FUT_DV01_HK",  "fut:dv01")

# Venues (advisory)
VENUE_REPO = os.getenv("REPO_VENUE_REPO", "REPO").upper()
VENUE_CASH = os.getenv("REPO_VENUE_CASH", "CASH").upper()
VENUE_BOND = os.getenv("REPO_VENUE_BOND", "EXCH").upper()
VENUE_FUT  = os.getenv("REPO_VENUE_FUT",  "FUT").upper()

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================ helpers ============================
def _hgetf(hk: str, field: str) -> Optional[float]:
    v = r.hget(hk, field)
    if v is None: return None
    try: return float(v)
    except Exception:
        try: return float(json.loads(v))
        except Exception: return None

def _hget_json(hk: str) -> Optional[dict]:
    raw = r.hgetall(hk)
    return {k: (float(v) if v.replace(".","",1).isdigit() else v) for k,v in raw.items()} if raw else None

def _now_ms() -> int: return int(time.time()*1000)

def _apr_to_period(apr: float, days: int) -> float:
    # simple money market day-count (ACT/360); adjust if you want ACT/365
    return apr * (days / 360.0)

def _fees_guard_apr() -> float:
    v = _hgetf(FEES_HKEY, "guard_apr")
    return v if v is not None else 0.0005  # 5 bps/yr

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
    return f"repo:ewma:{tag}"

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
    side: str
    usd: float
    entry_apr: float
    entry_z: float
    ts_ms: int
    # specials extras:
    par_qty: float = 0.0
    fut_qty: float = 0.0

def _poskey(name: str, tag: str) -> str:
    return f"repo:open:{name}:{tag}"

# ============================ strategy ============================
class RepoRateArbitrage(Strategy):
    """
    GC spread lend/borr and Specials capture with CTD hedge.
    """
    def __init__(self, name: str = "repo_rate_arbitrage", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_tick(self, tick: Dict) -> None:
        if (r.get(HALT_KEY) or "0") == "1":
            return
        now = time.time()
        if now - self._last < RECHECK_SECS:
            return
        self._last = now

        if MODE == "GC_SPREAD":
            self._eval_gc()
        else:
            self._eval_specials()

    # ---------------- GC spread mode ----------------
    def _eval_gc(self) -> None:
        tag = f"GC:{CCY}:{ISO}"
        repo_gc_bid = _hgetf(GC_HKEY.format(ccy=CCY), "bid")
        fund_ask    = _hgetf(FUND_HKEY.format(ccy=CCY), "ask")
        if repo_gc_bid is None or fund_ask is None:
            return

        fees = _fees_guard_apr()
        net_apr = repo_gc_bid - fund_ask - fees

        ew = _load_ewma(tag); m,v = ew.update(net_apr); _save_ewma(tag, ew)
        z = (net_apr - m) / math.sqrt(max(v, 1e-12))

        # monitoring signal (scaled to entry)
        self.emit_signal(max(-1.0, min(1.0, net_apr / max(1e-6, ENTRY_APR))))

        st = self._load_state(tag)

        # exit
        if st:
            if (net_apr <= EXIT_APR) or (abs(z) <= EXIT_Z):
                self._close_gc(tag, st)
            return

        if r.get(_poskey(self.ctx.name, tag)) is not None: return
        if not (net_apr >= ENTRY_APR and abs(z) >= ENTRY_Z): return

        usd = USD_NOTIONAL
        if usd < MIN_TICKET_USD: return

        # place paper legs: lend in reverse repo (GC), fund by borrowing cash
        self.order(f"REPO:REV:GC:{ISO}", "lend",
                   qty=usd, order_type="term", venue=VENUE_REPO,
                   flags={"rate_apr": repo_gc_bid, "tenor_days": TENOR_DAYS})
        self.order(f"CASH:{CCY}", "borrow",
                   qty=usd, order_type="term", venue=VENUE_CASH,
                   flags={"rate_apr": fund_ask, "tenor_days": TENOR_DAYS})

        self._save_state(tag, OpenState(mode="GC_SPREAD", side="lend_gc_borrow_cash",
                                        usd=usd, entry_apr=net_apr, entry_z=z, ts_ms=_now_ms()))

    def _close_gc(self, tag: str, st: OpenState) -> None:
        # Paper: let terms mature; here we simply mark state closed.
        r.delete(_poskey(self.ctx.name, tag))

    # ---------------- Specials mode ----------------
    def _eval_specials(self) -> None:
        tag = f"SPECIALS:{CUSIP}"
        repo_gc_bid   = _hgetf(GC_HKEY.format(ccy=CCY), "bid")
        repo_spc_bid  = _hgetf(SPC_HKEY.format(cusip=CUSIP), "bid")
        meta = _hget_json(BOND_META_HK.format(cusip=CUSIP))
        fut_dv01 = _hgetf(FUT_DV01_HK, CTD_FUT)
        if None in (repo_gc_bid, repo_spc_bid) or not meta or fut_dv01 is None:
            return

        specialness = max(0.0, repo_gc_bid - repo_spc_bid)  # how much richer GC is vs special
        fees = _fees_guard_apr()

        # Hedge cost: very small for short future; model as 0 here or add tiny carry guard if needed
        hedge_cost_apr = 0.0

        net_apr = specialness - fees - hedge_cost_apr

        ew = _load_ewma(tag); m,v = ew.update(net_apr); _save_ewma(tag, ew)
        z = (net_apr - m) / math.sqrt(max(v, 1e-12))
        self.emit_signal(max(-1.0, min(1.0, net_apr / max(1e-6, ENTRY_APR))))

        st = self._load_state(tag)
        if st:
            if (net_apr <= EXIT_APR) or (abs(z) <= EXIT_Z):
                self._close_specials(tag, st)
            return

        if r.get(_poskey(self.ctx.name, tag)) is not None: return
        if not (net_apr >= ENTRY_APR and abs(z) >= ENTRY_Z): return

        price = float(meta.get("price", 100.0))     # clean price per 100 par
        dv01_bond = float(meta.get("dv01", 0.0))    # $ per 1bp per 100 par
        conv_fac  = float(meta.get("conv_fac", 1.0))

        usd = USD_NOTIONAL
        if usd < MIN_TICKET_USD or price <= 0 or dv01_bond <= 0 or fut_dv01 <= 0:
            return

        # Size collateral purchase: par = usd / (price/100)
        par_qty = (usd / (max(1e-9, price/100.0)))

        # DV01 hedge with future: contracts ≈ (dv01_bond * par_qty / 100) / (fut_dv01 * conv_fac)
        dv01_total = (dv01_bond * par_qty) / 100.0
        fut_qty = dv01_total / max(1e-9, (fut_dv01 * max(0.1, conv_fac)))

        # Enter:
        # 1) Buy bond
        self.order(f"BOND:{CUSIP}", "buy", qty=par_qty, order_type="market", venue=VENUE_BOND)
        # 2) Lend the bond in specials repo
        self.order(f"REPO:REV:SPECIAL:{CUSIP}", "lend",
                   qty=par_qty, order_type="term", venue=VENUE_REPO,
                   flags={"rate_apr": repo_spc_bid, "tenor_days": TENOR_DAYS})
        # 3) Hedge duration (sell future if long bond DV01)
        self.order(CTD_FUT, "sell", qty=fut_qty, order_type="market", venue=VENUE_FUT)

        self._save_state(tag, OpenState(mode="SPECIALS", side="buy_collat_lend_special",
                                        usd=usd, entry_apr=net_apr, entry_z=z, ts_ms=_now_ms(),
                                        par_qty=par_qty, fut_qty=fut_qty))

    def _close_specials(self, tag: str, st: OpenState) -> None:
        # Unwind: buy back repo (paper settle), buy back future, sell the bond
        self.order(CTD_FUT, "buy", qty=st.fut_qty, order_type="market", venue=VENUE_FUT)
        self.order(f"BOND:{CUSIP}", "sell", qty=st.par_qty, order_type="market", venue=VENUE_BOND)
        r.delete(_poskey(self.ctx.name, tag))

    # ---------------- state I/O ----------------
    def _load_state(self, tag: str) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, tag))
        if not raw: return None
        try:
            o = json.loads(raw)
            return OpenState(mode=str(o["mode"]), side=str(o["side"]),
                             usd=float(o["usd"]), entry_apr=float(o["entry_apr"]),
                             entry_z=float(o["entry_z"]), ts_ms=int(o["ts_ms"]),
                             par_qty=float(o.get("par_qty", 0.0)), fut_qty=float(o.get("fut_qty", 0.0)))
        except Exception:
            return None

    def _save_state(self, tag: str, st: Optional[OpenState]) -> None:
        if st is None: return
        r.set(_poskey(self.ctx.name, tag), json.dumps({
            "mode": st.mode, "side": st.side, "usd": st.usd,
            "entry_apr": st.entry_apr, "entry_z": st.entry_z, "ts_ms": st.ts_ms,
            "par_qty": st.par_qty, "fut_qty": st.fut_qty
        }))