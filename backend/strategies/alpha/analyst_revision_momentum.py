# backend/strategies/diversified/analyst_revision_momentum.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import redis
from backend.engine.strategy_base import Strategy

"""
Analyst Revision Momentum — paper
---------------------------------
Idea: Positive changes in consensus (NTM/FY1 EPS), net rating upgrades, and net price-target
revisions predict near‑term outperformance; negatives predict underperformance.

Data you publish in Redis (examples):

  # Universe (tickers we scan)
  SADD universe:eq AAPL MSFT NVDA AMZN META TSLA ...

  # Per‑symbol revision deltas over a lookback window (e.g., 7 days)
  HSET rev:delta "AAPL" '{"ntm_pct":0.022, "fy1_pct":0.018,
                          "rating_up":3, "rating_down":1, "pt_up":4, "pt_down":1,
                          "cov":42, "days":7, "updated_ms": 1755200000000}'

  # Last prices
  HSET last_price "EQ:AAPL" '{"price": 230.15}'

  # Optional: sector / industry classification for caps & neutralization
  HSET ref:sector "AAPL" "TECH"

  # Fees (bps on notional, paper guard)
  HSET fees:eq EXCH 10

  # Global halt
  SET risk:halt 0|1

Routing (paper; adapters wire later):
  order("EQ:<SYM>", side, qty, order_type="market", venue="EXCH")
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("ARM_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("ARM_REDIS_PORT", "6379"))

# Thresholds / gates
ENTRY_Z        = float(os.getenv("ARM_ENTRY_Z", "1.0"))
EXIT_Z         = float(os.getenv("ARM_EXIT_Z",  "0.3"))
MAX_HOLD_DAYS  = int(os.getenv("ARM_MAX_HOLD_DAYS", "20"))
RECHECK_SECS   = float(os.getenv("ARM_RECHECK_SECS", "2.0"))

# Sizing / risk
USD_NOTIONAL_PER_NAME = float(os.getenv("ARM_USD_NOTIONAL_PER_NAME", "5000"))
MIN_TICKET_USD        = float(os.getenv("ARM_MIN_TICKET_USD", "200"))
MAX_NAMES             = int(os.getenv("ARM_MAX_NAMES", "25"))
MAX_PER_SECTOR        = int(os.getenv("ARM_MAX_PER_SECTOR", "6"))
LOT                   = float(os.getenv("ARM_LOT", "1"))

# Weights for composite revision score
W_NTM   = float(os.getenv("ARM_W_NTM", "0.6"))
W_FY1   = float(os.getenv("ARM_W_FY1", "0.2"))
W_RATE  = float(os.getenv("ARM_W_RATE", "0.15"))
W_PT    = float(os.getenv("ARM_W_PT",  "0.05"))

# Redis keys
HALT_KEY   = os.getenv("ARM_HALT_KEY", "risk:halt")
UNIV_KEY   = os.getenv("ARM_UNIV_KEY", "universe:eq")
DELTA_HK   = os.getenv("ARM_DELTA_HK", "rev:delta")
LAST_HK    = os.getenv("ARM_LAST_HK",  "last_price")
SECTOR_HK  = os.getenv("ARM_SECTOR_HK", "ref:sector")
FEES_HK    = os.getenv("ARM_FEES_HK",  "fees:eq")

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================ helpers ============================
def _hget_json(hk: str, field: str) -> Optional[dict]:
    raw = r.hget(hk, field)
    if not raw: return None
    try:
        j = json.loads(raw);  return j if isinstance(j, dict) else None # type: ignore
    except Exception:
        return None

def _px(sym: str) -> Optional[float]:
    raw = r.hget(LAST_HK, f"EQ:{sym}")
    if not raw: return None
    try:
        j = json.loads(raw);  return float(j.get("price", 0.0)) # type: ignore
    except Exception:
        try: return float(raw) # type: ignore
        except Exception: return None

def _fees_bps(venue: str="EXCH") -> float:
    v = r.hget(FEES_HK, venue)
    try: return float(v) if v is not None else 10.0 # type: ignore
    except Exception: return 10.0

def _now_ms() -> int: return int(time.time()*1000)

def _days_ago(ms: int) -> float:
    return max(0.0, (_now_ms() - ms) / 86400000.0)

# ============================ state ============================
@dataclass
class OpenState:
    side: str     # "long" | "short"
    qty: float
    z_at_entry: float
    ts_ms: int
    sector: str

def _poskey(name: str, sym: str) -> str:
    return f"arm:open:{name}:{sym}"

# ============================ Strategy ============================
class AnalystRevisionMomentum(Strategy):
    """
    Cross‑sectional analyst revision momentum with sector caps and z‑gates (paper).
    """
    def __init__(self, name: str = "analyst_revision_momentum", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_tick(self, tick: Dict) -> None:
        if (r.get(HALT_KEY) or "0") == "1": return
        now = time.time()
        if now - self._last < RECHECK_SECS: return
        self._last = now

        # ----- 1) Pull universe & build raw composite scores -----
        syms: List[str] = list(r.smembers(UNIV_KEY) or []) # type: ignore
        if not syms: return

        raw_scores: Dict[str, float] = {}
        sectors: Dict[str, str] = {}
        ages: Dict[str, float] = {}

        for s in syms:
            d = _hget_json(DELTA_HK, s)
            if not d: continue
            cov = max(1.0, float(d.get("cov", 1)))
            ntm = float(d.get("ntm_pct", 0.0))         # % change in NTM EPS
            fy1 = float(d.get("fy1_pct", 0.0))         # % change in FY1 EPS
            rate = (float(d.get("rating_up", 0.0)) - float(d.get("rating_down", 0.0))) / cov
            pt   = (float(d.get("pt_up", 0.0)) - float(d.get("pt_down", 0.0))) / cov
            age  = _days_ago(int(d.get("updated_ms", _now_ms())))
            # age decay (older revisions get down‑weighted linearly to 0 by 21 days)
            decay = max(0.0, min(1.0, 1.0 - age/21.0))

            score = decay * (W_NTM*ntm + W_FY1*fy1 + W_RATE*rate + W_PT*pt)
            raw_scores[s] = score
            sectors[s] = (r.hget(SECTOR_HK, s) or "UNKNOWN").upper() # type: ignore
            ages[s] = age

        if not raw_scores: return

        # ----- 2) Cross‑sectional z‑score (universe) -----
        vals = list(raw_scores.values())
        mu = sum(vals)/len(vals)
        var = sum((x-mu)*(x-mu) for x in vals) / max(1.0, len(vals)-1.0)
        sd = math.sqrt(max(1e-12, var))
        zscores = {s: (raw_scores[s] - mu)/sd for s in raw_scores}

        # ----- 3) Manage existing positions (exits) -----
        open_names = []
        sector_loads: Dict[str, int] = {}

        for s in syms:
            st = self._load_state(s)
            if not st: continue
            open_names.append(s)
            sector_loads[st.sector] = sector_loads.get(st.sector, 0) + 1
            z = zscores.get(s, 0.0)

            # exit rules: z mean‑reverted, or max holding days reached, or data too stale
            hold_days = _days_ago(st.ts_ms)
            stale = ages.get(s, 99) >= 30
            if (abs(z) <= EXIT_Z) or (hold_days >= MAX_HOLD_DAYS) or stale:
                self._close_position(s, st)

        # ----- 4) Entries (respect global & sector caps) -----
        # Build candidates sorted by |z| descending, prefer fresh
        cands = sorted(
            [ (s, zscores[s], ages.get(s, 99), sectors[s]) for s in zscores.keys() ],
            key=lambda t: (abs(t[1]), -max(0.0, 30.0 - t[2])) , reverse=True
        )

        # current total names
        n_open = len(open_names)
        fee_bps = _fees_bps("EXCH") * 1e-4

        for s, z, age, sec in cands:
            if n_open >= MAX_NAMES: break
            if abs(z) < ENTRY_Z: continue
            if s in open_names: continue
            if sector_loads.get(sec, 0) >= MAX_PER_SECTOR: continue

            px = _px(s)
            if not px or px <= 0: continue
            qty = math.floor((USD_NOTIONAL_PER_NAME / px) / max(1.0, LOT)) * LOT
            if qty <= 0 or qty * px < MIN_TICKET_USD: continue

            side = "buy" if z > 0 else "sell"
            self.order(f"EQ:{s}", side, qty=qty, order_type="market", venue="EXCH")

            self._save_state(s, OpenState(
                side=("long" if z>0 else "short"),
                qty=qty,
                z_at_entry=z,
                ts_ms=_now_ms(),
                sector=sec
            ))
            sector_loads[sec] = sector_loads.get(sec, 0) + 1
            n_open += 1

        # Optional dashboard: average absolute z among held names
        if open_names:
            avg_abs = sum(abs(zscores.get(s,0.0)) for s in open_names)/len(open_names)
            self.emit_signal(max(-1.0, min(1.0, avg_abs/3.0)))
        else:
            self.emit_signal(0.0)

    # ---------------- helpers ----------------
    def _load_state(self, sym: str) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, sym))
        if not raw: return None
        try:
            o = json.loads(raw) # type: ignore
            return OpenState(side=str(o["side"]), qty=float(o["qty"]),
                             z_at_entry=float(o["z_at_entry"]), ts_ms=int(o["ts_ms"]),
                             sector=str(o.get("sector","UNKNOWN")))
        except Exception:
            return None

    def _save_state(self, sym: str, st: OpenState) -> None:
        r.set(_poskey(self.ctx.name, sym), json.dumps({
            "side": st.side, "qty": st.qty, "z_at_entry": st.z_at_entry,
            "ts_ms": st.ts_ms, "sector": st.sector
        }))

    def _close_position(self, sym: str, st: OpenState) -> None:
        if st.side == "long":
            self.order(f"EQ:{sym}", "sell", qty=st.qty, order_type="market", venue="EXCH")
        else:
            self.order(f"EQ:{sym}", "buy",  qty=st.qty, order_type="market", venue="EXCH")
        r.delete(_poskey(self.ctx.name, sym))