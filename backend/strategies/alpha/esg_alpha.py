# backend/strategies/diversified/esg_alpha.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional, List

import redis
from backend.engine.strategy_base import Strategy

"""
ESG Alpha — paper
-----------------
Idea: Cross‑sectional long/short on **ESG score momentum** and **carbon‑/governance‑/green‑rev factors**,
with controversy penalties, sector caps, exclusions, and z‑gated entries.

You publish to Redis (examples):

  # Tradable universe
  SADD universe:eq AAPL MSFT TSLA NVDA ...

  # Current composite ESG datapoint per symbol (refresh daily)
  HSET esg:point "AAPL" '{
    "esg_score": 72.1,                // normalized 0..100
    "esg_score_30d_ago": 70.4,        // for momentum (%)
    "env_score": 68.0, "soc_score": 75.0, "gov_score": 80.0,
    "co2e_tons_per_mm_sales": 25.3,   // emissions intensity
    "co2e_tons_per_mm_sales_90d_ago": 28.1,
    "green_revenue_share_pct": 14.2,  // %
    "controversy_level": 0,           // 0=none, 1=minor, 2=moderate, 3+=severe  (halts above threshold)
    "updated_ms": 1765400000000
  }'

  # Prices
  HSET last_price "EQ:AAPL" '{"price":230.15}'

  # Sector map
  HSET ref:sector "AAPL" "TECH"

  # Exclusion list (we won't take new positions in these)
  SADD esg:exclude RUSAL TOBACCO_CO OIL_BAD

  # Optional sin‑sector tags
  HSET esg:sin "MO" "TOBACCO"

  # Fees (bps) and kill switch
  HSET fees:eq EXCH 10
  SET  risk:halt 0|1

Routing (paper; adapters wire later):
  order("EQ:<SYM>", side, qty, order_type="market", venue="EXCH")
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("ESGA_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("ESGA_REDIS_PORT", "6379"))

# Z‑gates & windows
ENTRY_Z        = float(os.getenv("ESGA_ENTRY_Z", "0.9"))
EXIT_Z         = float(os.getenv("ESGA_EXIT_Z",  "0.3"))
MAX_HOLD_DAYS  = int(os.getenv("ESGA_MAX_HOLD_DAYS", "30"))
RECHECK_SECS   = float(os.getenv("ESGA_RECHECK_SECS", "2.0"))

# Sizing / risk
USD_PER_NAME     = float(os.getenv("ESGA_USD_PER_NAME", "5000"))
MIN_TICKET_USD   = float(os.getenv("ESGA_MIN_TICKET_USD", "200"))
MAX_NAMES        = int(os.getenv("ESGA_MAX_NAMES", "25"))
MAX_PER_SECTOR   = int(os.getenv("ESGA_MAX_PER_SECTOR", "6"))
LOT              = float(os.getenv("ESGA_LOT", "1"))

# Factor weights (composite raw score before z‑scoring)
W_SCORE_MOM   = float(os.getenv("ESGA_W_SCORE_MOM", "0.45"))  # ΔESG%
W_GOV         = float(os.getenv("ESGA_W_GOV",       "0.20"))  # governance absolute (scaled)
W_GREEN_REV   = float(os.getenv("ESGA_W_GREEN",     "0.15"))  # % green revenue
W_EMI_TREND   = float(os.getenv("ESGA_W_EMI_TREND", "0.20"))  # negative weight (improving = down)

# Penalties / constraints
CONTRO_MAX      = int(os.getenv("ESGA_CONTRO_MAX", "2"))      # >= this → block entries
SIN_SECTOR_CAP  = int(os.getenv("ESGA_SIN_SECTOR_CAP", "0"))  # 0 = disallow; else limit positions
SHORT_SIN_NAMES = int(os.getenv("ESGA_SHORT_SIN_NAMES", "0")) # allow shorting sin names if >0

# Redis keys
HALT_KEY   = os.getenv("ESGA_HALT_KEY",   "risk:halt")
UNIV_KEY   = os.getenv("ESGA_UNIV_KEY",   "universe:eq")
POINT_HK   = os.getenv("ESGA_POINT_HK",   "esg:point")
LAST_HK    = os.getenv("ESGA_LAST_HK",    "last_price")
SECTOR_HK  = os.getenv("ESGA_SECTOR_HK",  "ref:sector")
FEES_HK    = os.getenv("ESGA_FEES_HK",    "fees:eq")
EXCL_SET   = os.getenv("ESGA_EXCL_SET",   "esg:exclude")
SIN_HK     = os.getenv("ESGA_SIN_HK",     "esg:sin")

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================ helpers ============================
def _hget_json(hk: str, field: str) -> Optional[dict]:
    raw = r.hget(hk, field)
    if not raw: return None
    try:
        j = json.loads(raw);  return j if isinstance(j, dict) else None # type: ignore
    except Exception: return None

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
def _days_ago(ms: int) -> float: return max(0.0, (_now_ms() - ms)/86400000.0)

# ============================ state ============================
@dataclass
class OpenState:
    side: str       # "long" | "short"
    qty: float
    z_at_entry: float
    sector: str
    ts_ms: int

def _poskey(name: str, sym: str) -> str:
    return f"esga:open:{name}:{sym}"

# ============================ Strategy ============================
class ESGAlpha(Strategy):
    """
    ESG score‑momentum + fundamentals (gov, green rev, emissions trend),
    cross‑sectional z‑scoring, sector caps, controversy + exclusion gates.
    """
    def __init__(self, name: str = "esg_alpha", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_tick(self, tick: Dict) -> None:
        if (r.get(HALT_KEY) or "0") == "1": return
        now = time.time()
        if now - self._last < RECHECK_SECS: return
        self._last = now

        syms: List[str] = list(r.smembers(UNIV_KEY) or []) # type: ignore
        if not syms: 
            self.emit_signal(0.0); 
            return

        excl = set(r.smembers(EXCL_SET) or []) # type: ignore
        raw_scores: Dict[str, float] = {}
        sectors: Dict[str, str] = {}
        sins: Dict[str, str] = {}

        # 1) Build raw composite score per name
        for s in syms:
            if s in excl: 
                continue
            p = _hget_json(POINT_HK, s)
            if not p: 
                continue

            # Controversy gate
            if int(p.get("controversy_level", 0)) >= CONTRO_MAX:
                # If already held, we’ll exit below; block new entries by skipping raw score
                continue

            # Factors
            esg = float(p.get("esg_score", 0.0))
            esg_prev = float(p.get("esg_score_30d_ago", esg))
            score_mom = ( (esg - esg_prev) / max(1.0, esg_prev) )  # % change

            gov = float(p.get("gov_score", 0.0)) / 100.0           # scale to 0..1

            green = float(p.get("green_revenue_share_pct", 0.0)) / 100.0

            emi = float(p.get("co2e_tons_per_mm_sales", 0.0))
            emi_prev = float(p.get("co2e_tons_per_mm_sales_90d_ago", emi))
            emi_trend = - ( (emi - emi_prev) / max(1e-6, emi_prev) )  # improvement (down) → positive

            # Composite
            score = (W_SCORE_MOM*score_mom) + (W_GOV*gov) + (W_GREEN_REV*green) + (W_EMI_TREND*emi_trend)

            raw_scores[s] = score
            sectors[s] = (r.hget(SECTOR_HK, s) or "UNKNOWN").upper() # type: ignore
            sins[s] = (r.hget(SIN_HK, s) or "").upper() # type: ignore

        if not raw_scores:
            self.emit_signal(0.0)
            return

        # 2) Cross‑sectional z‑score
        vals = list(raw_scores.values())
        mu = sum(vals)/len(vals)
        var = sum((x-mu)*(x-mu) for x in vals) / max(1.0, len(vals)-1.0)
        sd = math.sqrt(max(1e-12, var))
        zmap = {s: (raw_scores[s] - mu)/sd for s in raw_scores}

        # 3) Manage existing positions: exits on mean‑reversion, controversy spike, or max hold
        open_names = []
        sector_loads: Dict[str, int] = {}

        for s in syms:
            st = self._load_state(s)
            if not st: 
                continue
            open_names.append(s)
            sector_loads[st.sector] = sector_loads.get(st.sector, 0) + 1

            z = zmap.get(s, 0.0)
            hold_days = _days_ago(st.ts_ms)

            # controversy recheck (if now above threshold, force exit)
            p = _hget_json(POINT_HK, s) or {}
            contro = int(p.get("controversy_level", 0))

            if (abs(z) <= EXIT_Z) or (hold_days >= MAX_HOLD_DAYS) or (contro >= CONTRO_MAX):
                self._close_position(s, st)

        # 4) Entries: respect sector caps and sin/exclusion rules
        n_open = len(open_names)

        # order candidates by |z|
        cands = sorted(zmap.items(), key=lambda kv: abs(kv[1]), reverse=True)

        for s, z in cands:
            if n_open >= MAX_NAMES: break
            if s in open_names: continue
            if abs(z) < ENTRY_Z: continue

            sec = sectors.get(s, "UNKNOWN")
            if sector_loads.get(sec, 0) >= MAX_PER_SECTOR: 
                continue

            # sin handling
            sin_tag = sins.get(s, "")
            if sin_tag:
                # disallow new longs if SIN_SECTOR_CAP==0; allow shorts if SHORT_SIN_NAMES>0
                if z > 0 and SIN_SECTOR_CAP <= 0:
                    continue
                if z < 0 and SHORT_SIN_NAMES <= 0:
                    continue

            px = _px(s)
            if not px or px <= 0: continue
            qty = math.floor((USD_PER_NAME / px) / max(1.0, LOT)) * LOT
            if qty <= 0 or qty*px < MIN_TICKET_USD: 
                continue

            side = "buy" if z > 0 else "sell"
            self.order(f"EQ:{s}", side, qty=qty, order_type="market", venue="EXCH")

            self._save_state(s, OpenState(
                side=("long" if z>0 else "short"),
                qty=qty,
                z_at_entry=z,
                sector=sec,
                ts_ms=_now_ms()
            ))
            sector_loads[sec] = sector_loads.get(sec, 0) + 1
            n_open += 1

        # dashboard: average |z|
        avg_abs = sum(abs(v) for v in zmap.values())/len(zmap)
        self.emit_signal(max(-1.0, min(1.0, avg_abs/3.0)))

    # ---------------- helpers ----------------
    def _load_state(self, sym: str) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, sym))
        if not raw: return None
        try:
            o = json.loads(raw) # type: ignore
            return OpenState(side=str(o["side"]), qty=float(o["qty"]), z_at_entry=float(o["z_at_entry"]),
                             sector=str(o.get("sector","UNKNOWN")), ts_ms=int(o["ts_ms"]))
        except Exception:
            return None

    def _save_state(self, sym: str, st: OpenState) -> None:
        r.set(_poskey(self.ctx.name, sym), json.dumps({
            "side": st.side, "qty": st.qty, "z_at_entry": st.z_at_entry,
            "sector": st.sector, "ts_ms": st.ts_ms
        }))

    def _close_position(self, sym: str, st: OpenState) -> None:
        if st.side == "long":
            self.order(f"EQ:{sym}", "sell", qty=st.qty, order_type="market", venue="EXCH")
        else:
            self.order(f"EQ:{sym}", "buy",  qty=st.qty, order_type="market", venue="EXCH")
        r.delete(_poskey(self.ctx.name, sym))