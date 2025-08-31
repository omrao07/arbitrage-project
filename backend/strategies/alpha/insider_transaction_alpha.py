# backend/strategies/diversified/insider_transaction_alpha.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import redis
from backend.engine.strategy_base import Strategy

"""
Insider Transaction Alpha — paper
---------------------------------
Idea:
  • Clustered *net insider buying* (multiple officers/directors, meaningful $ value)
    predicts outperformance; clustered selling predicts underperformance.
  • Filter out: small trades, routine option exercises, tax-withholdings, 10b5-1 planned trades.
  • Weight by role seniority (CEO/CFO > Director), recency, breadth (# insiders), and $ value.

Publish to Redis (examples):

  # Universe
  SADD universe:eq AAPL MSFT NVDA AMZN META TSLA ...

  # Rolling 90d insider summary per symbol (refresh intraday/daily)
  HSET insider:roll "AAPL" '{
    "window_days": 90,
    "events": 24,                 // count of Form 4 lines included
    "insiders": 6,                // distinct insiders
    "net_usd": 8_500_000,         // net BUY $ (buys - sells), negative if net selling
    "gross_buy_usd": 9_200_000,
    "gross_sell_usd": 700_000,
    "buy_trades": 18, "sell_trades": 6,
    "avg_prem_to_30d": 0.012,     // exec price vs 30d VWAP (positive = paid premium)
    "roles": {"CEO":1,"CFO":1,"DIR":4},    // counts by seniority
    "plan_10b5_1_ratio": 0.10,    // share of $ flagged 10b5-1 planned
    "option_exercise_ratio": 0.08,// share of $ from exercises (E/A/M codes)
    "last_event_ms": 1765400000000
  }'

  # Prices
  HSET last_price "EQ:AAPL" '{"price": 230.15}'

  # Sector
  HSET ref:sector "AAPL" "TECH"

  # Ops
  HSET fees:eq EXCH 10
  SET  risk:halt 0|1

Routing (paper; adapters later):
  order("EQ:<SYM>", side, qty, order_type="market", venue="EXCH")
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("ITA_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("ITA_REDIS_PORT", "6379"))

ENTRY_Z        = float(os.getenv("ITA_ENTRY_Z", "1.0"))
EXIT_Z         = float(os.getenv("ITA_EXIT_Z",  "0.3"))
MAX_HOLD_DAYS  = int(os.getenv("ITA_MAX_HOLD_DAYS", "30"))
RECHECK_SECS   = float(os.getenv("ITA_RECHECK_SECS", "1.5"))

USD_PER_NAME     = float(os.getenv("ITA_USD_PER_NAME", "6000"))
MIN_TICKET_USD   = float(os.getenv("ITA_MIN_TICKET_USD", "200"))
MAX_NAMES        = int(os.getenv("ITA_MAX_NAMES", "25"))
MAX_PER_SECTOR   = int(os.getenv("ITA_MAX_PER_SECTOR", "6"))
LOT              = float(os.getenv("ITA_LOT", "1"))

# Feature weights for raw conviction score (before z-score)
W_NET_USD_SIG      = float(os.getenv("ITA_W_NET_USD_SIG", "0.45"))  # size-adjusted net $
W_BREADTH          = float(os.getenv("ITA_W_BREADTH",     "0.20"))  # distinct insiders
W_SENIORITY        = float(os.getenv("ITA_W_SENIORITY",   "0.20"))  # CEO/CFO premium
W_PRICE_SIGNAL     = float(os.getenv("ITA_W_PRICE_SIG",   "0.10"))  # paid above VWAP, etc.
W_RECENCY_DECAY    = float(os.getenv("ITA_W_REC_DECAY",   "0.05"))

# Penalties (subtract from raw score; they also gate entries if too large)
P_10B5_RATIO_PEN   = float(os.getenv("ITA_P_10B5_PEN",    "0.40"))  # penalty * plan_10b5_1_ratio
P_EXERCISE_RATIO_PEN= float(os.getenv("ITA_P_EXER_PEN",   "0.35"))  # penalty * option_exercise_ratio
MIN_NET_USD_SIGMA  = float(os.getenv("ITA_MIN_NET_USD_SIGMA", "0.5")) # min standardized size to consider
MIN_INSIDERS       = int(os.getenv("ITA_MIN_INSIDERS", "2"))

# Redis keys
HALT_KEY   = os.getenv("ITA_HALT_KEY",   "risk:halt")
UNIV_KEY   = os.getenv("ITA_UNIV_KEY",   "universe:eq")
ROLL_HK    = os.getenv("ITA_ROLL_HK",    "insider:roll")
LAST_HK    = os.getenv("ITA_LAST_HK",    "last_price")
SECTOR_HK  = os.getenv("ITA_SECTOR_HK",  "ref:sector")
FEES_HK    = os.getenv("ITA_FEES_HK",    "fees:eq")

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
def _days_ago(ms: int) -> float: return max(0.0, (_now_ms() - ms)/86400000.0)

# ============================ state ============================
@dataclass
class OpenState:
    side: str     # "long" | "short"
    qty: float
    z_at_entry: float
    ts_ms: int
    sector: str

def _poskey(name: str, sym: str) -> str:
    return f"ita:open:{name}:{sym}"

# ============================ scoring ============================
def _role_weight(roles: Dict[str,int]) -> float:
    # CEO:1.0, CFO:0.8, COO:0.6, DIR:0.4, OTHER:0.3 — count‑weighted
    RW = {"CEO":1.0, "CFO":0.8, "COO":0.6, "PRES":0.6, "DIR":0.4}
    num = 0.0; den = 0.0
    for k,c in (roles or {}).items():
        w = RW.get(k.upper(), 0.3); den += c; num += w * c
    return (num / den) if den > 0 else 0.0

def _standardize_net_usd(net_usd: float, gross_buy: float, gross_sell: float) -> float:
    # A rough sigma proxy: net / (sqrt(gross_buy + gross_sell) + 1)
    base = max(1.0, math.sqrt(max(0.0, gross_buy) + max(0.0, gross_sell)))
    return float(net_usd) / base

def _conviction(d: dict) -> Optional[float]:
    if not d: return None
    insiders = int(d.get("insiders", 0))
    if insiders < MIN_INSIDERS: return None

    net_usd  = float(d.get("net_usd", 0.0))
    gb, gs   = float(d.get("gross_buy_usd", 0.0)), float(d.get("gross_sell_usd", 0.0))
    net_sig  = _standardize_net_usd(net_usd, gb, gs)
    if abs(net_sig) < MIN_NET_USD_SIGMA: return None

    breadth  = math.log1p(max(0, insiders)) / math.log(1 + 10)  # 0..~1 for 1..10+
    senior   = _role_weight(d.get("roles") or {})
    price_sig= float(d.get("avg_prem_to_30d", 0.0))  # + if insiders paid premium vs 30d VWAP
    rec_days = _days_ago(int(d.get("last_event_ms", _now_ms())))
    recency  = max(0.0, 1.0 - rec_days/90.0)         # linear fade across window

    # penalties
    p10b5    = float(d.get("plan_10b5_1_ratio", 0.0))
    pex      = float(d.get("option_exercise_ratio", 0.0))
    penalty  = (P_10B5_RATIO_PEN * p10b5) + (P_EXERCISE_RATIO_PEN * pex)

    # raw score
    score = (W_NET_USD_SIG * net_sig) \
          + (W_BREADTH     * breadth) \
          + (W_SENIORITY   * senior) \
          + (W_PRICE_SIGNAL* price_sig) \
          + (W_RECENCY_DECAY * recency) \
          - penalty

    return score

# ============================ Strategy ============================
class InsiderTransactionAlpha(Strategy):
    """
    Cross‑sectional insider‑flow alpha with de‑noising (10b5‑1/exercises), sector caps, and z‑gates (paper).
    """
    def __init__(self, name: str = "insider_transaction_alpha", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
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

        raw_scores: Dict[str, float] = {}
        sectors: Dict[str, str] = {}

        # 1) Build raw conviction per symbol
        for s in syms:
            d = _hget_json(ROLL_HK, s)
            if not d: continue
            sc = _conviction(d)
            if sc is None: continue
            raw_scores[s] = sc
            sectors[s] = (r.hget(SECTOR_HK, s) or "UNKNOWN").upper() # type: ignore

        if not raw_scores:
            self.emit_signal(0.0); 
            return

        # 2) Cross‑sectional z
        vals = list(raw_scores.values())
        mu = sum(vals)/len(vals)
        var = sum((x-mu)*(x-mu) for x in vals) / max(1.0, len(vals)-1.0)
        sd = math.sqrt(max(1e-12, var))
        zmap = {s: (raw_scores[s] - mu)/sd for s in raw_scores}

        # 3) Manage existing positions (exits)
        open_names = []
        sector_loads: Dict[str,int] = {}
        for s in syms:
            st = self._load_state(s)
            if not st: continue
            open_names.append(s)
            sector_loads[st.sector] = sector_loads.get(st.sector, 0) + 1
            z = zmap.get(s, 0.0)
            hold_days = _days_ago(st.ts_ms)

            # exit on mean‑reversion or max hold or stale/no data anymore
            if (abs(z) <= EXIT_Z) or (hold_days >= MAX_HOLD_DAYS) or (s not in raw_scores):
                self._close(s, st)

        # 4) Entries (respect caps)
        n_open = len(open_names)
        cands = sorted(zmap.items(), key=lambda kv: abs(kv[1]), reverse=True)

        for s, z in cands:
            if n_open >= MAX_NAMES: break
            if s in open_names: continue
            if abs(z) < ENTRY_Z: continue

            sec = sectors.get(s, "UNKNOWN")
            if sector_loads.get(sec, 0) >= MAX_PER_SECTOR: continue

            px = _px(s)
            if not px or px <= 0: continue
            qty = math.floor((USD_PER_NAME / px) / max(1.0, LOT)) * LOT
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

        # dashboard: average |z| in candidate set
        avgabs = sum(abs(v) for v in zmap.values())/len(zmap)
        self.emit_signal(max(-1.0, min(1.0, avgabs/3.0)))

    # ---------------- state I/O ----------------
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

    def _close(self, sym: str, st: OpenState) -> None:
        if st.side == "long":
            self.order(f"EQ:{sym}", "sell", qty=st.qty, order_type="market", venue="EXCH")
        else:
            self.order(f"EQ:{sym}", "buy",  qty=st.qty, order_type="market", venue="EXCH")
        r.delete(_poskey(self.ctx.name, sym))