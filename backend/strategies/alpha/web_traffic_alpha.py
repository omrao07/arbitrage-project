# backend/strategies/diversified/web_traffic_alpha.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional, List

import redis
from backend.engine.strategy_base import Strategy

"""
Web Traffic Alpha — paper
-------------------------
Idea (stylized):
  • Rising site/app traffic and search interest tends to lead sales/GMV for many consumer/internet names.
  • We blend normalized level and trend signals (MoM / YoY / 7d-avg vs 28d-avg) with quality gates
    (referral mix, bounce, conversion proxy) and appdownload momentum. Crosssectional z for entries.

Expected Redis (publish from your ingestors — Similarweb/Trends/mobile apps or free scrapers):

# Universe (equities)
SADD universe:eq AMZN ETSY PINS SHOP UBER DASH ABNB ...

# Monthly/weekly aggregates (update as new data drops; keep updated_ms)
HSET web:agg "ETSY" '{
  "visits_millions": 95.2,        // latest period (e.g., month)
  "visits_mom": 0.062,            // MoM pct (e.g., +6.2% → 0.062)
  "visits_yoy": -0.035,           // YoY pct
  "avg_session_dur_s": 410,       // seconds
  "pages_per_visit": 6.2,
  "bounce_rate": 0.39,            // 0..1
  "referral_share_paid": 0.18,    // 0..1 (lower better)
  "referral_share_direct": 0.33,  // 0..1
  "search_interest_7d": 0.58,     // 0..1 scaled
  "search_mom_28d": 0.12,         // 28d vs prior 28d
  "app_dl_7d": 1.25,              // index vs 90d (=1 baseline)
  "gmv_conv_proxy": 0.07,         // conv proxy (checkouts/clicks etc., normalized 0..1)
  "data_lag_days": 7,             // age since end-of-period
  "updated_ms": 1765400000000
}'

# Event brakes (optional): earnings date proximity, major outages, PR shocks, etc.
HSET events:guard "ETSY" '{"earnings_t_ms":1766000000000,"blackout":0,"outage_flag":0,"buzz_spike_z":1.8,"updated_ms":1765390000000}'

# Prices (for sizing)
HSET last_price "EQ:ETSY" '{"price":68.4}'

# Sector map (for caps/neutrality)
HSET ref:sector "ETSY" "INTERNET"

# Ops
HSET fees:eq EXCH 10
SET  risk:halt 0|1

Routing (paper; adapters later):
  order("EQ:<SYM>", side, qty, order_type="market", venue="EXCH")
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("WTA_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("WTA_REDIS_PORT", "6379"))

HALT_KEY    = os.getenv("WTA_HALT_KEY",    "risk:halt")
UNIV_KEY    = os.getenv("WTA_UNIV_KEY",    "universe:eq")
WEB_HK      = os.getenv("WTA_WEB_HK",      "web:agg")
GUARD_HK    = os.getenv("WTA_GUARD_HK",    "events:guard")
LAST_HK     = os.getenv("WTA_LAST_HK",     "last_price")
SECTOR_HK   = os.getenv("WTA_SECTOR_HK",   "ref:sector")
FEES_HK     = os.getenv("WTA_FEES_HK",     "fees:eq")

RECHECK_SECS   = float(os.getenv("WTA_RECHECK_SECS", "5.0"))
STALE_HR       = float(os.getenv("WTA_STALE_HR",     "72"))     # web data can be slow; 3 days tolerance
ENTRY_Z        = float(os.getenv("WTA_ENTRY_Z",      "0.9"))
EXIT_Z         = float(os.getenv("WTA_EXIT_Z",       "0.35"))

USD_PER_NAME   = float(os.getenv("WTA_USD_PER_NAME", "4000"))
MIN_TICKET_USD = float(os.getenv("WTA_MIN_TICKET_USD","150"))
MAX_NAMES      = int(os.getenv("WTA_MAX_NAMES",      "24"))
MAX_PER_SECTOR = int(os.getenv("WTA_MAX_PER_SECTOR", "6"))
LOT            = float(os.getenv("WTA_LOT",          "1"))

# Feature weights (pre‑z composite; positive => long bias; negative => short bias)
W_VIS_MOM      = float(os.getenv("WTA_W_VIS_MOM",   "0.30"))
W_VIS_YOY      = float(os.getenv("WTA_W_VIS_YOY",   "0.15"))
W_SEARCH_LVL   = float(os.getenv("WTA_W_SEARCH",    "0.15"))
W_SEARCH_MOM   = float(os.getenv("WTA_W_SEARCHM",   "0.10"))
W_ENG_QUAL     = float(os.getenv("WTA_W_ENGQUAL",   "0.10"))
W_CONV_PROXY   = float(os.getenv("WTA_W_CONV",      "0.10"))
W_APP_DL       = float(os.getenv("WTA_W_APPDL",     "0.10"))

# Penalties (shrink magnitude)
P_PAID_MIX     = float(os.getenv("WTA_P_PAIDMIX",   "0.20"))  # penalize high paid share
P_BOUNCE       = float(os.getenv("WTA_P_BOUNCE",    "0.15"))
P_DATA_LAG_D   = float(os.getenv("WTA_P_LAG",       "0.01"))  # per day of lag beyond 3 days
FREE_LAG_DAYS  = float(os.getenv("WTA_FREE_LAG_D",  "3.0"))

# Hard brakes
EARN_BLACKOUT_MIN_D = float(os.getenv("WTA_EARN_BMIN","2.0"))  # skip new entries within ±2d of earnings
BUZZ_SPIKE_HARD_Z   = float(os.getenv("WTA_BUZZ_HARD","3.5"))  # skip if social spike extreme

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================ helpers ============================
def _hget_json(hk: str, field: str) -> Optional[dict]:
    raw = r.hget(hk, field)
    if not raw: return None
    try:
        j = json.loads(raw); return j if isinstance(j, dict) else None
    except Exception: return None

def _px(sym: str) -> Optional[float]:
    raw = r.hget(LAST_HK, f"EQ:{sym}")
    if not raw: return None
    try:
        j = json.loads(raw); return float(j.get("price", 0.0))
    except Exception:
        try: return float(raw)
        except Exception: return None

def _now_ms() -> int: return int(time.time()*1000)
def _hours_since(ms: int) -> float: return max(0.0, (_now_ms()-ms)/3_600_000.0)

# ============================ state ============================
@dataclass
class OpenState:
    side: str     # "long" | "short"
    qty: float
    z_at_entry: float
    ts_ms: int
    sector: str

def _poskey(name: str, sym: str) -> str:
    return f"wta:open:{name}:{sym}"

# ============================ Strategy ============================
class WebTrafficAlpha(Strategy):
    """
    Cross‑sectional equity alpha from web/app traffic, search trends, and engagement quality (paper).
    """
    def __init__(self, name: str = "web_traffic_alpha", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_tick(self, tick: Dict) -> None:
        if (r.get(HALT_KEY) or "0") == "1": return
        now = time.time()
        if now - self._last < RECHECK_SECS: return
        self._last = now

        syms: List[str] = list(r.smembers(UNIV_KEY) or [])
        if not syms:
            self.emit_signal(0.0); return

        raw_scores: Dict[str, float] = {}
        sectors: Dict[str, str] = {}
        hard_block: Dict[str, bool] = {}

        for s in syms:
            wa = _hget_json(WEB_HK, s)
            if not wa: continue
            if _hours_since(int(wa.get("updated_ms", _now_ms()))) > STALE_HR:
                continue

            # base features
            vis_mom = float(wa.get("visits_mom", 0.0))
            vis_yoy = float(wa.get("visits_yoy", 0.0))
            search_lvl = float(wa.get("search_interest_7d", 0.5))   # 0..1
            search_mom = float(wa.get("search_mom_28d", 0.0))
            app_dl_idx = float(wa.get("app_dl_7d", 1.0))            # 1 = baseline
            conv = float(wa.get("gmv_conv_proxy", 0.0))             # 0..1
            pages = float(wa.get("pages_per_visit", 0.0))
            dur_s = float(wa.get("avg_session_dur_s", 0.0))
            bounce = float(wa.get("bounce_rate", 0.0))
            paid = float(wa.get("referral_share_paid", 0.0))
            lag_d = float(wa.get("data_lag_days", 7.0))

            # normalize engagement quality roughly to 0..1 (more pages & time, lower bounce)
            eng = 0.5 * (1.0 - min(1.0, bounce)) \
                + 0.25 * (1.0 - math.exp(-pages/6.0)) \
                + 0.25 * (1.0 - math.exp(-dur_s/360.0))
            eng = max(0.0, min(1.0, eng))

            # app downloads — convert to symmetric contribution (>=1 good, <1 bad)
            app_sig = (app_dl_idx - 1.0) / (0.5 + abs(app_dl_idx - 1.0))  # ~[-1,+1]

            # composite (positive => long bias)
            core = (W_VIS_MOM    * vis_mom) \
                 + (W_VIS_YOY    * vis_yoy) \
                 + (W_SEARCH_LVL * (2*search_lvl - 1.0)) \
                 + (W_SEARCH_MOM * search_mom) \
                 + (W_ENG_QUAL   * (2*eng - 1.0)) \
                 + (W_CONV_PROXY * (2*conv - 1.0)) \
                 + (W_APP_DL     * app_sig)

            # penalties
            pen = 0.0
            pen += P_PAID_MIX * min(1.0, max(0.0, paid/0.5))   # >50% paid → heavy penalty
            pen += P_BOUNCE   * min(1.0, max(0.0, (bounce - 0.35)/0.35))  # worse than 35% → penalize
            lag_excess = max(0.0, lag_d - FREE_LAG_DAYS)
            pen += P_DATA_LAG_D * lag_excess

            signed = core - math.copysign(pen, core)

            # hard event brakes
            g = _hget_json(GUARD_HK, s) or {}
            buzz_z = float(g.get("buzz_spike_z", 0.0))
            blackout = int(g.get("blackout", 0))
            outage = int(g.get("outage_flag", 0))
            e_t = int(g.get("earnings_t_ms", 0)) if "earnings_t_ms" in g else 0
            days_to_earn = abs((_now_ms() - e_t)/86_400_000.0) if e_t>0 else 999

            block_new = blackout==1 or outage==1 or (buzz_z >= BUZZ_SPIKE_HARD_Z) or (days_to_earn <= EARN_BLACKOUT_MIN_D)
            hard_block[s] = bool(block_new)

            raw_scores[s] = float(signed)
            sectors[s] = (r.hget(SECTOR_HK, s) or "UNKNOWN").upper()

        if not raw_scores:
            self.emit_signal(0.0); return

        # cross‑sectional z
        vals = list(raw_scores.values())
        mu = sum(vals)/len(vals)
        var = sum((x-mu)*(x-mu) for x in vals) / max(1.0, len(vals)-1.0)
        sd = math.sqrt(max(1e-12, var))
        zmap = {s: (raw_scores[s] - mu)/sd for s in raw_scores}

        # manage existing (exits)
        open_names = []
        sector_loads: Dict[str, int] = {}
        for s in list(zmap.keys()):
            st = self._load_state(s)
            if not st: 
                continue
            open_names.append(s)
            sector_loads[st.sector] = sector_loads.get(st.sector, 0) + 1
            z = zmap.get(s, 0.0)
            # exit on mean reversion or if hard block appears
            if (abs(z) <= EXIT_Z) or hard_block.get(s, False):
                self._close(s, st)

        # entries
        n_open = len(open_names)
        cands = sorted(zmap.items(), key=lambda kv: abs(kv[1]), reverse=True)

        for s, z in cands:
            if n_open >= MAX_NAMES: break
            if s in open_names: continue
            if abs(z) < ENTRY_Z: continue
            if hard_block.get(s, False): continue

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
                qty=qty, z_at_entry=z, ts_ms=_now_ms(), sector=sec
            ))
            sector_loads[sec] = sector_loads.get(sec, 0) + 1
            n_open += 1

        # dashboard signal: avg |z|
        avg_abs = sum(abs(v) for v in zmap.values())/len(zmap)
        self.emit_signal(max(-1.0, min(1.0, avg_abs/3.0)))

    # ---------------- state I/O ----------------
    def _load_state(self, sym: str) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, sym))
        if not raw: return None
        try:
            o = json.loads(raw)
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