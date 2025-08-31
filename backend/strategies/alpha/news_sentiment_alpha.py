# backend/strategies/diversified/news_sentiment_alpha.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import redis
from backend.engine.strategy_base import Strategy

"""
News Sentiment Alpha — paper
----------------------------
Expected Redis schema you publish elsewhere (examples):

# Universe
SADD universe:eq AAPL MSFT NVDA AMZN META TSLA ...

# Aggregated news sentiment per symbol (rolling N hours/day)
# All scores in [-1, +1]; counts are integers; times are epoch ms.
HSET news:agg "AAPL" '{
  "sent_mean": 0.42,              // weighted by source credibility
  "sent_median": 0.40,
  "sent_std": 0.25,
  "bull_count": 18,
  "bear_count": 5,
  "neutral_count": 7,
  "source_cred_mean": 0.78,       // 0..1
  "conflict_ratio": 0.22,         // bear_count / (bull_count+bear_count+1e-9)
  "surprise_flag": 1,             // 1 if major unexpected news (M&A, guidance shock, probe, etc.)
  "event_intensity": 0.65,        // 0..1 (headline velocity, share velocity)
  "vol_rel_10d": 2.1,             // today’s notional vs 10d avg (optional)
  "updated_ms": 1765400000000
}'

# Prices & sectors
HSET last_price "EQ:AAPL" '{"price": 230.15}'
HSET ref:sector "AAPL" "TECH"

# Ops
HSET fees:eq EXCH 10
SET  risk:halt 0|1

Routing (paper; adapters later):
  order("EQ:<SYM>", side, qty, order_type="market", venue="EXCH")
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("NSA_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("NSA_REDIS_PORT", "6379"))

# Gates & cadence
ENTRY_Z        = float(os.getenv("NSA_ENTRY_Z", "0.9"))
EXIT_Z         = float(os.getenv("NSA_EXIT_Z",  "0.3"))
MAX_HOLD_DAYS  = int(os.getenv("NSA_MAX_HOLD_DAYS", "7"))
RECHECK_SECS   = float(os.getenv("NSA_RECHECK_SECS", "0.8"))

# Sizing / risk
USD_PER_NAME     = float(os.getenv("NSA_USD_PER_NAME", "5000"))
MIN_TICKET_USD   = float(os.getenv("NSA_MIN_TICKET_USD", "200"))
MAX_NAMES        = int(os.getenv("NSA_MAX_NAMES", "20"))
MAX_PER_SECTOR   = int(os.getenv("NSA_MAX_PER_SECTOR", "6"))
LOT              = float(os.getenv("NSA_LOT", "1"))

# Factor weights (pre‑z composite)
W_SENT_MEAN     = float(os.getenv("NSA_W_SENT_MEAN", "0.55"))
W_EVENT_INT     = float(os.getenv("NSA_W_EVENT_INT", "0.20"))
W_CRED          = float(os.getenv("NSA_W_CRED",      "0.15"))
W_VOL_CONFIRM   = float(os.getenv("NSA_W_VOL_CONF",  "0.10"))

# Penalties
P_CONFLICT      = float(os.getenv("NSA_P_CONFLICT",  "0.40"))  # * conflict_ratio
P_NOISE_STD     = float(os.getenv("NSA_P_NOISE_STD", "0.10"))  # * sent_std
P_STALENESS_HR  = float(os.getenv("NSA_P_STALE_HR",  "0.03"))  # per hour decay

# Surprise boost
SURPRISE_BONUS  = float(os.getenv("NSA_SURPRISE_BONUS","0.25"))  # added (signed by sent_mean) when surprise_flag=1

# Redis keys
HALT_KEY   = os.getenv("NSA_HALT_KEY",   "risk:halt")
UNIV_KEY   = os.getenv("NSA_UNIV_KEY",   "universe:eq")
NEWS_HK    = os.getenv("NSA_NEWS_HK",    "news:agg")
LAST_HK    = os.getenv("NSA_LAST_HK",    "last_price")
SECTOR_HK  = os.getenv("NSA_SECTOR_HK",  "ref:sector")
FEES_HK    = os.getenv("NSA_FEES_HK",    "fees:eq")

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================ helpers ============================
def _hget_json(hk: str, field: str) -> Optional[dict]:
    raw = r.hget(hk, field)
    if not raw: return None
    try:
        j = json.loads(raw); return j if isinstance(j, dict) else None # type: ignore
    except Exception: return None

def _px(sym: str) -> Optional[float]:
    raw = r.hget(LAST_HK, f"EQ:{sym}")
    if not raw: return None
    try:
        j = json.loads(raw); return float(j.get("price", 0.0)) # type: ignore
    except Exception:
        try: return float(raw) # type: ignore
        except Exception: return None

def _fees_bps(venue: str="EXCH") -> float:
    v = r.hget(FEES_HK, venue)
    try: return float(v) if v is not None else 10.0 # type: ignore
    except Exception: return 10.0

def _now_ms() -> int: return int(time.time()*1000)
def _hours_since(ms: int) -> float: return max(0.0, (_now_ms() - ms) / 3_600_000.0)

# ============================ state ============================
@dataclass
class OpenState:
    side: str     # "long" | "short"
    qty: float
    z_at_entry: float
    ts_ms: int
    sector: str

def _poskey(name: str, sym: str) -> str:
    return f"nsa:open:{name}:{sym}"

# ============================ Strategy ============================
class NewsSentimentAlpha(Strategy):
    """
    Real‑time news sentiment cross‑sectional strategy with freshness decay and conflict penalties (paper).
    """
    def __init__(self, name: str = "news_sentiment_alpha", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_tick(self, tick: Dict) -> None:
        if (r.get(HALT_KEY) or "0") == "1": return
        now = time.time()
        if now - self._last < RECHECK_SECS: return
        self._last = now

        syms: List[str] = list(r.smembers(UNIV_KEY) or []) # type: ignore
        if not syms:
            self.emit_signal(0.0); return

        raw_scores: Dict[str, float] = {}
        sectors: Dict[str, str] = {}
        freshness_hr: Dict[str, float] = {}

        # ---- build composite pre‑z score per name ----
        for s in syms:
            d = _hget_json(NEWS_HK, s)
            if not d: continue

            sent = float(d.get("sent_mean", 0.0))
            med  = float(d.get("sent_median", sent))
            std  = float(d.get("sent_std", 0.3))
            cred = float(d.get("source_cred_mean", 0.5))
            confl= float(d.get("conflict_ratio", 0.0))
            ev_i = float(d.get("event_intensity", 0.0))
            volr = float(d.get("vol_rel_10d", 1.0))
            hours= _hours_since(int(d.get("updated_ms", _now_ms())))
            surp = int(d.get("surprise_flag", 0))

            # sign from central tendency (mean; fallback to median)
            base_sent = sent if abs(sent) >= abs(med) else med

            # confirmation by volume: map vol_rel to 0..1 via 1 - exp(-k x)
            vol_conf = 1.0 - math.exp(-0.6 * max(0.0, volr - 1.0))  # =0 at 1x ADV, →1 as vol pops

            # core composite (signed)
            raw = (W_SENT_MEAN * base_sent) \
                + (W_EVENT_INT  * (ev_i * math.copysign(1.0, base_sent))) \
                + (W_CRED       * (cred * math.copysign(1.0, base_sent))) \
                + (W_VOL_CONFIRM* (vol_conf * math.copysign(1.0, base_sent)))

            # penalties (reduce magnitude)
            raw -= math.copysign(P_CONFLICT * confl, raw)
            raw -= math.copysign(P_NOISE_STD * min(1.0, std), raw)
            raw -= math.copysign(P_STALENESS_HR * min(48.0, hours), raw)

            # surprise kicker (aligned with sign)
            if surp == 1:
                raw += math.copysign(SURPRISE_BONUS, raw if raw != 0 else base_sent)

            raw_scores[s] = float(raw)
            sectors[s] = (r.hget(SECTOR_HK, s) or "UNKNOWN").upper() # type: ignore
            freshness_hr[s] = hours

        if not raw_scores:
            self.emit_signal(0.0); return

        # ---- cross‑sectional z‑score ----
        vals = list(raw_scores.values())
        mu = sum(vals)/len(vals)
        var = sum((x-mu)*(x-mu) for x in vals) / max(1.0, len(vals)-1.0)
        sd = math.sqrt(max(1e-12, var))
        zmap = {s: (raw_scores[s] - mu)/sd for s in raw_scores}

        # ---- manage existing (exits) ----
        open_names = []
        sector_loads: Dict[str, int] = {}

        for s in syms:
            st = self._load_state(s)
            if not st: continue
            open_names.append(s)
            sector_loads[st.sector] = sector_loads.get(st.sector, 0) + 1
            z = zmap.get(s, 0.0)
            hold_days = (_now_ms() - st.ts_ms)/86_400_000.0
            # exit on mean‑reversion, max hold, or stale (>24h)
            if (abs(z) <= EXIT_Z) or (hold_days >= MAX_HOLD_DAYS) or (freshness_hr.get(s, 99) > 24):
                self._close(s, st)

        # ---- entries (respect caps) ----
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

        # dashboard signal: average |z| in opp set
        avg_abs = sum(abs(v) for v in zmap.values())/len(zmap)
        self.emit_signal(max(-1.0, min(1.0, avg_abs/3.0)))

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