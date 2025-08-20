# backend/strategies/diversified/nlp_social_alpha.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional, List

import redis
from backend.engine.strategy_base import Strategy

"""
NLP Social Alpha — paper
------------------------
Expected Redis you publish elsewhere (your social ingestor writes this):

# Universe
SADD universe:eq AAPL MSFT NVDA AMZN META TSLA ...

# Rolling short-window (e.g., last 60–120 min) social aggregates per symbol
# All sentiment/stance in [-1,+1], probabilities in [0,1].
HSET social:agg "AAPL" '{
  "sent_mean": 0.38,            // engagement-weighted mean sentiment
  "sent_median": 0.34,
  "stance_bull": 0.71,          // model % bullish
  "stance_bear": 0.18,          // model % bearish
  "sarcasm_prob": 0.06,         // sarcasm/irony detector avg
  "bot_score_mean": 0.12,       // Botometer-like 0..1 (higher = botty)
  "author_rep_mean": 0.66,      // 0..1 (karma/followers/age quality)
  "engage_rate": 0.54,          // scaled 0..1 (likes+replies+RT per follower)
  "burst_z": 2.1,               // post-volume burst z-score vs 7d same hour
  "url_cred_mean": 0.73,        // 0..1 source credibility of linked URLs
  "conflict_px_30m": -0.12,     // sign(conflict) * |corr(sent vs 30m return)| ; negative ⇒ talk vs tape conflict
  "topic_risk": 0.15,           // 0..1 (regulatory/controversy lexicon proximity)
  "lang_mix_div": 0.22,         // 0..1 (language diversity; too high can add noise)
  "sample_n": 520,              // posts in window
  "updated_ms": 1765400000000
}'

# Prices & sector
HSET last_price "EQ:AAPL" '{"price":230.15}'
HSET ref:sector "AAPL" "TECH"

# Ops
HSET fees:eq EXCH 10
SET  risk:halt 0|1

Routing (paper; adapters later):
  order("EQ:<SYM>", side, qty, order_type="market", venue="EXCH")
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("SNA_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("SNA_REDIS_PORT", "6379"))

# Gates / cadence (social is fast → short recheck)
ENTRY_Z        = float(os.getenv("SNA_ENTRY_Z", "1.0"))
EXIT_Z         = float(os.getenv("SNA_EXIT_Z",  "0.35"))
MAX_HOLD_HRS   = float(os.getenv("SNA_MAX_HOLD_HRS", "12"))      # intradayish
RECHECK_SECS   = float(os.getenv("SNA_RECHECK_SECS", "0.6"))

# Sizing / risk
USD_PER_NAME     = float(os.getenv("SNA_USD_PER_NAME", "4000"))
MIN_TICKET_USD   = float(os.getenv("SNA_MIN_TICKET_USD", "150"))
MAX_NAMES        = int(os.getenv("SNA_MAX_NAMES", "24"))
MAX_PER_SECTOR   = int(os.getenv("SNA_MAX_PER_SECTOR", "6"))
LOT              = float(os.getenv("SNA_LOT", "1"))

# Feature weights for pre‑z composite (tuned later)
W_SENT          = float(os.getenv("SNA_W_SENT",        "0.45"))
W_STANCE        = float(os.getenv("SNA_W_STANCE",      "0.20"))
W_ENGAGE        = float(os.getenv("SNA_W_ENGAGE",      "0.15"))
W_BURST         = float(os.getenv("SNA_W_BURST",       "0.08"))
W_AUTHOR_REP    = float(os.getenv("SNA_W_AUTHOR_REP",  "0.07"))
W_URL_CRED      = float(os.getenv("SNA_W_URL_CRED",    "0.05"))

# Penalties (signed reduction of magnitude)
P_BOT           = float(os.getenv("SNA_P_BOT",         "0.50"))   # * bot_score_mean
P_SARCASM       = float(os.getenv("SNA_P_SARCASM",     "0.25"))   # * sarcasm_prob
P_TOPIC_RISK    = float(os.getenv("SNA_P_TOPIC",       "0.20"))   # * topic_risk
P_LANG_NOISE    = float(os.getenv("SNA_P_LANG",        "0.10"))   # * lang_mix_div
P_CONFLICT_PX   = float(os.getenv("SNA_P_CONFLICT",    "0.30"))   # * negative conflict with tape
P_STALE_MIN     = float(os.getenv("SNA_P_STALE_MIN",   "0.015"))  # per minute decay beyond 5 min
STALE_FREE_MIN  = float(os.getenv("SNA_STALE_FREE_MIN","5"))      # free freshness window

# Hard gates
MIN_SAMPLE_N     = int(os.getenv("SNA_MIN_SAMPLE_N", "50"))       # ignore tiny windows
MAX_BOT_MEAN     = float(os.getenv("SNA_MAX_BOT_MEAN", "0.6"))    # > → block entries
MIN_URL_CRED_MEAN= float(os.getenv("SNA_MIN_URL_CRED","0.35"))    # < → block entries

# Redis keys
HALT_KEY   = os.getenv("SNA_HALT_KEY",   "risk:halt")
UNIV_KEY   = os.getenv("SNA_UNIV_KEY",   "universe:eq")
SOC_HK     = os.getenv("SNA_SOC_HK",     "social:agg")
LAST_HK    = os.getenv("SNA_LAST_HK",    "last_price")
SECTOR_HK  = os.getenv("SNA_SECTOR_HK",  "ref:sector")
FEES_HK    = os.getenv("SNA_FEES_HK",    "fees:eq")

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
def _mins_since(ms: int) -> float: return max(0.0, (_now_ms() - ms) / 60_000.0)

# ============================ state ============================
@dataclass
class OpenState:
    side: str     # "long" | "short"
    qty: float
    z_at_entry: float
    ts_ms: int
    sector: str

def _poskey(name: str, sym: str) -> str:
    return f"sna:open:{name}:{sym}"

# ============================ Strategy ============================
class NLPSocialAlpha(Strategy):
    """
    Social‑NLP cross‑sectional alpha with freshness decay, bot/noise penalties, sector caps, z‑gates (paper).
    """
    def __init__(self, name: str = "nlp_social_alpha", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
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
        age_min: Dict[str, float] = {}

        for s in syms:
            d = _hget_json(SOC_HK, s)
            if not d: continue
            n = int(d.get("sample_n", 0))
            if n < MIN_SAMPLE_N: 
                continue

            sent = float(d.get("sent_mean", 0.0))
            med  = float(d.get("sent_median", sent))
            bull = float(d.get("stance_bull", 0.0))
            bear = float(d.get("stance_bear", 0.0))
            stance = bull - bear  # [-1, +1]
            sarcasm = float(d.get("sarcasm_prob", 0.0))
            botm = float(d.get("bot_score_mean", 0.0))
            arep = float(d.get("author_rep_mean", 0.0))
            engage = float(d.get("engage_rate", 0.0))
            burst_z = float(d.get("burst_z", 0.0))
            urlcred = float(d.get("url_cred_mean", 0.5))
            conflict = float(d.get("conflict_px_30m", 0.0))  # negative ⇒ talk vs tape
            topic = float(d.get("topic_risk", 0.0))
            langdiv = float(d.get("lang_mix_div", 0.0))
            mins = _mins_since(int(d.get("updated_ms", _now_ms())))

            # Hard intake gates (block obvious junk)
            if botm > MAX_BOT_MEAN: 
                continue
            if urlcred < MIN_URL_CRED_MEAN:
                continue

            base = sent if abs(sent) >= abs(med) else med

            # Composite (signed by base)
            raw = (W_SENT       * base) \
                + (W_STANCE     * (stance * math.copysign(1.0, base))) \
                + (W_ENGAGE     * (engage * math.copysign(1.0, base))) \
                + (W_BURST      * (max(0.0, burst_z)/3.0 * math.copysign(1.0, base))) \
                + (W_AUTHOR_REP * (arep * math.copysign(1.0, base))) \
                + (W_URL_CRED   * (urlcred * math.copysign(1.0, base)))

            # Penalties reduce magnitude
            raw -= math.copysign(P_BOT * botm, raw)
            raw -= math.copysign(P_SARCASM * sarcasm, raw)
            raw -= math.copysign(P_TOPIC_RISK * topic, raw)
            raw -= math.copysign(P_LANG_NOISE * langdiv, raw)
            if conflict < 0:
                raw -= math.copysign(P_CONFLICT_PX * min(1.0, -conflict), raw)

            # Freshness decay (after grace window)
            staleness = max(0.0, mins - STALE_FREE_MIN)
            raw -= math.copysign(P_STALE_MIN * staleness, raw)

            raw_scores[s] = float(raw)
            sectors[s] = (r.hget(SECTOR_HK, s) or "UNKNOWN").upper()
            age_min[s] = mins

        if not raw_scores:
            self.emit_signal(0.0); return

        # Cross‑sectional z
        vals = list(raw_scores.values())
        mu = sum(vals)/len(vals)
        var = sum((x-mu)*(x-mu) for x in vals) / max(1.0, len(vals)-1.0)
        sd = math.sqrt(max(1e-12, var))
        zmap = {s: (raw_scores[s] - mu)/sd for s in raw_scores}

        # Manage existing (exits)
        open_names = []
        sector_loads: Dict[str, int] = {}

        for s in syms:
            st = self._load_state(s)
            if not st: continue
            open_names.append(s)
            sector_loads[st.sector] = sector_loads.get(st.sector, 0) + 1
            z = zmap.get(s, 0.0)
            held_min = (_now_ms() - st.ts_ms)/60_000.0
            # exit on mean‑reversion, stale (> 90 min), or max hold
            if (abs(z) <= EXIT_Z) or (held_min >= MAX_HOLD_HRS*60.0) or (age_min.get(s, 999) > 90.0):
                self._close(s, st)

        # Entries
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

        # Dashboard signal: avg |z| in opp set
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