# backend/strategies/diversified/short_interest_alpha.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional, List

import redis
from backend.engine.strategy_base import Strategy

"""
Short Interest Alpha — paper
----------------------------
Idea (stylized):
  • Elevated short interest that is RISING (and not a squeeze) often precedes underperformance (short).
  • Falling short interest (covering) + benign borrow can precede mean reversion outperformance (long).
  • Avoid/hedge classic squeezes: very high utilization, high borrow fee (CTB), rapid price + vol spikes,
    and social/buzz bursts.

Expected Redis (your market-data/analytics writers publish these):

# Universe
SADD universe:eq AAPL TSLA GME AMC RBLX ...

# Short-interest & borrow analytics (update daily; intraday if you have it)
HSET short:agg "GME" '{
  "si_float": 0.185,            // short interest as % of free float (0..1)
  "dtc_days": 3.8,              // days-to-cover (short interest / ADV)
  "utilization": 0.93,          // securities lending utilization (0..1)
  "ctb_annual": 0.34,           // cost-to-borrow (annualized, 0..1+)
  "si_trend_7d": 0.025,         // 7d change in si_float (absolute, e.g., +2.5pp → 0.025)
  "si_trend_30d": 0.041,        // 30d change in si_float
  "shares_on_loan_chg_7d": 0.06,// 7d % change in shares on loan
  "ftd_ratio": 0.012,           // fails-to-deliver as % of float (optional)
  "updated_ms": 1765400000000
}'

# Tape/buzz risk (optional guards; publish intraday)
HSET buzz:agg "GME" '{
  "price_mom_3d": 0.22,         // 3d close/close return
  "price_mom_1d": 0.12,         // 1d return
  "vol_rel_5d": 3.4,            // turnover vs 5d median
  "social_burst_z": 2.8,        // posts/comments burst z-score vs 7d same-hour baseline
  "iv_30_rel": 1.7,             // 30d IV vs 6m median
  "updated_ms": 1765400000000
}'

# Last prices (for sizing)
HSET last_price "EQ:GME" '{"price":38.5}'
HSET last_price "EQ:AAPL" '{"price":230.2}'

# Sector map (for caps/neutrality if you want)
HSET ref:sector "GME" "CONS_DISC"
HSET ref:sector "AAPL" "TECH"

# Ops
HSET fees:eq EXCH 10
SET  risk:halt 0|1

Routing (paper; adapters later):
  order("EQ:<SYM>", side, qty, order_type="market", venue="EXCH")
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("SIA_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("SIA_REDIS_PORT", "6379"))

HALT_KEY   = os.getenv("SIA_HALT_KEY",   "risk:halt")
UNIV_KEY   = os.getenv("SIA_UNIV_KEY",   "universe:eq")
SHORT_HK   = os.getenv("SIA_SHORT_HK",   "short:agg")
BUZZ_HK    = os.getenv("SIA_BUZZ_HK",    "buzz:agg")
LAST_HK    = os.getenv("SIA_LAST_HK",    "last_price")
SECTOR_HK  = os.getenv("SIA_SECTOR_HK",  "ref:sector")
FEES_HK    = os.getenv("SIA_FEES_HK",    "fees:eq")

# Cadence
RECHECK_SECS   = float(os.getenv("SIA_RECHECK_SECS", "2.0"))

# Entry/exit gates (cross‑sectional z on composite score)
ENTRY_Z        = float(os.getenv("SIA_ENTRY_Z", "0.9"))
EXIT_Z         = float(os.getenv("SIA_EXIT_Z",  "0.3"))

# Risk / sizing
USD_PER_NAME   = float(os.getenv("SIA_USD_PER_NAME", "4000"))
MIN_TICKET_USD = float(os.getenv("SIA_MIN_TICKET_USD", "150"))
MAX_NAMES      = int(os.getenv("SIA_MAX_NAMES", "24"))
MAX_PER_SECTOR = int(os.getenv("SIA_MAX_PER_SECTOR", "6"))
LOT            = float(os.getenv("SIA_LOT", "1"))

# Feature weights (pre‑z composite, signed: + → short bias, − → long bias)
W_SI_LEVEL     = float(os.getenv("SIA_W_SI_LEVEL", "0.35"))  # si_float
W_SI_TREND     = float(os.getenv("SIA_W_SI_TREND", "0.25"))  # 7d/30d trend blend
W_DTC          = float(os.getenv("SIA_W_DTC",      "0.15"))  # days-to-cover
W_UTIL         = float(os.getenv("SIA_W_UTIL",     "0.10"))
W_CTB          = float(os.getenv("SIA_W_CTB",      "0.05"))
W_FTD          = float(os.getenv("SIA_W_FTD",      "0.05"))
W_SOL_CHG      = float(os.getenv("SIA_W_SOL",      "0.05"))  # shares-on-loan change

# Squeeze‑risk penalties (reduce magnitude toward 0)
P_SQ_PRICE_3D  = float(os.getenv("SIA_P_SQ_P3D",   "0.35"))  # * clamp(price_mom_3d,0..1)
P_SQ_VOL       = float(os.getenv("SIA_P_SQ_VOL",   "0.25"))  # * vol_conf
P_SQ_SOC       = float(os.getenv("SIA_P_SQ_SOC",   "0.20"))  # * burst factor
P_SQ_IV        = float(os.getenv("SIA_P_SQ_IV",    "0.15"))  # * iv_rel factor

# Hard squeeze gates (block new shorts entirely if any trip)
UTIL_HARD      = float(os.getenv("SIA_UTIL_HARD",  "0.975")) # utilization
CTB_HARD       = float(os.getenv("SIA_CTB_HARD",   "0.50"))  # 50% annualized CTB
BURST_Z_HARD   = float(os.getenv("SIA_BURST_HARD", "3.0"))   # social burst Z
VOLREL_HARD    = float(os.getenv("SIA_VOLREL_HARD","4.0"))   # turnover multiple

# Staleness
STALE_HR_SHORT = float(os.getenv("SIA_STALE_HR_SHORT", "48"))
STALE_HR_BUZZ  = float(os.getenv("SIA_STALE_HR_BUZZ",  "2"))

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
def _hours_since(ms: int) -> float: return max(0.0, (_now_ms() - ms)/3_600_000.0)

def _clamp01(x: float) -> float: return max(0.0, min(1.0, x))

# ============================ state ============================
@dataclass
class OpenState:
    side: str     # "long" | "short"
    qty: float
    z_at_entry: float
    ts_ms: int
    sector: str

def _poskey(name: str, sym: str) -> str:
    return f"sia:open:{name}:{sym}"

# ============================ Strategy ============================
class ShortInterestAlpha(Strategy):
    """
    Cross‑sectional alpha from short‑interest level/trend and borrow dynamics with squeeze‑risk guards (paper).
    """
    def __init__(self, name: str = "short_interest_alpha", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
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
        long_block: Dict[str, bool] = {}   # block new shorts (i.e., forced long/flat only) due to squeeze risk

        for s in syms:
            sh = _hget_json(SHORT_HK, s)
            if not sh: 
                continue
            if _hours_since(int(sh.get("updated_ms", _now_ms()))) > STALE_HR_SHORT:
                continue

            si = float(sh.get("si_float", 0.0))            # 0..1
            dtc = float(sh.get("dtc_days", 0.0))           # days
            util = float(sh.get("utilization", 0.0))       # 0..1
            ctb = float(sh.get("ctb_annual", 0.0))         # 0..1+ (annualized)
            t7 = float(sh.get("si_trend_7d", 0.0))
            t30 = float(sh.get("si_trend_30d", 0.0))
            sol7 = float(sh.get("shares_on_loan_chg_7d", 0.0))
            ftd = float(sh.get("ftd_ratio", 0.0))

            # Optional buzz/squeeze metrics
            bz = _hget_json(BUZZ_HK, s) or {}
            if bz and _hours_since(int(bz.get("updated_ms", _now_ms()))) > STALE_HR_BUZZ:
                bz = {}
            p3d = float(bz.get("price_mom_3d", 0.0))
            p1d = float(bz.get("price_mom_1d", 0.0))
            volrel = float(bz.get("vol_rel_5d", 1.0))
            sburst = float(bz.get("social_burst_z", 0.0))
            ivrel = float(bz.get("iv_30_rel", 1.0))

            # Composite core (positive => short bias; negative => long bias)
            # Trend blend: emphasize more recent but include 30d
            tblend = 0.6 * t7 + 0.4 * t30
            # Normalize some features to ~0..1
            dtc_n = dtc / (10.0 + abs(dtc))          # saturates ~1 near 10+ days
            ctb_n = ctb / (0.75 + abs(ctb))          # 75%+ CTB ~ strong
            util_n= util                               # already 0..1
            ftd_n = min(1.0, ftd * 10.0)             # 10% float FTD ≈ 1.0 (extreme)

            core = (W_SI_LEVEL * si) \
                 + (W_SI_TREND * tblend) \
                 + (W_DTC      * dtc_n) \
                 + (W_UTIL     * util_n) \
                 + (W_CTB      * ctb_n) \
                 + (W_FTD      * ftd_n) \
                 + (W_SOL_CHG  * sol7)

            # Squeeze‑risk penalties (shrink magnitude toward 0 when squeeze conditions present)
            # Volume confirmation: 0 at 1x, →1 as vol pops
            vol_conf = 1.0 - math.exp(-0.6 * max(0.0, volrel - 1.0))
            pen = 0.0
            pen += P_SQ_PRICE_3D * _clamp01(p3d)                    # strong recent pop → reduce short conviction
            pen += P_SQ_VOL      * vol_conf
            pen += P_SQ_SOC      * _clamp01(max(0.0, (sburst/3.0))) # scale bursts Z≈3 to 1.0
            pen += P_SQ_IV       * _clamp01(max(0.0, (ivrel - 1.0)/1.0)) # IV 2.0x → ~1.0

            signed = core - math.copysign(pen, core)  # reduce magnitude

            # Hard block for initiating NEW shorts (keep for exits): extreme borrow pressure & hype
            hard_block_short = (util >= UTIL_HARD) or (ctb >= CTB_HARD) or (sburst >= BURST_Z_HARD) or (volrel >= VOLREL_HARD)
            long_block[s] = bool(hard_block_short)

            raw_scores[s] = float(signed)
            sectors[s] = (r.hget(SECTOR_HK, s) or "UNKNOWN").upper()

        if not raw_scores:
            self.emit_signal(0.0); return

        # Cross‑sectional z
        vals = list(raw_scores.values())
        mu = sum(vals)/len(vals)
        var = sum((x-mu)*(x-mu) for x in vals) / max(1.0, len(vals)-1.0)
        sd = math.sqrt(max(1e-12, var))
        zmap = {s: (raw_scores[s] - mu)/sd for s in raw_scores}

        # Manage existing positions (exits)
        open_names = []
        sector_loads: Dict[str, int] = {}

        for s in list(zmap.keys()):
            st = self._load_state(s)
            if not st: 
                continue
            open_names.append(s)
            sector_loads[st.sector] = sector_loads.get(st.sector, 0) + 1
            z = zmap.get(s, 0.0)
            # Exit when mean‑reverted OR if a squeeze hard‑block appears while we’re short
            if (abs(z) <= EXIT_Z) or (st.side == "short" and long_block.get(s, False)):
                self._close(s, st)

        # Entries (respect caps & hard blocks)
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

            # Sign convention: positive z ⇒ short bias; negative z ⇒ long bias
            want_short = z > 0
            if want_short and long_block.get(s, False):
                # Hard squeeze risk — skip opening new short
                continue

            side = "sell" if want_short else "buy"
            self.order(f"EQ:{s}", side, qty=qty, order_type="market", venue="EXCH")

            self._save_state(s, OpenState(
                side=("short" if want_short else "long"),
                qty=qty, z_at_entry=z, ts_ms=_now_ms(), sector=sec
            ))
            sector_loads[sec] = sector_loads.get(sec, 0) + 1
            n_open += 1

        # Dashboard signal: average |z|
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