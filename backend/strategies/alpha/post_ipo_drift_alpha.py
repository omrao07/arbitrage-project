# backend/strategies/diversified/post_ipo_drift_alpha.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional, List

import redis
from backend.engine.strategy_base import Strategy

"""
Post‑IPO Drift Alpha — paper
----------------------------
Idea (stylized):
  • After the offering, some IPOs trend (up or down) for several weeks.
  • Drivers we use: 1st‑day return (signal of demand/underpricing), volume confirmation,
    float tightness (small float, high oversubscribe → scarcity), underwriter score,
    stabilization/quiet‑period timing, index‑inclusion, and relative valuation sanity.
  • We build a signed composite → cross‑sectional z → gated entries/exits.

You publish these (examples) to Redis from your IPO ingestor:

  # Universe (recent IPOs only)
  SADD universe:ipo ARM KVYO CART BIRK CAVA ...

  # Reference datapoint — mostly static or slow‑moving
  HSET ipo:ref "ARM" '{
    "list_ms": 1694736000000,          // listing date (ms)
    "offer_px": 51.0,
    "proceeds_usd": 4900000000,
    "float_pct": 0.095,                // free float / outstanding
    "underwriter_score": 0.78,         // 0..1 reputation/completion rank
    "quiet_expiry_ms": 1697433600000,  // ~25 days after
    "lockup_expiry_ms": 1706054400000, // ~180 days
    "sector": "TECH",
    "rel_val_z": 0.4                   // EV/Sales vs peers (z): positive = rich, negative = cheap
  }'

  # After‑market rolling datapoint — update daily or intraday
  HSET ipo:after "ARM" '{
    "day1_ret": 0.245,                 // close1 vs offer
    "day5_ret": 0.312,                 // close5 vs offer (or vs day1)
    "vol_rel_10d": 2.6,                // notional vs 10d median
    "gap_open_pct": 0.10,              // day1 open vs offer
    "stabilization_done": 1,           // market‑stabilization (greenshoe) likely done
    "index_incl": 0,                   // fast entry to index (0/1)
    "updated_ms": 1695340800000
  }'

  # Last prices and sectors (for sizing & caps)
  HSET last_price "EQ:ARM" '{"price": 62.4}'
  HSET ref:sector "ARM" "TECH"

Ops:
  HSET fees:eq EXCH 10
  SET  risk:halt 0|1

Routing (paper; adapters later):
  order("EQ:<SYM>", side, qty, order_type="market", venue="EXCH")
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("PID_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("PID_REDIS_PORT", "6379"))

# Windows & gates
MIN_LIST_DAYS     = float(os.getenv("PID_MIN_LIST_DAYS", "1.0"))    # wait ≥ day 1 close
MAX_LIST_DAYS     = float(os.getenv("PID_MAX_LIST_DAYS", "60.0"))   # only trade first ~2 months
ENTRY_Z           = float(os.getenv("PID_ENTRY_Z", "0.9"))
EXIT_Z            = float(os.getenv("PID_EXIT_Z",  "0.3"))
RECHECK_SECS      = float(os.getenv("PID_RECHECK_SECS", "1.2"))

# Sizing / risk
USD_PER_NAME      = float(os.getenv("PID_USD_PER_NAME", "5000"))
MIN_TICKET_USD    = float(os.getenv("PID_MIN_TICKET_USD", "200"))
MAX_NAMES         = int(os.getenv("PID_MAX_NAMES", "20"))
MAX_PER_SECTOR    = int(os.getenv("PID_MAX_PER_SECTOR", "6"))
LOT               = float(os.getenv("PID_LOT", "1"))

# Feature weights (pre‑z composite)
W_DAY1            = float(os.getenv("PID_W_DAY1", "0.35"))
W_DAY5            = float(os.getenv("PID_W_DAY5", "0.20"))
W_VOLCONF         = float(os.getenv("PID_W_VOL",  "0.15"))
W_UNDERWRITER     = float(os.getenv("PID_W_UW",   "0.10"))
W_FLOAT_TIGHT     = float(os.getenv("PID_W_FLOAT","0.10"))
W_INDEX           = float(os.getenv("PID_W_INDEX","0.05"))
W_STAB_DONE       = float(os.getenv("PID_W_STAB", "0.03"))
W_VAL_SANE        = float(os.getenv("PID_W_VAL",  "0.02"))

# Penalties / timers
P_PRE_QUIET_DAYS  = float(os.getenv("PID_P_PREQUIET", "0.03"))   # per day remaining until quiet expiry
P_PRE_LOCKUP_DAYS = float(os.getenv("PID_P_PRELOCK",  "0.01"))   # per day remaining until lock‑up
P_STALE_HR        = float(os.getenv("PID_P_STALE_HR", "0.02"))   # per hour staleness on after‑market blob

# Redis keys
HALT_KEY    = os.getenv("PID_HALT_KEY",  "risk:halt")
UNIV_KEY    = os.getenv("PID_UNIV_KEY",  "universe:ipo")  # IPO subset
REF_HK      = os.getenv("PID_REF_HK",    "ipo:ref")
AFTER_HK    = os.getenv("PID_AFTER_HK",  "ipo:after")
LAST_HK     = os.getenv("PID_LAST_HK",   "last_price")
SECTOR_HK   = os.getenv("PID_SECTOR_HK", "ref:sector")

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

def _now_ms() -> int: return int(time.time()*1000)
def _days_since(ms: int) -> float: return max(0.0, (_now_ms() - ms)/86_400_000.0)
def _hours_since(ms: int) -> float: return max(0.0, (_now_ms() - ms)/3_600_000.0)

# ============================ state ============================
@dataclass
class OpenState:
    side: str
    qty: float
    z_at_entry: float
    ts_ms: int
    sector: str

def _poskey(name: str, sym: str) -> str:
    return f"pid:open:{name}:{sym}"

# ============================ Strategy ============================
class PostIPODriftAlpha(Strategy):
    """
    Trades recent IPOs using demand/scarcity/confirmation features with event‑timing penalties (paper).
    """
    def __init__(self, name: str = "post_ipo_drift_alpha", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
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

        for s in syms:
            ref = _hget_json(REF_HK, s)
            aft = _hget_json(AFTER_HK, s)
            if not ref or not aft: 
                continue

            list_days = _days_since(int(ref.get("list_ms", _now_ms())))
            if not (MIN_LIST_DAYS <= list_days <= MAX_LIST_DAYS):
                continue

            # Base features
            day1   = float(aft.get("day1_ret", 0.0))     # + stronger pop
            day5   = float(aft.get("day5_ret", 0.0))
            volr   = float(aft.get("vol_rel_10d", 1.0))
            uw     = float(ref.get("underwriter_score", 0.5))
            floatp = float(ref.get("float_pct", 0.15))
            idx    = int(aft.get("index_incl", 0))
            stab   = int(aft.get("stabilization_done", 0))
            relz   = float(ref.get("rel_val_z", 0.0))    # >0 rich, <0 cheap

            # Scarcity proxy: tighter float → higher positive weight
            tight = 1.0 - max(0.0, min(1.0, floatp*2.0))  # float 0..50% → tight 1..0

            # Volume confirmation in 0..1 (0 at 1x, →1 as vol pops)
            vol_conf = 1.0 - math.exp(-0.6 * max(0.0, volr - 1.0))

            # Composite (signed by realized trend so far: day5 vs day1)
            base_sign = 1.0 if (day1 + 0.5*day5) >= 0 else -1.0

            raw = (W_DAY1        * day1) \
                + (W_DAY5        * day5) \
                + (W_VOLCONF     * (vol_conf * base_sign)) \
                + (W_UNDERWRITER * (uw * base_sign)) \
                + (W_FLOAT_TIGHT * (tight * base_sign)) \
                + (W_INDEX       * (1.0 if idx == 1 else 0.0) * base_sign) \
                + (W_STAB_DONE   * (1.0 if stab == 1 else 0.0) * base_sign) \
                + (W_VAL_SANE    * (-max(0.0, relz) + max(0.0, -relz)))  # cheap gets mild +, rich mild -

            # Event‑timing penalties (pre‑expiry supply risk)
            quiet_ms  = int(ref.get("quiet_expiry_ms", _now_ms()))
            lock_ms   = int(ref.get("lockup_expiry_ms", _now_ms()))
            q_days    = max(0.0, -_days_since(quiet_ms))   # days until quiet expiry (if future)
            l_days    = max(0.0, -_days_since(lock_ms))    # days until lock‑up expiry

            raw -= math.copysign(P_PRE_QUIET_DAYS  * q_days, raw)
            raw -= math.copysign(P_PRE_LOCKUP_DAYS * l_days, raw)

            # Staleness penalty on after‑market blob
            st_hr = _hours_since(int(aft.get("updated_ms", _now_ms())))
            raw -= math.copysign(P_STALE_HR * min(48.0, st_hr), raw)

            raw_scores[s] = float(raw)
            sectors[s] = (r.hget(SECTOR_HK, s) or str(ref.get("sector","UNKNOWN"))).upper() # type: ignore

        if not raw_scores:
            self.emit_signal(0.0); return

        # Cross‑sectional z across recent IPOs
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

            # Hard exit if symbol aged out of window or mean‑reverted
            ref = _hget_json(REF_HK, s) or {}
            list_days = _days_since(int(ref.get("list_ms", _now_ms())))
            if (abs(z) <= EXIT_Z) or (list_days > MAX_LIST_DAYS + 5):
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
            if qty <= 0 or qty*px < MIN_TICKET_USD: continue

            side = "buy" if z > 0 else "sell"
            self.order(f"EQ:{s}", side, qty=qty, order_type="market", venue="EXCH")

            self._save_state(s, OpenState(
                side=("long" if z>0 else "short"),
                qty=qty, z_at_entry=z, ts_ms=_now_ms(), sector=sec
            ))
            sector_loads[sec] = sector_loads.get(sec, 0) + 1
            n_open += 1

        # Dashboard: average |z|
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