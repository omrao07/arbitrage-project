# backend/strategies/diversified/earnings_surprise_momentum.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import redis
from backend.engine.strategy_base import Strategy

"""
Earnings Surprise Momentum — paper
----------------------------------
Idea: Post-earnings drift after large positive/negative surprises, strengthened by guidance,
opening gap, and abnormal volume.

You publish these to Redis (examples):

  # Universe
  SADD universe:eq AAPL MSFT NVDA AMZN META TSLA ...

  # Latest earnings event per symbol (refresh on each announcement)
  # All % are decimals (0.10 = +10%). Times are ms epoch.
  HSET earn:last "AAPL" '{"ann_ms": 1765584000000, "when":"AMC",           // BMO/AMC/INTRADAY
                          "sue": 2.1,             // standardized unexpected earnings (z)
                          "eps_surprise_pct": 0.08,
                          "rev_surprise_pct": 0.03,
                          "guide_delta_pct": 0.05, // midpoint guide vs street
                          "gap_open_pct": 0.04,    // open vs prev close
                          "day0_ret_pct": 0.06,    // close-to-close day 0
                          "vol_rel": 3.2,          // day0 volume / 20d avg
                          "currency":"USD"}'

  # Last prices
  HSET last_price "EQ:AAPL" '{"price": 230.15}'

  # Sector (for caps/neutralization)
  HSET ref:sector "AAPL" "TECH"

  # Ops / fees
  HSET fees:eq EXCH 10           # bps on notional (paper guard)
  SET  risk:halt 0|1

Routing (paper; adapters wire later):
  order("EQ:<SYM>", side, qty, order_type="market", venue="EXCH")
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("ESM_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("ESM_REDIS_PORT", "6379"))

# Entry windows
ENTRY_DAYS_MAX   = float(os.getenv("ESM_ENTRY_DAYS_MAX", "5"))   # enter within N trading days after ann
ENTRY_DAYS_MIN   = float(os.getenv("ESM_ENTRY_DAYS_MIN", "0.2")) # small delay to avoid opening print

# Hold/exit
MAX_HOLD_DAYS    = float(os.getenv("ESM_MAX_HOLD_DAYS", "15"))
EXIT_Z            = float(os.getenv("ESM_EXIT_Z", "0.25"))       # cross-sec reversion gate
RECHECK_SECS     = float(os.getenv("ESM_RECHECK_SECS", "1.5"))

# Cross-sectional gates
ENTRY_Z          = float(os.getenv("ESM_ENTRY_Z", "0.9"))        # |z| threshold for entries
MIN_VOL_REL      = float(os.getenv("ESM_MIN_VOL_REL", "1.5"))    # confirmation: abnormal volume
MIN_GAP_ABS      = float(os.getenv("ESM_MIN_GAP_ABS", "0.01"))   # 1% gap confirmation (signed)

# Sizing / risk
USD_PER_NAME     = float(os.getenv("ESM_USD_PER_NAME", "6000"))
MIN_TICKET_USD   = float(os.getenv("ESM_MIN_TICKET_USD", "200"))
MAX_NAMES        = int(os.getenv("ESM_MAX_NAMES", "20"))
MAX_PER_SECTOR   = int(os.getenv("ESM_MAX_PER_SECTOR", "6"))
LOT              = float(os.getenv("ESM_LOT", "1"))

# Weights for composite surprise score
W_SUE   = float(os.getenv("ESM_W_SUE", "0.5"))
W_EPS   = float(os.getenv("ESM_W_EPS", "0.25"))
W_REV   = float(os.getenv("ESM_W_REV", "0.10"))
W_GUIDE = float(os.getenv("ESM_W_GUIDE", "0.10"))
W_GAP   = float(os.getenv("ESM_W_GAP", "0.05"))

# Redis keys
HALT_KEY   = os.getenv("ESM_HALT_KEY",   "risk:halt")
UNIV_KEY   = os.getenv("ESM_UNIV_KEY",   "universe:eq")
EVENT_HK   = os.getenv("ESM_EVENT_HK",   "earn:last")
LAST_HK    = os.getenv("ESM_LAST_HK",    "last_price")
SECTOR_HK  = os.getenv("ESM_SECTOR_HK",  "ref:sector")
FEES_HK    = os.getenv("ESM_FEES_HK",    "fees:eq")

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================ helpers ============================
def _hget_json(hk: str, field: str) -> Optional[dict]:
    raw = r.hget(hk, field)
    if not raw: return None
    try:
        j = json.loads(raw); return j if isinstance(j, dict) else None # type: ignore
    except Exception:
        return None

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
def _days_since(ms: int) -> float: return max(0.0, (_now_ms() - ms)/86400000.0)

# ============================ state ============================
@dataclass
class OpenState:
    side: str     # "long" | "short"
    qty: float
    z_at_entry: float
    ann_ms: int
    sector: str
    ts_ms: int

def _poskey(name: str, sym: str) -> str:
    return f"esm:open:{name}:{sym}"

# ============================ Strategy ============================
class EarningsSurpriseMomentum(Strategy):
    """
    Post‑earnings drift strategy with composite surprise + confirmation and clean gates (paper).
    """
    def __init__(self, name: str = "earnings_surprise_momentum", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
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

        # ----- build composite surprise scores for names within entry window -----
        raw: Dict[str, float] = {}
        sectors: Dict[str, str] = {}
        recency: Dict[str, float] = {}
        conf_pass: Dict[str, bool] = {}

        for s in syms:
            e = _hget_json(EVENT_HK, s)
            if not e or "ann_ms" not in e: continue
            days = _days_since(int(e["ann_ms"]))
            if not (ENTRY_DAYS_MIN <= days <= ENTRY_DAYS_MAX): 
                continue  # only trade shortly after announcement

            sue   = float(e.get("sue", 0.0))
            eps   = float(e.get("eps_surprise_pct", 0.0))
            rev   = float(e.get("rev_surprise_pct", 0.0))
            guide = float(e.get("guide_delta_pct", 0.0))
            gap   = float(e.get("gap_open_pct", 0.0))
            d0ret = float(e.get("day0_ret_pct", 0.0))
            volr  = float(e.get("vol_rel", 1.0))

            # Composite surprise/confirmation score (signed)
            score = (W_SUE*sue) + (W_EPS*eps) + (W_REV*rev) + (W_GUIDE*guide) + (W_GAP*((gap + d0ret)/2.0))
            # Confirmation: abnormal volume and a gap in the same sign as score
            confirm = (volr >= MIN_VOL_REL) and (abs(gap) >= MIN_GAP_ABS) and ((score >= 0 and gap >= 0) or (score < 0 and gap <= 0))

            raw[s] = score
            sectors[s] = (r.hget(SECTOR_HK, s) or "UNKNOWN").upper() # type: ignore
            recency[s] = days
            conf_pass[s] = confirm

        if not raw:
            self.emit_signal(0.0)
            return

        # ----- cross‑sectional z across recent reporters -----
        vals = list(raw.values())
        mu = sum(vals)/len(vals)
        var = sum((x-mu)*(x-mu) for x in vals)/max(1.0, len(vals)-1.0)
        sd = math.sqrt(max(1e-9, var))
        zmap = {s: (raw[s]-mu)/sd for s in raw}

        # ----- manage existing (exits) -----
        open_names = []
        sector_loads: Dict[str,int] = {}
        for s in syms:
            st = self._load_state(s)
            if not st: continue
            open_names.append(s)
            sector_loads[st.sector] = sector_loads.get(st.sector, 0) + 1
            z = zmap.get(s, 0.0)
            held_days = _days_since(st.ts_ms)
            since_event = _days_since(st.ann_ms)
            if (abs(z) <= EXIT_Z) or (held_days >= MAX_HOLD_DAYS) or (since_event > (ENTRY_DAYS_MAX + MAX_HOLD_DAYS)):
                self._close(s, st)

        # ----- entries (respect caps, confirmations) -----
        n_open = len(open_names)
        fee = _fees_bps("EXCH") * 1e-4

        # Prefer larger |z| and nearer announcements
        cands = sorted(zmap.items(), key=lambda kv: (abs(kv[1]), -1.0/max(1e-6, recency[kv[0]])), reverse=True)

        for s, z in cands:
            if n_open >= MAX_NAMES: break
            if s in open_names: continue
            if abs(z) < ENTRY_Z: continue
            if not conf_pass.get(s, False): continue

            sec = sectors[s]
            if sector_loads.get(sec, 0) >= MAX_PER_SECTOR: continue

            px = _px(s)
            if not px or px <= 0: continue
            qty = math.floor((USD_PER_NAME / px) / max(1.0, LOT)) * LOT
            if qty <= 0 or qty*px < MIN_TICKET_USD: continue

            side = "buy" if z > 0 else "sell"
            self.order(f"EQ:{s}", side, qty=qty, order_type="market", venue="EXCH")

            self._save_state(s, OpenState(
                side=("long" if z>0 else "short"),
                qty=qty,
                z_at_entry=z,
                ann_ms=int((_hget_json(EVENT_HK, s) or {}).get("ann_ms", _now_ms())),
                sector=sec,
                ts_ms=_now_ms()
            ))
            sector_loads[sec] = sector_loads.get(sec, 0) + 1
            n_open += 1

        # Dashboard: average |z| of current opportunity set
        avgabs = sum(abs(z) for z in zmap.values())/len(zmap)
        self.emit_signal(max(-1.0, min(1.0, avgabs/3.0)))

    # ---------------- helpers ----------------
    def _load_state(self, sym: str) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, sym))
        if not raw: return None
        try:
            o = json.loads(raw) # type: ignore
            return OpenState(side=str(o["side"]), qty=float(o["qty"]), z_at_entry=float(o["z_at_entry"]),
                             ann_ms=int(o.get("ann_ms", _now_ms())), sector=str(o.get("sector","UNKNOWN")),
                             ts_ms=int(o["ts_ms"]))
        except Exception:
            return None

    def _save_state(self, sym: str, st: OpenState) -> None:
        r.set(_poskey(self.ctx.name, sym), json.dumps({
            "side": st.side, "qty": st.qty, "z_at_entry": st.z_at_entry,
            "ann_ms": st.ann_ms, "sector": st.sector, "ts_ms": st.ts_ms
        }))

    def _close(self, sym: str, st: OpenState) -> None:
        if st.side == "long":
            self.order(f"EQ:{sym}", "sell", qty=st.qty, order_type="market", venue="EXCH")
        else:
            self.order(f"EQ:{sym}", "buy",  qty=st.qty, order_type="market", venue="EXCH")
        r.delete(_poskey(self.ctx.name, sym))