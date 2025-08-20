# backend/strategies/diversified/volatility_event_driven.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import redis
from backend.engine.strategy_base import Strategy

"""
Volatility Event‑Driven Overlay — paper
--------------------------------------
Two sleeves:

(A) Index‑Vol (macro events: CPI/FOMC/NFP): 
    • Pre‑event: compare implied move vs modeled move; go LONG vol if implied < modeled (underpriced),
      SHORT vol if implied >> modeled (overpriced). Trade via VIX proxy ETF (e.g., VIXY/UVXY) and a small SPY hedge.
    • Post‑event: flatten or fade the spike/crush.

(B) Single‑Name Earnings:
    • After earnings drop, compare realized move vs implied move. 
      If realization << implied (classic crush), mean‑revert small; if >> implied with strong tape, momentum follow.
    • No options required; this sleeve trades the underlying with conservative sizing.

You publish these to Redis (examples):

# ---- Macro event calendar (per region) ----
HSET events:macro "CPI_2025-08-14T14:00Z" '{
  "t_ms": 1765778400000,             // UTC ms for event time
  "kind": "CPI",                     // CPI | FOMC | NFP | ECB | BOE ...
  "region": "US",
  "implied_move_pct_spy": 1.10,      // 1‑day implied move on SPY (from options) in %
  "modeled_move_pct_spy": 1.60,      // your historical model’s expected move in %
  "iv_rank_30d": 0.42,               // 0..1
  "pre_compression_flag": 1,         // realized vol compressed into event (optional)
  "importance": 0.9,                 // 0..1
  "updated_ms": 1765770000000
}'

# ---- Earnings events (per symbol) ----
HSET events:earnings "AAPL" '{
  "t_ms": 1765700000000,             // earnings announcement UTC ms
  "window_ms": 14400000,             // +/- 4h window considered "post-event"
  "implied_move_pct": 5.8,           // from options (±)
  "hist_move_pct": 4.2,              // median/mean past few quarters
  "iv_rank_30d": 0.65,
  "posted": 1,                       // 1 once results out
  "surprise_score": 0.30,            // [-1,+1] negative→miss, positive→beat
  "realized_move_pct": 3.1,          // abs move since close→post window (set after)
  "updated_ms": 1765710000000
}'

# ---- Prices (for sizing/hedging) ----
HSET last_price "ETF:SPY"  '{"price":530.0}'
HSET last_price "ETF:VIXY" '{"price":15.8}'
HSET last_price "EQ:AAPL"  '{"price":230.2}'

# ---- Ops ----
HSET fees:etf EXCH 2
HSET fees:eq  EXCH 10
SET  risk:halt 0|1

Routing (paper; adapters later):
  order("ETF:<TICKER>" | "EQ:<SYM>", side, qty, order_type="market", venue="EXCH")
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("VED_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("VED_REDIS_PORT", "6379"))

HALT_KEY    = os.getenv("VED_HALT_KEY",    "risk:halt")
MACRO_HK    = os.getenv("VED_MACRO_HK",    "events:macro")
ERN_HK      = os.getenv("VED_EARN_HK",     "events:earnings")
LAST_HK     = os.getenv("VED_LAST_HK",     "last_price")
FEES_EQ_HK  = os.getenv("VED_FEES_EQ",     "fees:eq")
FEES_ETF_HK = os.getenv("VED_FEES_ETF",    "fees:etf")

RECHECK_SECS   = float(os.getenv("VED_RECHECK_SECS", "1.5"))
STALE_HR       = float(os.getenv("VED_STALE_HR",     "24"))

# --- Sleeve A: Macro / Index Vol ---
PRE_WINDOW_SEC  = float(os.getenv("VED_PRE_WINDOW_SEC",  "5400"))    # 90 min before event
POST_WINDOW_SEC = float(os.getenv("VED_POST_WINDOW_SEC", "5400"))    # 90 min after event
UNDERPRICE_Z    = float(os.getenv("VED_UNDERPRICE_Z",    "0.6"))     # act if (modeled - implied)/sigma ≥ this
OVERPRICE_Z     = float(os.getenv("VED_OVERPRICE_Z",     "0.8"))
MOVE_SIGMA_PCT  = float(os.getenv("VED_MOVE_SIGMA_PCT",  "0.8"))     # rough sigma (% move)

SPY_TICKER   = os.getenv("VED_SPY_TICKER",   "ETF:SPY")
VIX_TICKER   = os.getenv("VED_VIX_TICKER",   "ETF:VIXY")

USD_BUDGET_MACRO = float(os.getenv("VED_USD_BUDGET_MACRO", "12000"))
MIN_TICKET_USD   = float(os.getenv("VED_MIN_TICKET_USD",   "200"))
LOT              = float(os.getenv("VED_LOT", "1"))
COOLDOWN_SEC     = float(os.getenv("VED_COOLDOWN_SEC", "1800"))

# --- Sleeve B: Earnings (single‑name) ---
POST_HOLD_MAX_SEC  = float(os.getenv("VED_EARN_POST_HOLD", "7200"))  # up to 2h
ENTRY_THRESH_R     = float(os.getenv("VED_EARN_ENTRY_R",   "0.35"))  # |realized - implied| / implied
MOM_THRESH         = float(os.getenv("VED_EARN_MOM_T",     "0.20"))  # surprise & realized both large → momentum
MR_THRESH          = float(os.getenv("VED_EARN_MR_T",      "0.15"))  # crush with small realized → mean reversion

USD_PER_EARN       = float(os.getenv("VED_USD_PER_EARN",   "3000"))
MAX_EARN_POS       = int(os.getenv("VED_MAX_EARN_POS",     "8"))

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================ helpers ============================
def _now_ms() -> int: return int(time.time()*1000)
def _hours_since(ms: int) -> float: return max(0.0, (_now_ms() - ms)/3_600_000.0)
def _s_to_ms(x: float) -> int: return int(x*1000)

def _hgetall_json(hk: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    raw = r.hgetall(hk) or {}
    for k,v in raw.items():
        try:
            j = json.loads(v)
            if isinstance(j, dict): out[k] = j
        except Exception:
            continue
    return out

def _px(sym: str) -> Optional[float]:
    raw = r.hget(LAST_HK, sym)
    if not raw: return None
    try:
        j = json.loads(raw)
        return float(j.get("price", 0.0))
    except Exception:
        try: return float(raw)
        except Exception: return None

def _qty_for_usd(px: float, usd: float) -> float:
    if not px or px <= 0: return 0.0
    q = math.floor((usd / px) / max(1.0, LOT)) * LOT
    return max(0.0, q)

# ============================ state ============================
@dataclass
class MacroState:
    active: bool
    side: str         # "long_vol" | "short_vol" | "flat"
    event_key: str
    t_enter_ms: int
    cooldown_until_ms: int

@dataclass
class EarnState:
    sym: str
    side: str         # "mr_long" | "mr_short" | "mom_long" | "mom_short"
    t_enter_ms: int

def _mkey(ctx: str) -> str: return f"ved:macro:{ctx}"
def _ekey(ctx: str, sym: str) -> str: return f"ved:earn:{ctx}:{sym}"

def _load_macro(ctx: str) -> Optional[MacroState]:
    raw = r.get(_mkey(ctx))
    if not raw: return None
    try:
        o = json.loads(raw)
        return MacroState(active=bool(o["active"]), side=str(o["side"]), event_key=str(o["event_key"]),
                          t_enter_ms=int(o["t_enter_ms"]), cooldown_until_ms=int(o.get("cooldown_until_ms", 0)))
    except Exception:
        return None

def _save_macro(ctx: str, st: MacroState) -> None:
    r.set(_mkey(ctx), json.dumps({
        "active": st.active, "side": st.side, "event_key": st.event_key,
        "t_enter_ms": st.t_enter_ms, "cooldown_until_ms": st.cooldown_until_ms
    }))

def _load_earn(ctx: str, sym: str) -> Optional[EarnState]:
    raw = r.get(_ekey(ctx, sym))
    if not raw: return None
    try:
        o = json.loads(raw)
        return EarnState(sym=sym, side=str(o["side"]), t_enter_ms=int(o["t_enter_ms"]))
    except Exception:
        return None

def _save_earn(ctx: str, st: EarnState) -> None:
    r.set(_ekey(ctx, st.sym), json.dumps({"sym": st.sym, "side": st.side, "t_enter_ms": st.t_enter_ms}))

def _del_earn(ctx: str, sym: str) -> None:
    r.delete(_ekey(ctx, sym))

# ============================ Strategy ============================
class VolatilityEventDriven(Strategy):
    """
    Index‑vol around macro events (VIX proxy + SPY hedge) and single‑name earnings reactions (paper).
    """
    def __init__(self, name: str = "volatility_event_driven", region: Optional[str] = "US", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_tick(self, tick: Dict) -> None:
        if (r.get(HALT_KEY) or "0") == "1": return
        now = time.time()
        if now - self._last < RECHECK_SECS: return
        self._last = now

        # Sleeve A — Macro index‑vol
        self._macro_sleeve()

        # Sleeve B — Single‑name earnings
        self._earnings_sleeve()

        # UI heartbeat: simple activity indicator
        mac = _load_macro(self.ctx.name)
        ui = 0.0
        if mac and mac.active:
            ui = 0.6 if mac.side == "long_vol" else -0.6
        self.emit_signal(max(-1.0, min(1.0, ui)))

    # --------- Sleeve A: Macro / Index Vol ----------
    def _macro_sleeve(self) -> None:
        mac = _load_macro(self.ctx.name)
        events = _hgetall_json(MACRO_HK)

        # pick the next/ongoing US event by time and importance
        next_key, next_ev = None, None
        tnow = _now_ms()
        best_score = -1.0
        for k,ev in events.items():
            if ev.get("region","US") != "US": 
                continue
            if _hours_since(int(ev.get("updated_ms", tnow))) > STALE_HR: 
                continue
            t_ev = int(ev.get("t_ms", tnow))
            # only consider within pre/post windows
            if abs(t_ev - tnow) > _s_to_ms(PRE_WINDOW_SEC + POST_WINDOW_SEC):
                continue
            score = float(ev.get("importance", 0.5))
            if score > best_score:
                best_score, next_key, next_ev = score, k, ev

        if not next_ev:
            # no actionable event, flatten if active and cooldown elapsed
            if mac and mac.active and tnow >= mac.cooldown_until_ms:
                self._macro_flatten(mac)
                _save_macro(self.ctx.name, MacroState(False, "flat", "", 0, tnow + _s_to_ms(COOLDOWN_SEC)))
            return

        t_ev = int(next_ev["t_ms"])
        pre_w = _s_to_ms(PRE_WINDOW_SEC)
        post_w = _s_to_ms(POST_WINDOW_SEC)
        in_pre  = (tnow >= t_ev - pre_w) and (tnow < t_ev)
        in_post = (tnow >= t_ev) and (tnow <= t_ev + post_w)

        implied = float(next_ev.get("implied_move_pct_spy", 0.0))
        modeled = float(next_ev.get("modeled_move_pct_spy", implied))
        ivr = float(next_ev.get("iv_rank_30d", 0.5))
        cmp_flag = int(next_ev.get("pre_compression_flag", 0))

        # normalized mispricing signal ~ z
        sigma = max(0.1, MOVE_SIGMA_PCT)
        z_mis = (modeled - implied) / sigma   # + → underpriced (go long vol), - → overpriced (short vol)

        if not mac:
            mac = MacroState(False, "flat", "", 0, 0)

        # PRE‑EVENT ENTRIES
        if in_pre and (tnow >= mac.cooldown_until_ms):
            want = None
            if z_mis >= UNDERPRICE_Z and (ivr <= 0.65 or cmp_flag == 1):
                want = "long_vol"
            elif z_mis <= -OVERPRICE_Z and (ivr >= 0.35):  # high IV rank → more believable overpricing
                want = "short_vol"

            if want and (not mac.active or mac.side != want or mac.event_key != next_key):
                self._macro_enter(want)
                _save_macro(self.ctx.name, MacroState(True, want, next_key, tnow, tnow + _s_to_ms(COOLDOWN_SEC)))
                return

        # POST‑EVENT EXITS (and opportunistic reversals)
        if in_post and mac and mac.active:
            # simple rule: realize the trade shortly after event
            self._macro_flatten(mac)
            _save_macro(self.ctx.name, MacroState(False, "flat", next_key, tnow, tnow + _s_to_ms(COOLDOWN_SEC)))
            return

        # Outside window — ensure flat
        if not in_pre and not in_post and mac and mac.active and tnow >= mac.cooldown_until_ms:
            self._macro_flatten(mac)
            _save_macro(self.ctx.name, MacroState(False, "flat", next_key or "", tnow, tnow + _s_to_ms(COOLDOWN_SEC)))

    def _macro_enter(self, side: str) -> None:
        # side: "long_vol" buys VIX proxy; "short_vol" sells VIX proxy; tiny SPY hedge to reduce beta bleed
        px_vix = _px(VIX_TICKER); px_spy = _px(SPY_TICKER)
        if not px_vix or not px_spy: return
        vix_usd = USD_BUDGET_MACRO
        spy_usd = USD_BUDGET_MACRO * 0.25  # small hedge
        q_vix = _qty_for_usd(px_vix, vix_usd)
        q_spy = _qty_for_usd(px_spy, spy_usd)
        if q_vix * px_vix < MIN_TICKET_USD: return

        if side == "long_vol":
            self.order(VIX_TICKER, "buy", qty=q_vix, order_type="market", venue="EXCH")
            if q_spy > 0: self.order(SPY_TICKER, "sell", qty=q_spy, order_type="market", venue="EXCH")
        else:
            self.order(VIX_TICKER, "sell", qty=q_vix, order_type="market", venue="EXCH")
            if q_spy > 0: self.order(SPY_TICKER, "buy", qty=q_spy, order_type="market", venue="EXCH")

    def _macro_flatten(self, mac: MacroState) -> None:
        # Send opposite direction once for paper; real OMS would track position and delta‑trade to flat
        px_vix = _px(VIX_TICKER); px_spy = _px(SPY_TICKER)
        if not px_vix or not px_spy: return
        q_vix = _qty_for_usd(px_vix, USD_BUDGET_MACRO)
        q_spy = _qty_for_usd(px_spy, USD_BUDGET_MACRO*0.25)
        if q_vix <= 0: return

        if mac.side == "long_vol":
            self.order(VIX_TICKER, "sell", qty=q_vix, order_type="market", venue="EXCH")
            if q_spy > 0: self.order(SPY_TICKER, "buy", qty=q_spy, order_type="market", venue="EXCH")
        elif mac.side == "short_vol":
            self.order(VIX_TICKER, "buy", qty=q_vix, order_type="market", venue="EXCH")
            if q_spy > 0: self.order(SPY_TICKER, "sell", qty=q_spy, order_type="market", venue="EXCH")

    # --------- Sleeve B: Earnings ----------
    def _earnings_sleeve(self) -> None:
        # Iterate earnings blobs and act only for ones "posted" and within post window, one shot per symbol
        blobs = _hgetall_json(ERN_HK)
        if not blobs: return
        tnow = _now_ms()

        open_count = 0
        # count existing earn positions
        for k in blobs.keys():
            st = _load_earn(self.ctx.name, k)
            if st: open_count += 1

        for sym, ev in blobs.items():
            if _hours_since(int(ev.get("updated_ms", tnow))) > STALE_HR:
                continue
            if int(ev.get("posted", 0)) != 1:
                continue

            t_ev   = int(ev.get("t_ms", tnow))
            win_ms = int(ev.get("window_ms", 3_600_000))  # default 1h
            in_post = (tnow >= t_ev) and (tnow <= t_ev + win_ms)
            if not in_post: 
                # flatten if we still hold after window
                st = _load_earn(self.ctx.name, sym)
                if st and (tnow - st.t_enter_ms >= POST_HOLD_MAX_SEC*1000):
                    self._earn_flatten(sym, st)
                continue

            st = _load_earn(self.ctx.name, sym)
            if st:
                # manage hold duration
                if (tnow - st.t_enter_ms) >= POST_HOLD_MAX_SEC*1000:
                    self._earn_flatten(sym, st)
                continue

            if open_count >= MAX_EARN_POS:
                break

            # Compute signal
            implied = float(ev.get("implied_move_pct", 0.0)) / 100.0
            hist_mv = float(ev.get("hist_move_pct", implied*100)) / 100.0
            realized = float(ev.get("realized_move_pct", 0.0)) / 100.0
            surpr = float(ev.get("surprise_score", 0.0))  # [-1,+1]

            if implied <= 0: 
                continue

            miss_ratio = abs(realized - implied) / implied        # how different reality is from pricing
            mom_ok = (realized >= implied*(1.0 + MOM_THRESH)) and (abs(surpr) >= 0.25)  # strong beat/miss & big move
            mr_ok  = (implied >= hist_mv*(1.0 + MR_THRESH)) and (realized <= implied*(1.0 - MR_THRESH))

            # size
            px = _px(f"EQ:{sym}")
            if not px or px <= 0: 
                continue
            qty = _qty_for_usd(px, USD_PER_EARN)
            if qty * px < MIN_TICKET_USD: 
                continue

            # choose side
            if mom_ok:
                side = "mom_long" if surpr > 0 and realized > 0 else "mom_short"
                self.order(f"EQ:{sym}", ("buy" if side=="mom_long" else "sell"), qty=qty, order_type="market", venue="EXCH")
                _save_earn(self.ctx.name, EarnState(sym, side, _now_ms()))
                open_count += 1
            elif (miss_ratio >= ENTRY_THRESH_R) and mr_ok:
                # mean‑revert after overpriced implied‑move crush
                side = "mr_long" if realized < 0 else "mr_short"
                self.order(f"EQ:{sym}", ("buy" if side=="mr_long" else "sell"), qty=qty, order_type="market", venue="EXCH")
                _save_earn(self.ctx.name, EarnState(sym, side, _now_ms()))
                open_count += 1
            else:
                continue

    def _earn_flatten(self, sym: str, st: EarnState) -> None:
        # Send opposite side once (paper)
        if st.side in ("mom_long", "mr_long"):
            self.order(f"EQ:{sym}", "sell", qty=_qty_for_usd(_px(f"EQ:{sym}") or 0.0, USD_PER_EARN),
                       order_type="market", venue="EXCH")
        else:
            self.order(f"EQ:{sym}", "buy",  qty=_qty_for_usd(_px(f"EQ:{sym}") or 0.0, USD_PER_EARN),
                       order_type="market", venue="EXCH")
        _del_earn(self.ctx.name, sym)