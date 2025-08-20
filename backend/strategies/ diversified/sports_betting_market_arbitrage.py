# backend/strategies/diversified/sports_betting_market_arbitrage.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import redis
from backend.engine.strategy_base import Strategy

"""
Sports Betting Market Arbitrage — paper
---------------------------------------
Modes:
  1) CROSS_BOOK (2-way markets): take A@Book1 and B@Book2 if 1/oA + 1/oB + frictions < 1
  2) DUTCH_BOOK (N-way): aggregate best odds per outcome; if sum(1/odds_best) + frictions < 1 → bet
  3) MIDDLE_SPREAD: capture middles on spreads/totals between two books (paper evaluation)

Odds format: **decimal** (e.g., 2.10) in Redis.

Redis feeds (publish via adapters):
  # Executable odds (decimal) & limits (in account currency, e.g., USD)
  HSET sb:odds:<EVENT_ID> "<BOOK>|<OUTCOME_ID>" '{"odds":2.10,"max_stake":500,"fee_bps":50,"fx_ccy":"USD","withhold_bps":0}'
  HSET sb:meta:<EVENT_ID> 'meta' '{"sport":"TENNIS","market":"ML","outcomes":["A","B"],"ccy":"USD","start_ms":<epoch_ms>}'
  # For middles on spreads/totals (per book line & juice)
  HSET sb:line:<EVENT_ID> "<BOOK>|<MARKET>" '{"line": -2.5, "odds":1.95, "max_stake":300}'   # e.g., MARKET="SPREAD:HOME"
  HSET sb:line:<EVENT_ID> "<BOOK>|<MARKET2>" '{"line": +3.5, "odds":1.95, "max_stake":300}'

  # Ops / risk
  HSET sb:fees 'guard_bps' 30             # extra bps safety for hidden frictions
  SET  risk:halt 0|1

Paper routing (adapters map to real APIs if ever wired):
  • Back bet:  "SB:<BOOK>:<EVENT_ID>:<OUTCOME_ID>"  side="back", qty=<stake>, price=<odds>
  • For lines: "SBLINE:<BOOK>:<EVENT_ID>:<MARKET>"  side="back", qty=<stake>, price=<odds>, flags={"line": <line>}
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("SB_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("SB_REDIS_PORT", "6379"))

MODE       = os.getenv("SB_MODE", "CROSS_BOOK").upper()     # CROSS_BOOK | DUTCH_BOOK | MIDDLE_SPREAD
EVENT_ID   = os.getenv("SB_EVENT_ID", "EVT-123").upper()
BOOK_A     = os.getenv("SB_BOOK_A", "BOOK1").upper()
BOOK_B     = os.getenv("SB_BOOK_B", "BOOK2").upper()

# Thresholds / cadence
ENTRY_BPS  = float(os.getenv("SB_ENTRY_BPS", "40"))   # minimum net edge (bps of total stakes) to enter
EXIT_BPS   = float(os.getenv("SB_EXIT_BPS",  "10"))   # for stateful patterns; here mostly one-shot
ENTRY_Z    = float(os.getenv("SB_ENTRY_Z",   "1.1"))
EXIT_Z     = float(os.getenv("SB_EXIT_Z",    "0.5"))
RECHECK_SECS = float(os.getenv("SB_RECHECK_SECS", "0.7"))
EWMA_ALPHA = float(os.getenv("SB_EWMA_ALPHA", "0.10"))

# Sizing / risk
USD_BANKROLL    = float(os.getenv("SB_USD_BANKROLL", "1000"))
USD_TICKET_CAP  = float(os.getenv("SB_USD_TICKET_CAP", "300"))
MIN_STAKE       = float(os.getenv("SB_MIN_STAKE", "5"))
MAX_CONCURRENT  = int(os.getenv("SB_MAX_CONCURRENT", "1"))

# Redis keys
HALT_KEY   = os.getenv("SB_HALT_KEY", "risk:halt")
ODDS_HK    = os.getenv("SB_ODDS_HK",  "sb:odds:{event}")
META_HK    = os.getenv("SB_META_HK",  "sb:meta:{event}")
LINE_HK    = os.getenv("SB_LINE_HK",  "sb:line:{event}")
FEES_HK    = os.getenv("SB_FEES_HK",  "sb:fees")

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================ helpers ============================
def _hget_json(hk: str, field: str) -> Optional[dict]:
    raw = r.hget(hk, field)
    if not raw: return None
    try:
        j = json.loads(raw)
        return j if isinstance(j, dict) else None
    except Exception:
        return None

def _meta(event: str) -> Optional[dict]:
    return _hget_json(META_HK.format(event=event), "meta")

def _guard_bps() -> float:
    v = r.hget(FEES_HK, "guard_bps")
    try: return float(v) if v is not None else 30.0
    except Exception: return 30.0

def _now_ms() -> int: return int(time.time() * 1000)

def _sum_inv_odds(odds: List[float]) -> float:
    return sum(1.0/max(1e-9, o) for o in odds)

def _net_inv(o: float, fee_bps: float, withhold_bps: float, fx_mult: float) -> float:
    """
    Convert odds to effective inverse odds after frictions:
      Effective payout ≈ o * (1 - fee) * (1 - withhold)
      We incorporate fx by scaling stakes externally; keeping inverse odds impact light.
    """
    eff = max(1.0, o) * (1.0 - fee_bps*1e-4) * (1.0 - withhold_bps*1e-4)
    return 1.0 / max(1e-9, eff)

def _stake_split(total: float, odds: List[float]) -> List[float]:
    """
    Proportional split to equalize **gross returns**.
    For decimal odds o_i, choose s_i = total * (1/o_i) / sum(1/o_j).
    Profit if outcome i wins: s_i*o_i - total = total*(1 - sum(1/o_j)).
    """
    inv_sum = _sum_inv_odds(odds)
    return [ total * (1.0/max(1e-9,o)) / max(1e-9, inv_sum) for o in odds ]

# ============================ EWMA ============================
@dataclass
class EwmaMV:
    mean: float
    var: float
    alpha: float
    def update(self, x: float) -> Tuple[float,float]:
        m0 = self.mean
        self.mean = (1 - self.alpha)*self.mean + self.alpha*x
        self.var  = max(1e-12, (1 - self.alpha)*(self.var + (x - m0)*(x - self.mean)))
        return self.mean, self.var

def _ewma_key(tag: str) -> str:
    return f"sb:ewma:{tag}"

def _load_ewma(tag: str) -> EwmaMV:
    raw = r.get(_ewma_key(tag))
    if raw:
        try:
            o = json.loads(raw)
            return EwmaMV(mean=float(o["m"]), var=float(o["v"]), alpha=float(o.get("a", EWMA_ALPHA)))
        except Exception: pass
    return EwmaMV(mean=0.0, var=1.0, alpha=EWMA_ALPHA)

def _save_ewma(tag: str, ew: EwmaMV) -> None:
    r.set(_ewma_key(tag), json.dumps({"m": ew.mean, "v": ew.var, "a": ew.alpha}))

# ============================ state ============================
@dataclass
class OpenState:
    mode: str
    tag: str
    legs: List[dict]   # [{"book":..., "outcome":..., "odds":..., "stake":..., "sym":...}]
    edge_bps: float
    entry_z: float
    ts_ms: int

def _poskey(name: str, tag: str) -> str:
    return f"sb:open:{name}:{tag}"

# ============================ Strategy ============================
class SportsBettingMarketArb(Strategy):
    """
    Cross-book & Dutch-book back-arbs, plus spread/total middles (paper).
    """
    def __init__(self, name: str = "sports_betting_market_arbitrage", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_tick(self, tick: Dict) -> None:
        if (r.get(HALT_KEY) or "0") == "1": return
        now = time.time()
        if now - self._last < RECHECK_SECS: return
        self._last = now

        meta = _meta(EVENT_ID) or {}
        if not meta: return

        if MODE == "CROSS_BOOK":
            self._eval_cross(meta)
        elif MODE == "DUTCH_BOOK":
            self._eval_dutch(meta)
        else:
            self._eval_middle(meta)

    # --------------- CROSS_BOOK (2-way) ---------------
    def _eval_cross(self, meta: dict) -> None:
        tag = f"CB:{EVENT_ID}"
        outcomes = list(meta.get("outcomes") or [])
        if len(outcomes) != 2: return
        a, b = outcomes[0], outcomes[1]

        oa = _hget_json(ODDS_HK.format(event=EVENT_ID), f"{BOOK_A}|{a}")
        ob = _hget_json(ODDS_HK.format(event=EVENT_ID), f"{BOOK_B}|{b}")
        # try flipped sides too (maybe BOOK_A is better for b and BOOK_B for a)
        oa2 = _hget_json(ODDS_HK.format(event=EVENT_ID), f"{BOOK_A}|{b}")
        ob2 = _hget_json(ODDS_HK.format(event=EVENT_ID), f"{BOOK_B}|{a}")

        best = None
        candidates = []
        if oa and ob:  candidates.append(( (BOOK_A,a,oa), (BOOK_B,b,ob) ))
        if oa2 and ob2: candidates.append(( (BOOK_B,a,ob2), (BOOK_A,b,oa2) ))
        if not candidates: return

        guard = _guard_bps() * 1e-4
        for (book1, out1, j1), (book2, out2, j2) in candidates:
            o1 = float(j1.get("odds", 0.0)); o2 = float(j2.get("odds", 0.0))
            if o1<=1 or o2<=1: continue
            fee1 = float(j1.get("fee_bps", 0.0)); fee2 = float(j2.get("fee_bps", 0.0))
            wh1  = float(j1.get("withhold_bps", 0.0)); wh2 = float(j2.get("withhold_bps", 0.0))
            inv = _net_inv(o1, fee1, wh1, 1.0) + _net_inv(o2, fee2, wh2, 1.0) + guard
            edge_bps = 1e4 * (1.0 - inv)  # positive if arbitrage exists
            if best is None or edge_bps > best[0]:
                best = (edge_bps, (book1,out1,o1,j1), (book2,out2,o2,j2))

        if not best: return
        edge_bps, leg1, leg2 = best

        ew = _load_ewma(tag); m,v = ew.update(edge_bps); _save_ewma(tag, ew)
        z = (edge_bps - m)/math.sqrt(max(v,1e-12))
        self.emit_signal(max(-1.0, min(1.0, edge_bps/max(1.0, ENTRY_BPS))))

        if r.get(_poskey(self.ctx.name, tag)) is not None: return
        if not (edge_bps >= ENTRY_BPS and abs(z) >= ENTRY_Z): return

        # Stake sizing with limit guards
        total = min(USD_TICKET_CAP, USD_BANKROLL)
        s1, s2 = _stake_split(total, [leg1[2], leg2[2]])
        s1 = min(s1, float(leg1[3].get("max_stake", s1)))
        s2 = min(s2, float(leg2[3].get("max_stake", s2)))
        # If limits bind, rescale to equalized returns using actual s1,s2
        inv_sum = (1.0/leg1[2]) + (1.0/leg2[2])
        total_used = max(MIN_STAKE, s1 + s2)
        # Recompute expected profit per outcome using chosen stakes
        prof1 = s1*leg1[2] - total_used
        prof2 = s2*leg2[2] - total_used
        min_prof = min(prof1, prof2)
        edge_real_bps = 1e4 * (min_prof / max(1e-9, total_used))
        if edge_real_bps < ENTRY_BPS: return  # abort if limits eroded the edge

        # Place legs
        sym1 = f"SB:{leg1[0]}:{EVENT_ID}:{leg1[1]}"
        sym2 = f"SB:{leg2[0]}:{EVENT_ID}:{leg2[1]}"
        self.order(sym1, "back", qty=s1, price=leg1[2], order_type="market", venue=leg1[0])
        self.order(sym2, "back", qty=s2, price=leg2[2], order_type="market", venue=leg2[0])

        legs = [
            {"book": leg1[0], "outcome": leg1[1], "odds": leg1[2], "stake": s1, "sym": sym1},
            {"book": leg2[0], "outcome": leg2[1], "odds": leg2[2], "stake": s2, "sym": sym2},
        ]
        self._save_state(tag, OpenState(mode="CROSS_BOOK", tag=tag, legs=legs,
                                        edge_bps=edge_real_bps, entry_z=z, ts_ms=_now_ms()))

    # --------------- DUTCH_BOOK (N-way) ---------------
    def _eval_dutch(self, meta: dict) -> None:
        tag = f"DB:{EVENT_ID}"
        outs = list(meta.get("outcomes") or [])
        if len(outs) < 2: return

        # For each outcome, pick **best** odds across available books in Redis
        best: List[Tuple[str,str,float,dict]] = []  # (book, outcome, odds, json)
        guard = _guard_bps()*1e-4

        for o in outs:
            # scan all fields for this outcome
            hk = ODDS_HK.format(event=EVENT_ID)
            fields = r.hkeys(hk)
            top = None
            for f in fields:
                if not f or "|" not in f: continue
                book, outcome = f.split("|",1)
                if outcome != o: continue
                j = _hget_json(hk, f)
                if not j: continue
                odds = float(j.get("odds", 0.0))
                if odds <= 1: continue
                if top is None or odds > top[2]:
                    top = (book, o, odds, j)
            if not top: return
            best.append(top)

        # Friction‑adjusted inverse odds sum
        inv = 0.0
        for (_,_,odds,j) in best:
            inv += _net_inv(odds, float(j.get("fee_bps",0.0)), float(j.get("withhold_bps",0.0)), 1.0)
        inv += guard
        edge_bps = 1e4 * (1.0 - inv)

        ew = _load_ewma(tag); m,v = ew.update(edge_bps); _save_ewma(tag, ew)
        z = (edge_bps - m)/math.sqrt(max(v,1e-12))
        self.emit_signal(max(-1.0, min(1.0, edge_bps/max(1.0, ENTRY_BPS))))

        if r.get(_poskey(self.ctx.name, tag)) is not None: return
        if not (edge_bps >= ENTRY_BPS and abs(z) >= ENTRY_Z): return

        # Stakes for guaranteed profit
        total = min(USD_TICKET_CAP, USD_BANKROLL)
        stakes = _stake_split(total, [odds for (_,_,odds,_) in best])

        # Respect per-book limits; recompute realized min profit
        used = 0.0
        legs = []
        for (stk, (book, outcome, odds, j)) in zip(stakes, best):
            s = min(max(MIN_STAKE, stk), float(j.get("max_stake", stk)))
            used += s
            sym = f"SB:{book}:{EVENT_ID}:{outcome}"
            self.order(sym, "back", qty=s, price=odds, order_type="market", venue=book)
            legs.append({"book":book, "outcome":outcome, "odds":odds, "stake":s, "sym":sym})

        # realized edge:
        profits = [ leg["stake"]*leg["odds"] - used for leg in legs ]
        min_prof = min(profits) if profits else 0.0
        edge_real_bps = 1e4 * (min_prof / max(1e-9, used))
        if edge_real_bps < EXIT_BPS:
            # Optional: no clean cancel API in paper mode; in real life you'd void or cash‑out if possible
            pass

        self._save_state(tag, OpenState(mode="DUTCH_BOOK", tag=tag, legs=legs,
                                        edge_bps=edge_real_bps, entry_z=z, ts_ms=_now_ms()))

    # --------------- MIDDLE_SPREAD (paper) ---------------
    def _eval_middle(self, meta: dict) -> None:
        tag = f"MID:{EVENT_ID}"
        # Example for totals (OVER/UNDER) or spreads: we look for disjoint lines
        # e.g., BookA Under 48.5 @ 1.95 and BookB Over 47.5 @ 1.95 → 1‑point middle
        la = _hget_json(LINE_HK.format(event=EVENT_ID), f"{BOOK_A}|TOTAL:UNDER")
        lb = _hget_json(LINE_HK.format(event=EVENT_ID), f"{BOOK_B}|TOTAL:OVER")
        if not (la and lb): return

        line_under = float(la.get("line", 0.0)); odds_u = float(la.get("odds", 0.0)); lim_u = float(la.get("max_stake", 0.0))
        line_over  = float(lb.get("line",  0.0)); odds_o = float(lb.get("odds", 0.0)); lim_o = float(lb.get("max_stake", 0.0))

        if odds_u<=1 or odds_o<=1: return
        # Require a positive middle window
        window = line_under - line_over   # points where both tickets can win if total lands in (line_over, line_under)
        if window <= 0: return

        # Paper EV proxy: assume the distribution of totals has density ~ 6% per point near the lines (toy)
        win_both_prob = min(0.20, max(0.00, 0.06 * window))  # crude proxy for demonstration
        guard = _guard_bps()*1e-4
        # EV per $ total stake (very rough)
        total = min(USD_TICKET_CAP, USD_BANKROLL, lim_u + lim_o)
        s_u = min(total/2.0, lim_u); s_o = min(total - s_u, lim_o)
        ev = win_both_prob*(s_u*(odds_u-1)+s_o*(odds_o-1)) - (1-win_both_prob)*0.0  # other cases: one wins one loses ~ flat if odds≈1.91
        edge_bps = 1e4 * (ev / max(1e-9, s_u+s_o)) - (guard*1e4)

        ew = _load_ewma(tag); m,v = ew.update(edge_bps); _save_ewma(tag, ew)
        z = (edge_bps - m)/math.sqrt(max(v,1e-12))
        self.emit_signal(max(-1.0, min(1.0, edge_bps/max(1.0, ENTRY_BPS))))

        if r.get(_poskey(self.ctx.name, tag)) is not None: return
        if not (edge_bps >= ENTRY_BPS and abs(z) >= ENTRY_Z): return

        sym_u = f"SBLINE:{BOOK_A}:{EVENT_ID}:TOTAL:UNDER"
        sym_o = f"SBLINE:{BOOK_B}:{EVENT_ID}:TOTAL:OVER"
        self.order(sym_u, "back", qty=s_u, price=odds_u, order_type="market", venue=BOOK_A, flags={"line": line_under})
        self.order(sym_o, "back", qty=s_o, price=odds_o, order_type="market", venue=BOOK_B, flags={"line": line_over})

        legs = [
            {"book":BOOK_A, "market":"TOTAL:UNDER", "line":line_under, "odds":odds_u, "stake":s_u, "sym":sym_u},
            {"book":BOOK_B, "market":"TOTAL:OVER",  "line":line_over,  "odds":odds_o, "stake":s_o, "sym":sym_o},
        ]
        self._save_state(tag, OpenState(mode="MIDDLE_SPREAD", tag=tag, legs=legs,
                                        edge_bps=edge_bps, entry_z=z, ts_ms=_now_ms()))

    # --------------- state I/O ---------------
    def _load_state(self, tag: str) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, tag))
        if not raw: return None
        try:
            o = json.loads(raw)
            return OpenState(mode=str(o["mode"]), tag=str(o["tag"]),
                             legs=list(o.get("legs") or []),
                             edge_bps=float(o["edge_bps"]), entry_z=float(o["entry_z"]),
                             ts_ms=int(o["ts_ms"]))
        except Exception:
            return None

    def _save_state(self, tag: str, st: Optional[OpenState]) -> None:
        if st is None: return
        r.set(_poskey(self.ctx.name, tag), json.dumps({
            "mode": st.mode, "tag": st.tag, "legs": st.legs,
            "edge_bps": st.edge_bps, "entry_z": st.entry_z, "ts_ms": st.ts_ms
        }))