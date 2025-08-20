# backend/strategies/diversified/capital_flow_rotation.py
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import redis

from backend.engine.strategy_base import Strategy

# ------------------------- Config via env -------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# Stream that publishes flow metrics per symbol
# Expected payload example (JSON string under field "json" on a Redis Stream):
# {
#   "ts_ms": 1723500000000,
#   "symbol": "XLK",
#   "venue": "ARCA",
#   "region": "US",
#   "price": 227.31,
#   "net_flow_usd": 125000000.0,         # (+) inflow, (-) outflow
#   "flow_1d": 1.2e8, "flow_5d": 3.8e8,   # optional multi-horizon
#   "oi_change_usd": 4.5e7,               # futures OI change (optional)
#   "vol_dollar": 9.1e9                   # dollar volume (optional)
# }
FLOW_STREAM = os.getenv("CAPFLO_STREAM", "metrics.capital_flows")

# Universe (comma‑sep). Symbols must match whatever your last_price/flows use.
# Common examples: US sector ETFs "XLK,XLF,XLY,XLI,XLP,XLV,XLE,XLC,XLU,XLB", or indices "SPY,QQQ,IWM,TLT,HYG".
UNIVERSE = [s.strip().upper() for s in os.getenv("CAPFLO_UNIVERSE", "SPY,QQQ,IWM,TLT,HYG,GLD,USO,XLK,XLF,XLV").split(",") if s.strip()]

# Rebalance cadence (seconds)
REBALANCE_SECS = int(os.getenv("CAPFLO_REBALANCE_SECS", "900"))    # 15 min default

# Portfolio sizing
GROSS_USD     = float(os.getenv("CAPFLO_GROSS_USD", "100000"))     # total gross target
LONG_BUCKET   = int(os.getenv("CAPFLO_LONG_BUCKET", "4"))          # # to long
SHORT_BUCKET  = int(os.getenv("CAPFLO_SHORT_BUCKET", "4"))         # # to short (set 0 for long-only)
NEUTRALIZE    = os.getenv("CAPFLO_NEUTRALIZE", "true").lower() in ("1","true","yes")

# Signal weighting
W_FLOW        = float(os.getenv("CAPFLO_W_FLOW", "0.65"))          # flow momentum weight
W_PRICE       = float(os.getenv("CAPFLO_W_PRICE", "0.35"))         # price strength weight

# Flow transform params
USE_OI        = os.getenv("CAPFLO_USE_OI", "true").lower() in ("1","true","yes")
FLOW_HALFLIFE = int(os.getenv("CAPFLO_FLOW_HALFLIFE", "3"))        # EWMA halflife (events) for flow smooth
LOOKBACK_MAX  = int(os.getenv("CAPFLO_LOOKBACK_MAX", "50"))        # max cached observations per symbol

# Safety
MIN_PRICE     = float(os.getenv("CAPFLO_MIN_PRICE", "2.0"))        # skip penny stuff
MIN_LIQ_USD   = float(os.getenv("CAPFLO_MIN_LIQ_USD", "5e6"))      # skip illiquid if vol_dollar provided

# Region hint (helps router & policy)
REGION_HINT   = os.getenv("CAPFLO_REGION_HINT", "US").upper()

# ------------------------- Redis wiring -------------------------
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def _last_price(symbol: str) -> Optional[float]:
    raw = r.hget("last_price", symbol.upper())
    if not raw: return None
    try:
        return float(json.loads(raw)["price"])
    except Exception:
        return None

def _load_pos_by_strategy(strategy: str, symbol: str) -> Dict:
    raw = r.hget(f"positions:by_strategy:{strategy}", symbol)
    if not raw: return {"symbol": symbol, "qty": 0.0, "avg_price": 0.0}
    try: return json.loads(raw)
    except Exception: return {"symbol": symbol, "qty": 0.0, "avg_price": 0.0}

def _sign(x: float) -> int:
    return (x > 0) - (x < 0)

# ------------------------- EWMA helper -------------------------
class Ewma:
    def __init__(self, halflife_events: int):
        self.alpha = 1.0 - math.exp(math.log(0.5) / max(1, halflife_events))
        self.v = None

    def update(self, x: float) -> float:
        if self.v is None:
            self.v = float(x)
        else:
            self.v = (1 - self.alpha) * self.v + self.alpha * float(x)
        return self.v if self.v is not None else 0.0

# ------------------------- Strategy -------------------------
@dataclass
class SymState:
    ew_flow: Ewma
    ew_price_chg: Ewma
    last_ts: int = 0
    last_price: float = float("nan")
    last_flow: float = 0.0

class CapitalFlowRotation(Strategy):
    """
    Rotates into assets with strong **inflows + price confirmation** and out of those with persistent outflows.
    Uses EWMA of net_flow_usd (optionally OI) and price strength; ranks and forms long/short buckets.

    Inputs: FLOW_STREAM messages (see header). You can publish them from your ClickHouse/ETL.
    Orders: Rebalance every REBALANCE_SECS toward target weights using current last_price marks.
    """

    def __init__(self, name: str = "capital_flow_rotation", region: Optional[str] = REGION_HINT, default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self.sym: Dict[str, SymState] = {}
        self.last_rebalance = 0.0

        # in-memory book of current target weights (for smoother trading)
        self.current_weights: Dict[str, float] = {u: 0.0 for u in UNIVERSE}

    # ---- lifecycle ----
    def on_start(self) -> None:
        super().on_start()
        # seed states
        for s in UNIVERSE:
            self.sym[s] = SymState(ew_flow=Ewma(FLOW_HALFLIFE), ew_price_chg=Ewma(max(1, FLOW_HALFLIFE//2)))
        # announce universe for UI
        r.hset("strategy:universe", self.ctx.name, json.dumps({"symbols": UNIVERSE, "ts": int(time.time()*1000)}))

    # ---- core tick (consumes FLOW_STREAM events) ----
    def on_tick(self, tick: Dict) -> None:
        """
        This strategy should be attached to FLOW_STREAM via the strategy_router.
        """
        sym = str(tick.get("symbol") or "").upper()
        if sym not in UNIVERSE:
            return

        px  = float(tick.get("price") or _last_price(sym) or 0.0)
        if px < MIN_PRICE:
            return

        # liquidity guard (optional)
        liq = tick.get("vol_dollar")
        if liq is not None:
            try:
                if float(liq) < MIN_LIQ_USD:
                    return
            except Exception:
                pass

        # combine flows
        flow = float(tick.get("net_flow_usd") or 0.0)
        if USE_OI:
            oi = float(tick.get("oi_change_usd") or 0.0)
            flow += 0.5 * oi  # small weight on OI change

        st = self.sym.get(sym)
        if st is None:
            st = SymState(ew_flow=Ewma(FLOW_HALFLIFE), ew_price_chg=Ewma(max(1, FLOW_HALFLIFE//2)))
            self.sym[sym] = st

        # price strength as short-horizon pct change sign/magnitude (event-wise EWMA)
        if st.last_price and st.last_price > 0:
            ret = (px - st.last_price) / st.last_price
        else:
            ret = 0.0

        ewf = st.ew_flow.update(flow)
        ewp = st.ew_price_chg.update(ret)

        st.last_ts = int(tick.get("ts_ms") or time.time() * 1000)
        st.last_price = px
        st.last_flow = flow

        # emit a per‑symbol signal in [-1,1] for allocator visibility
        # z‑like via sign(flow)*sqrt(|flow|) scaled + price confirmation
        flow_score = math.tanh(ewf / (1e7))  # scale denominator to your flow magnitudes
        price_score = math.tanh(50.0 * ewp)  # 2% event jump -> ~tanh(1) magnitude
        score = W_FLOW * flow_score + W_PRICE * price_score
        self.emit_signal(score)  # strategy-level aggregate; optional: could store per-symbol in Redis if needed

        # periodic rebalance
        now = time.time()
        if now - self.last_rebalance >= REBALANCE_SECS:
            self._rebalance()
            self.last_rebalance = now

    # ---- ranking & target weights ----
    def _rank_scores(self) -> List[Tuple[str, float]]:
        scores: List[Tuple[str, float]] = []
        for s in UNIVERSE:
            st = self.sym.get(s)
            px = _last_price(s)
            if st is None or px is None or px < MIN_PRICE:
                continue
            flow_score = math.tanh((st.ew_flow.v or 0.0) / (1e7))
            price_score = math.tanh(50.0 * (st.ew_price_chg.v or 0.0))
            scores.append((s, W_FLOW * flow_score + W_PRICE * price_score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def _target_weights(self) -> Dict[str, float]:
        scores = self._rank_scores()
        if not scores:
            return {s: 0.0 for s in UNIVERSE}

        longs = [s for s, _ in scores[:LONG_BUCKET]]
        shorts = [s for s, _ in scores[-SHORT_BUCKET:]] if SHORT_BUCKET > 0 else []

        w: Dict[str, float] = {s: 0.0 for s in UNIVERSE}
        if NEUTRALIZE and SHORT_BUCKET > 0:
            if longs:
                wl = 0.5 / max(1, len(longs))
                for s in longs: w[s] = +wl
            if shorts:
                ws = 0.5 / max(1, len(shorts))
                for s in shorts: w[s] = -ws
        else:
            if longs:
                wl = 1.0 / max(1, len(longs))
                for s in longs: w[s] = +wl
        return w

    # ---- order generation ----
    def _rebalance(self) -> None:
        tgt_w = self._target_weights()
        if not tgt_w:
            return

        # translate weights to dollar targets, then to qty deltas
        # read current per‑symbol position for this strategy
        for s, w in tgt_w.items():
            px = _last_price(s)
            if px is None or px < MIN_PRICE:
                continue

            target_notional = w * GROSS_USD
            pos = _load_pos_by_strategy(self.ctx.name, s)
            cur_qty = float(pos.get("qty", 0.0))
            cur_notional = cur_qty * px

            delta_notional = target_notional - cur_notional
            if abs(delta_notional) <= max(5.0, 0.0005 * GROSS_USD):  # skip tiny trades
                continue

            qty = delta_notional / px
            side = "buy" if qty > 0 else "sell"
            self.order(s, side, qty=abs(qty), order_type="market", venue=tick_venue_hint(s))

        # remember current weights
        self.current_weights = tgt_w

# ---- venue hint helper (region-aware symbols if you mix assets) ----
def tick_venue_hint(symbol: str) -> Optional[str]:
    # very light heuristic; refine if needed
    if symbol.endswith(".HK"): return "HKEX"
    if symbol.endswith(".NS") or symbol.endswith(".BSE") or symbol.endswith(".BO"): return "NSE"
    if symbol in ("BTCUSDT","ETHUSDT"): return "BINANCE"
    return "ARCA" if symbol.isalpha() and len(symbol) <= 5 else None