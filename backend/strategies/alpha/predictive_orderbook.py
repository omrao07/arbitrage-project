# backend/strategies/alpha/predictive_orderbook.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import redis
from backend.engine.strategy_base import Strategy

"""
Predictive Order Book (microstructure) — paper
----------------------------------------------
Expected Redis (publish from your feed handler; examples below):

# Universe (symbols you stream L1/L2 for)
SADD universe:micro ES BTCUSD AAPL

# Best book
HSET lob:best "AAPL" '{"bid":229.98,"bsize":1800,"ask":230.02,"asize":1500,"ts_ms":1765400000000,"spread":0.04}'

# Depth (first few levels; sizes in shares/contracts; px ascending on asks/descending on bids)
HSET lob:depth "AAPL" '{
  "bids":[[229.98,1800],[229.97,2600],[229.96,2200]],
  "asks":[[230.02,1500],[230.03,2500],[230.04,2000]],
  "ts_ms":1765400000000
}'

# Recent flow features (rolling 0.5s–2s; updated by your market data process)
HSET lob:flow "AAPL" '{
  "ofi_1s": 3200,                  // Order Flow Imbalance (shares): +buy, -sell
  "ofi_2s": 4100,
  "trade_imb_1s": 0.32,            // (buys - sells) / (buys + sells)
  "cancel_ratio_1s": 0.18,         // cancels / (adds + trades + cancels)
  "refresh_events_1s": 3,          // best-quote flips
  "vol_rel_5m": 1.4                // turnover vs 5m avg
}'

# Price (for sizing)
HSET last_price "EQ:AAPL" '{"price":230.00}'

# Fees (bps) and kill
HSET fees:eq EXCH 2
SET  risk:halt 0|1

Routing (paper; adapters wire later):
  order("EQ:<SYM>" | "FUT:<SYM>" | "CEX:<VENUE>:<SYM>", side, qty, order_type="market", venue="EXCH")
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("POB_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("POB_REDIS_PORT", "6379"))

UNIV_KEY     = os.getenv("POB_UNIV_KEY", "universe:micro")
BEST_HK      = os.getenv("POB_BEST_HK", "lob:best")
DEPTH_HK     = os.getenv("POB_DEPTH_HK", "lob:depth")
FLOW_HK      = os.getenv("POB_FLOW_HK", "lob:flow")
LAST_HK      = os.getenv("POB_LAST_HK", "last_price")
HALT_KEY     = os.getenv("POB_HALT_KEY", "risk:halt")
FEES_HK      = os.getenv("POB_FEES_HK", "fees:eq")

# Cadence
RECHECK_SECS   = float(os.getenv("POB_RECHECK_SECS", "0.15"))  # ~150ms
STALE_MS       = int(os.getenv("POB_STALE_MS", "1200"))        # data must be fresh

# Entry/Exit gates (score is roughly -3..+3)
ENTRY_SCORE    = float(os.getenv("POB_ENTRY_SCORE", "1.1"))
EXIT_SCORE     = float(os.getenv("POB_EXIT_SCORE",  "0.3"))

# Risk / sizing
USD_PER_TRADE  = float(os.getenv("POB_USD_PER_TRADE", "2500"))
MIN_TICKET_USD = float(os.getenv("POB_MIN_TICKET_USD", "100"))
MAX_CONC_OPEN  = int(os.getenv("POB_MAX_CONC_OPEN", "4"))     # max concurrent names
LOT            = float(os.getenv("POB_LOT", "1"))
TAKE_PROFIT_BP = float(os.getenv("POB_TP_BP", "6"))           # 6 bps
STOP_LOSS_BP   = float(os.getenv("POB_SL_BP", "10"))          # 10 bps
MAX_HOLD_SEC   = float(os.getenv("POB_MAX_HOLD_SEC", "45"))

# Microstructure weights
W_IMB_MICRO    = float(os.getenv("POB_W_IMB_MICRO", "0.40"))
W_OFI          = float(os.getenv("POB_W_OFI",       "0.30"))
W_SPREAD_REG   = float(os.getenv("POB_W_SPREAD",    "0.10"))
W_REFRESH      = float(os.getenv("POB_W_REFRESH",   "0.10"))
W_VOLCONF      = float(os.getenv("POB_W_VOLCONF",   "0.10"))

# ============================ Redis ============================
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ============================ helpers ============================
def _now_ms() -> int: return int(time.time()*1000)

def _hget_json(hk: str, field: str) -> Optional[dict]:
    raw = r.hget(hk, field)
    if not raw: return None
    try:
        j = json.loads(raw); return j if isinstance(j, dict) else None # type: ignore
    except Exception: return None

def _last_px(symkey: str) -> Optional[float]:
    raw = r.hget(LAST_HK, symkey)
    if not raw: return None
    try:
        j = json.loads(raw); return float(j.get("price", 0.0)) # type: ignore
    except Exception:
        try: return float(raw) # type: ignore
        except Exception: return None

def _fees_bps(venue: str="EXCH") -> float:
    v = r.hget(FEES_HK, venue)
    try: return float(v) if v is not None else 2.0 # type: ignore
    except Exception: return 2.0

# ============================ state ============================
@dataclass
class OpenState:
    side: str           # long/short
    qty: float
    entry_px: float
    entry_ms: int
    entry_score: float

def _poskey(ctx: str, sym: str) -> str:
    return f"pob:open:{ctx}:{sym}"

# ============================ feature calc ============================
def _micro_features(best: dict, depth: dict, flow: dict) -> Tuple[float, Dict[str,float]]:
    """
    Returns (signed score, debug dict). Sign: +bull (buy), -bear (sell).
    """
    # L1
    bid = float(best.get("bid", 0.0)); ask = float(best.get("ask", 0.0))
    bsz = float(best.get("bsize", 0.0)); asz = float(best.get("asize", 0.0))
    spread = float(best.get("spread", max(0.0, ask - bid)))
    mid = (bid + ask) / 2.0 if (bid>0 and ask>0) else _safe_mid(depth)
    # Microprice premium (scaled by spread)
    micro = (ask*bsz + bid*asz) / max(1e-9, (bsz + asz)) if (bsz>0 and asz>0) else mid
    micro_prem = (micro - mid) / max(1e-9, spread)  # ~[-1,+1]

    # Depth imbalance near touch (use first 3 levels)
    bids = depth.get("bids") or []
    asks = depth.get("asks") or []
    sum_b = sum(x[1] for x in bids[:3]) if bids else bsz
    sum_a = sum(x[1] for x in asks[:3]) if asks else asz
    imb = (sum_b - sum_a) / max(1.0, sum_b + sum_a)  # [-1,+1]

    # Order Flow Imbalance (normalized)
    ofi1 = float(flow.get("ofi_1s", 0.0))
    ofi2 = float(flow.get("ofi_2s", 0.0))
    ofi = 0.6*ofi1 + 0.4*ofi2
    # squash to [-1,1] by softsign against a scale (choose 5k shares default)
    ofi_norm = ofi / (5000.0 + abs(ofi))

    trade_imb = float(flow.get("trade_imb_1s", 0.0))  # already [-1,+1]
    cancel_r = float(flow.get("cancel_ratio_1s", 0.0))
    refresh_n = float(flow.get("refresh_events_1s", 0.0))
    volrel = float(flow.get("vol_rel_5m", 1.0))

    # Spread regime (tighter spreads favor momentum at micro horizons)
    spread_reg = -1.0 if spread <= 0 else min(1.0, 0.5 / spread)  # rough; tighter → higher positive value

    # Composite score
    core = (W_IMB_MICRO * (0.6*imb + 0.4*micro_prem)) \
         + (W_OFI       * (0.7*ofi_norm + 0.3*trade_imb)) \
         + (W_SPREAD_REG* spread_reg) \
         + (W_REFRESH   * (-min(1.0, refresh_n/6.0))) 

    # Penalize cancel storms
    core -= 0.15 * min(1.0, cancel_r)

    return float(core), {
        "micro_prem": micro_prem, "imb": imb, "ofi": ofi_norm, "trade_imb": trade_imb,
        "spread": spread, "spread_reg": spread_reg, "refresh": refresh_n, "volrel": volrel, "cancel_r": cancel_r
    }

def _safe_mid(depth: dict) -> float:
    bids = depth.get("bids") or []; asks = depth.get("asks") or []
    b = bids[0][0] if bids else 0.0; a = asks[0][0] if asks else 0.0
    return (b + a)/2.0 if (b>0 and a>0) else 0.0

# ============================ Strategy ============================
class PredictiveOrderBook(Strategy):
    """
    Short‑horizon microstructure alpha using L1/L2 features (paper).
    """
    def __init__(self, name: str = "predictive_orderbook", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
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

        # Track how many open
        open_keys = [k for k in r.scan_iter(match=_poskey(self.ctx.name, "*"))]
        n_open = len(open_keys)

        # Iterate symbols
        best_name = None; best_abs = 0.0; best_signed = 0.0

        for s in syms:
            best = _hget_json(BEST_HK, s); depth = _hget_json(DEPTH_HK, s); flow = _hget_json(FLOW_HK, s)
            if not best or not depth or not flow: 
                continue
            ts = int(best.get("ts_ms", _now_ms()))
            if _now_ms() - ts > STALE_MS: 
                continue

            score, dbg = _micro_features(best, depth, flow)
            signed = float(score)
            abs_score = abs(signed)

            # Manage an existing position first
            st = self._load_state(s)
            if st:
                # take-profit / stop-loss / time
                mid = (float(best.get("bid",0))+float(best.get("ask",0)))/2.0
                if mid <= 0: 
                    continue
                pnl_bp = 1e4 * ( (mid - st.entry_px)/st.entry_px if st.side=="long"
                                  else (st.entry_px - mid)/st.entry_px )
                held = (_now_ms() - st.entry_ms)/1000.0
                if (pnl_bp >= TAKE_PROFIT_BP) or (pnl_bp <= -STOP_LOSS_BP) or (abs(signed) <= EXIT_SCORE) or (held >= MAX_HOLD_SEC):
                    self._flat(s, st)
                    n_open = max(0, n_open-1)
                continue

            # Consider entries (only if capacity left)
            if n_open >= MAX_CONC_OPEN: 
                continue
            if abs_score < ENTRY_SCORE: 
                continue

            # Size & route
            # Map symbol namespace to last_price key; if your symbol already prefixed, publish accordingly.
            px = _last_px(f"EQ:{s}") or best.get("ask") or best.get("bid")
            if not px or float(px) <= 0: 
                continue
            qty = math.floor((USD_PER_TRADE / float(px)) / max(1.0, LOT)) * LOT
            if qty <= 0 or qty * float(px) < MIN_TICKET_USD: 
                continue

            side = "buy" if signed > 0 else "sell"
            self.order(f"EQ:{s}", side, qty=qty, order_type="market", venue="EXCH")

            self._save_state(s, OpenState(
                side=("long" if signed>0 else "short"),
                qty=qty, entry_px=float(px), entry_ms=_now_ms(), entry_score=signed
            ))
            n_open += 1

            # keep best for dashboard
            if abs_score > best_abs:
                best_abs = abs_score; best_signed = signed; best_name = s

        # Emit a UI heartbeat (signed best score scaled)
        ui = max(-1.0, min(1.0, (best_signed/3.0) if best_abs>0 else 0.0))
        self.emit_signal(ui)

    # ---------------- state I/O ----------------
    def _load_state(self, sym: str) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, sym))
        if not raw: return None
        try:
            o = json.loads(raw) # type: ignore
            return OpenState(side=str(o["side"]), qty=float(o["qty"]), entry_px=float(o["entry_px"]),
                             entry_ms=int(o["entry_ms"]), entry_score=float(o.get("entry_score", 0.0)))
        except Exception:
            return None

    def _save_state(self, sym: str, st: OpenState) -> None:
        r.set(_poskey(self.ctx.name, sym), json.dumps({
            "side": st.side, "qty": st.qty, "entry_px": st.entry_px,
            "entry_ms": st.entry_ms, "entry_score": st.entry_score
        }))

    def _flat(self, sym: str, st: OpenState) -> None:
        if st.side == "long":
            self.order(f"EQ:{sym}", "sell", qty=st.qty, order_type="market", venue="EXCH")
        else:
            self.order(f"EQ:{sym}", "buy",  qty=st.qty, order_type="market", venue="EXCH")
        r.delete(_poskey(self.ctx.name, sym))