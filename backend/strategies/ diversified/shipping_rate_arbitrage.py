# backend/strategies/diversified/shipping_rate_arbitrage.py
from __future__ import annotations

import json, math, os, time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import redis
from backend.engine.strategy_base import Strategy

"""
Shipping Rate Arbitrage — paper
-------------------------------
Modes:

1) FFA_BASIS (FFA vs. Spot Nowcast)
   For a route R and delivery period P:
     fair_ffa ≈ avg_E[ spot_TCE(R, t) over P ] - adj_costs
   Edge_bps ≈ (FFA_quote - fair_ffa) / fair_ffa
   • If positive & large → SELL FFA (rich) / BUY spot proxy
   • If negative & large → BUY FFA (cheap) / SELL spot proxy

2) CAL_SPREAD (Calendar spread)
   Compare near vs far FFA vs a seasonality/fair curve:
     fair_spread = fair_ffa_far - fair_ffa_near (or model input)
   Trade the spread when deviation clears thresholds.

Paper routing (adapters map to exchange APIs if/when wired):
  • "FFA:<ROUTE>:<PERIOD>"       (order_type="market"/"limit", qty in **lots**)
  • "SPOTPX:<ROUTE>"             (synthetic hedge proxy; could map to ETF/basket later)
  • "FFA_SPREAD:<ROUTE>:<NEAR>-<FAR>" (paper convenience)

Redis feeds (examples after code):
  # Marks
  HSET last_price "FFA:TD3C_DEC25" '{"price": 60000}'     # $/day for the period (or $/day index points)
  HSET last_price "SPOTPX:TD3C"     '{"price": 58000}'    # $/day nowcast for the route

  # Meta & lot sizing
  HSET ffa:meta "FFA:TD3C_DEC25" '{"route":"TD3C","period":"DEC25","start_ms":1764547200000,"end_ms":1767139200000,"lot_usd_per_$perday": 1000}'
  HSET route:carry "TD3C" '{"bunker_adj_per_day": 300, "ops_adj_per_day": 50}'  # optional fair-value adjustments

  # For calendar spreads
  HSET last_price "FFA:TD3C_JAN26" '{"price": 60500}'
  HSET ffa:seasonality "TD3C" '{"spread_bias_bps":{"DEC25_JAN26": -35}}'   # optional (bps of near's fair)

  # Fees/ops
  HSET fees:ffa "ICE" 4          # taker bps equivalent per notional (rough guard)
  SET  risk:halt 0|1
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("SHIP_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("SHIP_REDIS_PORT", "6379"))

MODE     = os.getenv("SHIP_MODE", "FFA_BASIS").upper()  # FFA_BASIS | CAL_SPREAD
FFA_SYM  = os.getenv("SHIP_FFA", "FFA:TD3C_DEC25").upper()
SPOT_SYM = os.getenv("SHIP_SPOT", "SPOTPX:TD3C").upper()

NEAR_FFA = os.getenv("SHIP_NEAR", "FFA:TD3C_DEC25").upper()
FAR_FFA  = os.getenv("SHIP_FAR",  "FFA:TD3C_JAN26").upper()

VENUE_FFA  = os.getenv("SHIP_VENUE_FFA", "ICE").upper()
VENUE_SPOT = os.getenv("SHIP_VENUE_SPOT","SYNTH").upper()

# Thresholds & risk
ENTRY_BPS = float(os.getenv("SHIP_ENTRY_BPS", "60"))  # required net edge (bps)
EXIT_BPS  = float(os.getenv("SHIP_EXIT_BPS",  "25"))
ENTRY_Z   = float(os.getenv("SHIP_ENTRY_Z",   "1.2"))
EXIT_Z    = float(os.getenv("SHIP_EXIT_Z",    "0.5"))

USD_NOTIONAL   = float(os.getenv("SHIP_USD_NOTIONAL", "50000"))
MIN_TICKET_USD = float(os.getenv("SHIP_MIN_TICKET_USD", "500"))
HEDGE_BETA     = float(os.getenv("SHIP_HEDGE_BETA", "1.0"))  # spot proxy beta vs FFA (≈1)

RECHECK_SECS   = float(os.getenv("SHIP_RECHECK_SECS", "1.5"))
EWMA_ALPHA     = float(os.getenv("SHIP_EWMA_ALPHA", "0.06"))

# Redis keys
HALT_KEY       = os.getenv("SHIP_HALT_KEY", "risk:halt")
LAST_HK        = os.getenv("SHIP_LAST_HK", "last_price")
FFA_META_HK    = os.getenv("SHIP_FFA_META_HK", "ffa:meta")
ROUTE_CARRY_HK = os.getenv("SHIP_ROUTE_CARRY_HK", "route:carry")
FEES_HK        = os.getenv("SHIP_FEES_HK", "fees:ffa")
SEAS_HK        = os.getenv("SHIP_SEAS_HK", "ffa:seasonality")

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

def _px(sym: str) -> Optional[float]:
    raw = r.hget(LAST_HK, sym)
    if not raw: return None
    try:
        j = json.loads(raw)
        return float(j.get("price", 0))
    except Exception:
        try:
            return float(raw)
        except Exception:
            return None

def _fees_bps(venue: str) -> float:
    v = r.hget(FEES_HK, venue)
    try:
        return float(v) if v is not None else 5.0
    except Exception:
        return 5.0

def _now_ms() -> int: return int(time.time() * 1000)

def _tenor_years(start_ms: int, end_ms: int) -> float:
    if not (start_ms and end_ms): return 0.25
    days = max(1.0, (end_ms - start_ms) / 86400000.0)
    return days / 365.0

def _lot_usd(meta: dict, price_per_day: float) -> float:
    # Notional per 1 lot ≈ price_per_day * lot_usd_per_$perday
    lot_mult = float(meta.get("lot_usd_per_$perday", 1000.0))
    return price_per_day * lot_mult

def _carry_adj(route: str) -> float:
    j = _hget_json(ROUTE_CARRY_HK, route) or {}
    bunker = float(j.get("bunker_adj_per_day", 0.0))
    ops    = float(j.get("ops_adj_per_day", 0.0))
    return bunker + ops

# ============================ EWMA ============================
@dataclass
class EwmaMV:
    mean: float
    var: float
    alpha: float
    def update(self, x: float) -> Tuple[float, float]:
        m0 = self.mean
        self.mean = (1 - self.alpha)*self.mean + self.alpha*x
        self.var  = max(1e-12, (1 - self.alpha)*(self.var + (x - m0)*(x - self.mean)))
        return self.mean, self.var

def _ewma_key(tag: str) -> str:
    return f"ship:ewma:{tag}"

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
    side: str  # "sell_ffa_buy_spot" | "buy_ffa_sell_spot" | "sell_spread" | "buy_spread"
    lots: float
    hedge_qty: float
    entry_bps: float
    entry_z: float
    ts_ms: int

def _poskey(name: str, tag: str) -> str:
    return f"ship:open:{name}:{tag}"

# ============================ strategy ============================
class ShippingRateArbitrage(Strategy):
    """
    FFA vs Spot nowcast basis, and Calendar spread vs seasonality fair.
    """
    def __init__(self, name: str = "shipping_rate_arbitrage", region: Optional[str] = "GLOBAL", default_qty: float = 1.0):
        super().__init__(name=name, region=region, default_qty=default_qty)
        self._last = 0.0

    def on_tick(self, tick: Dict) -> None:
        if (r.get(HALT_KEY) or "0") == "1": return
        now = time.time()
        if now - self._last < RECHECK_SECS: return
        self._last = now

        if MODE == "FFA_BASIS":
            self._eval_ffa_basis()
        else:
            self._eval_cal_spread()

    # ---------------- FFA vs Spot basis ----------------
    def _eval_ffa_basis(self) -> None:
        tag = f"FFA_BASIS:{FFA_SYM}"
        meta = _hget_json(FFA_META_HK, FFA_SYM)
        ffa_px = _px(FFA_SYM)
        spot   = _px(SPOT_SYM)
        if not meta or ffa_px is None or spot is None: return

        route = str(meta.get("route", "ROUTE"))
        start_ms = int(meta.get("start_ms", 0) or 0)
        end_ms   = int(meta.get("end_ms", 0) or 0)
        T = _tenor_years(start_ms, end_ms)

        # Fair value: spot nowcast minus daily adj costs (simple; your router can publish a richer fair)
        adj_per_day = _carry_adj(route)  # $/day
        fair = max(1e-6, spot - adj_per_day)

        # Net edge (bps of fair) after fees
        fees = _fees_bps(VENUE_FFA) * 1e-4
        edge_bps = 1e4 * ((ffa_px*(1 - fees) - fair) / fair)

        ew = _load_ewma(tag); m,v = ew.update(edge_bps); _save_ewma(tag, ew)
        z = (edge_bps - m) / math.sqrt(max(v, 1e-12))

        # monitoring
        self.emit_signal(max(-1.0, min(1.0, edge_bps / max(1.0, ENTRY_BPS))))

        st = self._load_state(tag)
        if st:
            if (abs(edge_bps) <= EXIT_BPS) or (abs(z) <= EXIT_Z):
                self._close(tag, st)
            return

        if r.get(_poskey(self.ctx.name, tag)) is not None: return
        if not (abs(edge_bps) >= ENTRY_BPS and abs(z) >= ENTRY_Z): return

        # Sizing: convert USD_NOTIONAL to lots via current notional per lot
        lot_notional = _lot_usd(meta, max(ffa_px, fair))
        lots = max(1.0, USD_NOTIONAL / max(lot_notional, 1.0))
        if lots * lot_notional < MIN_TICKET_USD: return

        # Hedge qty for spot proxy (USD notionally, paper)
        hedge_usd = HEDGE_BETA * lots * lot_notional
        spot_qty = hedge_usd / max(fair, 1.0)

        if edge_bps > 0:
            # FFA rich → SELL FFA / BUY spot proxy
            self.order(FFA_SYM, "sell", qty=lots, order_type="market", venue=VENUE_FFA)
            self.order(SPOT_SYM, "buy",  qty=spot_qty, order_type="market", venue=VENUE_SPOT)
            side = "sell_ffa_buy_spot"
        else:
            # FFA cheap → BUY FFA / SELL spot proxy
            self.order(FFA_SYM, "buy",  qty=lots, order_type="market", venue=VENUE_FFA)
            self.order(SPOT_SYM, "sell", qty=spot_qty, order_type="market", venue=VENUE_SPOT)
            side = "buy_ffa_sell_spot"

        self._save_state(tag, OpenState(mode="FFA_BASIS", side=side, lots=lots, hedge_qty=spot_qty,
                                        entry_bps=edge_bps, entry_z=z, ts_ms=_now_ms()))

    # ---------------- Calendar spread ----------------
    def _eval_cal_spread(self) -> None:
        tag = f"CAL_SPREAD:{NEAR_FFA}:{FAR_FFA}"
        meta_n = _hget_json(FFA_META_HK, NEAR_FFA)
        meta_f = _hget_json(FFA_META_HK, FAR_FFA)
        px_n = _px(NEAR_FFA); px_f = _px(FAR_FFA)
        spot = _px(SPOT_SYM)  # for optional fair anchoring
        if None in (meta_n, meta_f) or None in (px_n, px_f) or spot is None: return

        route = str(meta_n.get("route", "ROUTE"))
        # Simple fair spread from seasonality map (bps of near fair)
        seas = _hget_json(SEAS_HK, route) or {}
        bias_map = seas.get("spread_bias_bps") or {}
        key = f"{meta_n.get('period','NEAR')}_{meta_f.get('period','FAR')}"
        seas_bps = float(bias_map.get(key, 0.0))

        # Build a fair for each leg as spot - adj, then apply seasonal bias to far vs near
        adj = _carry_adj(route)
        fair_n = max(1e-6, spot - adj)
        fair_f = fair_n * (1.0 + seas_bps * 1e-4)  # apply bias to far

        spread_mkt = px_f - px_n
        spread_fair = fair_f - fair_n
        # Net edge (bps of |fair_spread| with guard)
        denom = max(1.0, abs(spread_fair))
        edge_bps = 1e4 * ((spread_mkt - spread_fair) / denom)

        ew = _load_ewma(tag); m,v = ew.update(edge_bps); _save_ewma(tag, ew)
        z = (edge_bps - m) / math.sqrt(max(v, 1e-12))
        self.emit_signal(max(-1.0, min(1.0, edge_bps / max(1.0, ENTRY_BPS))))

        st = self._load_state(tag)
        if st:
            if (abs(edge_bps) <= EXIT_BPS) or (abs(z) <= EXIT_Z):
                self._close(tag, st)
            return

        if r.get(_poskey(self.ctx.name, tag)) is not None: return
        if not (abs(edge_bps) >= ENTRY_BPS and abs(z) >= ENTRY_Z): return

        # Sizing: use near leg for lot notional baseline
        lot_notional = _lot_usd(meta_n, max(px_n, fair_n))
        lots = max(1.0, USD_NOTIONAL / max(lot_notional, 1.0))
        if lots * lot_notional < MIN_TICKET_USD: return

        if edge_bps > 0:
            # Spread rich vs fair → SELL (far - near) → SELL FAR / BUY NEAR
            self.order(FAR_FFA,  "sell", qty=lots, order_type="market", venue=VENUE_FFA)
            self.order(NEAR_FFA, "buy",  qty=lots, order_type="market", venue=VENUE_FFA)
            side = "sell_spread"
        else:
            # Spread cheap → BUY (far - near) → BUY FAR / SELL NEAR
            self.order(FAR_FFA,  "buy",  qty=lots, order_type="market", venue=VENUE_FFA)
            self.order(NEAR_FFA, "sell", qty=lots, order_type="market", venue=VENUE_FFA)
            side = "buy_spread"

        # Hedge_qty unused here (0); kept for consistent state shape
        self._save_state(tag, OpenState(mode="CAL_SPREAD", side=side, lots=lots, hedge_qty=0.0,
                                        entry_bps=edge_bps, entry_z=z, ts_ms=_now_ms()))

    # ---------------- state I/O & close ----------------
    def _load_state(self, tag: str) -> Optional[OpenState]:
        raw = r.get(_poskey(self.ctx.name, tag))
        if not raw: return None
        try:
            o = json.loads(raw)
            return OpenState(mode=str(o["mode"]), side=str(o["side"]),
                             lots=float(o["lots"]), hedge_qty=float(o["hedge_qty"]),
                             entry_bps=float(o["entry_bps"]), entry_z=float(o["entry_z"]),
                             ts_ms=int(o["ts_ms"]))
        except Exception:
            return None

    def _save_state(self, tag: str, st: Optional[OpenState]) -> None:
        if st is None: return
        r.set(_poskey(self.ctx.name, tag), json.dumps({
            "mode": st.mode, "side": st.side, "lots": st.lots,
            "hedge_qty": st.hedge_qty, "entry_bps": st.entry_bps,
            "entry_z": st.entry_z, "ts_ms": st.ts_ms
        }))

    def _close(self, tag: str, st: OpenState) -> None:
        # Unwind positions (reverse the legs)
        if st.mode == "FFA_BASIS":
            if st.side == "sell_ffa_buy_spot":
                self.order(FFA_SYM, "buy",  qty=st.lots, order_type="market", venue=VENUE_FFA)
                self.order(SPOT_SYM, "sell", qty=st.hedge_qty, order_type="market", venue=VENUE_SPOT)
            else:
                self.order(FFA_SYM, "sell", qty=st.lots, order_type="market", venue=VENUE_FFA)
                self.order(SPOT_SYM, "buy",  qty=st.hedge_qty, order_type="market", venue=VENUE_SPOT)
        else:
            if st.side == "sell_spread":
                # reverse → BUY FAR / SELL NEAR
                self.order(FAR_FFA,  "buy",  qty=st.lots, order_type="market", venue=VENUE_FFA)
                self.order(NEAR_FFA, "sell", qty=st.lots, order_type="market", venue=VENUE_FFA)
            else:
                self.order(FAR_FFA,  "sell", qty=st.lots, order_type="market", venue=VENUE_FFA)
                self.order(NEAR_FFA, "buy",  qty=st.lots, order_type="market", venue=VENUE_FFA)
        r.delete(_poskey(self.ctx.name, tag))