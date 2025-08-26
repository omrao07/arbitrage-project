# backend/execution_plus/strategies/market_maker.py
"""
Simple cost-aware market maker (two-sided quoting with inventory skew).

Plugs into your execution stack:
- Uses AdapterBase from execution_plus.adapters
- Optional cost awareness via DefaultCostModel
- Risk controls: position cap, max loss, kill switch
- Inventory skewing: tighter on the side that reduces inventory

Baseline algorithm (per symbol/venue loop):
  1) Fetch quote -> mid + spread
  2) Compute fair value (fv = mid)
  3) Compute base spread from:
       - market spread (half-spread)
       - cost model (fees + latency + impact converted to bps)
       - min_spread_bps
  4) Inventory skew:
       - widen on the side that increases inventory
       - tighten on the side that reduces inventory
  5) Post limit orders at:
       bid = fv - spread_bid ; ask = fv + spread_ask
  6) Cancel/replace if drift > reprice_threshold_bps, size mismatch, or stale

Configuration via MarketMakerConfig (can be per symbol).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from backend.execution_plus.adapters import ( # type: ignore
    AdapterBase, Order, OrderType, Side, Quote, AdapterRegistry
)
from backend.execution_plus.cost_model import DefaultCostModel, get_default_model # type: ignore


# ---------------------------------------------------------------------
# Config & State
# ---------------------------------------------------------------------

@dataclass
class MarketMakerConfig:
    symbol: str
    venue_id: str
    lot_size: float                     # base order size
    max_position: float                 # absolute cap (units)
    min_spread_bps: float = 4.0         # floor spread
    inv_skew_bps_per_unit: float = 0.5  # extra bps per unit inventory
    reprice_threshold_bps: float = 3.0  # cancel/replace threshold
    post_only: bool = True              # keep passive; price <= bid for sells, >= ask for buys
    max_loss_usd: float = 10_000.0      # session kill-switch
    max_order_age_sec: float = 30.0     # refresh TTL
    topup_when_filled: bool = True      # keep top of book presence


@dataclass
class OrderRef:
    order_id: Optional[str]
    side: Side
    px: float
    qty: float
    ts: float


@dataclass
class MMState:
    position: float = 0.0
    cash_pnl_usd: float = 0.0
    open: Dict[str, OrderRef] = field(default_factory=dict)  # "bid"/"ask" -> OrderRef
    last_mid: Optional[float] = None


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _half_spread(q: Quote, fallback_bps: float = 8.0) -> float:
    # value in price units
    if q.bid is not None and q.ask is not None and q.ask > q.bid > 0:
        return 0.5 * (q.ask - q.bid)
    mid = q.mid or ((q.bid or 0.0) + (q.ask or 0.0)) / 2.0
    return (fallback_bps / 10_000.0) * max(0.0, mid)

def _bps_from_px(px: float, mid: float) -> float:
    if mid <= 0:
        return 0.0
    return 10_000.0 * abs(px - mid) / mid

def _now() -> float:
    return time.time()


# ---------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------

class MarketMaker:
    def __init__(
        self,
        adapter: AdapterBase,
        cfg: MarketMakerConfig,
        *,
        cost_model: Optional[DefaultCostModel] = None,
        fx_usd_per_base: float = 1.0,
    ) -> None:
        self.adapter = adapter
        self.cfg = cfg
        self.cost = cost_model or get_default_model()
        self.fx = float(fx_usd_per_base)
        self.state = MMState()

    # --------------- core loop ---------------

    def run_once(self) -> None:
        """
        One maintenance tick: quote/update risk/cancel-replace.
        """
        # 0) Kill switch
        if self.state.cash_pnl_usd <= -abs(self.cfg.max_loss_usd):
            self._cancel_both(reason="kill_switch")
            return

        # 1) Get market
        q = self.adapter.get_quote(self.cfg.symbol)
        mid = q.mid or ((q.bid or 0.0) + (q.ask or 0.0)) / 2.0
        if not mid or mid <= 0:
            self._cancel_both(reason="no_mid")
            return

        self.state.last_mid = mid

        # 2) Compute baseline spread from market + costs
        hs = _half_spread(q)
        # Convert cost model USD total to per-unit price add-on
        fake_order_qty = max(self.cfg.lot_size, 1e-9)
        cb = self.cost.estimate(self.adapter,
                                Order(symbol=self.cfg.symbol, side=Side.BUY, qty=fake_order_qty, type=OrderType.MARKET),
                                q,
                                fx_usd_per_base=self.fx)
        per_unit_cost = cb.total / fake_order_qty  # price units
        floor = (self.cfg.min_spread_bps / 10_000.0) * mid
        base_half = max(hs, floor) + per_unit_cost

        # 3) Inventory skew (in price units)
        inv = self.state.position
        skew_bps = self.cfg.inv_skew_bps_per_unit * inv
        skew_px = (skew_bps / 10_000.0) * mid
        # If long inventory, we want to sell -> tighten ask, widen bid
        bid_px = mid - base_half - max(0.0, skew_px)
        ask_px = mid + base_half - max(0.0, -skew_px)

        # 4) Respect post-only (stay passive vs current book)
        if self.cfg.post_only:
            if q.bid is not None:
                bid_px = min(bid_px, q.bid)  # do not cross
            if q.ask is not None:
                ask_px = max(ask_px, q.ask)

        # 5) Guardrail: reprice/cancel if drift large
        self._refresh_order("bid", Side.BUY, bid_px, self._calc_bid_qty(mid), q)
        self._refresh_order("ask", Side.SELL, ask_px, self._calc_ask_qty(mid), q)

    # --------------- sizing ---------------

    def _calc_bid_qty(self, mid: float) -> float:
        # Reduce buy size if near max long
        headroom = max(0.0, self.cfg.max_position - self.state.position)
        return max(0.0, min(self.cfg.lot_size, headroom))

    def _calc_ask_qty(self, mid: float) -> float:
        # Reduce sell size if near max short
        headroom = max(0.0, self.cfg.max_position + self.state.position)
        return max(0.0, min(self.cfg.lot_size, headroom))

    # --------------- order lifecycle ---------------

    def _refresh_order(self, key: str, side: Side, px: float, qty: float, q: Quote) -> None:
        """
        Create or adjust the standing order on one side.
        """
        if qty <= 0:
            self._cancel_side(key)
            return

        ref = self.state.open.get(key)
        now = _now()

        # If no order, place one
        if ref is None:
            self._place_side(key, side, px, qty)
            return

        # If stale or price drifted enough, replace
        drift = _bps_from_px(px, self.state.last_mid or (q.mid or 0.0))
        age = now - ref.ts
        need_reprice = (drift >= self.cfg.reprice_threshold_bps) or (abs(ref.px - px) / max(px, 1e-12) > 0.0003)
        need_refresh = age >= self.cfg.max_order_age_sec
        size_changed = abs(ref.qty - qty) / max(1.0, ref.qty) > 0.25

        if need_reprice or need_refresh or size_changed:
            self._cancel_side(key)
            self._place_side(key, side, px, qty)

    def _place_side(self, key: str, side: Side, px: float, qty: float) -> None:
        res = self.adapter.place_order(Order(
            symbol=self.cfg.symbol, side=side, qty=qty, type=OrderType.LIMIT, limit_price=px, venue_id=self.cfg.venue_id
        ))
        # If adapter fills instantly (mock), update PnL/position; otherwise store the working order
        if res.ok and res.filled_qty > 0 and res.avg_price is not None:
            self._apply_fill(side, float(res.filled_qty), float(res.avg_price), float(res.fees))
            if self.cfg.topup_when_filled:
                # Immediately top-up again at same side (will be repriced next tick)
                self.state.open.pop(key, None)
            else:
                self.state.open[key] = OrderRef(order_id=res.order_id, side=side, px=px, qty=qty, ts=_now())
        elif res.ok and res.status in ("accepted",):
            self.state.open[key] = OrderRef(order_id=res.order_id, side=side, px=px, qty=qty, ts=_now())
        else:
            # rejected/error -> drop side
            self.state.open.pop(key, None)

    def _cancel_side(self, key: str) -> None:
        ref = self.state.open.pop(key, None)
        if ref and ref.order_id:
            try:
                self.adapter.cancel_order(ref.order_id)
            except Exception:
                pass

    def _cancel_both(self, reason: str = "") -> None:
        self._cancel_side("bid")
        self._cancel_side("ask")

    # --------------- fills & PnL ---------------

    def _apply_fill(self, side: Side, qty: float, px: float, fees: float) -> None:
        if side == Side.BUY:
            self.state.position += qty
            self.state.cash_pnl_usd -= (px * qty * self.fx)  # spend cash
        else:
            self.state.position -= qty
            self.state.cash_pnl_usd += (px * qty * self.fx)  # receive cash
        self.state.cash_pnl_usd -= fees * self.fx

    # --------------- utilities ---------------

    def snapshot(self) -> Dict[str, float]:
        return {
            "position": self.state.position,
            "cash_pnl_usd": self.state.cash_pnl_usd,
            "last_mid": float(self.state.last_mid or 0.0),
            "has_bid": 1.0 if "bid" in self.state.open else 0.0,
            "has_ask": 1.0 if "ask" in self.state.open else 0.0,
        }


# ---------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------

def run_market_maker_loop(
    symbol: str,
    venue_id: str,
    *,
    lot_size: float = 0.1,
    max_position: float = 2.0,
    sleep_sec: float = 1.0,
) -> None:
    """
    Tiny self-contained loop using built-in mock adapters (or real ones if registered).
    """
    adapter = AdapterRegistry.get(venue_id)
    mm = MarketMaker(
        adapter=adapter,
        cfg=MarketMakerConfig(
            symbol=symbol,
            venue_id=venue_id,
            lot_size=lot_size,
            max_position=max_position,
        ),
    )
    print(f"[MM] starting symbol={symbol} venue={venue_id}")
    try:
        while True:
            mm.run_once()
            snap = mm.snapshot()
            print(f"[MM] mid={snap['last_mid']:.4f} pos={snap['position']:.4f} pnl=${snap['cash_pnl_usd']:.2f} bid={int(snap['has_bid'])} ask={int(snap['has_ask'])}")
            time.sleep(sleep_sec)
    except KeyboardInterrupt:
        print("\n[MM] stopping; cancelling orders…")
        mm._cancel_both("shutdown")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Example: crypto mock
    run_market_maker_loop(symbol="BTCUSDT", venue_id="BINANCE", lot_size=0.05, max_position=0.5, sleep_sec=1.0)