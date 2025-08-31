# backend/engine/strategies/cross_market_adr.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, DefaultDict
from collections import defaultdict

from backend.engine.strategy_base import Strategy
from backend.bus.streams import hset  # metadata, errors, etc.


# ----------------------- Config -----------------------
@dataclass
class Pair:
    adr: str                 # ADR ticker (e.g., "RELIANCE.NY")
    local: str               # Local ticker (e.g., "RELIANCE.NS")
    fx: str                  # FX symbol base->adr_ccy (e.g., "INRUSD" meaning INR->USD)
    ratio: float             # shares per ADR (e.g., 2 local shares = 1 ADR -> ratio = 0.5 ? see below)
    # NOTE: Define ratio as: ADR_price_theoretical = local_px * ratio * fx_rate
    # Example: If 2 local shares = 1 ADR, and local quoted in INR, ADR in USD, and FX = INRUSD,
    # then ratio = 0.5 and ADR_theo = local_px * 0.5 * INRUSD

@dataclass
class ADRConfig:
    pairs: tuple[Pair, ...] = (
        # Example (dummy; replace with real):
        # Pair(adr="INFY", local="INFY.NS", fx="INRUSD", ratio=0.5),
    )
    venues_adr: tuple[str, ...] = ("IBKR", "PAPER")
    venues_local: tuple[str, ...] = ("ZERODHA", "PAPER")
    venues_fx: tuple[str, ...] = ("IBKR", "PAPER")

    default_qty_adr: float = 1.0
    # Risk & economics
    min_edge_bps: float = 5.0        # post-haircut threshold
    taker_fee_bps_per_leg: float = 1.0
    slippage_bps: float = 0.6
    latency_bps: float = 0.3

    symbol_cooldown_ms: int = 1500
    venue_cooldown_ms: int = 500
    max_gross_pos_adr: float = 10.0
    max_notional_usd: float = 250_000.0

    allow_short: bool = True
    hard_kill: bool = False

    # Optional: force which venue to use for each leg
    force_venue_adr: Optional[str] = None
    force_venue_local: Optional[str] = None
    force_venue_fx: Optional[str] = None


# ----------------------- Strategy -----------------------
class CrossMarketADR(Strategy):
    """
    ADR vs Local Line arbitrage:
      - Tracks ADR last/quotes, Local last/quotes, and FX
      - Theoretical ADR = local_px * ratio * fx
      - If ADR >> theo by edge -> SELL ADR, BUY local (+ hedge FX if needed)
      - If ADR << theo by edge -> BUY ADR, SELL local
      - Sends legs via self.order(); OMS/risk handle validation/fills.

    Tick tolerance:
      - {symbol|s, venue|v, bid, ask} OR {symbol|s, venue|v, price|p}
      - FX ticks: same, with symbol = cfg.fx (e.g., "INRUSD"), quoted as ADR_ccy per local_ccy
    """

    def __init__(self, name="alpha_adr_cross", region=None, cfg: Optional[ADRConfig] = None):
        cfg = cfg or ADRConfig()
        super().__init__(name=name, region=region, default_qty=cfg.default_qty_adr)
        self.cfg = cfg

        # store last best bid/ask per symbol per venue
        self.book: DefaultDict[str, Dict[str, Tuple[float, float, int]]] = defaultdict(dict)
        # last mid per symbol for quick math
        self.mid: Dict[str, float] = defaultdict(float)
        # pos tracker (ADR units; local units implied by ratio)
        self.pos_adr: Dict[str, float] = defaultdict(float)

        # cooldowns
        self._last_sym_ms: Dict[str, int] = defaultdict(lambda: 0)
        self._last_ven_ms: Dict[str, int] = defaultdict(lambda: 0)

        # quick index for pair lookup by any symbol
        self._by_symbol: Dict[str, Pair] = {}
        for p in self.cfg.pairs:
            self._by_symbol[p.adr.upper()] = p
            self._by_symbol[p.local.upper()] = p
            self._by_symbol[p.fx.upper()] = p

    # ---------------- lifecycle ----------------
    def on_start(self) -> None:
        super().on_start()
        hset("strategy:meta", self.ctx.name, {
            "tags": ["arbitrage", "adr", "cross_market"],
            "region": self.ctx.region or "GLOBAL",
            "notes": "ADR vs Local line with FX/ratio hedge"
        })

    # ---------------- helpers ------------------
    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _mid(bid: float, ask: float) -> float:
        if bid > 0 and ask > 0:
            return 0.5 * (bid + ask)
        return max(bid, ask, 0.0)

    def _haircut_bps(self) -> float:
        # 2 equity legs + (optional) FX leg: approximate total costs
        legs = 2
        total_fees = self.cfg.taker_fee_bps_per_leg * legs
        return total_fees + self.cfg.slippage_bps + self.cfg.latency_bps

    def _cooldown(self, sym: str, v1: str, v2: str, now: int) -> bool:
        if now - self._last_sym_ms[sym] < self.cfg.symbol_cooldown_ms:
            return True
        if now - self._last_ven_ms[v1] < self.cfg.venue_cooldown_ms:
            return True
        if now - self._last_ven_ms[v2] < self.cfg.venue_cooldown_ms:
            return True
        return False

    def _update_book(self, sym: str, ven: str, bid: float, ask: float) -> None:
        now = self._now_ms()
        self.book[sym][ven] = (bid, ask, now)
        self.mid[sym] = self._mid(bid, ask)

    def _best_ask(self, sym: str, allowed: tuple[str, ...]) -> Optional[Tuple[str, float]]:
        best = None
        for v, (b, a, ts) in self.book.get(sym, {}).items():
            if v not in allowed or a <= 0:
                continue
            if best is None or a < best[1]:
                best = (v, a)
        return best

    def _best_bid(self, sym: str, allowed: tuple[str, ...]) -> Optional[Tuple[str, float]]:
        best = None
        for v, (b, a, ts) in self.book.get(sym, {}).items():
            if v not in allowed or b <= 0:
                continue
            if best is None or b > best[1]:
                best = (v, b)
        return best

    # ---------------- main ---------------------
    def on_tick(self, tick: Dict[str, Any]) -> None:
        if self.cfg.hard_kill or not self.cfg.pairs:
            return

        sym = (tick.get("symbol") or tick.get("s") or "").upper()
        if sym not in self._by_symbol:
            return
        ven = (tick.get("venue") or tick.get("v") or "").upper()

        # normalize to bid/ask (tolerate trade prints)
        bid = tick.get("bid")
        ask = tick.get("ask")
        px  = tick.get("price") or tick.get("p")
        side = (tick.get("side") or "").lower()
        if bid is None or ask is None:
            # reconstruct rough BBO
            last = self.book.get(sym, {}).get(ven, (0.0, 0.0, 0))
            if px is not None:
                if side == "buy":
                    bid, ask = last[0], float(px)
                elif side == "sell":
                    bid, ask = float(px), last[1]
        try:
            bid = float(bid or 0.0)
            ask = float(ask or 0.0)
        except Exception:
            return
        if bid <= 0 and ask <= 0:
            return
        if ask > 0 and bid > ask:
            bid = ask = max(bid, ask)  # be conservative

        self._update_book(sym, ven, bid, ask)

        pair = self._by_symbol[sym]
        adr, local, fx_sym, ratio = pair.adr.upper(), pair.local.upper(), pair.fx.upper(), float(pair.ratio)

        # Need ADR, LOCAL, FX mids
        mid_adr   = self.mid.get(adr, 0.0)
        mid_local = self.mid.get(local, 0.0)
        mid_fx    = self.mid.get(fx_sym, 0.0)
        if mid_adr <= 0 or mid_local <= 0 or mid_fx <= 0:
            return

        # Theoretical ADR in ADR currency
        theo_adr = mid_local * ratio * mid_fx

        # Compute raw mispricing in bps relative to theo
        raw_edge_bps = (mid_adr - theo_adr) / theo_adr * 1e4
        net_edge_bps = raw_edge_bps - self._haircut_bps()

        # Emit signal for allocator/UI
        # scale: +/-10 bps -> +/-1.0
        self.emit_signal(max(-1.0, min(1.0, net_edge_bps / 10.0)))

        if abs(net_edge_bps) < self.cfg.min_edge_bps:
            return

        # Best venues for actionable prices
        best_ask_adr = self._best_ask(adr, self.cfg.venues_adr)
        best_bid_adr = self._best_bid(adr, self.cfg.venues_adr)
        best_ask_loc = self._best_ask(local, self.cfg.venues_local)
        best_bid_loc = self._best_bid(local, self.cfg.venues_local)

        if not (best_ask_adr and best_bid_adr and best_ask_loc and best_bid_loc):
            return

        # FX venue (optional if you want explicit FX hedge leg)
        fx_ven = self.cfg.force_venue_fx or (self.cfg.venues_fx[0] if self.cfg.venues_fx else None)

        now = self._now_ms()

        # Position & notional checks (rough)
        qty_adr = self.ctx.default_qty or self.cfg.default_qty_adr
        if abs(self.pos_adr[adr] + (qty_adr if net_edge_bps < 0 else -qty_adr)) > self.cfg.max_gross_pos_adr:
            return
        # Use ADR mid for notional check
        if mid_adr * qty_adr > self.cfg.max_notional_usd:
            return

        # Decide legs
        if net_edge_bps >= self.cfg.min_edge_bps:
            # ADR rich: SELL ADR, BUY local (and potentially SELL FX to hedge)
            sell_ven_adr, sell_px_adr = best_bid_adr
            buy_ven_loc,  buy_px_loc  = best_ask_loc
            if self._cooldown(adr, sell_ven_adr, buy_ven_loc, now):
                return

            # optional forced venues
            sell_ven_adr = (self.cfg.force_venue_adr or sell_ven_adr).upper()
            buy_ven_loc  = (self.cfg.force_venue_local or buy_ven_loc).upper()

            # fire legs
            self.order(adr,   "sell", qty_adr, order_type="market", venue=sell_ven_adr,
                       mark_price=sell_px_adr,
                       extra={"reason": "adr>theo", "edge_bps": net_edge_bps, "pair": f"{adr}/{local}"})
            # Convert ADR qty to local qty via ratio: local_qty = qty_adr / ratio
            local_qty = max(1.0, qty_adr / max(1e-9, ratio))
            self.order(local, "buy",  local_qty, order_type="market", venue=buy_ven_loc,
                       mark_price=buy_px_loc,
                       extra={"reason": "adr>theo_hedge", "pair": f"{adr}/{local}"})

            # FX hedge (SELL local_ccy / BUY adr_ccy if needed) — optional stub:
            # if fx_ven:
            #     self.order(fx_sym, "sell", qty=local_qty * mid_local * ratio, order_type="market", venue=fx_ven,
            #                mark_price=mid_fx, extra={"reason":"fx_hedge"})

            # cooldown & pos
            self._last_sym_ms[adr] = now
            self._last_ven_ms[sell_ven_adr] = now
            self._last_ven_ms[buy_ven_loc]  = now
            self.pos_adr[adr] -= qty_adr

        elif net_edge_bps <= -self.cfg.min_edge_bps:
            # ADR cheap: BUY ADR, SELL local (and potentially BUY FX)
            buy_ven_adr,  buy_px_adr  = best_ask_adr
            sell_ven_loc, sell_px_loc = best_bid_loc
            if self._cooldown(adr, buy_ven_adr, sell_ven_loc, now):
                return

            buy_ven_adr  = (self.cfg.force_venue_adr or buy_ven_adr).upper()
            sell_ven_loc = (self.cfg.force_venue_local or sell_ven_loc).upper()

            self.order(adr,   "buy",  qty_adr, order_type="market", venue=buy_ven_adr,
                       mark_price=buy_px_adr,
                       extra={"reason": "adr<theo", "edge_bps": net_edge_bps, "pair": f"{adr}/{local}"})
            local_qty = max(1.0, qty_adr / max(1e-9, ratio))
            self.order(local, "sell", local_qty, order_type="market", venue=sell_ven_loc,
                       mark_price=sell_px_loc,
                       extra={"reason": "adr<theo_hedge", "pair": f"{adr}/{local}"})

            # if fx_ven:
            #     self.order(fx_sym, "buy", qty=local_qty * mid_local * ratio, order_type="market", venue=fx_ven,
            #                mark_price=mid_fx, extra={"reason":"fx_hedge"})

            self._last_sym_ms[adr] = now
            self._last_ven_ms[buy_ven_adr]   = now
            self._last_ven_ms[sell_ven_loc]  = now
            self.pos_adr[adr] += qty_adr


# ---------------------- Example runner (optional) ----------------------
if __name__ == "__main__":
    """
    Quick local run (replace pairs with real ones):
      export REDIS_HOST=localhost REDIS_PORT=6379
      python -m backend.engine.strategies.cross_market_adr
    Typically run via: Strategy.run(stream="ticks.global") elsewhere.
    """
    cfg = ADRConfig(pairs=(
        # Pair(adr="INFY", local="INFY.NS", fx="INRUSD", ratio=0.5),
    ))
    strat = CrossMarketADR(cfg=cfg)
    # strat.run(stream="ticks.global")