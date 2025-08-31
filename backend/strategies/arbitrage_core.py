# backend/engine/strategies/arbitrage_core.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, DefaultDict
from collections import defaultdict

from backend.engine.strategy_base import Strategy
from backend.bus.streams import hset  # to register metadata, errors, etc.


# ----------------------- Config -----------------------
@dataclass
class ArbConfig:
    symbols: tuple[str, ...] = ("AAPL", "MSFT")     # symbols to monitor
    venues: tuple[str, ...]  = ("ZERODHA", "IBKR", "PAPER")

    # Trading sizes & limits
    default_qty: float = 1.0
    max_gross_pos: float = 10.0                     # absolute |long| or |short| cap per symbol
    max_gross_notional: float = 100_000.0           # hard safety

    # Edge requirements (bps of mid) AFTER fees/slippage/latency haircut
    min_edge_bps: float = 3.0                       # trade only if >= this
    taker_fee_bps: float = 1.0                      # assume taker both legs
    slippage_bps: float = 0.5
    latency_bps: float = 0.3

    # Cooldowns (ms)
    symbol_cooldown_ms: int = 1_000                 # after a trade on symbol
    venue_cooldown_ms: int  = 400                   # throttle per venue

    # Risk knobs
    allow_short: bool = True
    hard_kill: bool = False                         # if true, strategy refuses to trade

    # Optional: force hedge venue for the off-leg (None = choose best)
    hedge_venue: Optional[str] = None


# ----------------------- Strategy -----------------------
class ArbitrageCore(Strategy):
    """
    Cross-venue spot arbitrage:
      - Track best bid/ask per venue for configured symbols
      - When ask_v1 + fees < bid_v2 - fees by >= min_edge_bps, BUY at v1, SELL at v2
      - Sends two independent market orders (OMS/risk will coordinate/validate)
    Input tick formats tolerated:
      { symbol|s, venue|v, bid, ask }  OR  {symbol|s, venue|v, side, price}
    """

    def __init__(self, name="alpha_arb_core", region=None, cfg: Optional[ArbConfig] = None):
        cfg = cfg or ArbConfig()
        super().__init__(name=name, region=region, default_qty=cfg.default_qty)
        self.cfg = cfg

        # per-venue best quotes: book[symbol][venue] = (bid, ask, ts_ms)
        self.book: DefaultDict[str, Dict[str, Tuple[float, float, int]]] = defaultdict(dict)

        # simple positions tracker (net) — OMS can also be source of truth
        self.pos: Dict[str, float] = defaultdict(float)

        # cooldown trackers
        self._last_trade_symbol_ms: Dict[str, int] = defaultdict(lambda: 0)
        self._last_trade_venue_ms: Dict[str, int] = defaultdict(lambda: 0)

    # ------------- lifecycle ----------------
    def on_start(self) -> None:
        super().on_start()
        # register metadata so allocators / UI know what this does
        hset("strategy:meta", self.ctx.name, {
            "tags": ["arbitrage", "microstructure"],
            "region": self.ctx.region or "GLOBAL",
            "notes": "Cross-venue spot arb with fee/latency haircuts"
        })

    # ------------- helpers ------------------
    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    def _haircut_bps(self) -> float:
        return (self.cfg.taker_fee_bps * 2.0) + self.cfg.slippage_bps + self.cfg.latency_bps

    def _edge_bps(self, buy_px: float, sell_px: float) -> float:
        """
        Positive if profitable before haircuts: (sell - buy) / mid * 1e4
        """
        if buy_px <= 0 or sell_px <= 0:
            return -1e9
        mid = 0.5 * (buy_px + sell_px)
        return (sell_px - buy_px) / mid * 1e4

    def _cooldown_hit(self, sym: str, v1: str, v2: str, now: int) -> bool:
        if now - self._last_trade_symbol_ms[sym] < self.cfg.symbol_cooldown_ms:
            return True
        if now - self._last_trade_venue_ms[v1] < self.cfg.venue_cooldown_ms:
            return True
        if now - self._last_trade_venue_ms[v2] < self.cfg.venue_cooldown_ms:
            return True
        return False

    def _within_limits(self, sym: str, qty: float, buy_then_sell: bool) -> bool:
        """
        Rough check using local pos view; OMS/risk will enforce true limits.
        """
        proposed = self.pos[sym] + (qty if buy_then_sell else -qty)
        if abs(proposed) > self.cfg.max_gross_pos:
            return False
        # notional safety (use rough last mid from any venue)
        any_book = self.book.get(sym, {})
        if any_book:
            # pick one venue snapshot
            bid, ask, _ = next(iter(any_book.values()))
            mid = 0.5 * (bid + ask) if (bid > 0 and ask > 0) else max(bid, ask, 0)
        else:
            mid = 0.0
        if mid * abs(qty) > self.cfg.max_gross_notional:
            return False
        return True

    # ------------- main tick handler --------
    def on_tick(self, tick: Dict[str, Any]) -> None:
        if self.cfg.hard_kill:
            return

        # parse
        sym = (tick.get("symbol") or tick.get("s") or "").upper()
        if sym not in self.cfg.symbols:
            return
        ven = (tick.get("venue") or tick.get("v") or "").upper()
        if ven not in self.cfg.venues:
            return

        # Normalize to best bid/ask
        bid = tick.get("bid")
        ask = tick.get("ask")
        px  = tick.get("price") or tick.get("p")
        side = (tick.get("side") or "").lower()

        # tolerate prints-only feed: reconstruct rough BBO
        if bid is None or ask is None:
            if side == "buy":
                # trade buyer-initiated -> uptick at ask
                bid = float(self.book.get(sym, {}).get(ven, (0.0, 0.0, 0))[0])
                ask = float(px or 0.0)
            elif side == "sell":
                bid = float(px or 0.0)
                ask = float(self.book.get(sym, {}).get(ven, (0.0, 0.0, 0))[1])
            else:
                # fallback to last known
                b, a, _ = self.book.get(sym, {}).get(ven, (0.0, 0.0, 0))
                bid = float(b)
                ask = float(a)

        try:
            bid = float(bid or 0.0)
            ask = float(ask or 0.0)
        except Exception:
            return

        if bid <= 0 or ask <= 0 or ask < bid:
            return

        now = self._now_ms()
        self.book[sym][ven] = (bid, ask, now)

        # find cross-venue opportunity
        best_buy: Optional[Tuple[str, float]] = None   # (venue, ask)
        best_sell: Optional[Tuple[str, float]] = None  # (venue, bid)

        for v, (b, a, ts) in self.book[sym].items():
            if a > 0 and (best_buy is None or a < best_buy[1]):
                best_buy = (v, a)
            if b > 0 and (best_sell is None or b > best_sell[1]):
                best_sell = (v, b)

        if not best_buy or not best_sell:
            return

        buy_v, buy_px = best_buy
        sell_v, sell_px = best_sell

        # avoid same-venue false arbitrage
        if buy_v == sell_v:
            return

        # honor forced hedge venue if configured
        if self.cfg.hedge_venue and sell_v != self.cfg.hedge_venue:
            # If hedge venue is defined, prefer that venue for the sell leg if it has a bid
            hv = self.cfg.hedge_venue.upper()
            hb = self.book[sym].get(hv, (0.0, 0.0, 0))[0]
            if hb > 0:
                sell_v, sell_px = hv, hb

        # compute edge after haircuts
        raw_edge_bps = self._edge_bps(buy_px, sell_px)
        net_edge_bps = raw_edge_bps - self._haircut_bps()

        # inform allocator about instantaneous attractiveness (scaled)
        self.emit_signal(max(-1.0, min(1.0, net_edge_bps / 10.0)))

        if net_edge_bps < self.cfg.min_edge_bps:
            return
        if self._cooldown_hit(sym, buy_v, sell_v, now):
            return

        qty = self.ctx.default_qty or self.cfg.default_qty
        if not self._within_limits(sym, qty, buy_then_sell=True):
            return

        # --- Fire legs (buy low, sell high) ---
        # BUY @ cheap venue
        self.order(
            symbol=sym, side="buy", qty=qty,
            order_type="market", venue=buy_v,
            mark_price=buy_px,
            extra={"reason": "xvenue_arb", "leg": "buy", "sell_venue": sell_v, "edge_bps": net_edge_bps}
        )
        # SELL @ rich venue
        self.order(
            symbol=sym, side="sell", qty=qty,
            order_type="market", venue=sell_v,
            mark_price=sell_px,
            extra={"reason": "xvenue_arb", "leg": "sell", "buy_venue": buy_v, "edge_bps": net_edge_bps}
        )

        # update local pos assuming ideal fill (OMS will reconcile real fills)
        self.pos[sym] += qty  # buy then sell → flat; we keep this simple
        self.pos[sym] -= qty

        # cooldowns
        self._last_trade_symbol_ms[sym] = now
        self._last_trade_venue_ms[buy_v] = now
        self._last_trade_venue_ms[sell_v] = now


# ---------------------- optional: register a quick runner ----------------------
if __name__ == "__main__":
    """
    Example quick attach:
      export REDIS_HOST=localhost REDIS_PORT=6379
      python -m backend.engine.strategies.arbitrage_core
    """
    strat = ArbitrageCore()
    # Typically you run via Strategy.run(stream="ticks.<region/symbol>") elsewhere
    # e.g., strat.run(stream="ticks.equities.us")