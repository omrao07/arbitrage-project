# backend/engine/strategies/etf_nav_arb.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, DefaultDict
from collections import defaultdict

from backend.engine.strategy_base import Strategy
from backend.bus.streams import hset


# ----------------------- Config -----------------------
@dataclass
class ETFNavConfig:
    etf: str = "SPY"
    # Basket in same currency as ETF; weights should sum ~1 (auto-normalized if not)
    basket: tuple[str, ...] = ("AAPL", "MSFT", "AMZN", "GOOGL", "NVDA")
    weights: Optional[tuple[float, ...]] = None

    # Venues
    venues_etf: tuple[str, ...] = ("IBKR", "PAPER")
    venues_basket: tuple[str, ...] = ("IBKR", "PAPER")

    # Use streaming iNAV if available; otherwise compute from basket mids
    use_inav: bool = True
    inav_symbol: Optional[str] = None  # e.g., "SPY.INAV" (mid-only feed)

    # Sizing / limits
    target_notional: float = 25_000.0     # per action cycle
    default_qty: float = 1.0              # fallback when price missing
    max_gross_notional: float = 250_000.0
    symbol_cooldown_ms: int = 1_500
    venue_cooldown_ms: int = 400

    # Economics (bps of mid)
    min_edge_bps: float = 4.0
    taker_fee_bps_per_leg: float = 1.0
    slippage_bps: float = 0.5
    latency_bps: float = 0.3

    # Behavior & risk
    allow_short: bool = True
    hard_kill: bool = False
    forced_etf_venue: Optional[str] = None
    forced_equity_venue: Optional[str] = None

    # Tolerance
    max_missing_basket: int = 2   # allow a couple missing prices before skipping


# ----------------------- Strategy -----------------------
class ETFNavArb(Strategy):
    """
    ETF NAV arbitrage:
      - Tracks ETF mid and either iNAV (if provided) or computes basket NAV from constituents.
      - Premium  : ETF_mid >> NAV  -> SELL ETF, BUY basket
      - Discount : ETF_mid << NAV  -> BUY ETF, SELL basket
      - Sends legs via self.order(); OMS/risk validate & route.

    Tick tolerance:
      - {symbol|s, venue|v, bid, ask} or {symbol|s, price|p}
      - iNAV tick: symbol == inav_symbol with {price|p|mid}
    """

    def __init__(self, name="alpha_etf_nav_arb", region=None, cfg: Optional[ETFNavConfig] = None):
        cfg = cfg or ETFNavConfig()
        super().__init__(name=name, region=region, default_qty=cfg.default_qty)
        self.cfg = cfg

        self.etf = cfg.etf.upper()
        self.basket = tuple(s.upper() for s in cfg.basket)
        self.weights = self._normalize_weights(cfg.weights, len(self.basket))

        # orderbook snapshots
        self.book: DefaultDict[str, Dict[str, Tuple[float, float, int]]] = defaultdict(dict)
        self.mid: Dict[str, float] = defaultdict(float)

        # cooldowns
        self._last_sym_ms: Dict[str, int] = defaultdict(lambda: 0)
        self._last_ven_ms: Dict[str, int] = defaultdict(lambda: 0)

        # inav symbol canonicalization
        self.inav_symbol = (cfg.inav_symbol or f"{self.etf}.INAV").upper() if cfg.use_inav else None

    # ------------- lifecycle ----------------
    def on_start(self) -> None:
        super().on_start()
        hset("strategy:meta", self.ctx.name, {
            "tags": ["arbitrage", "etf", "nav"],
            "region": self.ctx.region or "US",
            "notes": "ETF vs iNAV/basket implied NAV arb"
        })

    # ------------- helpers ------------------
    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def _normalize_weights(w: Optional[tuple[float, ...]], n: int) -> tuple[float, ...]:
        if not w or len(w) != n:
            return tuple([1.0 / n] * n)
        s = sum(abs(x) for x in w) or 1.0
        return tuple(float(x) / s for x in w)

    @staticmethod
    def _mid_from(bid: float, ask: float) -> float:
        if bid > 0 and ask > 0:
            return 0.5 * (bid + ask)
        return max(bid, ask, 0.0)

    def _update_book(self, sym: str, ven: str, bid: float, ask: float) -> None:
        now = self._now_ms()
        self.book[sym][ven] = (bid, ask, now)
        self.mid[sym] = self._mid_from(bid, ask)

    def _best_bid(self, sym: str, allowed: tuple[str, ...]) -> Optional[Tuple[str, float]]:
        best = None
        for v, (b, a, _) in self.book.get(sym, {}).items():
            if v not in allowed or b <= 0:
                continue
            if best is None or b > best[1]:
                best = (v, b)
        return best

    def _best_ask(self, sym: str, allowed: tuple[str, ...]) -> Optional[Tuple[str, float]]:
        best = None
        for v, (b, a, _) in self.book.get(sym, {}).items():
            if v not in allowed or a <= 0:
                continue
            if best is None or a < best[1]:
                best = (v, a)
        return best

    def _haircut_bps(self, legs: int) -> float:
        return legs * self.cfg.taker_fee_bps_per_leg + self.cfg.slippage_bps + self.cfg.latency_bps

    def _cooldown(self, sym: str, v1: str, v2: str, now: int) -> bool:
        if now - self._last_sym_ms[sym] < self.cfg.symbol_cooldown_ms:
            return True
        if now - self._last_ven_ms[v1] < self.cfg.venue_cooldown_ms:
            return True
        if now - self._last_ven_ms[v2] < self.cfg.venue_cooldown_ms:
            return True
        return False

    def _basket_nav(self) -> Optional[float]:
        """Compute basket NAV from last mids & weights (requires most names present)."""
        missing = [s for s in self.basket if self.mid.get(s, 0.0) <= 0.0]
        if len(missing) > self.cfg.max_missing_basket:
            return None
        nav = 0.0
        for w, s in zip(self.weights, self.basket):
            px = self.mid.get(s, 0.0)
            if px <= 0:  # skip missing few
                continue
            nav += w * px
        return nav if nav > 0 else None

    # ------------- main ---------------------
    def on_tick(self, tick: Dict[str, Any]) -> None:
        if self.cfg.hard_kill:
            return

        sym = (tick.get("symbol") or tick.get("s") or "").upper()
        ven = (tick.get("venue") or tick.get("v") or "").upper()

        # Only react to ETF, basket, or iNAV symbol
        if sym not in (self.etf,) + self.basket + ((self.inav_symbol,) if self.inav_symbol else tuple()):
            return

        # Normalize to bid/ask/mid
        bid = tick.get("bid")
        ask = tick.get("ask")
        px = tick.get("price") or tick.get("p") or tick.get("mid")

        if bid is None or ask is None:
            if px is not None:
                try:
                    px = float(px)
                except Exception:
                    return
                bid = bid or px
                ask = ask or px
            else:
                # no usable price
                return

        try:
            bid = float(bid or 0.0)
            ask = float(ask or 0.0)
        except Exception:
            return

        if bid <= 0 and ask <= 0:
            return
        if ask > 0 and bid > ask:
            # be conservative on bad ticks
            bid = ask = max(bid, ask)

        # Update books
        if sym != (self.inav_symbol or ""):
            # iNAV is mid-only; don't store venue book for it
            if sym == self.etf and (ven in self.cfg.venues_etf):
                self._update_book(sym, ven, bid, ask)
            elif sym in self.basket and (ven in self.cfg.venues_basket):
                self._update_book(sym, ven, bid, ask)
        else:
            # iNAV symbol: store as mid
            self.mid[self.inav_symbol] = self._mid_from(bid, ask) # type: ignore

        # We need ETF mid and NAV
        etf_best_bid = self._best_bid(self.etf, self.cfg.venues_etf)
        etf_best_ask = self._best_ask(self.etf, self.cfg.venues_etf)
        if not (etf_best_bid and etf_best_ask):
            return
        etf_mid = 0.5 * (etf_best_bid[1] + etf_best_ask[1])

        if self.cfg.use_inav and self.inav_symbol and self.mid.get(self.inav_symbol, 0.0) > 0:
            nav = self.mid[self.inav_symbol]
        else:
            nav = self._basket_nav()

        if not nav or nav <= 0:
            return

        # Premium/discount in bps vs NAV
        raw_edge_bps = (etf_mid - nav) / nav * 1e4

        # Approximate cost: 1 ETF leg + N basket legs (or 2 legs if using iNAV proxy trades both sides in equity)
        legs = 1 + (len(self.basket) if not self.cfg.use_inav else len(self.basket))
        net_edge_bps = raw_edge_bps - self._haircut_bps(legs)

        # Emit signal for allocators/UI (scale 10bps → 1.0)
        self.emit_signal(max(-1.0, min(1.0, net_edge_bps / 10.0)))

        if abs(net_edge_bps) < self.cfg.min_edge_bps:
            return

        # Decide direction
        now = self._now_ms()
        if net_edge_bps > 0:
            # ETF rich → SELL ETF, BUY basket
            etf_venue, etf_px = etf_best_bid  # sell at bid
            etf_venue = (self.cfg.forced_etf_venue or etf_venue).upper()
            # Cooldown uses first basket venue encountered
            best_buy = None
            for s in self.basket:
                ba = self._best_ask(s, self.cfg.venues_basket)
                if ba:
                    best_buy = ba
                    break
            if not best_buy:
                return
            if self._cooldown(self.etf, etf_venue, best_buy[0], now):
                return

            # Size
            qty_etf = max(1.0, self.cfg.target_notional / max(etf_mid, 1e-9))
            if qty_etf * etf_mid > self.cfg.max_gross_notional:
                return

            # Fire ETF leg
            self.order(self.etf, "sell", qty=qty_etf, order_type="market", venue=etf_venue,
                       mark_price=etf_px,
                       extra={"reason": "etf_nav_premium", "edge_bps": net_edge_bps})

            # Fire basket buys proportional to weights
            remaining = float(self.cfg.target_notional)
            for w, s in zip(self.weights, self.basket):
                px_i = self.mid.get(s, 0.0)
                ven_i_px = self._best_ask(s, self.cfg.venues_basket)
                if px_i <= 0 or not ven_i_px:
                    # fallback small qty if missing
                    self.order(s, "buy", qty=self.ctx.default_qty, order_type="market",
                               extra={"reason": "basket_buy_fallback"})
                    continue
                v = max(0.0, remaining * abs(w))
                qty_i = max(1.0, v / px_i)
                ven_i = (self.cfg.forced_equity_venue or ven_i_px[0]).upper()
                self.order(s, "buy", qty=qty_i, order_type="market", venue=ven_i,
                           mark_price=ven_i_px[1],
                           extra={"reason": "basket_buy", "w": w, "px": px_i})
            # Cooldowns
            self._last_sym_ms[self.etf] = now
            self._last_ven_ms[etf_venue] = now
            self._last_ven_ms[best_buy[0]] = now

        else:
            # ETF cheap → BUY ETF, SELL basket
            etf_venue, etf_px = etf_best_ask  # buy at ask
            etf_venue = (self.cfg.forced_etf_venue or etf_venue).upper()
            best_sell = None
            for s in self.basket:
                bb = self._best_bid(s, self.cfg.venues_basket)
                if bb:
                    best_sell = bb
                    break
            if not best_sell:
                return
            if self._cooldown(self.etf, etf_venue, best_sell[0], now):
                return

            qty_etf = max(1.0, self.cfg.target_notional / max(etf_mid, 1e-9))
            if qty_etf * etf_mid > self.cfg.max_gross_notional:
                return

            self.order(self.etf, "buy", qty=qty_etf, order_type="market", venue=etf_venue,
                       mark_price=etf_px,
                       extra={"reason": "etf_nav_discount", "edge_bps": net_edge_bps})

            for w, s in zip(self.weights, self.basket):
                px_i = self.mid.get(s, 0.0)
                ven_i_px = self._best_bid(s, self.cfg.venues_basket)
                if px_i <= 0 or not ven_i_px:
                    self.order(s, "sell", qty=self.ctx.default_qty, order_type="market",
                               extra={"reason": "basket_sell_fallback"})
                    continue
                v = max(0.0, self.cfg.target_notional * abs(w))
                qty_i = max(1.0, v / px_i)
                ven_i = (self.cfg.forced_equity_venue or ven_i_px[0]).upper()
                self.order(s, "sell", qty=qty_i, order_type="market", venue=ven_i,
                           mark_price=ven_i_px[1],
                           extra={"reason": "basket_sell", "w": w, "px": px_i})

            self._last_sym_ms[self.etf] = now
            self._last_ven_ms[etf_venue] = now
            self._last_ven_ms[best_sell[0]] = now


# ---------------------- optional runner ----------------------
if __name__ == "__main__":
    """
    Example quick attach:
      python -m backend.engine.strategies.etf_nav_arb
    Typically run via Strategy.run(...) elsewhere (e.g., "ticks.equities.us").
    """
    strat = ETFNavArb()
    # strat.run(stream="ticks.equities.us")