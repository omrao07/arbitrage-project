# backend/market/quotes.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

try:
    # Prefer your shared schema if present
    from backend.common.schemas import Quote as QuoteSchema # type: ignore
except Exception:
    @dataclass
    class QuoteSchema:
        symbol: str
        ts_ms: int
        bid: Optional[float] = None
        ask: Optional[float] = None
        bid_size: Optional[float] = None
        ask_size: Optional[float] = None
        venue: Optional[str] = None
        def mid(self) -> Optional[float]:
            if self.bid is not None and self.ask is not None and self.ask >= self.bid:
                return 0.5 * (self.bid + self.ask)
            return None
        def spread_bps(self) -> Optional[float]:
            m = self.mid()
            if m and self.ask is not None and self.bid is not None and m > 0:
                return (self.ask - self.bid) / m * 1e4
            return None


@dataclass
class NBBO:
    symbol: str
    ts_ms: int
    best_bid: Optional[float]
    best_ask: Optional[float]
    best_bid_sz: Optional[float]
    best_ask_sz: Optional[float]
    bid_venue: Optional[str]
    ask_venue: Optional[str]

    def mid(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None and self.best_ask >= self.best_bid:
            return 0.5 * (self.best_bid + self.best_ask)
        return None

    def spread_bps(self) -> Optional[float]:
        m = self.mid()
        if m and self.best_bid is not None and self.best_ask is not None and m > 0:
            return (self.best_ask - self.best_bid) / m * 1e4
        return None


class QuoteBook:
    """
    Tracks per-venue best quotes for a symbol and computes NBBO-like consolidation.

    API:
      • set_quote(venue, ts_ms, bid, ask, bid_size, ask_size)
      • drop_stale(now_ms)
      • nbbo(now_ms=None) -> NBBO
      • snapshot() -> dict of venue -> quote dict
      • venues() -> list[str]
    """
    def __init__(self, symbol: str, *, ttl_ms: int = 5_000):
        self.symbol = symbol
        self.ttl_ms = int(ttl_ms)
        self._by_venue: Dict[str, QuoteSchema] = {}
        self._last_ts: int = 0

    # ---- updates ----
    def set_quote(
        self,
        *,
        venue: str,
        ts_ms: int,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        bid_size: Optional[float] = None,
        ask_size: Optional[float] = None,
    ) -> None:
        q = QuoteSchema(
            symbol=self.symbol,
            ts_ms=int(ts_ms),
            bid=float(bid) if bid is not None else None,
            ask=float(ask) if ask is not None else None,
            bid_size=float(bid_size) if bid_size is not None else None,
            ask_size=float(ask_size) if ask_size is not None else None,
            venue=venue,
        )
        self._by_venue[venue] = q
        if q.ts_ms > self._last_ts:
            self._last_ts = q.ts_ms

    def drop_stale(self, now_ms: int) -> None:
        dead = [v for v, q in self._by_venue.items() if now_ms - q.ts_ms > self.ttl_ms]
        for v in dead:
            self._by_venue.pop(v, None)

    # ---- reads ----
    def venues(self) -> List[str]:
        return list(self._by_venue.keys())

    def nbbo(self, now_ms: Optional[int] = None) -> NBBO:
        if now_ms is not None:
            self.drop_stale(now_ms)

        best_bid = None
        best_ask = None
        best_bid_sz = None
        best_ask_sz = None
        bid_venue = None
        ask_venue = None
        ts_max = 0

        for v, q in self._by_venue.items():
            ts_max = max(ts_max, q.ts_ms)
            if q.bid is not None:
                if best_bid is None or q.bid > best_bid or (q.bid == best_bid and (q.bid_size or 0) > (best_bid_sz or 0)):
                    best_bid = q.bid
                    best_bid_sz = q.bid_size
                    bid_venue = v
            if q.ask is not None:
                if best_ask is None or q.ask < best_ask or (q.ask == best_ask and (q.ask_size or 0) > (best_ask_sz or 0)):
                    best_ask = q.ask
                    best_ask_sz = q.ask_size
                    ask_venue = v

        return NBBO(
            symbol=self.symbol,
            ts_ms=ts_max or self._last_ts,
            best_bid=best_bid,
            best_ask=best_ask,
            best_bid_sz=best_bid_sz,
            best_ask_sz=best_ask_sz,
            bid_venue=bid_venue,
            ask_venue=ask_venue,
        )

    def snapshot(self) -> Dict[str, Dict]:
        """All venue quotes as dict for UI/WS."""
        return {v: q.__dict__ for v, q in self._by_venue.items()}


class MultiSymbolQuotes:
    """
    Holds QuoteBook per symbol. Useful for a consolidated feed process.
    """
    def __init__(self, *, ttl_ms: int = 5_000):
        self.ttl_ms = int(ttl_ms)
        self._books: Dict[str, QuoteBook] = {}

    def upsert(self, symbol: str, venue: str, ts_ms: int,
               bid: Optional[float], ask: Optional[float],
               bid_size: Optional[float] = None, ask_size: Optional[float] = None) -> None:
        qb = self._books.get(symbol)
        if qb is None:
            qb = QuoteBook(symbol, ttl_ms=self.ttl_ms)
            self._books[symbol] = qb
        qb.set_quote(venue=venue, ts_ms=ts_ms, bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size)

    def nbbo(self, symbol: str, now_ms: Optional[int] = None) -> Optional[NBBO]:
        qb = self._books.get(symbol)
        if qb is None: return None
        return qb.nbbo(now_ms=now_ms)

    def venues(self, symbol: str) -> List[str]:
        qb = self._books.get(symbol)
        return qb.venues() if qb else []

    def snapshot(self, symbol: str) -> Dict[str, Dict]:
        qb = self._books.get(symbol)
        return qb.snapshot() if qb else {}


# ---- tiny smoke test ----
if __name__ == "__main__":  # pragma: no cover
    q = QuoteBook("AAPL", ttl_ms=2_000)
    q.set_quote(venue="NYSE", ts_ms=1000, bid=189.98, ask=190.02, bid_size=500, ask_size=400)
    q.set_quote(venue="ARCA", ts_ms=1001, bid=189.99, ask=190.03, bid_size=200, ask_size=600)
    nbbo = q.nbbo()
    print("NBBO:", nbbo)
    print("mid:", nbbo.mid(), "spread_bps:", nbbo.spread_bps())