# backend/analytics/venue_spreads.py
"""
Venue Spreads & Cross-Venue Edge
--------------------------------
Maintains per-venue top-of-book, computes:
  • NBBO (national best bid/offer) per symbol
  • Venue edge (bid/ask distance to NBBO)
  • Cross-venue spread & inverted markets (arb hints)
  • Rolling stats (avg spread, p90, inverted rate)

Listens (shape is flexible, best effort):
  quotes.*  messages with at least:
    {
      "symbol": "AAPL" | "RELIANCE.NS",
      "venue": "NASDAQ" | "BSE" | "NSE" | "BINANCE",
      "bid": 190.01, "ask": 190.03,
      "ts":  1690000000000  # ms (or 'ts_ms')
    }

Publishes (when bus is available):
  - metrics.venue_spreads:
      {
        "ts_ms", "symbol", "nbbo": {"bid","ask","bid_venue","ask_venue"},
        "per_venue": { VENUE: {"bid","ask","mid","edge_bps_bid","edge_bps_ask"} },
        "cross": {"best_bid_venue","best_ask_venue","cross_spread_bps","inverted": bool}
      }
  - ai.insight (compact bullets on persistent inversions / wide spreads)

Also exposes a small in-process API for dashboards:
  get_snapshot(symbol) -> dict
  get_rollup(symbol, window=300) -> dict of rolling stats

CLI:
  python -m backend.analytics.venue_spreads --probe
  python -m backend.analytics.venue_spreads --run
"""

from __future__ import annotations

import math
import time
import json
from collections import defaultdict, deque
from typing import Any, Dict, Optional, Tuple, List

# Optional bus (graceful if missing)
try:
    from backend.bus.streams import consume_stream, publish_stream
except Exception:
    consume_stream = publish_stream = None  # type: ignore


def _utc_ms() -> int:
    return int(time.time() * 1000)

def _mid(b: float, a: float) -> Optional[float]:
    if b is None or a is None or b <= 0 or a <= 0:
        return None
    if a < b:  # inverted book from feed; swap for mid calc only
        b, a = a, b
    return (b + a) / 2.0

def _bps(base: float, x: float) -> float:
    if base is None or base <= 0 or x is None:
        return 0.0
    return (x - base) / base * 1e4


class VenueSpreads:
    """
    Keeps a small rolling window of top-of-book per symbol x venue.
    """
    def __init__(self, window: int = 600):
        # quotes[sym][venue] = deque of (ts, bid, ask) length ~window
        self.quotes: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=max(10, window))))
        # last computed snapshot for quick read
        self.last: Dict[str, Dict[str, Any]] = {}
        # rolling stats
        self.stats: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max(30, window)))  # (ts, spr_bps, inverted)

    # ------------- ingest -------------
    def on_quote(self, m: Dict[str, Any]) -> None:
        try:
            sym = (m.get("symbol") or m.get("s") or "").upper()
            ven = str(m.get("venue") or m.get("exchange") or m.get("v") or "")
            b = float(m.get("bid") if m.get("bid") is not None else m.get("b") or 0.0) # type: ignore
            a = float(m.get("ask") if m.get("ask") is not None else m.get("a") or 0.0) # type: ignore
            ts = int(m.get("ts_ms") or m.get("ts") or _utc_ms())
        except Exception:
            return
        if not sym or not ven or b <= 0 or a <= 0:
            return
        self.quotes[sym][ven].append((ts, b, a))
        snap = self._compute_snapshot(sym)
        if snap:
            self.last[sym] = snap
            # keep rolling spread & inverted info for stats
            spr = snap["nbbo"]["ask"] - snap["nbbo"]["bid"] if snap["nbbo"]["ask"] and snap["nbbo"]["bid"] else None
            if spr is not None and snap["nbbo"]["mid"] > 0:
                self.stats[sym].append((ts, (spr / snap["nbbo"]["mid"]) * 1e4, snap["cross"]["inverted"]))

            # publish if bus exists & meaningful change
            if publish_stream and (snap["cross"]["inverted"] or snap["cross"]["cross_spread_bps"] > 2.0):
                publish_stream("metrics.venue_spreads", snap)
                if snap["cross"]["inverted"]:
                    publish_stream("ai.insight", {
                        "ts_ms": snap["ts_ms"],
                        "kind": "venue_spread",
                        "summary": f"Inverted market {sym}: {snap['nbbo']['bid_venue']} bid > {snap['nbbo']['ask_venue']} ask",
                        "details": [f"Cross spread {snap['cross']['cross_spread_bps']:.1f} bps"],
                        "tags": ["venue","spread","inversion", sym]
                    })

    # ------------- compute -------------
    def _compute_snapshot(self, sym: str) -> Optional[Dict[str, Any]]:
        books = self.quotes.get(sym)
        if not books:
            return None

        best_bid = (-1.0, None)   # (px, venue)
        best_ask = (1e99, None)
        per_venue: Dict[str, Dict[str, float]] = {}

        # latest top per venue
        for ven, dq in books.items():
            if not dq:
                continue
            ts, b, a = dq[-1]
            per_venue[ven] = {"bid": b, "ask": a, "mid": _mid(b, a)} # type: ignore

            if b > 0 and b > best_bid[0]:
                best_bid = (b, ven)
            if a > 0 and a < best_ask[0]:
                best_ask = (a, ven)

        if best_bid[1] is None or best_ask[1] is None:
            return None

        nbbo_bid, nbbo_bid_ven = best_bid
        nbbo_ask, nbbo_ask_ven = best_ask
        inverted = nbbo_ask < nbbo_bid
        nbbo_mid = _mid(nbbo_bid, nbbo_ask)

        # venue edge vs NBBO
        for ven, row in per_venue.items():
            row["edge_bps_bid"] = _bps(nbbo_bid, row["bid"])   # >0 means venue has better bid than NBBO (rare unless local book)
            row["edge_bps_ask"] = _bps(row["ask"], nbbo_ask)   # >0 means venue has better ask (lower) than NBBO

        cross_spread_bps = 0.0
        if nbbo_mid:
            cross_spread_bps = (nbbo_ask - nbbo_bid) / nbbo_mid * 1e4

        snap = {
            "ts_ms": _utc_ms(),
            "symbol": sym,
            "nbbo": {
                "bid": nbbo_bid, "ask": nbbo_ask, "mid": nbbo_mid,
                "bid_venue": nbbo_bid_ven, "ask_venue": nbbo_ask_ven
            },
            "per_venue": per_venue,
            "cross": {
                "best_bid_venue": nbbo_bid_ven,
                "best_ask_venue": nbbo_ask_ven,
                "cross_spread_bps": float(cross_spread_bps),
                "inverted": bool(inverted),
            }
        }
        return snap

    # ------------- queries/statistics -------------
    def get_snapshot(self, symbol: str) -> Optional[Dict[str, Any]]:
        return self.last.get(symbol.upper())

    def get_rollup(self, symbol: str, window_ms: int = 300_000) -> Dict[str, Any]:
        sym = symbol.upper()
        now = _utc_ms()
        pts = [p for p in self.stats.get(sym, []) if now - p[0] <= window_ms]
        if not pts:
            return {"symbol": sym, "window_ms": window_ms, "avg_bps": None, "p90_bps": None, "inverted_rate": 0.0}
        arr = [x[1] for x in pts]
        arr_sorted = sorted(arr)
        p90 = arr_sorted[int(0.9 * (len(arr_sorted)-1))]
        inv_rate = sum(1 for _, _, inv in pts if inv) / len(pts)
        avg_bps = sum(arr) / len(arr)
        return {
            "symbol": sym,
            "window_ms": window_ms,
            "avg_bps": round(avg_bps, 2),
            "p90_bps": round(p90, 2),
            "inverted_rate": round(inv_rate, 4),
            "n": len(pts)
        }


# ----------------------------- runner -----------------------------

def run_loop():
    """
    Attach to bus and process quotes.* streams.
    We’ll scan a small set of common channels; customize as needed.
    """
    assert consume_stream, "bus streams not available"
    vs = VenueSpreads()
    cursors = {"q": "$"}

    # If your bus splits per asset, you can add more topics here
    topics = ["quotes.equities", "quotes.fno", "quotes.fx", "quotes.crypto", "quotes"]
    while True:
        for t in topics:
            try:
                for _, msg in consume_stream(t, start_id=cursors["q"], block_ms=200, count=500):
                    cursors["q"] = "$"
                    try:
                        if isinstance(msg, str):
                            msg = json.loads(msg)
                    except Exception:
                        continue
                    vs.on_quote(msg)
            except Exception:
                # topic might not exist; continue
                pass
        time.sleep(0.02)


# ----------------------------- CLI -----------------------------

def main():
    import argparse, random
    ap = argparse.ArgumentParser(description="Venue Spreads & Cross-Venue Edge")
    ap.add_argument("--run", action="store_true", help="Run against bus streams (quotes.*)")
    ap.add_argument("--probe", action="store_true", help="Synthetic probe to stdout")
    ap.add_argument("--symbol", type=str, default="RELIANCE.NS")
    args = ap.parse_args()

    if args.probe:
        vs = VenueSpreads()
        now = _utc_ms()
        sym = args.symbol.upper()
        # synthetic: two venues with slight differences + a brief inversion
        demo = [
            {"symbol": sym, "venue": "NSE", "bid": 2900.00, "ask": 2900.50, "ts": now},
            {"symbol": sym, "venue": "BSE", "bid": 2899.90, "ask": 2900.45, "ts": now+50},
            {"symbol": sym, "venue": "NSE", "bid": 2900.20, "ask": 2900.40, "ts": now+100},
            {"symbol": sym, "venue": "BSE", "bid": 2900.35, "ask": 2900.30, "ts": now+150},  # inverted
            {"symbol": sym, "venue": "NSE", "bid": 2900.10, "ask": 2900.25, "ts": now+200},
        ]
        for m in demo:
            vs.on_quote(m)
        snap = vs.get_snapshot(sym)
        roll = vs.get_rollup(sym, window_ms=10_000)
        print(json.dumps({"snapshot": snap, "rollup": roll}, indent=2))
        return

    if args.run:
        try:
            run_loop()
        except KeyboardInterrupt:
            pass
        return

    print("Nothing to do. Use --probe for a demo or --run to attach to bus.")

if __name__ == "__main__":
    main()