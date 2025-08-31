# backend/ai/agents/concrete/insight_agent.py
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ------------------------------------------------------------
# Framework fallbacks
# ------------------------------------------------------------
try:
    from ..core.base_agent import BaseAgent  # type: ignore
except Exception:
    class BaseAgent:
        name: str = "insight_agent"
        def plan(self, *a, **k): ...
        def act(self, *a, **k): ...
        def explain(self, *a, **k): ...
        def heartbeat(self, *a, **k): return {"ok": True}

# Real quotes skill (optional)
try:
    # expected: get_candles(symbol, interval="1m"|"5m"|"1h"|"1d", lookback=int) -> [{ts,o,h,l,c,v}, ...]
    from ..skills.market.quotes import get_candles  # type: ignore
except Exception:
    def get_candles(symbol: str, *, interval: str = "1d", lookback: int = 60) -> List[Dict[str, Any]]:
        """Synthetic candles so the agent runs without connectors."""
        random.seed(hash(symbol) & 0xFFFF)
        base = 100.0 + (hash(symbol) % 500) / 10.0
        out: List[Dict[str, Any]] = []
        px = base
        now = int(time.time() * 1000)
        for i in range(lookback):
            ret = random.gauss(0.0003, 0.012)
            px = max(0.5, px * (1.0 + ret))
            rng = abs(random.gauss(0, 0.01)) * px
            ts = now - (lookback - i) * 60_000
            out.append({"ts": ts, "o": px / (1 + ret), "h": px + rng, "l": max(0.5, px - rng), "c": px, "v": abs(random.gauss(0, 1))*1_000_0})
        return out

# Research factors (optional)
try:
    from ..skills.research.signals import compute_factors  # type: ignore
except Exception:
    def compute_factors(symbol: str, closes: List[float]) -> Dict[str, float]:
        if len(closes) < 5:
            return {"momentum_20d": 0.0, "vol_20d": 0.0, "zscore_5d": 0.0}
        mom = closes[-1] / max(1e-9, closes[-21] if len(closes) >= 21 else closes[0]) - 1.0
        mean20 = sum(closes[-20:]) / 20.0 if len(closes) >= 20 else sum(closes) / len(closes)
        vol20 = (sum((x - mean20) ** 2 for x in closes[-20:]) / max(1, min(20, len(closes)) - 1)) ** 0.5
        z5 = _zscore(closes[-5:]) or 0.0
        return {"momentum_20d": mom, "vol_20d": vol20, "zscore_5d": z5}

# News (optional)
try:
    from ..skills.news.news_yahoo import fetch_news  # type: ignore
except Exception:
    def fetch_news(symbol: str, limit: int = 3) -> List[Dict[str, Any]]:
        return [{"headline": f"{symbol}: (stub) No live news", "summary": "Wire not connected."}]

# ------------------------------------------------------------
# Data models
# ------------------------------------------------------------
@dataclass
class InsightRequest:
    symbols: List[str]
    interval: str = "1m"
    lookback: int = 120
    threshold_z: float = 2.5   # anomaly threshold on close z-score
    threshold_vol_jump: float = 2.0  # z-score threshold on volume
    notes: Optional[str] = None

@dataclass
class InsightItem:
    symbol: str
    last_price: float
    anomaly: Optional[str]
    zscore_price: Optional[float]
    zscore_volume: Optional[float]
    factors: Dict[str, Any]
    news: List[Dict[str, Any]]

@dataclass
class InsightResponse:
    generated_at: int
    interval: str
    lookback: int
    items: List[InsightItem]
    summary: str

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _zscore(xs: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    s2 = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    s = s2 ** 0.5
    if s <= 0:
        return None
    return (xs[-1] - m) / s

def _pct(x0: float, x1: float) -> float:
    return (x1 / max(1e-9, x0)) - 1.0

# ------------------------------------------------------------
# Insight Agent
# ------------------------------------------------------------
class InsightAgent(BaseAgent): # type: ignore
    """
    Detects noteworthy changes:
      • price/volume z-score spikes
      • factor shifts (momentum/vol)
      • attaches recent headlines
    Emits concise items + a portfolio-style summary.
    """

    name = "insight_agent"

    def plan(self, req: InsightRequest | Dict[str, Any]) -> InsightRequest:
        if isinstance(req, InsightRequest):
            return req
        return InsightRequest(
            symbols=list(req.get("symbols", [])),
            interval=req.get("interval", "1m"),
            lookback=int(req.get("lookback", 120)),
            threshold_z=float(req.get("threshold_z", 2.5)),
            threshold_vol_jump=float(req.get("threshold_vol_jump", 2.0)),
            notes=req.get("notes"),
        )

    def act(self, request: InsightRequest | Dict[str, Any]) -> InsightResponse:
        req = self.plan(request)
        items: List[InsightItem] = []
        flags: List[str] = []

        for sym in req.symbols:
            candles = get_candles(sym, interval=req.interval, lookback=req.lookback)
            if not candles:
                items.append(InsightItem(
                    symbol=sym, last_price=float("nan"),
                    anomaly="no-data", zscore_price=None, zscore_volume=None,
                    factors={}, news=[]
                ))
                continue

            closes = [float(c["c"]) for c in candles]
            vols   = [float(c["v"]) for c in candles]
            last_px = closes[-1]

            z_p = _zscore(closes[-min(len(closes), 60):])  # price z on recent window
            z_v = _zscore(vols[-min(len(vols), 60):])      # volume z on recent window
            factors = compute_factors(sym, closes)
            news = fetch_news(sym, limit=3)

            anomaly = None
            if z_p is not None and abs(z_p) >= req.threshold_z:
                anomaly = f"price-z={z_p:.2f}"
            if z_v is not None and abs(z_v) >= req.threshold_vol_jump:
                anomaly = (anomaly + "; " if anomaly else "") + f"vol-z={z_v:.2f}"

            items.append(InsightItem(
                symbol=sym,
                last_price=last_px,
                anomaly=anomaly,
                zscore_price=z_p,
                zscore_volume=z_v,
                factors=factors,
                news=news
            ))

            if anomaly:
                dir_txt = "↑" if (z_p or 0) > 0 else "↓"
                flags.append(f"{sym} {dir_txt} {anomaly}")

        summary = self._summarize(items, flags)

        return InsightResponse(
            generated_at=int(time.time() * 1000),
            interval=req.interval,
            lookback=req.lookback,
            items=items,
            summary=summary
        )

    # --------------------------------------------------------
    # Text helpers
    # --------------------------------------------------------
    def _summarize(self, items: List[InsightItem], flags: List[str]) -> str:
        if not items:
            return "No symbols analyzed."
        highlights = ", ".join(flags[:6]) if flags else "No z-score anomalies."
        # quick breadth: pct of names with positive last move
        ups = 0
        for it in items:
            # cheap last move: use last two closes if available
            # (if candles missing, skip)
            ups += 1 if (it.zscore_price or 0) > 0 else 0
        breadth = f"{ups}/{len(items)} positive z-move"
        return f"{highlights} | Breadth: {breadth}."

    def explain(self) -> str:
        return (
            "InsightAgent computes recent price/volume z-scores, basic factor reads, "
            "and attaches headlines to surface anomalies. Use it to drive alerts, "
            "news panels, and 'why now' badges in the terminal."
        )

    def heartbeat(self) -> Dict[str, Any]:
        return {"ok": True, "agent": self.name, "ts": int(time.time())}


# ------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    agent = InsightAgent()
    resp = agent.act({"symbols": ["AAPL", "MSFT", "SPY"], "interval": "1m", "lookback": 120})
    print(resp.summary)
    for it in resp.items:
        print(f"- {it.symbol}: px={it.last_price:.2f} anomaly={it.anomaly} zP={it.zscore_price} zV={it.zscore_volume}")