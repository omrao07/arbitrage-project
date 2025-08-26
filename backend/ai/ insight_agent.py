# backend/ai/insight_agent.py
"""
InsightAgent: fuse news + market + execution + risk into human-readable insights.

Listens to:
  - news.events            (normalized from news_yahoo.py / news_moneycontrol.py)
  - marks                  (price marks)
  - oms.fill               (fills)
  - exec.router.decisions  (router choices)
  - risk.var / risk.dd / risk.es  (risk metrics)
  - governor.events        (automatic actions taken)

Emits:
  - ai.insight  (short text + tags + confidence + refs)
  - ai.state    (optional rolling state snapshot)

No external dependencies required. If 'transformers' is installed, will use a tiny
sentiment pipeline; otherwise falls back to a lexicon-based scorer.

Message schemas (inputs are forgiving; best-effort extraction):
  news.events: {
      "ts_ms": 169..., "source": "yahoo|moneycontrol|...", "title": "...",
      "summary": "...", "tickers": ["RELIANCE.NS","AAPL"], "url": "...",
      "sentiment": { "score": -1..+1 }   # optional
  }
  marks: { "ts": 169..., "symbol": "RELIANCE.NS", "price": 2874.5 }
  exec.router.decisions: { "ts":..., "symbol":"AAPL", "route":"POV", "params": {...} }
  oms.fill: { "ts":..., "symbol":"AAPL","side":"buy","qty":100,"price":189.7,"strategy":"alpha.momo" }
  risk.var: { "ts":..., "firm_var_pct_nav": 0.028, "per_strategy": {"alpha.momo":0.007, ...} }
  risk.dd:  { "ts":..., "firm_dd": 0.043, "per_strategy": {"alpha.momo":0.012, ...} }
  governor.events: { "ts":..., "action":"reduce_gross", "args":{"frac":0.2}, "reason":"dd_ladder_0.10" }

Output example:
  ai.insight: {
    "ts_ms": ..., "kind":"news_impact", "summary":"RELIANCE Q1 beat; momo ↑; hedge: sell call spread",
    "tickers":["RELIANCE.NS"], "score":0.62, "confidence":0.71,
    "tags":["news","bullish","india","options-hedge"],
    "refs":{"news_id":"...","risk":"risk.var@ts","router":"POV"}
  }
"""

from __future__ import annotations

import os
import re
import time
import math
import json
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

# Your existing bus helpers
try:
    from backend.bus.streams import consume_stream, publish_stream, hset
except Exception as e:  # dev fallback for tests
    consume_stream = publish_stream = hset = None  # type: ignore

# Optional HF sentiment (auto-disabled if not installed)
_USE_HF = False
try:
    from transformers import pipeline  # type: ignore
    _USE_HF = True
except Exception:
    _USE_HF = False


# ----------------------------- Utilities -------------------------------------

_TICK = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z]{1,3})?\b")  # crude ticker snag e.g., AAPL, RELIANCE.NS

_LEX_POS = {
    "beat","surge","upgrade","outperform","record","raise","strong","growth","tailwind",
    "approval","win","acquire","profitable","guidance up","launch","partnership"
}
_LEX_NEG = {
    "miss","plunge","downgrade","underperform","fraud","probe","investigation","guidance cut",
    "halt","ban","recall","lawsuit","default","delay","layoff","shutdown"
}

def _utc_ms() -> int:
    return int(time.time() * 1000)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def softsig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ----------------------------- Sentiment -------------------------------------

class SentimentScorer:
    """
    Pluggable sentiment scoring.
    If transformers is present, use a small pipeline; else lexicon + heuristics.
    """
    def __init__(self):
        self._pipe = None
        if _USE_HF:
            try:
                self._pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            except Exception:
                self._pipe = None

    def score(self, title: str, summary: str = "") -> float:
        text = (title or "") + " " + (summary or "")
        if not text.strip():
            return 0.0

        if self._pipe is not None:
            try:
                out = self._pipe(text[:512])[0]  # {'label':'POSITIVE','score':0.98}
                s = out.get("score", 0.5)
                return s if out.get("label","POSITIVE").upper().startswith("POS") else -s
            except Exception:
                pass

        # fallback: lexicon hits
        t = text.lower()
        pos = sum(1 for k in _LEX_POS if k in t)
        neg = sum(1 for k in _LEX_NEG if k in t)
        raw = (pos - neg) / max(1.0, (pos + neg))
        # add tiny emphasis for numbers like "+12%" or "record"
        if re.search(r"\+\d+(\.\d+)?\s*%", t): raw += 0.1
        if "record" in t: raw += 0.05
        return clamp(raw, -1.0, 1.0)


# ----------------------------- Rolling State ---------------------------------

class RollingState:
    """
    Keep light rolling context per symbol and strategy for better insights.
    """
    def __init__(self, maxlen: int = 2000):
        self.marks: Dict[str, float] = {}
        self.returns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.fills_by_sym: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.router_last: Dict[str, Dict[str, Any]] = {}
        self.risk_snap: Dict[str, Any] = {}
        self.dd_snap: Dict[str, Any] = {}
        self.var_firm: float = 0.0
        self.dd_firm: float = 0.0
        self.events: deque = deque(maxlen=maxlen)

    def update_mark(self, sym: str, px: float):
        prev = self.marks.get(sym)
        self.marks[sym] = px
        if prev and prev > 0:
            r = (px - prev) / prev
            self.returns[sym].append(r)

    def update_fill(self, fill: Dict[str, Any]):
        sym = str(fill.get("symbol","")).upper()
        self.fills_by_sym[sym].append(fill)

    def update_router(self, dec: Dict[str, Any]):
        sym = str(dec.get("symbol","")).upper()
        self.router_last[sym] = dec

    def update_var(self, snap: Dict[str, Any]):
        self.var_firm = float(snap.get("firm_var_pct_nav", 0.0))
        self.risk_snap = snap

    def update_dd(self, snap: Dict[str, Any]):
        self.dd_firm = float(snap.get("firm_dd", 0.0))
        self.dd_snap = snap

    def as_state(self) -> Dict[str, Any]:
        return {
            "ts_ms": _utc_ms(),
            "marks": self.marks,
            "var": self.var_firm,
            "dd": self.dd_firm,
            "router_last": self.router_last,
        }


# ----------------------------- Insight Agent ---------------------------------

class InsightAgent:
    """
    Fuse disparate signals into concise, actionable insights.

    Usage:
        agent = InsightAgent()
        agent.run()  # blocking; loops over streams and emits ai.insight
    """
    def __init__(
        self,
        streams: Dict[str, str] | None = None,
        publish_stream_name: str = "ai.insight",
        state_stream_name: Optional[str] = "ai.state",
        region_hint: Optional[str] = None,
    ):
        self.streams = streams or {
            "news": "news.events",
            "marks": "marks",
            "fills": "oms.fill",
            "router": "exec.router.decisions",
            "risk_var": "risk.var",
            "risk_dd": "risk.dd",
            "gov": "governor.events",
        }
        self.out_stream = publish_stream_name
        self.state_stream = state_stream_name
        self.state = RollingState()
        self.sentiment = SentimentScorer()
        self.region = region_hint or os.getenv("REGION", "GLOBAL")

    # ---------- Core loop ----------
    def run(self, poll_ms: int = 250) -> None:
        """
        Blocking loop that polls all configured streams and emits insights.
        Relies on backend.bus.streams.consume_stream to support '$' tailing.
        """
        assert consume_stream and publish_stream, "backend.bus.streams not available"

        cursors = {k: "$" for k in self.streams}
        while True:
            # Pull each stream non-blocking-ish
            for key, stream in self.streams.items():
                for _, msg in consume_stream(stream, start_id=cursors[key], block_ms=poll_ms, count=100):
                    cursors[key] = "$"
                    try:
                        if isinstance(msg, str):
                            msg = json.loads(msg)
                    except Exception:
                        pass
                    self._ingest(key, msg) # type: ignore

            # Periodically publish rolling state
            if self.state_stream:
                publish_stream(self.state_stream, self.state.as_state())

    # ---------- Ingest ----------
    def _ingest(self, source: str, msg: Dict[str, Any]) -> None:
        if source == "news":
            self._handle_news(msg)
        elif source == "marks":
            sym = (msg.get("symbol") or msg.get("s") or "").upper()
            px = float(msg.get("price") or msg.get("p") or 0.0)
            if sym and px > 0:
                self.state.update_mark(sym, px)
        elif source == "fills":
            self.state.update_fill(msg)
            self._emit_fill_insight(msg)
        elif source == "router":
            self.state.update_router(msg)
        elif source == "risk_var":
            self.state.update_var(msg)
            self._emit_risk_insight(kind="var", snap=msg)
        elif source == "risk_dd":
            self.state.update_dd(msg)
            self._emit_risk_insight(kind="dd", snap=msg)
        elif source == "gov":
            self._emit_governor_insight(msg)

    # ---------- News → (sentiment, impacted tickers, suggestion) ----------
    def _handle_news(self, m: Dict[str, Any]) -> None:
        title = (m.get("title") or "").strip()
        summary = (m.get("summary") or "").strip()
        tickers = list({t.upper() for t in (m.get("tickers") or [])})
        if not tickers:
            # try to extract crude tickers from title/summary
            tickers = [t for t in _TICK.findall(title + " " + summary) if len(t) >= 3]

        s_news = m.get("sentiment", {}).get("score")
        if s_news is None:
            s_news = self.sentiment.score(title, summary)

        # Magnify/attenuate by live return if we have recent marks
        imp = 0.0
        for t in tickers[:4]:
            px = self.state.marks.get(t)
            hist = self.state.returns.get(t)
            if px and hist:
                # use short-term return average to modulate confidence
                rbar = sum(hist) / max(1, len(hist))
                imp += rbar
        imp = clamp(imp, -0.05, 0.05)
        # Confidence blends sentiment strength and data recency proxy
        conf = softsig(2.0 * s_news) * (0.6 + 4.0 * abs(imp))

        tags = ["news"]
        tags += (["bullish"] if s_news > 0.15 else ["bearish"] if s_news < -0.15 else ["neutral"])
        if self.region: tags.append(self.region.lower())
        if any(x.endswith(".NS") for x in tickers): tags.append("india")

        # Simple suggestion engine
        suggestion = self._suggest_trade(tickers, s_news)

        insight = {
            "ts_ms": m.get("ts_ms") or _utc_ms(),
            "kind": "news_impact",
            "summary": self._summarize_news(title, tickers, s_news, suggestion),
            "tickers": tickers,
            "score": float(clamp(s_news, -1.0, 1.0)),
            "confidence": float(clamp(conf, 0.0, 1.0)),
            "tags": tags + (["options-hedge"] if suggestion and "put" in suggestion.lower() else []),
            "refs": {"source": m.get("source"), "url": m.get("url")},
        }
        publish_stream(self.out_stream, insight) # type: ignore

    def _summarize_news(self, title: str, tickers: List[str], s: float, suggestion: Optional[str]) -> str:
        dir_word = "↑" if s > 0.15 else "↓" if s < -0.15 else "→"
        tk = ", ".join(tickers[:3]) or "market"
        base = f"{tk} {dir_word} — {title.strip()[:160]}"
        if suggestion:
            base += f" | idea: {suggestion}"
        return base

    def _suggest_trade(self, tickers: List[str], s: float) -> Optional[str]:
        if not tickers:
            return None
        t0 = tickers[0]
        # If bearish and options likely available → propose put spread; bullish → call spread
        if s <= -0.25:
            return f"buy {t0} put spread (cheap tail hedge)"
        if s >= 0.25:
            return f"buy {t0} call spread (momentum carry)"
        return None

    # ---------- Fills / Execution insights ----------
    def _emit_fill_insight(self, fill: Dict[str, Any]) -> None:
        sym = str(fill.get("symbol","")).upper()
        side = fill.get("side")
        qty  = float(fill.get("qty", 0))
        px   = float(fill.get("price", 0))
        strat = fill.get("strategy") or "unknown"
        route = self.state.router_last.get(sym, {}).get("route")
        last_px = self.state.marks.get(sym, px)
        slip_bps = (px - last_px) / last_px * 1e4 if last_px else 0.0

        tags = ["execution","fill"]
        if abs(slip_bps) > 8.0: tags.append("high-slippage")

        insight = {
            "ts_ms": _utc_ms(),
            "kind": "exec_fill",
            "summary": f"{sym} {side} {qty:g} @ {px:.4f} via {route or 'router'} | slippage {slip_bps:.1f} bps | strat {strat}",
            "tickers": [sym],
            "score": float(clamp(-abs(slip_bps)/20.0, -1.0, 1.0)),  # worse slippage → more negative
            "confidence": 0.7,
            "tags": tags,
            "refs": {"route": route, "strategy": strat},
        }
        publish_stream(self.out_stream, insight) # type: ignore

    # ---------- Risk / Governor insights ----------
    def _emit_risk_insight(self, kind: str, snap: Dict[str, Any]) -> None:
        if kind == "var":
            firm = float(snap.get("firm_var_pct_nav", 0.0))
            tag = "risk-var"
            sev = "⚠️" if firm >= 0.035 else "OK"
            summary = f"Firm VaR: {firm*100:.2f}% NAV ({sev})"
            score = -clamp((firm - 0.02) / 0.03, -1.0, 1.0)  # penalize if above ~2%
        else:
            dd = float(snap.get("firm_dd", 0.0))
            tag = "risk-dd"
            sev = "🚨" if dd >= 0.1 else "⚠️" if dd >= 0.05 else "OK"
            summary = f"Drawdown: {dd*100:.1f}% ({sev})"
            score = -clamp((dd - 0.03) / 0.10, -1.0, 1.0)

        insight = {
            "ts_ms": snap.get("ts") or _utc_ms(),
            "kind": tag,
            "summary": summary,
            "tickers": [],
            "score": float(score),
            "confidence": 0.8,
            "tags": ["risk","governance"],
            "refs": {},
        }
        publish_stream(self.out_stream, insight) # type: ignore

    def _emit_governor_insight(self, m: Dict[str, Any]) -> None:
        action = m.get("action")
        reason = m.get("reason")
        args   = m.get("args", {})
        summary = f"Governor: {action} {args} (reason: {reason})"
        insight = {
            "ts_ms": m.get("ts") or _utc_ms(),
            "kind": "governor",
            "summary": summary,
            "tickers": [],
            "score": 0.0,
            "confidence": 0.9,
            "tags": ["governor","risk-action"],
            "refs": {},
        }
        publish_stream(self.out_stream, insight) # type: ignore


# ----------------------------- CLI -------------------------------------------

def main():
    """
    Optional CLI entrypoint:
        python -m backend.ai.insight_agent
    """
    agent = InsightAgent()
    try:
        agent.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()