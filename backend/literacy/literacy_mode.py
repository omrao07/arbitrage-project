# backend/literacy/literacy_mode.py
"""
Learning Mode (Financial Literacy)
----------------------------------
Transforms complex portfolio/risk/news data into plain-English insights.

What it does:
- Summarizes today's performance in simple language
- Explains drivers: by strategy, by region, by top symbols
- Emits risk warnings (DD/VaR/exposure) and "what this means" tips
- Converts news events + sentiment into easy signals for humans
- Produces JSON-safe dicts you can render in web/CLI

Inputs it can consume (all optional):
- PnL snapshot: from backend.analytics.pnl_attribution.PnLAttributor.snapshot()
- Risk metrics: from backend.analytics.risk_metrics.RiskMetrics.snapshot()
- Positions & exposures: dicts {symbol: qty}, {region: exposure}, etc.
- Recent news events: List[NewsEvent] (with optional sentiment scores you attach)

Usage:
    lm = LearningMode()
    story = lm.explain_day(
        pnl_snapshot=pnl_snapshot,
        risk_snapshot=risk_snapshot,
        positions=positions_dict,
        region_exposure={"india": 0.42, "us": 0.58},
        news_events=recent_news_list  # optional
    )
    # story is a dict with keys: headline, summary, drivers, risks, tips, feed
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Optional type import; works without news module too.
try:
    from backend.ingestion.news.news_base import NewsEvent # type: ignore
except Exception:
    @dataclass
    class NewsEvent:  # minimal stub
        id: str
        source: str
        headline: str
        url: str
        published_at: float
        summary: str = ""
        symbol: Optional[str] = None


# ---------------- Plain-English helpers ----------------

def _fmt_pct(x: float, places: int = 2) -> str:
    try:
        return f"{x*100:.{places}f}%"
    except Exception:
        return "0.00%"

def _signword(x: float) -> str:
    return "up" if x > 0 else ("down" if x < 0 else "flat")

def _nice_num(x: float) -> str:
    # Short currency-like printing without committing to a currency
    abx = abs(x)
    sign = "-" if x < 0 else ""
    if abx >= 1_000_000_000:
        return f"{sign}{abx/1_000_000_000:.2f}B"
    if abx >= 1_000_000:
        return f"{sign}{abx/1_000_000:.2f}M"
    if abx >= 1_000:
        return f"{sign}{abx/1_000:.2f}K"
    return f"{x:.2f}"

def _today_str(ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts or time.time()))

def _safe(d: Dict, *path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


# ---------------- Core class ----------------

class LearningMode:
    def __init__(
        self,
        dd_warn: float = -0.03,     # warn if drawdown worse than -3%
        var_warn: float = 0.02,     # warn if daily VaR > 2% of equity
        expo_warn: float = 0.60,    # warn if any single region > 60% exposure
        top_n_drivers: int = 3,
        news_max: int = 6,
    ):
        self.dd_warn = dd_warn
        self.var_warn = var_warn
        self.expo_warn = expo_warn
        self.top_n = top_n_drivers
        self.news_max = news_max

    # -------- Public API --------

    def explain_day(
        self,
        pnl_snapshot: Optional[Dict[str, Any]] = None,
        risk_snapshot: Optional[Dict[str, Any]] = None,
        positions: Optional[Dict[str, Any]] = None,
        region_exposure: Optional[Dict[str, float]] = None,
        news_events: Optional[List[NewsEvent]] = None,
        equity_now: Optional[float] = None,
        equity_start: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Builds a Learning Mode story:
        - headline: one-liner for the day
        - summary: 2-3 sentences in plain English
        - drivers: top strategies/regions/symbols contributing
        - risks: list of warnings with simple explanations
        - tips: short literacy notes to teach the audience
        - feed: simplified news bullets (Bullish/Bearish/Neutral)
        """
        # --- compute simple perf today ---
        if equity_now is None and pnl_snapshot is not None:
            # approximate equity from totals if available: realized+unrealized
            tot = _safe(pnl_snapshot, "totals", default={}) or {}
            equity_now = float(tot.get("realized", 0.0) + tot.get("unrealized", 0.0))
        # fallback
        eq_now = float(equity_now or 0.0)

        # Pull max drawdown (if present) and VaR from risk snapshot
        total_risk = risk_snapshot.get("total") if isinstance(risk_snapshot, dict) and "total" in risk_snapshot \
                     else risk_snapshot.get("total", {}) if isinstance(risk_snapshot, dict) \
                     else None

        # Some users will pass RiskMetrics.snapshot() which returns {strategy: {...}}
        # Try to compute a blended "total" if missing.
        if not total_risk and isinstance(risk_snapshot, dict):
            # average across keys
            vals = [v for v in risk_snapshot.values() if isinstance(v, dict) and "vol" in v]
            if vals:
                import statistics as _st
                total_risk = {
                    "sharpe": _st.fmean([v.get("sharpe", 0.0) for v in vals]),
                    "max_drawdown": _st.fmean([v.get("max_drawdown", 0.0) for v in vals]),
                    "var_95": _st.fmean([v.get("var_95", 0.0) for v in vals]),
                    "vol": _st.fmean([v.get("vol", 0.0) for v in vals]),
                }

        # --- drivers ---
        drivers = self._drivers(pnl_snapshot)

        # --- risks & warnings ---
        risks = self._risks(total_risk, region_exposure)

        # --- news bullets ---
        feed = self._news_bullets(news_events or [])

        # --- headline + summary ---
        headline, summary = self._headline_summary(drivers, risks, eq_now)

        # --- tips (financial literacy one-liners) ---
        tips = self._tips(risks, drivers)

        return {
            "date": _today_str(),
            "headline": headline,
            "summary": summary,
            "drivers": drivers,
            "risks": risks,
            "tips": tips,
            "feed": feed,
        }

    # -------- Internals --------

    def _drivers(self, pnl_snapshot: Optional[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        if not pnl_snapshot:
            return {"strategies": [], "regions": [], "symbols": []}

        def topn(bucket: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
            # rank by total pnl = realized + unrealized - fees
            rows = []
            for k, v in bucket.items():
                pnl = float(v.get("realized", 0.0) + v.get("unrealized", 0.0) - v.get("fees", 0.0))
                rows.append({"name": str(k), "pnl": pnl})
            rows.sort(key=lambda r: r["pnl"], reverse=True)
            return rows[: self.top_n]

        by_strat = _safe(pnl_snapshot, "by_strategy", default={}) or {}
        by_region = _safe(pnl_snapshot, "by_region", default={}) or {}
        by_symbol = _safe(pnl_snapshot, "by_symbol", default={}) or {}

        return {
            "strategies": topn(by_strat),
            "regions": topn(by_region),
            "symbols": topn(by_symbol),
        }

    def _risks(self, total_risk: Optional[Dict[str, float]], region_exposure: Optional[Dict[str, float]]) -> List[Dict[str, Any]]:
        risks: List[Dict[str, Any]] = []

        if total_risk:
            dd = float(total_risk.get("max_drawdown", 0.0) or 0.0)  # negative number
            var = float(total_risk.get("var_95", 0.0) or 0.0)
            if dd <= self.dd_warn:
                risks.append({
                    "type": "drawdown",
                    "level": "warning",
                    "detail": f"Max drawdown {dd:.2%} exceeds comfort band",
                    "explain": "Drawdown means how far your equity fell from its peak. Big drawdowns are stressful and risky."
                })
            if abs(var) >= self.var_warn:
                risks.append({
                    "type": "var",
                    "level": "warning",
                    "detail": f"Daily VaR around {abs(var):.2%}",
                    "explain": "VaR estimates a worst-case daily loss with 95% confidence. Bigger VaR means larger daily swings."
                })

        if region_exposure:
            # any region above threshold?
            for reg, w in sorted(region_exposure.items(), key=lambda kv: kv[1], reverse=True):
                if w >= self.expo_warn:
                    risks.append({
                        "type": "concentration",
                        "level": "caution",
                        "detail": f"High concentration in {reg}: {_fmt_pct(w)} of portfolio",
                        "explain": "When too much is in one region, a local shock can hit your whole portfolio at once."
                    })
                    break

        if not risks:
            risks.append({
                "type": "ok",
                "level": "info",
                "detail": "Risk looks reasonable today",
                "explain": "Diversification and position sizes appear balanced for normal market conditions."
            })
        return risks

    def _news_bullets(self, events: List[NewsEvent]) -> List[Dict[str, Any]]:
        # Expect sentiment fields attached upstream if you want (e.g., ev.meta['sentiment'])
        # We'll present a simple label based on a 'sentiment_score' in [-1,1] if present.
        out: List[Dict[str, Any]] = []
        # sort most recent first
        events = sorted(events, key=lambda e: getattr(e, "published_at", 0.0), reverse=True)[: self.news_max]
        for ev in events:
            score = 0.0
            label = "Neutral"
            # optional: ev.raw/meta may contain sentiment; check common places
            s = None
            raw = getattr(ev, "raw", None)
            if raw and isinstance(raw, dict):
                s = raw.get("sentiment_score") or raw.get("sentiment")
            if s is None and hasattr(ev, "score"):
                s = getattr(ev, "score")
            try:
                score = float(s) # type: ignore
            except Exception:
                score = 0.0

            if score >= 0.25:
                label = "Bullish"
            elif score <= -0.25:
                label = "Bearish"

            out.append({
                "when": time.strftime("%H:%M", time.localtime(getattr(ev, "published_at", time.time()))),
                "label": label,
                "headline": getattr(ev, "headline", ""),
                "symbol": getattr(ev, "symbol", None),
                "source": getattr(ev, "source", ""),
                "score": round(score, 2),
                "url": getattr(ev, "url", ""),
            })
        return out

    def _headline_summary(self, drivers: Dict[str, List[Dict[str, Any]]], risks: List[Dict[str, Any]], equity_now: float):
        # Build a human one-liner like: "Portfolio up today; top help from India & mean_rev; risk normal."
        top_reg = drivers["regions"][0]["name"] if drivers["regions"] else None
        top_strat = drivers["strategies"][0]["name"] if drivers["strategies"] else None

        risk_line = "risk looks normal"
        for r in risks:
            if r["level"] in ("warning", "caution"):
                if r["type"] == "drawdown":
                    risk_line = "drawdown is elevated"
                elif r["type"] == "var":
                    risk_line = "daily risk is elevated"
                elif r["type"] == "concentration":
                    risk_line = "concentration is high"
                break

        # headline
        head_bits = ["Portfolio update"]
        if top_reg:
            head_bits.append(f"— {top_reg.capitalize()} led")
        if top_strat:
            head_bits.append(f"— strategy: {top_strat}")
        headline = " ".join(head_bits)

        # summary (2–3 sentences)
        driver_bits = []
        if drivers["regions"]:
            regs = ", ".join([d["name"] for d in drivers["regions"][:2]])
            driver_bits.append(f"Top regions: {regs}.")
        if drivers["strategies"]:
            strs = ", ".join([d["name"] for d in drivers["strategies"][:2]])
            driver_bits.append(f"Top strategies: {strs}.")

        summary = f"Today’s {headline.lower()}; {risk_line}. " + " ".join(driver_bits)
        return headline, summary

    def _tips(self, risks: List[Dict[str, Any]], drivers: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        tips: List[str] = []
        for r in risks:
            if r["type"] == "drawdown":
                tips.append("Tip: Smaller position sizes and wider diversification can reduce drawdowns.")
            if r["type"] == "var":
                tips.append("Tip: VaR rises when volatility rises. Consider reducing leverage in volatile markets.")
            if r["type"] == "concentration":
                tips.append("Tip: Add positions from other regions/sectors to spread risk.")
        if not tips:
            tips.append("Tip: Stay diversified and size positions so a single loss never sinks the portfolio.")
        # If a single symbol dominates drivers, add a teaching moment
        if drivers["symbols"]:
            sym = drivers["symbols"][0]["name"]
            tips.append(f"Note: {sym} drove performance. Relying on one name increases idiosyncratic risk.")
        return tips