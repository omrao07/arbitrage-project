# backend/ai/risk_explainer.py
"""
RiskExplainer
-------------
Turns raw risk signals (VaR, ES, Drawdown, breaches, stress results) into
human-readable insights and simple hedge suggestions.

Listens (best-effort; all optional):
  - risk.var                { firm_var_pct_nav, per_strategy, ts }
  - risk.es                 { firm_es_pct_nav, per_strategy, ts }
  - risk.dd                 { firm_dd, per_strategy, ts }
  - risk.limits.breach      { kind, level, metric, strategy|book|symbol, value, limit, ts }
  - pnl.snap                { firm_pnl, per_strategy, trailing:{day,week,month}, ts }
  - stress.results          { name, shocks[], pnl_total, per_symbol{}, per_bucket{}, ts }
  - exposures.snap          { per_symbol{qty,notional,ccy,asset_class,region}, per_bucket{}, ts }

Emits:
  - ai.insight              (concise bullets suitable for the right-rail UI)
  - ai.risk_explainer       (same payload; can be consumed by literacy dashboard)

Output example:
{
  "ts_ms": 169..., "kind":"risk_explainer",
  "summary":"VaR 3.1% (⚠️) — main drivers: alpha.momo, statarb.pairs; hedge: +INR calls / NIFTY puts",
  "details":[
     "VaR 3.1% of NAV (↑ 0.6pp vs 24h). Top contrib: alpha.momo (0.9%), statarb.pairs (0.7%).",
     "Drawdown 4.8% (OK). ES(97.5) = 4.9%.",
     "Regime: high vol equities; INR FX exposure high vs budget."
  ],
  "recommendations":[
     "Trim alpha.momo gross by 10% (within governor band).",
     "Hedge 20-30% INR exposure via USDINR futures or NIFTY put spread (1-2wk)."
  ],
  "tags":["risk","var","hedge","governor","india"],
  "refs":{"var_ts":169..., "dd_ts":169..., "es_ts":169...}
}
"""

from __future__ import annotations

import os
import json
import time
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---- Bus (degrades if unavailable) ------------------------------------------
try:
    from backend.bus.streams import consume_stream, publish_stream
except Exception:  # tests/dev
    consume_stream = publish_stream = None  # type: ignore


# ---- Utilities ---------------------------------------------------------------

def _utc_ms() -> int:
    return int(time.time() * 1000)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def pct(x: float, digits: int = 1) -> str:
    try:
        return f"{x*100:.{digits}f}%"
    except Exception:
        return "n/a"

def money(x: float, digits: int = 0) -> str:
    try:
        s = f"{x:,.{digits}f}"
        return s
    except Exception:
        return str(x)

def topn(d: Dict[str, float], n: int = 3) -> List[Tuple[str, float]]:
    return sorted([(k, float(v)) for k, v in d.items()], key=lambda kv: -abs(kv[1]))[:n]

def delta_pp(now: float, prev: float | None) -> float:
    if prev is None:
        return 0.0
    return (now - prev) * 100  # percentage points


# ---- Rolling state -----------------------------------------------------------

@dataclass
class RiskSnap:
    ts: int = 0
    firm: float = 0.0
    per_strategy: Dict[str, float] = field(default_factory=dict)

@dataclass
class DrawdownSnap:
    ts: int = 0
    firm: float = 0.0
    per_strategy: Dict[str, float] = field(default_factory=dict)

@dataclass
class ExposureSnap:
    ts: int = 0
    per_symbol: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # {sym:{notional, region, asset_class}}
    per_bucket: Dict[str, float] = field(default_factory=dict)           # e.g., {"FX:INR": 3_200_000}

class RollingState:
    def __init__(self):
        self.var = RiskSnap()
        self.es = RiskSnap()
        self.dd = DrawdownSnap()
        self.pnl_trailing: Dict[str, float] = {}   # {"day":..., "week":..., "month":...}
        self.expo = ExposureSnap()
        self.last_var_firm_prev: Optional[float] = None

    def update_var(self, msg: Dict[str, Any]):
        self.last_var_firm_prev = self.var.firm if self.var.firm else None
        self.var = RiskSnap(
            ts=int(msg.get("ts") or _utc_ms()),
            firm=float(msg.get("firm_var_pct_nav", 0.0)),
            per_strategy={k: float(v) for k, v in (msg.get("per_strategy") or {}).items()}
        )

    def update_es(self, msg: Dict[str, Any]):
        self.es = RiskSnap(
            ts=int(msg.get("ts") or _utc_ms()),
            firm=float(msg.get("firm_es_pct_nav", msg.get("firm_es", 0.0))),
            per_strategy={k: float(v) for k, v in (msg.get("per_strategy") or {}).items()}
        )

    def update_dd(self, msg: Dict[str, Any]):
        self.dd = DrawdownSnap(
            ts=int(msg.get("ts") or _utc_ms()),
            firm=float(msg.get("firm_dd", 0.0)),
            per_strategy={k: float(v) for k, v in (msg.get("per_strategy") or {}).items()}
        )

    def update_pnl(self, msg: Dict[str, Any]):
        tr = msg.get("trailing") or {}
        self.pnl_trailing = {k: float(v) for k, v in tr.items()}

    def update_exposure(self, msg: Dict[str, Any]):
        self.expo = ExposureSnap(
            ts=int(msg.get("ts") or _utc_ms()),
            per_symbol=msg.get("per_symbol") or {},
            per_bucket={k: float(v) for k, v in (msg.get("per_bucket") or {}).items()}
        )


# ---- Risk Explainer ----------------------------------------------------------

class RiskExplainer:
    """
    Converts risk telemetry to explanations + hedge suggestions.
    """
    def __init__(
        self,
        in_streams: Optional[Dict[str, str]] = None,
        out_stream: str = "ai.insight",
        mirror_stream: Optional[str] = "ai.risk_explainer",
        region_hint: Optional[str] = None,
        dd_ladder: Optional[List[float]] = None,     # drawdown thresholds (fraction)
        var_warn: float = 0.03,                      # 3% VaR warn level
        es_warn: float = 0.05                        # 5% ES warn level
    ):
        self.streams = in_streams or {
            "var": "risk.var",
            "es": "risk.es",
            "dd": "risk.dd",
            "breach": "risk.limits.breach",
            "pnl": "pnl.snap",
            "exp": "exposures.snap",
            "stress": "stress.results",
        }
        self.out = out_stream
        self.mirror = mirror_stream
        self.state = RollingState()
        self.region = (region_hint or os.getenv("REGION") or "GLOBAL").lower()
        self.dd_ladder = dd_ladder or [0.05, 0.10, 0.15]
        self.var_warn = float(var_warn)
        self.es_warn = float(es_warn)

    # --------- Loop ---------
    def run(self, poll_ms: int = 300):
        assert consume_stream and publish_stream, "bus streams not wired"
        cursors = {k: "$" for k in self.streams}
        while True:
            for key, stream in self.streams.items():
                for _, msg in consume_stream(stream, start_id=cursors[key], block_ms=poll_ms, count=200):
                    cursors[key] = "$"
                    try:
                        if isinstance(msg, str):
                            msg = json.loads(msg)
                    except Exception:
                        continue
                    self._ingest(key, msg)

    # --------- Ingest ---------
    def _ingest(self, key: str, msg: Dict[str, Any]):
        if key == "var":
            self.state.update_var(msg); self._emit_var()
        elif key == "es":
            self.state.update_es(msg); self._emit_es()
        elif key == "dd":
            self.state.update_dd(msg); self._emit_dd()
        elif key == "pnl":
            self.state.update_pnl(msg)
        elif key == "exp":
            self.state.update_exposure(msg)
        elif key == "breach":
            self._emit_breach(msg)
        elif key == "stress":
            self._emit_stress(msg)

    # --------- Emitters ---------
    def _emit_var(self):
        st = self.state
        v = st.var.firm
        dv_pp = delta_pp(v, st.last_var_firm_prev)
        top = topn(st.var.per_strategy, 3)
        sev = "🚨" if v >= (self.var_warn * 1.5) else "⚠️" if v >= self.var_warn else "OK"
        bullets = [
            f"VaR {pct(v, 1)} of NAV ({'↑' if dv_pp>0 else '↓' if dv_pp<0 else '→'} {abs(dv_pp):.1f}pp vs 24h). "
            + (f"Top contrib: " + ", ".join(f"{k} ({pct(val,1)})" for k, val in top) + "." if top else "")
        ]

        # Context from DD/ES
        bullets.append(f"Drawdown {pct(st.dd.firm,1)}; ES {pct(st.es.firm,1)}.")

        # Exposure hints
        hints = self._exposure_hints()
        bullets += hints["hints"]

        recs = self._hedge_suggestions(hints["hot_buckets"])
        self._publish(
            summary=f"VaR {pct(v,1)} ({sev}) — drivers: " + ", ".join(k for k,_ in top) + (f"; hedge: {recs[0]}" if recs else ""),
            details=bullets,
            recommendations=recs,
            tags=["risk","var", self.region] + (["india"] if any("INR" in b or ".NS" in b for b in hints["hot_buckets"]) else []),
            refs={"var_ts": st.var.ts, "dd_ts": st.dd.ts, "es_ts": st.es.ts}
        )

    def _emit_es(self):
        st = self.state
        e = st.es.firm
        sev = "🚨" if e >= (self.es_warn * 1.5) else "⚠️" if e >= self.es_warn else "OK"
        bullets = [
            f"Expected Shortfall (97.5) {pct(e,1)} ({sev}).",
            f"VaR {pct(st.var.firm,1)}, Drawdown {pct(st.dd.firm,1)}."
        ]
        self._publish(
            summary=f"ES {pct(e,1)} ({sev}) — tail risk snapshot.",
            details=bullets,
            recommendations=[],
            tags=["risk","es", self.region],
            refs={"es_ts": st.es.ts}
        )

    def _emit_dd(self):
        st = self.state
        d = st.dd.firm
        lvl = self._dd_level(d)
        sev = ["OK","⚠️","🚨","🛑"][lvl]
        actions = [
            "Monitor only.",
            "Tighten gross caps 10–20%; reduce risk-on tilts.",
            "Cut gross 25–40%; add crash hedges.",
            "Hard de-risk to core; freeze risk adds until recovery."
        ]
        bullets = [
            f"Drawdown {pct(d,1)} ({sev}). Ladder level {lvl}/{len(self.dd_ladder)}.",
            actions[min(lvl, len(actions)-1)]
        ]
        self._publish(
            summary=f"Drawdown {pct(d,1)} — {actions[min(lvl, len(actions)-1)]}",
            details=bullets,
            recommendations=(["Add protective puts (1–2wk) on index; trim highest VAR strategies."] if lvl >= 2 else []),
            tags=["risk","drawdown", self.region],
            refs={"dd_ts": st.dd.ts}
        )

    def _emit_breach(self, m: Dict[str, Any]):
        kind = (m.get("kind") or m.get("metric") or "limit").lower()
        value = float(m.get("value", 0.0))
        limit = float(m.get("limit", 0.0))
        subject = m.get("strategy") or m.get("book") or m.get("symbol") or "portfolio"
        action = self._breach_action(kind, value, limit, subject)
        summary = f"Limit breach: {kind} {value} > {limit} on {subject}. Action: {action}"
        self._publish(
            summary=summary,
            details=[summary],
            recommendations=[action],
            tags=["risk","breach"],
            refs={"ts": int(m.get("ts") or _utc_ms())}
        )

    def _emit_stress(self, res: Dict[str, Any]):
        name = res.get("name","scenario")
        pnl_total = float(res.get("pnl_total", 0.0))
        per_sym = {k: float(v) for k, v in (res.get("per_symbol") or {}).items()}
        worst = topn(per_sym, 5)
        bullets = [f"Stress '{name}': total PnL {money(pnl_total)}."]
        if worst:
            bullets.append("Worst symbols: " + ", ".join(f"{s} ({money(v)})" for s, v in worst))
        recs = []
        if pnl_total < 0:
            buckets = [b for b in (res.get("per_bucket") or {}).keys()]
            hints = self._hedge_suggestions(buckets)
            recs = hints
        self._publish(
            summary=f"Stress '{name}' → {money(pnl_total)}; hedge: {recs[0] if recs else 'review tilts'}",
            details=bullets,
            recommendations=recs,
            tags=["risk","stress", self.region],
            refs={"ts": int(res.get("ts") or _utc_ms())}
        )

    # --------- Heuristics / Hedges ---------
    def _dd_level(self, d: float) -> int:
        lvl = 0
        for i, thr in enumerate(self.dd_ladder, start=1):
            if d >= thr: lvl = i
        return lvl

    def _exposure_hints(self) -> Dict[str, Any]:
        """
        Inspect exposure buckets to find hot areas.
        Expect keys like 'FX:INR', 'EQ:US', 'EQ:IN', 'RATE:IN', etc.
        """
        buckets = self.state.expo.per_bucket or {}
        if not buckets:
            return {"hints": [], "hot_buckets": []}
        # take top absolute notional buckets
        tops = topn(buckets, 4)
        hints = []
        hot = []
        for b, v in tops:
            hot.append(b)
            tag = b.replace(":", " ")
            hints.append(f"Exposure heavy in {tag} (≈ {money(v)}).")
        return {"hints": hints, "hot_buckets": hot}

    def _hedge_suggestions(self, buckets: List[str]) -> List[str]:
        recs: List[str] = []
        for b in buckets[:3]:
            u = b.upper()
            if "FX:INR" in u or ("FX" in u and "INR" in u):
                recs.append("Hedge INR via USDINR futures (25–50% notional) or NIFTY put spread (1–2wk).")
            elif u.startswith("EQ:IN") or ".NS" in u:
                recs.append("Add NIFTY/BankNIFTY put spread or reduce high beta longs by 10–20%.")
            elif u.startswith("EQ:US"):
                recs.append("Add SPY put spread or short ES micro futures (5–10% notional).")
            elif u.startswith("RATE:"):
                recs.append("Receive rates via IR futures/swaps or add duration via bond ETFs.")
            elif u.startswith("COMMO") or "OIL" in u:
                recs.append("Use calendar put spreads or collars on key commodities.")
        # ensure unique and concise
        out = []
        for r in recs:
            if r not in out:
                out.append(r)
        return out[:3]

    def _breach_action(self, kind: str, value: float, limit: float, subject: str) -> str:
        if "var" in kind:
            return f"Trim gross on {subject} by 10–20% until VaR < {pct(limit,1)}."
        if "dd" in kind or "drawdown" in kind:
            return f"Auto-cut risk and add index puts (1–2wk) until DD < {pct(limit,1)}."
        if "leverage" in kind or "gross" in kind:
            return f"Reduce leverage on {subject} to comply with {pct(limit,1)} cap."
        return "Pause adds; review exposures and hedges."

    # --------- Publisher ---------
    def _publish(self, summary: str, details: List[str], recommendations: List[str], tags: List[str], refs: Dict[str, Any]):
        payload = {
            "ts_ms": _utc_ms(),
            "kind": "risk_explainer",
            "summary": summary[:240],
            "details": [d[:280] for d in details][:6],
            "recommendations": [r[:160] for r in recommendations][:4],
            "tags": sorted(list(set(tags)))[:8],
            "refs": refs
        }
        if publish_stream:
            publish_stream(self.out, payload)
            if self.mirror:
                publish_stream(self.mirror, payload)

# ---- Simple functional API ---------------------------------------------------

_explainer_singleton: Optional[RiskExplainer] = None

def explain_snapshot(var: Dict[str, Any] | None = None,
                     es: Dict[str, Any] | None = None,
                     dd: Dict[str, Any] | None = None,
                     exposures: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    One-shot helper: feed raw snapshots and get a single explanation dict back.
    Does not require the bus; useful for tests or synchronous UI calls.
    """
    rx = RiskExplainer(mirror_stream=None)
    if var: rx.state.update_var(var)
    if es:  rx.state.update_es(es)
    if dd:  rx.state.update_dd(dd)
    if exposures: rx.state.update_exposure(exposures)
    # Compose a single summary similar to _emit_var
    v = rx.state.var.firm; top = topn(rx.state.var.per_strategy, 3)
    hints = rx._exposure_hints()
    recs  = rx._hedge_suggestions(hints["hot_buckets"])
    return {
        "ts_ms": _utc_ms(),
        "kind": "risk_explainer",
        "summary": f"VaR {pct(v,1)} — drivers: " + ", ".join(k for k,_ in top) + (f"; hedge: {recs[0]}" if recs else ""),
        "details": [
            f"VaR {pct(v,1)}; Drawdown {pct(rx.state.dd.firm,1)}; ES {pct(rx.state.es.firm,1)}.",
            *hints["hints"]
        ],
        "recommendations": recs,
        "tags": ["risk","snapshot"],
        "refs": {"var_ts": rx.state.var.ts, "dd_ts": rx.state.dd.ts, "es_ts": rx.state.es.ts},
    }

# ---- CLI ---------------------------------------------------------------------

def main():
    """
    Run loop:
        python -m backend.ai.risk_explainer
    """
    try:
        rx = RiskExplainer()
        rx.run()
    except KeyboardInterrupt:
        pass
    except AssertionError as e:
        # Bus not available; print a single synthetic explanation as a hint
        demo = explain_snapshot(
            var={"firm_var_pct_nav": 0.031, "per_strategy": {"alpha.momo":0.009,"statarb.pairs":0.007}},
            es={"firm_es_pct_nav": 0.048},
            dd={"firm_dd": 0.048},
            exposures={"per_bucket": {"FX:INR": 2_500_000, "EQ:US": 1_200_000}}
        )
        print(json.dumps(demo, indent=2))

if __name__ == "__main__":
    main()