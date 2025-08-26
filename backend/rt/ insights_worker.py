# backend/insights/insights_worker.py
"""
Insights Worker
---------------
Continuously consumes events (news, PnL, risk, OMS/TCA, strategy metrics),
detects interesting patterns, and emits concise human-readable insights.

Inputs (best-effort; any subset)
- Streams (via backend.bus.streams):
    * news.moneycontrol, news.yahoo, news.sentiment
    * pnl.events, risk.metrics, risk.bank_stress, risk.contagion, risk.sovereign
    * oms.router, tca.metrics, engine.metrics, engine.strategy
- Hashes (optional snapshots):
    * strategy:signal / :vol / :drawdown / strategy:health
    * esg:scores

Outputs
- Stream: insights.events  ({"ts_ms","severity","topic","title","summary","tags","refs"})
- Hash:   insights:last::<topic>  (latest insight per topic for UI)

CLI
  python -m backend.insights.insights_worker --probe
  python -m backend.insights.insights_worker --run
  python -m backend.insights.insights_worker --run --topics news,pnl,risk,oms
"""

from __future__ import annotations

import json
import math
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# -------- Optional deps / glue (graceful fallbacks) ----------
try:
    import redis  # noqa: F401
except Exception:
    redis = None  # type: ignore

try:
    from backend.bus.streams import consume_stream, publish_stream, hset # type: ignore
except Exception:
    # Minimal fallbacks for probe mode
    def consume_stream(stream: str, start_id: str = "$", block_ms: int = 1000, count: int = 100):
        if False:
            yield None
        return []
    def publish_stream(stream: str, payload: Dict[str, Any]):
        print(f"[INSIGHTS-> {stream}] {payload}")
    def hset(key: str, field: str, value: Any):
        pass

# ------------- Small utils -------------
def _now_ms() -> int: return int(time.time() * 1000)

@dataclass
class WorkerConfig:
    topics: Sequence[str] = ("news","pnl","risk","oms")
    poll_ms: int = 500
    min_gap_ms_same_title: int = 60_000  # de-dup by title within 60s
    sentiment_heat_threshold: float = 0.7
    pnl_spike_sigma: float = 3.0
    slippage_bps_warn: float = 20.0
    drawdown_warn: float = 0.06
    dd_kill: float = 0.10
    vol_warn: float = 0.35
    sovereign_spread_warn_bps: float = 150
    bank_breach_emit: bool = True
    keep_last_n_per_topic: int = 50

    @staticmethod
    def from_env() -> "WorkerConfig":
        def _f(name, typ, default):
            v = os.getenv(name)
            if v is None: return default
            try:
                if typ is int: return int(v)
                if typ is float: return float(v)
                if typ is bool: return v.lower() in ("1","true","yes","on")
                if typ is list: return [t.strip() for t in v.split(",") if t.strip()]
            except Exception:
                pass
            return default
        return WorkerConfig(
            topics=tuple(_f("INS_TOPICS", list, []) or ("news","pnl","risk","oms")), # type: ignore
            poll_ms=_f("INS_POLL_MS", int, 500), # type: ignore
            min_gap_ms_same_title=_f("INS_GAP_MS", int, 60_000), # type: ignore
            sentiment_heat_threshold=_f("INS_SENT_HEAT", float, 0.7), # type: ignore
            pnl_spike_sigma=_f("INS_PNL_SIGMA", float, 3.0), # type: ignore
            slippage_bps_warn=_f("INS_SLIP_WARN", float, 20.0), # type: ignore
            drawdown_warn=_f("INS_DD_WARN", float, 0.06), # type: ignore
            dd_kill=_f("INS_DD_KILL", float, 0.10), # type: ignore
            vol_warn=_f("INS_VOL_WARN", float, 0.35), # type: ignore
            sovereign_spread_warn_bps=_f("INS_SOV_WARN", float, 150.0), # type: ignore
            bank_breach_emit=_f("INS_BANK_BREACH", bool, True), # type: ignore
            keep_last_n_per_topic=_f("INS_KEEP_N", int, 50), # type: ignore
        )

# ------------- Insight type -------------
@dataclass
class Insight:
    severity: str
    topic: str
    title: str
    summary: str
    tags: List[str] = field(default_factory=list)
    refs: Dict[str, Any] = field(default_factory=dict)  # compact payload refs

    def as_dict(self) -> Dict[str, Any]:
        return {"ts_ms": _now_ms(), "severity": self.severity, "topic": self.topic,
                "title": self.title, "summary": self.summary, "tags": self.tags, "refs": self.refs}

# ------------- Stateful helpers -------------
class Deduper:
    def __init__(self, gap_ms: int):
        self.gap = gap_ms
        self.last: Dict[str, int] = {}
    def ok(self, key: str) -> bool:
        now = _now_ms()
        t = self.last.get(key, 0)
        if now - t < self.gap:
            return False
        self.last[key] = now
        return True

class RollingPNL:
    def __init__(self, maxn: int = 200):
        self.buf: List[float] = []
        self.maxn = maxn
    def push(self, x: float):
        self.buf.append(float(x))
        if len(self.buf) > self.maxn:
            self.buf.pop(0)
    def stats(self) -> Tuple[float,float]:
        if not self.buf: return (0.0, 1.0)
        m = sum(self.buf)/len(self.buf)
        v = sum((y-m)*(y-m) for y in self.buf) / max(1, len(self.buf)-1)
        sd = math.sqrt(max(1e-9, v))
        return (m, sd)

# ------------- Worker core -------------
class InsightsWorker:
    def __init__(self, cfg: Optional[WorkerConfig] = None):
        self.cfg = cfg or WorkerConfig.from_env()
        self._running = False
        self._dedup = Deduper(self.cfg.min_gap_ms_same_title)
        self._pnl_roll: Dict[str, RollingPNL] = {}  # per-strategy

        # cursors per stream
        self._cursors: Dict[str, str] = {}

        # stream map by topic
        self._streams_by_topic: Dict[str, List[str]] = {
            "news": ["news.moneycontrol", "news.yahoo", "news.sentiment"],
            "pnl":  ["pnl.events", "engine.metrics"],
            "risk": ["risk.bank_stress", "risk.contagion", "risk.sovereign", "risk.metrics"],
            "oms":  ["oms.router", "tca.metrics"]
        }

    # ---- main loop ----
    def start(self):
        self._running = True
        self._install_signals()
        try:
            while self._running:
                for topic in self.cfg.topics:
                    for stream in self._streams_by_topic.get(topic, []):
                        self._drain_stream(topic, stream)
                time.sleep(max(0.05, self.cfg.poll_ms/1000.0))
        except KeyboardInterrupt:
            pass

    def stop(self):
        self._running = False

    # ---- consume one stream tick ----
    def _drain_stream(self, topic: str, stream: str):
        cursor = self._cursors.get(stream, "$")
        try:
            for _id, raw in consume_stream(stream, start_id=cursor, block_ms=1, count=200): # type: ignore
                ev = raw if isinstance(raw, dict) else self._safe_json(raw)
                self._handle_event(topic, stream, ev)
            self._cursors[stream] = "$"
        except Exception:
            # keep going even if a stream isn’t available
            pass

    # ---- per-topic handlers ----
    def _handle_event(self, topic: str, stream: str, ev: Dict[str, Any]):
        if topic == "news":
            self._on_news(stream, ev)
        elif topic == "pnl":
            self._on_pnl(stream, ev)
        elif topic == "risk":
            self._on_risk(stream, ev)
        elif topic == "oms":
            self._on_oms(stream, ev)

    # ---- detectors ----
    def _on_news(self, stream: str, ev: Dict[str, Any]):
        title = (ev.get("title") or ev.get("headline") or "").strip()
        sym   = (ev.get("symbol") or ev.get("ticker") or ev.get("symbols") or "") or ""
        senti = self._to_float(ev.get("sentiment"))
        heat  = self._to_float(ev.get("heat") or ev.get("news_heat"))

        if heat is not None and heat >= self.cfg.sentiment_heat_threshold:
            t = f"🔥 High-impact news: {sym or 'Market'}"
            if self._dedup.ok(f"news:{sym}:{title or t}"):
                ins = Insight(
                    severity="high",
                    topic="news",
                    title=title or t,
                    summary=f"News heat={round(heat,2)}; sentiment={round(senti,2) if senti is not None else 'n/a'}",
                    tags=["news","high_impact"] + ([sym] if sym else []),
                    refs={"stream": stream, "sym": sym, "heat": heat, "sentiment": senti}
                )
                self._emit(ins, key=f"news::{sym or 'market'}")

    def _on_pnl(self, stream: str, ev: Dict[str, Any]):
        name = ev.get("strategy") or ev.get("name") or "unknown"
        pnl  = self._to_float(ev.get("pnl") or ev.get("pnl_delta") or ev.get("pnl_after_tax"))
        if pnl is None:
            return
        roll = self._pnl_roll.setdefault(name, RollingPNL())
        roll.push(pnl)
        m, sd = roll.stats()
        if sd <= 1e-9:
            return
        z = (pnl - m) / sd
        if abs(z) >= self.cfg.pnl_spike_sigma:
            sev = "high" if z < 0 else "medium"
            sign = "loss" if z < 0 else "gain"
            title = f"{name}: {sign} spike (z={z:.1f})"
            if self._dedup.ok(f"pnl:{name}:{sign}:{int(z)}"):
                ins = Insight(
                    severity=sev, topic="pnl", title=title,
                    summary=f"PnL={pnl:,.0f} vs mean {m:,.0f} (σ={sd:,.0f})",
                    tags=["pnl","anomaly",name],
                    refs={"stream": stream, "strategy": name, "pnl": pnl, "z": z}
                )
                self._emit(ins, key=f"pnl::{name}")

    def _on_risk(self, stream: str, ev: Dict[str, Any]):
        dd  = self._to_float(ev.get("dd") or ev.get("drawdown"))
        vol = self._to_float(ev.get("vol") or ev.get("volatility"))
        slip= self._to_float(ev.get("slip_bps") or ev.get("slippage_bps"))
        strat = ev.get("strategy") or ev.get("bank") or ev.get("country") or "system"

        if dd is not None and dd >= self.cfg.dd_kill:
            self._emit(Insight(
                severity="critical", topic="risk",
                title=f"{strat}: drawdown {dd:.1%} (kill threshold)",
                summary="Kill-switch threshold crossed. Governor should pause or de-risk.",
                tags=["risk","drawdown","governor",strat],
                refs={"stream":stream,"dd":dd}
            ), key=f"risk::dd::{strat}")
        elif dd is not None and dd >= self.cfg.drawdown_warn:
            self._emit(Insight(
                severity="high", topic="risk",
                title=f"{strat}: drawdown {dd:.1%}",
                summary="Elevated drawdown; consider reducing exposure or widening stops.",
                tags=["risk","drawdown",strat],
                refs={"stream":stream,"dd":dd}
            ), key=f"risk::dd::{strat}")

        if vol is not None and vol >= self.cfg.vol_warn:
            self._emit(Insight(
                severity="medium", topic="risk",
                title=f"{strat}: volatility {vol:.2f} (high)",
                summary="Realized/est. vol elevated; tighten risk and throttle router.",
                tags=["risk","vol",strat],
                refs={"stream":stream,"vol":vol}
            ), key=f"risk::vol::{strat}")

        if "risk.sovereign" in stream:
            sp_bps = self._to_float(ev.get("spread_bps"))
            if sp_bps and sp_bps >= self.cfg.sovereign_spread_warn_bps:
                cty = ev.get("country","CTY")
                self._emit(Insight(
                    severity="high", topic="risk",
                    title=f"{cty}: sovereign spread {sp_bps:.0f} bps",
                    summary="Sovereign risk elevated; check bank holdings, hedges.",
                    tags=["risk","sovereign",cty],
                    refs={"stream":stream,"spread_bps":sp_bps}
                ), key=f"risk::sov::{cty}")

        if self.cfg.bank_breach_emit and "risk.bank_stress" in stream:
            breaches = ev.get("breaches") or []
            if breaches:
                bank = ev.get("bank","BANK")
                self._emit(Insight(
                    severity="high", topic="risk",
                    title=f"{bank}: stress breaches {', '.join(breaches)}",
                    summary="Bank stress test flags regulatory ratio breach(es).",
                    tags=["risk","bank",bank],
                    refs={"stream":stream,"breaches":breaches}
                ), key=f"risk::bank::{bank}")

    def _on_oms(self, stream: str, ev: Dict[str, Any]):
        slip = self._to_float(ev.get("slip_bps") or ev.get("slippage_bps"))
        lat  = self._to_float(ev.get("lat_ms") or ev.get("latency_ms"))
        strat= ev.get("strategy") or "unknown"
        if slip is not None and slip >= self.cfg.slippage_bps_warn:
            self._emit(Insight(
                severity="medium", topic="oms",
                title=f"{strat}: slippage {slip:.1f} bps",
                summary="Execution slippage high; consider POST_ONLY / lower POV / different venue.",
                tags=["oms","tca",strat],
                refs={"stream":stream,"slip_bps":slip}
            ), key=f"oms::slip::{strat}")
        if lat is not None and lat >= 50:
            self._emit(Insight(
                severity="low", topic="oms",
                title=f"{strat}: latency {lat:.0f} ms",
                summary="Latency elevated; check network/venue or throttle.",
                tags=["oms","latency",strat],
                refs={"stream":stream,"lat_ms":lat}
            ), key=f"oms::lat::{strat}")

    # ---- emit to bus + latest cache ----
    def _emit(self, ins: Insight, *, key: Optional[str] = None):
        payload = ins.as_dict()
        try:
            publish_stream("insights.events", payload)
        except Exception:
            pass
        if key and hset:
            try:
                hset("insights:last", key, payload)
                hset("insights:topics", f"{ins.topic}:last", payload)
            except Exception:
                pass

    # ---- helpers ----
    @staticmethod
    def _safe_json(x: Any) -> Dict[str, Any]:
        if isinstance(x, dict): return x
        try: return json.loads(x)
        except Exception: return {}

    @staticmethod
    def _to_float(x: Any) -> Optional[float]:
        try:
            if x is None: return None
            f = float(x)
            if math.isfinite(f): return f
        except Exception:
            return None
        return None

    # ---- signals ----
    def _install_signals(self):
        def _stop(signum, frame):
            self.stop(); sys.exit(0)
        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)


# ---------------- CLI ----------------

def _probe():
    w = InsightsWorker(WorkerConfig())
    # fake high-heat news
    w._on_news("news.moneycontrol", {"title":"RBI policy surprise", "symbol":"NIFTY","heat":0.85,"sentiment":-0.4})
    # fake pnl spike
    for x in [1000, 1200, 900, 1100, -800, 950, 1050]: w._on_pnl("pnl.events", {"strategy":"momo_in","pnl":x})
    w._on_pnl("pnl.events", {"strategy":"momo_in","pnl": -5000})
    # fake risk events
    w._on_risk("risk.sovereign", {"country":"IN","spread_bps":180})
    w._on_risk("risk.metrics", {"strategy":"meanrev_us","dd":0.11})
    # fake OMS/TCA
    w._on_oms("tca.metrics", {"strategy":"arb_x","slip_bps":26.4,"lat_ms":72})
    print("Probe complete (see printed INSIGHTS lines).")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Insights Worker")
    ap.add_argument("--probe", action="true", default=False)  # keep compat with -m style
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--topics", type=str, help="Comma list: news,pnl,risk,oms")
    args, _ = ap.parse_known_args()

    cfg = WorkerConfig.from_env()
    if args.topics:
        cfg.topics = tuple(s.strip() for s in args.topics.split(",") if s.strip())

    if args.run:
        w = InsightsWorker(cfg)
        w.start()
    else:
        _probe()

if __name__ == "__main__":
    main()