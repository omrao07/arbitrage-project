# backend/analytics/alpha_dna.py
"""
Alpha DNA Report
----------------
Summarizes where your performance really comes from:
- Top / bottom strategy contributors (YTD, 30D, 7D)
- Risk/quality metrics per strategy (Sharpe, Max DD, Turnover, Hit rate)
- Simple factor hints (e.g., beta to market, carry, momentum proxy)
- Concentration (Herfindahl) and fragility checks
- Plain-English bullets + Markdown for investor-style letter

Inputs (best-effort; any subset works):
  pnl_timeseries: dict[strategy] -> list[{"ts":int,"pnl":float}]
  metrics: dict[strategy] -> {"sharpe":..,"max_dd":..,"turnover":..,"hit_rate":..}
  exposures: dict[strategy] -> {"beta_mkt":..,"beta_rate":..,"beta_fx":..,"beta_momo":..}  (optional)
  buckets: {"book":{...}, "region":{...}, "asset_class":{...}}  (optional breakdowns)
  firm: {"nav": float, "ytd": float, "mtd": float, "wtd": float} (optional headline)

Outputs:
  report = {
    "asof": ts_ms,
    "headline": {...},                  # firm-level
    "top_contributors": [(name, pnl)],  # YTD
    "top_bleeders":    [(name, pnl)],
    "by_window": { "30D": {...}, "7D": {...} },
    "risk_table": {strategy: {...metrics...}},
    "factor_hints": {strategy: {"mkt":..,"rate":..,"fx":..,"momo":..}},
    "concentration": {"herfindahl": float, "top_k_share": float},
    "bullets": [ ... ],
    "markdown": "..."                   # pretty letter section
  }

If Redis/event bus exists, you can publish the summary to `ai.insight`.
"""

from __future__ import annotations

import math
import os
import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional

try:
    import numpy as np  # type: ignore
except Exception:
    np = None

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

# Optional publish
try:
    from backend.bus.streams import publish_stream
except Exception:
    publish_stream = None  # type: ignore


# ------------------------ utilities ------------------------

def _utc_ms() -> int:
    return int(time.time() * 1000)

def money(x: float, digits: int = 0) -> str:
    try:
        return f"{x:,.{digits}f}"
    except Exception:
        return str(x)

def pct(x: float, digits: int = 2) -> str:
    try:
        return f"{x*100:.{digits}f}%"
    except Exception:
        return "n/a"

def head_tail_sorted(d: Dict[str, float], k: int = 5) -> Tuple[List[Tuple[str,float]], List[Tuple[str,float]]]:
    arr = sorted([(k2, float(v)) for k2, v in d.items()], key=lambda t: t[1], reverse=True)
    return arr[:k], arr[-k:]

def herfindahl(weights: List[float]) -> float:
    if not weights:
        return 0.0
    s = sum(max(0.0, float(w)) for w in weights)
    if s <= 0:
        return 0.0
    return sum((w/s)**2 for w in weights)

def safe_mean(xs: List[float]) -> float:
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / max(1, len(xs))

# ------------------------ core compute ------------------------

def _window_pnl(pnl_ts: Dict[str, List[Dict[str, Any]]], days: int) -> Dict[str, float]:
    """
    Sum PnL over trailing N days (approx; assumes daily-ish entries).
    """
    out: Dict[str, float] = {}
    if not pnl_ts:
        return out
    cutoff = time.time() - days*86400
    for strat, rows in pnl_ts.items():
        s = 0.0
        for r in rows[-2000:]:
            ts = float(r.get("ts", 0)) / (1000.0 if r.get("ts", 0) > 10_000_000_000 else 1.0)
            if ts >= cutoff:
                s += float(r.get("pnl", 0.0))
        out[strat] = s
    return out

def _ytd_pnl(pnl_ts: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not pnl_ts:
        return out
    t = time.gmtime()
    y_start = time.mktime((t.tm_year, 1, 1, 0, 0, 0, 0, 0, -1))
    for strat, rows in pnl_ts.items():
        s = 0.0
        for r in rows[-5000:]:
            ts = float(r.get("ts", 0)) / (1000.0 if r.get("ts", 0) > 10_000_000_000 else 1.0)
            if ts >= y_start:
                s += float(r.get("pnl", 0.0))
        out[strat] = s
    return out

def _factor_hints(exposures: Optional[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """
    Normalize factor exposures to -1..+1 range heuristically.
    """
    if not exposures:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for strat, ex in exposures.items():
        mkt = float(ex.get("beta_mkt", 0.0))
        rate = float(ex.get("beta_rate", 0.0))
        fx = float(ex.get("beta_fx", 0.0))
        momo = float(ex.get("beta_momo", 0.0))
        # squash with tanh to -1..1
        out[strat] = {
            "mkt": math.tanh(mkt),
            "rate": math.tanh(rate),
            "fx": math.tanh(fx),
            "momo": math.tanh(momo),
        }
    return out

def _concentration(ytd: Dict[str, float], top_k: int = 5) -> Dict[str, float]:
    weights = [max(0.0, v) for v in ytd.values()]
    hh = herfindahl(weights)
    arr = sorted(weights, reverse=True)
    top_share = (sum(arr[:top_k]) / max(1e-9, sum(arr))) if arr else 0.0
    return {"herfindahl": float(hh), "top_k_share": float(top_share)}

# ------------------------ main API ------------------------

def build_report(
    pnl_timeseries: Dict[str, List[Dict[str, Any]]],
    metrics: Optional[Dict[str, Dict[str, float]]] = None,
    exposures: Optional[Dict[str, Dict[str, float]]] = None,
    firm: Optional[Dict[str, float]] = None,
    buckets: Optional[Dict[str, Dict[str, float]]] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Core entrypoint – returns the Alpha DNA report dict.
    """
    ytd = _ytd_pnl(pnl_timeseries)
    w30 = _window_pnl(pnl_timeseries, 30)
    w7  = _window_pnl(pnl_timeseries, 7)

    top, bottom = head_tail_sorted(ytd, k=top_k)
    conc = _concentration(ytd, top_k=top_k)
    factors = _factor_hints(exposures)

    # Risk table – merge basic metrics & windows
    risk_table: Dict[str, Dict[str, Any]] = {}
    for strat in set(list(ytd.keys()) + list((metrics or {}).keys())):
        m = (metrics or {}).get(strat, {})
        risk_table[strat] = {
            "ytd": float(ytd.get(strat, 0.0)),
            "30d": float(w30.get(strat, 0.0)),
            "7d": float(w7.get(strat, 0.0)),
            "sharpe": _round(m.get("sharpe")),
            "max_dd": _round(m.get("max_dd")),
            "turnover": _round(m.get("turnover")),
            "hit_rate": _round(m.get("hit_rate")),
        }

    bullets = _bullets(firm or {}, top, bottom, conc, factors)

    report = {
        "asof": _utc_ms(),
        "headline": firm or {},
        "top_contributors": top,
        "top_bleeders": bottom,
        "by_window": {"30D": w30, "7D": w7},
        "risk_table": risk_table,
        "factor_hints": factors,
        "concentration": conc,
        "bullets": bullets,
    }
    report["markdown"] = to_markdown(report, buckets=buckets)
    return report

def to_markdown(report: Dict[str, Any], buckets: Optional[Dict[str, Dict[str, float]]] = None) -> str:
    """
    Pretty investor-letter style section.
    """
    firm = report.get("headline", {})
    ytd = firm.get("ytd"); mtd = firm.get("mtd"); wtd = firm.get("wtd")
    top = report.get("top_contributors", [])[:5]
    bleeders = report.get("top_bleeders", [])[:5]
    conc = report.get("concentration", {})
    lines = []

    lines.append(f"### Alpha DNA — as of {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(report.get('asof', _utc_ms())/1000))}\n")
    if firm:
        parts = []
        if ytd is not None: parts.append(f"YTD PnL **{money(ytd)}**")
        if mtd is not None: parts.append(f"MTD **{money(mtd)}**")
        if wtd is not None: parts.append(f"WTD **{money(wtd)}**")
        if parts:
            lines.append("- " + " · ".join(parts))

    if top:
        lines.append(f"- **Top contributors:** " + ", ".join(f"`{n}` ({money(v)})" for n,v in top))
    if bleeders:
        lines.append(f"- **Top bleeders:** " + ", ".join(f"`{n}` ({money(v)})" for n,v in bleeders))
    if conc:
        lines.append(f"- Concentration: Herfindahl {conc.get('herfindahl',0):.3f}, Top-{len(top)} share {pct(conc.get('top_k_share',0),1)}")

    # Optional bucket breakdowns
    if buckets:
        for k, d in buckets.items():
            s = sum(d.values()) if d else 0.0
            if s:
                topb = sorted(d.items(), key=lambda t: -abs(t[1]))[:5]
                lines.append(f"- By **{k}**: " + ", ".join(f"`{kk}` ({money(v)})" for kk, v in topb))

    # Factor hints (summarize strongest)
    fh = report.get("factor_hints") or {}
    if fh:
        hot = strongest_factor_summary(fh)
        if hot:
            lines.append(f"- Factor tilt: {hot}")

    # Bullets from engine
    engine_bullets = report.get("bullets") or []
    if engine_bullets:
        lines.append("\n**Notes:**")
        lines += [f"- {b}" for b in engine_bullets[:4]]

    return "\n".join(lines) + "\n"

def strongest_factor_summary(fh: Dict[str, Dict[str, float]]) -> str:
    """
    Pick the most common strong factor across strategies.
    """
    if not fh:
        return ""
    agg: Dict[str, List[float]] = {"mkt":[], "rate":[], "fx":[], "momo":[]}
    for strat, fac in fh.items():
        for k in agg:
            agg[k].append(abs(float(fac.get(k,0.0))))
    means = {k: safe_mean(v) for k, v in agg.items()}
    top = sorted(means.items(), key=lambda t: t[1], reverse=True)[:2]
    return ", ".join(f"{k} ({v:.2f})" for k, v in top if v > 0.15)

def _bullets(
    firm: Dict[str, float],
    top: List[Tuple[str, float]],
    bottom: List[Tuple[str, float]],
    conc: Dict[str, float],
    factors: Dict[str, Dict[str, float]],
) -> List[str]:
    bullets: List[str] = []
    if firm:
        nav = firm.get("nav")
        if nav:
            bullets.append(f"NAV {money(nav)}; portfolio {('green' if firm.get('ytd',0)>=0 else 'red')} YTD.")
    if top:
        names = ", ".join(n for n,_ in top[:3])
        bullets.append(f"Top alpha engines: {names}.")
    if bottom:
        names = ", ".join(n for n,_ in bottom[:2])
        bullets.append(f"Underperformers to review: {names}.")
    if conc:
        if conc.get("top_k_share",0) > 0.6:
            bullets.append("Performance concentration is high; consider diversifying gross caps.")
    # factor tilt
    s = strongest_factor_summary(factors)
    if s:
        bullets.append(f"Factor tilt elevated in: {s}.")
    return bullets

def _round(x: Any, d: int = 3) -> Optional[float]:
    try:
        return round(float(x), d)
    except Exception:
        return None

# ------------------------ convenience I/O ------------------------

def publish_insight(report: Dict[str, Any], stream: str = "ai.insight") -> None:
    """
    Emit a short insight item for the right-rail UI.
    """
    if not publish_stream:
        return
    top = report.get("top_contributors", [])[:3]
    bleeders = report.get("top_bleeders", [])[:2]
    summary = "Alpha DNA — " + ", ".join(f"{n}:{money(v)}" for n,v in top) \
              + (f"; bleeders: " + ", ".join(f"{n}:{money(v)}" for n,v in bleeders) if bleeders else "")
    publish_stream(stream, {
        "ts_ms": report.get("asof", _utc_ms()),
        "kind": "alpha_dna",
        "summary": summary[:240],
        "details": report.get("bullets", [])[:5],
        "tags": ["pnl","attribution","alpha-dna"],
        "refs": {}
    })

def save_report(report: Dict[str, Any], path: str) -> None:
    """
    Save the report JSON + a .md neighbor for human reading.
    """
    try:
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        base, _ = os.path.splitext(path)
        with open(base + ".md", "w") as f:
            f.write(report.get("markdown",""))
    except Exception:
        pass

# ------------------------ example CLI ------------------------

def main():
    """
    Quick demo:
      python -m backend.analytics.alpha_dna --out runtime/alpha_dna.json
    """
    import argparse, random
    ap = argparse.ArgumentParser(description="Alpha DNA report builder")
    ap.add_argument("--out", type=str, default="runtime/alpha_dna.json")
    args = ap.parse_args()

    # synthetic demo data
    now = _utc_ms()
    rng = random.Random(42)
    strategies = ["alpha.momo","meanrev.core","statarb.pairs","hrp.portfolio","carry.fx"]
    pnl_ts = {
        s: [{"ts": now - i*86400_000, "pnl": rng.uniform(-5e4, 8e4)} for i in range(120)][::-1]
        for s in strategies
    }
    metrics = {
        s: {"sharpe": rng.uniform(-0.2, 2.5), "max_dd": rng.uniform(0.03, 0.18),
            "turnover": rng.uniform(0.2, 3.0), "hit_rate": rng.uniform(0.35, 0.65)}
        for s in strategies
    }
    exposures = {
        s: {"beta_mkt": rng.uniform(-0.5, 1.2), "beta_rate": rng.uniform(-0.4, 0.8),
            "beta_fx": rng.uniform(-0.5, 0.5), "beta_momo": rng.uniform(-0.3, 0.9)}
        for s in strategies
    }
    firm = {"nav": 10_000_000, "ytd": sum(_ytd_pnl(pnl_ts).values()),
            "mtd": sum(_window_pnl(pnl_ts, 30).values()),
            "wtd": sum(_window_pnl(pnl_ts, 7).values())}

    rep = build_report(pnl_ts, metrics=metrics, exposures=exposures, firm=firm)
    save_report(rep, args.out)
    publish_insight(rep)
    print(f"Wrote {args.out} and {os.path.splitext(args.out)[0]+'.md'}")

if __name__ == "__main__":
    main()