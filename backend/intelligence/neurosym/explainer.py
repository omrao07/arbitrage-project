# backend/common/explainer.py
"""
Human-readable explainers for agents, risk, and coordinator decisions.

What it covers
--------------
- explain_proposal(): one agent's Proposal (+ optional RiskReport)
- explain_decision(): consolidated Coordinator slate (ExecutionDecision)
- explain_signals_delta(): compact diff between two signal snapshots
- table(): fixed-width ASCII table helper (no external deps)
- md_report(): quick Markdown report builder
- tiny sparkline(): unicode sparkline for quick trend visuals

All inputs are plain dict-like or your dataclasses with attributes;
the explainer will try both (attr and dict access) for convenience.
"""

from __future__ import annotations

import math
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ------------------------ helpers: safe getters ------------------------

def _get(obj: Any, key: str, default=None):
    if obj is None:
        return default
    if is_dataclass(obj):
        try:
            return getattr(obj, key)
        except Exception:
            return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    try:
        return getattr(obj, key)
    except Exception:
        return default

def _as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if is_dataclass(obj):
        try:
            return asdict(obj) # type: ignore
        except Exception:
            pass
    # best effort
    try:
        return dict(obj)  # type: ignore[arg-type]
    except Exception:
        # last resort: introspect common attrs
        out = {}
        for k in ("orders", "thesis", "score", "confidence", "horizon_sec", "tags"):
            v = _get(obj, k, None)
            if v is not None:
                out[k] = v
        return out


# ------------------------ formatting primitives ------------------------

def table(headers: Sequence[str], rows: Sequence[Sequence[Any]], *, pad: int = 1) -> str:
    """Render a simple fixed-width ASCII table."""
    cols = len(headers)
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(str(row[i])) if i < len(row) else 0)
    def fmt_row(vals: Sequence[Any]) -> str:
        return " | ".join(str(vals[i])[: widths[i]].ljust(widths[i]) for i in range(cols))
    sep = "-+-".join("-" * w for w in widths)
    lines = [fmt_row(headers), sep]
    for r in rows:
        lines.append(fmt_row(r))
    return "\n".join(lines)

_SPARK_BARS = "▁▂▃▄▅▆▇█"
def sparkline(xs: Sequence[float]) -> str:
    """Unicode sparkline for a small series."""
    if not xs:
        return ""
    lo, hi = min(xs), max(xs)
    rng = (hi - lo) or 1.0
    out = []
    for x in xs:
        idx = int((x - lo) / rng * (len(_SPARK_BARS) - 1))
        out.append(_SPARK_BARS[max(0, min(idx, len(_SPARK_BARS) - 1))])
    return "".join(out)

def _pct(x: Optional[float]) -> str:
    try:
        return f"{float(x)*100:.2f}%" # type: ignore
    except Exception:
        return "n/a"

def _bps(x: Optional[float]) -> str:
    try:
        return f"{float(x)*10_000:.0f}bps" # type: ignore
    except Exception:
        return "n/a"

def _fmt(x: Any, nd=4) -> str:
    try:
        f = float(x)
        if abs(f) >= 1000 or f.is_integer():
            return f"{f:.0f}"
        return f"{f:.{nd}f}"
    except Exception:
        return str(x)


# ------------------------ agent proposal explainer ---------------------

def explain_proposal(agent_name: str, proposal: Any, risk: Optional[Any] = None, *, markdown: bool = False) -> str:
    """
    Render a single agent's proposal + (optional) risk in a readable form.
    """
    p = _as_dict(proposal)
    orders = p.get("orders") or []
    score = p.get("score", 0.0)
    conf = p.get("confidence", 0.0)
    horizon = p.get("horizon_sec", 0.0)
    thesis = p.get("thesis", "") or ""
    tags = p.get("tags") or []

    rows = []
    for o in orders:
        rows.append([
            (o.get("side") if isinstance(o, dict) else _get(o, "side")) or "",
            (o.get("qty") if isinstance(o, dict) else _get(o, "qty")) or 0,
            (o.get("symbol") if isinstance(o, dict) else _get(o, "symbol")) or "",
            (o.get("venue") if isinstance(o, dict) else _get(o, "venue")) or "",
        ])

    head = f"[{agent_name}] score={_fmt(score,2)} conf={_fmt(conf,2)} horizon≈{int(horizon)}s"
    legs = table(["Side", "Qty", "Symbol", "Venue"], rows) if rows else "(no legs)"
    risk_txt = ""
    if risk is not None:
        ok = bool(_get(risk, "ok", False))
        gross = _fmt(_get(risk, "gross_notional_usd", 0.0), 2)
        net = _fmt(_get(risk, "exposure_usd", 0.0), 2)
        notes = _get(risk, "notes", "") or ""
        risk_txt = f"risk={'PASS' if ok else 'FAIL'} gross=${gross} net=${net}" + (f" | {notes}" if notes else "")

    body = f"{head}\n{legs}\n\nthesis: {thesis}"
    if tags:
        body += f"\nlabels: {', '.join(tags)}"
    if risk_txt:
        body += f"\n{risk_txt}"

    if markdown:
        # wrap in fenced code for monospace table
        return f"### {agent_name} Proposal\n\n```\n{body}\n```"
    return body


# ------------------------ coordinator decision explainer ---------------

def explain_decision(decision: Any, *, markdown: bool = False) -> str:
    """
    Render Coordinator's ExecutionDecision into a concise report.
    """
    ok = bool(_get(decision, "ok", False))
    legs = _get(decision, "legs", []) or []
    notes = _get(decision, "notes", "") or ""
    diags = _get(decision, "diagnostics", {}) or {}

    rows = []
    for L in legs:
        if isinstance(L, dict):
            side = L.get("side"); qty = L.get("qty"); sym = L.get("symbol"); ven = L.get("venue")
            r = L.get("rationale", ""); ns = L.get("meta", {}).get("net_score", 0.0)
            contribs = ", ".join(L.get("contributors", [])[:3])
        else:
            side = _get(L, "side"); qty = _get(L, "qty"); sym = _get(L, "symbol"); ven = _get(L, "venue")
            r = _get(L, "rationale", ""); meta = _get(L, "meta", {}) or {}
            ns = meta.get("net_score", 0.0); contribs = ", ".join((_get(L, "contributors", []) or [])[:3])
        rows.append([side, _fmt(qty, 6), sym, ven or "ANY", _fmt(ns, 2), r, contribs])

    title = f"[COORDINATOR] {'OK' if ok else 'NO-GO'} | {notes}"
    legs_txt = table(["Side", "Qty", "Symbol", "Venue", "NetScore", "Rationale", "Contribs"], rows) if rows else "(no legs)"

    # optional summary from diagnostics
    out_dbg = ""
    if diags:
        try:
            outs = diags.get("outcomes") or {}
            brief = []
            for name, info in outs.items():
                if not isinstance(info, dict): 
                    continue
                ok2 = "ok" if info.get("ok") else "fail"
                brief.append(f"{name}:{ok2}:{_fmt(info.get('score',0),2)}/{_fmt(info.get('confidence',0),2)}")
            if brief:
                out_dbg = "agents: " + "; ".join(brief)
        except Exception:
            pass

    body = f"{title}\n{legs_txt}"
    if out_dbg:
        body += f"\n{out_dbg}"

    if markdown:
        return f"## Consolidated Slate\n\n```\n{body}\n```"
    return body


# ------------------------ signal delta explainer -----------------------

def explain_signals_delta(prev: Dict[str, float], curr: Dict[str, float], *, top_k: int = 12, markdown: bool = False) -> str:
    """
    Show the largest movers between two signal snapshots.
    """
    keys = set(prev.keys()) | set(curr.keys())
    rows = []
    for k in keys:
        a = float(prev.get(k, float("nan")))
        b = float(curr.get(k, float("nan")))
        if math.isnan(a) or math.isnan(b):
            continue
        d = b - a
        rows.append((k, a, b, d, abs(d)))
    rows.sort(key=lambda t: t[4], reverse=True)
    rows = rows[:top_k]
    tbl = table(["Signal", "Prev", "Curr", "Δ"], [[k, _fmt(a,4), _fmt(b,4), _fmt(d,4)] for k,a,b,d,_ in rows])
    if markdown:
        return f"### Signal Movers\n\n```\n{tbl}\n```"
    return tbl


# ------------------------ markdown report ------------------------------

def md_report(
    *,
    title: str,
    context_summary: Optional[Dict[str, Any]] = None,
    agent_blocks: Optional[List[str]] = None,
    decision_block: Optional[str] = None,
    notes: Optional[str] = None,
) -> str:
    """
    Assemble a single Markdown report from building blocks.

    Example:
        md = md_report(
            title="Daily Arb Run",
            context_summary={"prices": {"BTCUSDT": 65200}, "risk_z": -0.2},
            agent_blocks=[explain_proposal("crypto", prop, risk, markdown=False)],
            decision_block=explain_decision(decision, markdown=False),
            notes="Run id=abc123"
        )
    """
    lines = [f"# {title}"]
    if context_summary:
        try:
            pretty = json.dumps(context_summary, indent=2, sort_keys=True)
            lines += ["\n## Context", "```json", pretty, "```"]
        except Exception:
            lines += ["\n## Context", "```\n" + str(context_summary) + "\n```"]
    if agent_blocks:
        for blk in agent_blocks:
            lines += ["\n## Agent", "```\n" + blk + "\n```"]
    if decision_block:
        lines += ["\n## Decision", "```\n" + decision_block + "\n```"]
    if notes:
        lines += ["\n## Notes", notes]
    return "\n".join(lines)


# ------------------------ tiny demo ------------------------------------

if __name__ == "__main__":
    # Minimal smoke using simple dicts (works with your dataclasses too)
    fake_prop = {
        "orders": [
            {"side": "BUY", "qty": 0.05, "symbol": "BTCUSDT", "venue": "BINANCE", "meta": {"score": 0.6}},
            {"side": "SELL", "qty": 5, "symbol": "AAPL", "venue": "NYSE", "meta": {"score": -0.3}},
        ],
        "thesis": "BTC carry + sentiment; AAPL rich valuation fade.",
        "score": 0.45,
        "confidence": 0.62,
        "horizon_sec": 7200,
        "tags": ["crypto", "equities"],
    }
    fake_risk = {"ok": True, "gross_notional_usd": 25000, "exposure_usd": 1200, "notes": "checks ok"}
    fake_decision = {
        "ok": True,
        "notes": "selected 2 legs",
        "legs": [
            {"symbol": "BTCUSDT", "side": "BUY", "qty": 0.05, "venue": "BINANCE", "rationale": "net=0.52 scale=1.2 base_qty≈0.05",
             "contributors": ["crypto"], "meta": {"net_score": 0.52}},
            {"symbol": "AAPL", "side": "SELL", "qty": 5, "venue": "NYSE", "rationale": "net=-0.31 scale=1.0 base_qty≈5",
             "contributors": ["equities"], "meta": {"net_score": -0.31}},
        ],
        "diagnostics": {"outcomes": {"crypto": {"ok": True, "score": 0.6, "confidence": 0.7}}}
    }

    print(explain_proposal("crypto", fake_prop, fake_risk))
    print()
    print(explain_decision(fake_decision))
    print()
    prev = {"social_sent_btc": 0.1, "mom_z_AAPL": 0.2}
    curr = {"social_sent_btc": 0.36, "mom_z_AAPL": -0.5}
    print(explain_signals_delta(prev, curr))