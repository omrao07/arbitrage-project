# backend/compliance/compliance_narrative.py
from __future__ import annotations
"""
Compliance Narrative Generator
------------------------------
Produce regulator-/investor-ready narratives summarizing trading activity,
controls, best-execution, surveillance, data privacy, and model governance.

Inputs (dicts or JSON files):
- trades:      list of executed orders/fills with minimal schema (see Trade type)
- tca:         transaction cost analysis metrics per venue/strategy (optional)
- best_ex:     best-execution evidence snapshots (quotes/venues considered) (optional)
- surveillance:alerts and resolutions (optional)
- lineage:     data lineage steps for models/features used (optional)
- policies:    current active policies / risk limits / attestations (optional)
- model_cards: model info (name, version, owner, last_validation, risk_notes) (optional)
- privacy:     data privacy controls, consents, retention windows (optional)

Outputs:
- Markdown narrative string
- JSON narrative object (sectioned)
- Optional emit to bus (Redis stream) if backend.bus.streams.publish_stream exists

No hard dependencies:
- If `jinja2` is installed, it will be used to render a nicer template.
- Otherwise falls back to a built-in string template engine.

CLI:
  python -m backend.compliance.compliance_narrative \
      --in inputs.json \
      --out narrative.md \
      --json_out narrative.json \
      --redact

Schema tips are in the docstrings below.
"""
import json, hashlib, os, time, re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# Optional bus
try:
    from backend.bus.streams import publish_stream  # type: ignore
except Exception:
    def publish_stream(stream: str, payload: Dict[str, Any]) -> None:
        pass

# Optional jinja2
try:
    from jinja2 import Template  # type: ignore
    _has_jinja = True
except Exception:
    _has_jinja = False

OUT_STREAM = os.getenv("COMPLIANCE_NARRATIVE_STREAM", "compliance.narrative")

# ----------------------------- Data models -----------------------------

@dataclass
class Trade:
    """
    Minimal trade schema. Extra fields are preserved in 'extra'.
    """
    id: str
    ts_ms: int
    strategy: str
    symbol: str
    side: str          # buy/sell
    qty: float
    price: float
    venue: str
    order_type: str = "market"
    parent_order_id: Optional[str] = None
    account: Optional[str] = None
    region: Optional[str] = None
    tags: Optional[List[str]] = None
    explain: Optional[str] = None         # human explanation (if any)
    signal_score: Optional[float] = None  # [-1,1]
    risk_checks: Optional[Dict[str, Any]] = None
    extra: Optional[Dict[str, Any]] = None

# ----------------------------- Utilities ------------------------------

_PII_RE = re.compile(r"\b([A-Z0-9._%+-]+)@([A-Z0-9.-]+\.[A-Z]{2,})\b", re.I)
_ACCT_RE = re.compile(r"\b(\d{6,})\b")

def redact(text: Any) -> Any:
    """
    Redact emails and long account-like numbers.
    Works on strings, dicts, and lists.
    """
    if text is None:
        return None
    if isinstance(text, str):
        x = _PII_RE.sub("[redacted-email]", text)
        x = _ACCT_RE.sub("[redacted-acct]", x)
        return x
    if isinstance(text, dict):
        return {k: redact(v) for k, v in text.items()}
    if isinstance(text, list):
        return [redact(v) for v in text]
    return text

def hash_payload(obj: Any) -> str:
    j = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(j.encode("utf-8")).hexdigest()[:16]

def _fmt_money(x: Optional[float]) -> str:
    try:
        return f"${x:,.2f}"
    except Exception:
        return str(x)

def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def _now_ms() -> int:
    return int(time.time() * 1000)

# ----------------------------- Core builder ---------------------------

class ComplianceNarrative:
    """
    Build a narrative & JSON from structured inputs.
    """

    def __init__(self, *, emit: bool = False, stream: str = OUT_STREAM):
        self.emit = emit
        self.stream = stream

    # ---- Public API ----
    def build(
        self,
        *,
        trades: List[Dict[str, Any]],
        tca: Optional[Dict[str, Any]] = None,
        best_ex: Optional[Dict[str, Any]] = None,
        surveillance: Optional[Dict[str, Any]] = None,
        lineage: Optional[Dict[str, Any]] = None,
        policies: Optional[Dict[str, Any]] = None,
        model_cards: Optional[List[Dict[str, Any]]] = None,
        privacy: Optional[Dict[str, Any]] = None,
        redact_pii: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Returns (markdown, json_object)
        """
        # Normalize & optionally redact
        tr_rows = [asdict(self._coerce_trade(r)) for r in trades]
        payload = {
            "meta": {
                "ts_ms": _now_ms(),
                "session_hash": hash_payload({"trades": tr_rows}),
                **(metadata or {})
            },
            "summary": self._summary_block(tr_rows, tca, best_ex),
            "governance": self._governance_block(policies, model_cards),
            "best_execution": self._best_ex_block(best_ex, tca),
            "surveillance": self._surv_block(surveillance),
            "risk_management": self._risk_block(tr_rows, policies),
            "privacy": self._privacy_block(privacy),
            "lineage": self._lineage_block(lineage),
            "explainability": self._explain_block(tr_rows),
            "changes": self._change_log_block(policies, model_cards),
            "appendix": {"tca": tca or {}, "raw_best_ex": best_ex or {}},
        }
        if redact_pii:
            payload = redact(payload)

        md = self._render_markdown(payload)

        if self.emit:
            try:
                publish_stream(self.stream, {"ts_ms": _now_ms(), "kind": "narrative", "session": payload["meta"]["session_hash"]})
            except Exception:
                pass

        return md, payload

    # ---- Blocks ----
    def _summary_block(self, trades: List[Dict[str, Any]], tca: Optional[Dict[str, Any]], best_ex: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(trades)
        notional = sum((t.get("price", 0.0) or 0.0) * (t.get("qty", 0.0) or 0.0) for t in trades)
        syms = sorted({t.get("symbol") for t in trades if t.get("symbol")}) # type: ignore
        strategies = sorted({t.get("strategy") for t in trades if t.get("strategy")}) # type: ignore
        venues = sorted({t.get("venue") for t in trades if t.get("venue")}) # type: ignore
        return {
            "trades_count": n,
            "symbols": syms,
            "strategies": strategies,
            "venues": venues,
            "gross_notional": notional,
            "tca_brief": {
                "avg_slippage_bps": (tca or {}).get("summary", {}).get("avg_slippage_bps"),
                "fill_rate": (tca or {}).get("summary", {}).get("fill_rate"),
                "venue_hit_ratio": (tca or {}).get("summary", {}).get("venue_hit_ratio"),
            },
            "best_ex_snapshot": (best_ex or {}).get("snapshot", {})
        }

    def _governance_block(self, policies: Optional[Dict[str, Any]], model_cards: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        return {
            "policies_active": (policies or {}).get("active", []),
            "risk_limits": (policies or {}).get("limits", {}),
            "attestations": (policies or {}).get("attestations", []),
            "models": model_cards or []
        }

    def _best_ex_block(self, best_ex: Optional[Dict[str, Any]], tca: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        be = best_ex or {}
        return {
            "quote_set_count": len(be.get("quotes", [])),
            "venues_considered": sorted({q.get("venue") for q in be.get("quotes", []) if q.get("venue")}),
            "routing_reason": be.get("routing_reason"),
            "priority_rules": be.get("priority_rules", ["price", "liquidity", "latency", "fill_prob"]),
            "ex_post_tca": (tca or {}).get("per_venue", {})
        }

    def _surv_block(self, surveillance: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        s = surveillance or {}
        return {
            "alerts_count": len(s.get("alerts", [])),
            "alerts_open": [a for a in s.get("alerts", []) if str(a.get("status")).lower() not in ("closed","resolved")],
            "mitigations": s.get("mitigations", [])
        }

    def _risk_block(self, trades: List[Dict[str, Any]], policies: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        limits = (policies or {}).get("limits", {})
        checks = []
        breaches = []
        for t in trades:
            rc = t.get("risk_checks") or {}
            checks.append({"trade_id": t["id"], "checks": rc})
            if rc.get("breach"):
                breaches.append({"trade_id": t["id"], "detail": rc.get("detail")})
        return {
            "limits": limits,
            "per_trade_checks": checks,
            "breaches": breaches
        }

    def _privacy_block(self, privacy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        p = privacy or {}
        return {
            "personnel_access": p.get("access_model", "least-privilege"),
            "pii_controls": p.get("pii_controls", ["masking","tokenization","logging"]),
            "retention_days": p.get("retention_days", 365),
            "consents": p.get("consents", [])
        }

    def _lineage_block(self, lineage: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        ln = lineage or {}
        return {
            "sources": ln.get("sources", []),
            "feature_store": ln.get("feature_store", {}),
            "transformations": ln.get("transformations", []),
            "hash": hash_payload(ln) if ln else None
        }

    def _explain_block(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Aggregate explainability per strategy
        by_strat: Dict[str, Dict[str, Any]] = {}
        for t in trades:
            st = t.get("strategy") or "unknown"
            d = by_strat.setdefault(st, {"count": 0, "examples": [], "signal_stats": []})
            d["count"] += 1
            if t.get("explain"):
                d["examples"].append({"trade_id": t["id"], "text": t["explain"][:280]})
            if t.get("signal_score") is not None:
                d["signal_stats"].append(float(t["signal_score"]))
        # summarize signal stats
        for st, d in by_strat.items():
            xs = d["signal_stats"]
            if xs:
                m = sum(xs)/len(xs)
                lo, hi = min(xs), max(xs)
                d["signal_summary"] = {"mean": m, "min": lo, "max": hi}
        return {"per_strategy": by_strat}

    def _change_log_block(self, policies: Optional[Dict[str, Any]], model_cards: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        return {
            "policy_changes": (policies or {}).get("changes", []),
            "model_changes": (model_cards or []),
        }

    # ---- Template rendering ----
    def _render_markdown(self, p: Dict[str, Any]) -> str:
        if _has_jinja:
            return Template(_JINJA_TMPL).render(p=p, money=_fmt_money, json=json.dumps)
        return _fallback_render(p)

    # ---- Coercion ----
    def _coerce_trade(self, obj: Dict[str, Any]) -> Trade:
        return Trade(
            id=str(obj.get("id") or obj.get("order_id") or obj.get("fill_id")),
            ts_ms=int(obj.get("ts_ms") or obj.get("timestamp") or _now_ms()),
            strategy=str(obj.get("strategy") or "unknown"),
            symbol=str(obj.get("symbol") or ""),
            side=str(obj.get("side") or "").lower() or "buy",
            qty=float(obj.get("qty") or obj.get("quantity") or 0.0),
            price=float(obj.get("price") or 0.0),
            venue=str(obj.get("venue") or obj.get("exchange") or "NA"),
            order_type=str(obj.get("order_type") or "market"),
            parent_order_id=(obj.get("parent_order_id")),
            account=(obj.get("account")),
            region=(obj.get("region")),
            tags=(obj.get("tags")),
            explain=(obj.get("explain")),
            signal_score=_safe_float(obj.get("signal_score")),
            risk_checks=(obj.get("risk_checks") or {}),
            extra={k: v for k, v in obj.items() if k not in {
                "id","order_id","fill_id","ts_ms","timestamp","strategy","symbol","side","qty","quantity","price",
                "venue","exchange","order_type","parent_order_id","account","region","tags","explain","signal_score","risk_checks"
            }}
        )

# ----------------------------- Templates ------------------------------

_JINJA_TMPL = r"""
# Compliance Narrative

**Session:** `{{ p.meta.session_hash }}`  
**Generated:** {{ p.meta.ts_ms }} (ms since epoch)

## 1) Executive Summary
- Trades: **{{ p.summary.trades_count }}**  
- Symbols: {{ p.summary.symbols }}  
- Strategies: {{ p.summary.strategies }}  
- Venues: {{ p.summary.venues }}  
- Gross Notional: **{{ money(p.summary.gross_notional) }}**

### TCA Snapshot
- Avg Slippage (bps): {{ p.summary.tca_brief.avg_slippage_bps }}
- Fill Rate: {{ p.summary.tca_brief.fill_rate }}
- Venue Hit Ratio: {{ p.summary.tca_brief.venue_hit_ratio }}

## 2) Governance & Policies
- Active Policies: {{ p.governance.policies_active }}
- Risk Limits: {{ p.governance.risk_limits }}
- Attestations: {{ p.governance.attestations }}
- Models: {% for m in p.governance.models %}
  - **{{ m.name }}** v{{ m.version }} (owner: {{ m.owner }}, last_validation: {{ m.last_validation }})
    - Intended Use: {{ m.intended_use }}
    - Risk Notes: {{ m.risk_notes }}
  {% endfor %}

## 3) Best Execution
- Venues considered: {{ p.best_execution.venues_considered }}
- Quote sets: {{ p.best_execution.quote_set_count }}
- Routing reason: {{ p.best_execution.routing_reason }}
- Priority rules: {{ p.best_execution.priority_rules }}

## 4) Surveillance & Conduct
- Alerts (total): **{{ p.surveillance.alerts_count }}**
- Open alerts: {{ p.surveillance.alerts_open }}
- Mitigations: {{ p.surveillance.mitigations }}

## 5) Risk Management
- Limits: {{ p.risk_management.limits }}
- Per-trade checks: {{ p.risk_management.per_trade_checks }}
- Breaches: {{ p.risk_management.breaches }}

## 6) Privacy & Data Protection
- Access model: {{ p.privacy.personnel_access }}
- PII controls: {{ p.privacy.pii_controls }}
- Retention (days): {{ p.privacy.retention_days }}
- Consents: {{ p.privacy.consents }}

## 7) Data Lineage & Integrity
- Sources: {{ p.lineage.sources }}
- Feature Store: {{ p.lineage.feature_store }}
- Transformations: {{ p.lineage.transformations }}
- Lineage Hash: `{{ p.lineage.hash }}`

## 8) Explainability (Per Strategy)
{% for st, d in p.explainability.per_strategy.items() %}
### Strategy: {{ st }}
- Trades: {{ d.count }}
- Signal summary: {{ d.signal_summary }}
- Examples:
{% for ex in d.examples %}
  - {{ ex.trade_id }}: {{ ex.text }}
{% endfor %}
{% endfor %}

## 9) Change Log
- Policy changes: {{ p.changes.policy_changes }}
- Model changes: {{ p.changes.model_changes }}

---

### Appendix
- TCA (per venue): {{ p.appendix.tca }}
- Raw best-ex: (available in JSON export)
"""

def _fallback_render(p: Dict[str, Any]) -> str:
    # Simple deterministic markdown without jinja2
    lines = []
    lines.append(f"# Compliance Narrative")
    lines.append("")
    lines.append(f"**Session:** `{p['meta']['session_hash']}`")
    lines.append(f"**Generated:** {p['meta']['ts_ms']} (ms since epoch)")
    lines.append("")
    s = p["summary"]
    lines += [
        "## 1) Executive Summary",
        f"- Trades: **{s['trades_count']}**",
        f"- Symbols: {s['symbols']}",
        f"- Strategies: {s['strategies']}",
        f"- Venues: {s['venues']}",
        f"- Gross Notional: **{_fmt_money(s['gross_notional'])}**",
        "",
        "### TCA Snapshot",
        f"- Avg Slippage (bps): {s['tca_brief'].get('avg_slippage_bps')}",
        f"- Fill Rate: {s['tca_brief'].get('fill_rate')}",
        f"- Venue Hit Ratio: {s['tca_brief'].get('venue_hit_ratio')}",
        "",
        "## 2) Governance & Policies",
        f"- Active Policies: {p['governance']['policies_active']}",
        f"- Risk Limits: {p['governance']['risk_limits']}",
        f"- Attestations: {p['governance']['attestations']}",
        "- Models:",
    ]
    for m in p["governance"]["models"]:
        lines.append(f"  - {m.get('name')} v{m.get('version')} (owner: {m.get('owner')}, last_validation: {m.get('last_validation')})")
        lines.append(f"    - Intended Use: {m.get('intended_use')}")
        lines.append(f"    - Risk Notes: {m.get('risk_notes')}")
    be = p["best_execution"]
    lines += [
        "",
        "## 3) Best Execution",
        f"- Venues considered: {be['venues_considered']}",
        f"- Quote sets: {be['quote_set_count']}",
        f"- Routing reason: {be.get('routing_reason')}",
        f"- Priority rules: {be['priority_rules']}",
        "",
        "## 4) Surveillance & Conduct",
        f"- Alerts (total): **{p['surveillance']['alerts_count']}**",
        f"- Open alerts: {p['surveillance']['alerts_open']}",
        f"- Mitigations: {p['surveillance']['mitigations']}",
        "",
        "## 5) Risk Management",
        f"- Limits: {p['risk_management']['limits']}",
        f"- Per-trade checks: {p['risk_management']['per_trade_checks']}",
        f"- Breaches: {p['risk_management']['breaches']}",
        "",
        "## 6) Privacy & Data Protection",
        f"- Access model: {p['privacy']['personnel_access']}",
        f"- PII controls: {p['privacy']['pii_controls']}",
        f"- Retention (days): {p['privacy']['retention_days']}",
        f"- Consents: {p['privacy']['consents']}",
        "",
        "## 7) Data Lineage & Integrity",
        f"- Sources: {p['lineage']['sources']}",
        f"- Feature Store: {p['lineage']['feature_store']}",
        f"- Transformations: {p['lineage']['transformations']}",
        f"- Lineage Hash: `{p['lineage']['hash']}`",
        "",
        "## 8) Explainability (Per Strategy)",
    ]
    for st, d in p["explainability"]["per_strategy"].items():
        lines.append(f"### Strategy: {st}")
        lines.append(f"- Trades: {d['count']}")
        lines.append(f"- Signal summary: {d.get('signal_summary')}")
        lines.append("- Examples:")
        for ex in d.get("examples", []):
            lines.append(f"  - {ex['trade_id']}: {ex['text']}")
    lines += [
        "",
        "## 9) Change Log",
        f"- Policy changes: {p['changes']['policy_changes']}",
        f"- Model changes: {p['changes']['model_changes']}",
        "",
        "---",
        "",
        "### Appendix",
        f"- TCA (per venue): {p['appendix']['tca']}",
        "- Raw best-ex: (available in JSON export)",
    ]
    return "\n".join(lines)

# ----------------------------- CLI ------------------------------------

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def _save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _main():
    import argparse
    p = argparse.ArgumentParser(description="Generate a compliance narrative (Markdown + JSON).")
    p.add_argument("--in", dest="inp", required=True, help="Input JSON with keys: trades, tca, best_ex, surveillance, lineage, policies, model_cards, privacy")
    p.add_argument("--out", dest="out_md", required=True, help="Output Markdown path")
    p.add_argument("--json_out", dest="out_json", required=False, help="Output JSON path")
    p.add_argument("--redact", action="store_true", help="Redact PII (emails, account numbers)")
    p.add_argument("--emit", action="store_true", help="Publish a stub event to the bus")
    args = p.parse_args()

    data = _load_json(args.inp)
    gen = ComplianceNarrative(emit=args.emit)
    md, obj = gen.build(
        trades=data.get("trades", []),
        tca=data.get("tca"),
        best_ex=data.get("best_ex"),
        surveillance=data.get("surveillance"),
        lineage=data.get("lineage"),
        policies=data.get("policies"),
        model_cards=data.get("model_cards"),
        privacy=data.get("privacy"),
        redact_pii=args.redact,
        metadata=data.get("meta")
    )
    _save(args.out_md, md)
    if args.out_json:
        _save_json(args.out_json, obj)

if __name__ == "__main__":  # pragma: no cover
    _main()