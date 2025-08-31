# backend/ai/narratives/macro_narratives.py
from __future__ import annotations

"""
Macro Narratives
----------------
Turn macro features into a compact regime + narrative for dashboards, alerts, or reports.

Inputs: dicts of simple time series (lists of floats; last element = latest) and optional stance/sentiment.
Outputs: scores in [-1, +1], regime tags, a summary string, bullets, and diagnostics.

Usage (programmatic)
--------------------
from backend.ai.narratives.macro_narratives import MacroNarratives, NarrativeInput

inputs = NarrativeInput(
    series = {
        "pmi": [49.2, 50.1, 50.7, 51.0],
        "cpi_surprise": [-0.1, 0.0, -0.2, -0.3],
        "breakevens_5y": [2.2, 2.1, 2.0, 1.95],
        "vix": [13.5, 14.2, 13.8, 12.9],
        "credit_oas": [120, 118, 115, 112],
        "fci": [0.15, 0.12, 0.08, 0.05],
        "oil_1m_pct": [0.03, -0.01, -0.02, -0.04],
        "dxy": [105.0, 104.6, 103.9, 103.2],
        "payroll_surprise": [50, 30, 20, 15],   # k over consensus
        "ism_new_orders": [47.0, 49.5, 50.2, 51.1],
    },
    cb_stance = -0.2,          # dovish (-1) ↔ hawkish (+1) from your CB monitor (optional)
    news_sent = 0.1,           # -1..+1 (optional)
    earnings_sent = 0.2        # -1..+1 (optional)
)
res = MacroNarratives().analyze(inputs)
print(res.regime_tags, res.summary)

CLI
---
python -m backend.ai.narratives.macro_narratives --in features.json --out narrative.json

features.json:
{
  "series": {
    "pmi": [ ... ],
    "cpi_surprise": [ ... ]
  },
  "cb_stance": -0.1,
  "news_sent": 0.05,
  "earnings_sent": 0.1
}
"""

import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Iterable, Any

# ---------------------------- Models ----------------------------

@dataclass
class NarrativeInput:
    # series: each key -> list[float], with latest at the end
    series: Dict[str, List[float]]
    cb_stance: Optional[float] = None      # -1 (dovish) … +1 (hawkish)
    news_sent: Optional[float] = None      # -1 … +1
    earnings_sent: Optional[float] = None  # -1 … +1
    notes: Dict[str, Any] = None           # type: ignore # passthrough metadata (country, date, etc.)

@dataclass
class NarrativeResult:
    scores: Dict[str, float]               # growth, inflation, liquidity, policy, risk_stress, risk_on, sentiment
    regime_tags: List[str]
    summary: str
    bullets: List[str]
    diagnostics: Dict[str, Any]            # z-scores per indicator, coverage, confidence

# ---------------------- Robust Stats Helpers ----------------------

def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / max(1, len(xs))

def _stdev(xs: Iterable[float]) -> float:
    xs = list(xs); n = len(xs)
    if n < 2: return 0.0
    m = _mean(xs)
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(max(0.0, v))

def _z_latest(vals: List[float]) -> float:
    """
    Z-score of the latest value vs prior history (exclude last for mean/stdev).
    Returns 0.0 if insufficient history.
    """
    if not vals: return 0.0
    if len(vals) < 3:  # too short to be meaningful
        return 0.0
    latest = vals[-1]
    base = vals[:-1]
    m = _mean(base)
    s = _stdev(base)
    if s <= 1e-12:
        return 0.0
    return (latest - m) / s

def _squash(x: float, k: float = 1.5) -> float:
    """Map real line → [-1,1] smoothly (tanh). k controls steepness."""
    return float(math.tanh(x / max(1e-9, k)))

def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

# ---------------------- Indicator → Dimension ----------------------

# Weights and signs for each dimension.
# Sign convention: we compute z = zscore(latest); contribution = sign * weight * z
# so that positive dimension score means: Growth↑, Inflation↑, Liquidity↑(looser), Policy↑(hawkish), RiskStress↑ (worse), RiskOn↑ (better)
WEIGHTS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "growth": {
        "pmi": (1.0, +1.0),
        "ism_new_orders": (1.1, +1.0),
        "gdp_nowcast": (1.2, +1.0),
        "real_sales": (0.8, +1.0),
        "payroll_surprise": (0.9, +1.0),
        "eps_revisions": (0.9, +1.0),
        "global_trade": (0.7, +1.0),
    },
    "inflation": {
        "cpi_surprise": (1.2, +1.0),
        "ppi_mom": (0.8, +1.0),
        "core_cpi_yoy": (1.0, +1.0),
        "breakevens_5y": (0.9, +1.0),
        "oil_1m_pct": (0.7, +1.0),
    },
    "liquidity": {
        "fci": (1.2, -1.0),      # higher FCI = tighter → negative liquidity
        "dxy": (0.6, -1.0),      # stronger USD → tighter global liquidity
        "term_premium": (0.6, -1.0),
    },
    "risk_stress": {
        "vix": (1.0, +1.0),
        "credit_oas": (1.0, +1.0),
        "ted_spread": (0.7, +1.0),
        "funding_stress": (0.8, +1.0),
    },
}

def _dimension_score(name: str, series: Dict[str, List[float]], diag: Dict[str, Any]) -> float:
    spec = WEIGHTS.get(name, {})
    if not spec:
        return 0.0
    num, den = 0.0, 0.0
    for key, (w, sgn) in spec.items():
        vals = series.get(key)
        if not vals: 
            continue
        z = _z_latest(vals)
        num += w * sgn * z
        den += w
        diag.setdefault("z", {})[key] = z
    if den == 0.0:
        return 0.0
    return _clip(_squash(num / den, k=1.6))

def _combine_sentiment(news: Optional[float], earns: Optional[float]) -> float:
    xs = [x for x in (news, earns) if isinstance(x, (int, float))]
    if not xs:
        return 0.0
    m = sum(xs) / len(xs)
    return _clip(m)

def _policy_score(cb_stance: Optional[float], series: Dict[str, List[float]], diag: Dict[str, Any]) -> float:
    """
    Policy stance: hawkish (+) vs dovish (-).
    Prefer explicit cb_stance; otherwise infer from short-rate z if provided.
    """
    if isinstance(cb_stance, (int, float)):
        return _clip(float(cb_stance))
    # inference: use "policy_rate_change_bp" or "2y_yield" z as proxy
    if "policy_rate" in series and len(series["policy_rate"]) >= 3:
        z = _z_latest(series["policy_rate"])
        diag.setdefault("z", {})["policy_rate"] = z
        return _clip(_squash(z))
    if "y2y" in series and len(series["y2y"]) >= 3:
        z = _z_latest(series["y2y"])
        diag.setdefault("z", {})["y2y"] = z
        return _clip(_squash(z))
    return 0.0

# ---------------------- Regime Classification ----------------------

def _regime_tags(growth: float, inflation: float, liquidity: float, risk_on: float) -> List[str]:
    tags: List[str] = []
    g, inf = growth, inflation
    if g > 0.35 and inf < -0.20:
        tags += ["Goldilocks", "Disinflation Rally"]
    elif g > 0.35 and inf > 0.20:
        tags += ["Reflation", "Hot Growth"]
    elif g < -0.35 and inf > 0.20:
        tags += ["Stagflation Risk"]
    elif g < -0.35 and inf < -0.20:
        tags += ["Growth Slowdown"]
    else:
        tags += ["Mixed Regime"]

    if liquidity > 0.25:
        tags.append("Loose Liquidity")
    elif liquidity < -0.25:
        tags.append("Tight Liquidity")

    if risk_on > 0.25:
        tags.append("Risk-On")
    elif risk_on < -0.25:
        tags.append("Risk-Off")
    return tags

def _confidence(diag: Dict[str, Any]) -> float:
    zmap = diag.get("z", {})
    if not zmap: 
        return 0.25
    zs = [abs(v) for v in zmap.values() if isinstance(v, (int, float))]
    if not zs:
        return 0.25
    # coverage = fraction of configured indicators present
    total_keys = sum(len(v) for v in WEIGHTS.values())
    cov = len(zmap) / max(1, total_keys)
    mag = sum(zs) / len(zs)
    # soft map: more coverage & magnitude → higher confidence
    conf = 0.3 * cov + 0.7 * _squash(mag, k=2.0)
    return float(_clip(conf, 0.0, 1.0))

# --------------------------- Main Engine ---------------------------

class MacroNarratives:
    def analyze(self, inp: NarrativeInput) -> NarrativeResult:
        diag: Dict[str, Any] = {"z": {}}
        series = inp.series or {}

        growth = _dimension_score("growth", series, diag)
        inflation = _dimension_score("inflation", series, diag)
        liquidity = _dimension_score("liquidity", series, diag)
        risk_stress = _dimension_score("risk_stress", series, diag)
        policy = _policy_score(inp.cb_stance, series, diag)
        # Risk-on is the inverse of stress plus liquidity contribution
        risk_on = _clip(_squash((-risk_stress + 0.5 * liquidity), k=1.2))
        sentiment = _combine_sentiment(inp.news_sent, inp.earnings_sent)

        scores = dict(
            growth=growth,
            inflation=inflation,
            liquidity=liquidity,
            policy=policy,
            risk_stress=risk_stress,
            risk_on=risk_on,
            sentiment=sentiment,
        )

        tags = _regime_tags(growth, inflation, liquidity, risk_on)

        bullets = self._bullets(scores)
        summary = self._summary(scores, tags)

        diag["coverage"] = len(diag["z"])
        diag["confidence"] = _confidence(diag)
        if inp.notes:
            diag["notes"] = inp.notes

        return NarrativeResult(
            scores=scores,
            regime_tags=tags,
            summary=summary,
            bullets=bullets,
            diagnostics=diag
        )

    # ------------------ Text Builders ------------------

    def _summary(self, s: Dict[str, float], tags: List[str]) -> str:
        parts: List[str] = []

        # Core regime
        parts.append(f"Regime: {', '.join(tags)}.")

        # One-liners per axis
        def _tone(x: float, pos: str, neg: str, mild: str="flat") -> str:
            if x > 0.35: return pos
            if x < -0.35: return neg
            if x > 0.10: return f"mild {pos}"
            if x < -0.10: return f"mild {neg}"
            return mild

        parts.append(f"Growth is {_tone(s['growth'],'expanding','slowing')}, "
                     f"inflation is {_tone(s['inflation'],'hotter','cooling')}.")

        parts.append(f"Liquidity looks {_tone(s['liquidity'],'looser','tighter')}, "
                     f"policy tone is {_tone(s['policy'],'hawkish','dovish')}.")

        parts.append(f"Risk regime is {_tone(s['risk_on'],'risk-on','risk-off')}, "
                     f"tape sentiment {_tone(s['sentiment'],'constructive','cautious')}.")

        return " ".join(parts)

    def _bullets(self, s: Dict[str, float]) -> List[str]:
        def b(label: str, v: float) -> str:
            return f"{label}: {v:+.2f}"
        out = [
            b("Growth", s["growth"]),
            b("Inflation", s["inflation"]),
            b("Liquidity", s["liquidity"]),
            b("Policy (hawkish↔dovish)", s["policy"]),
            b("Risk Stress", s["risk_stress"]),
            b("Risk-On", s["risk_on"]),
            b("Sentiment", s["sentiment"]),
        ]
        # Flavor tags based on extremes
        if s["inflation"] < -0.4 and s["growth"] >= 0.0:
            out.append("Disinflation tailwind in place.")
        if s["inflation"] > 0.4 and s["growth"] < 0.0:
            out.append("Stagflation risk rising (hot inflation, soft growth).")
        if s["liquidity"] > 0.4:
            out.append("Financial conditions easing supports multiples.")
        if s["risk_stress"] > 0.4:
            out.append("Risk stress elevated; watch spreads/vol.")
        return out

# ------------------------------ CLI ---------------------------------

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _main():
    import argparse
    p = argparse.ArgumentParser(description="Macro Narratives (regime + summary from simple features)")
    p.add_argument("--in", dest="inp", required=True, help="Input JSON with {series:{...}, cb_stance?, news_sent?, earnings_sent?}")
    p.add_argument("--out", dest="out", required=False, help="Write narrative JSON to this path")
    args = p.parse_args()

    raw = _load_json(args.inp)
    ni = NarrativeInput(
        series=raw.get("series", {}),
        cb_stance=raw.get("cb_stance"),
        news_sent=raw.get("news_sent"),
        earnings_sent=raw.get("earnings_sent"),
        notes=raw.get("notes", {})
    )
    res = MacroNarratives().analyze(ni)
    payload = {
        "scores": res.scores,
        "regime_tags": res.regime_tags,
        "summary": res.summary,
        "bullets": res.bullets,
        "diagnostics": res.diagnostics,
    }
    if args.out:
        _save_json(args.out, payload)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))

if __name__ == "__main__":  # pragma: no cover
    _main()