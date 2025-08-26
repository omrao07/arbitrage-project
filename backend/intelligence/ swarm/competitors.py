# backend/research/competitiors.py
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Optional pandas for neat DataFrame exports
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore


# ============================== Data models ===============================

@dataclass
class FeatureScore:
    """
    A single feature’s raw score and optional metadata.
    Scores are on a 0..1 scale unless a normalizer is provided in the schema.
    """
    value: float
    note: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Competitor:
    """
    Generic competitor entity: broker, venue, OMS, alt-data vendor, fund peer, etc.
    All features go in `features` keyed by the schema names.
    """
    name: str
    category: str                    # e.g., "broker", "data", "oms", "fund"
    region: str = ""                 # e.g., "US", "IN", "EU", "Global"
    pricing_model: str = ""          # e.g., "bps", "per-API-call", "flat"
    features: Dict[str, FeatureScore] = field(default_factory=dict)
    strengths: List[str] = field(default_factory=list)  # free text
    weaknesses: List[str] = field(default_factory=list) # free text
    url: Optional[str] = None

    def get(self, key: str, default: float = 0.0) -> float:
        fs = self.features.get(key)
        return float(fs.value) if fs else float(default)


@dataclass
class SchemaField:
    """
    Defines one comparable dimension and how to normalize / weight it.
    """
    key: str
    weight: float                    # importance weight (relative)
    higher_is_better: bool = True
    # Optional min/max for linear normalization of raw (non 0..1) inputs
    min_raw: Optional[float] = None
    max_raw: Optional[float] = None
    # Optional transform (log, sqrt) for raw → normalized before direction/weight
    transform: Optional[str] = None  # "log", "sqrt", None


@dataclass
class ScoringSchema:
    """
    Collection of fields with helpers for normalization and composite scoring.
    """
    fields: List[SchemaField]
    name: str = "default"

    def field_map(self) -> Dict[str, SchemaField]:
        return {f.key: f for f in self.fields}

    def normalize(self, key: str, raw: float) -> float:
        f = self.field_map()[key]
        x = float(raw)

        # pre-transform
        if f.transform == "log":
            x = math.log(max(1e-12, x + 1.0))
        elif f.transform == "sqrt":
            x = math.sqrt(max(0.0, x))

        # linear scale if bounds provided
        if f.min_raw is not None and f.max_raw is not None and f.max_raw > f.min_raw:
            x = (x - f.min_raw) / (f.max_raw - f.min_raw)
            x = max(0.0, min(1.0, x))

        # if no bounds and value looks already in [0,1], trust it; else clamp 0..1 softly
        if f.min_raw is None and f.max_raw is None:
            if not (0.0 <= x <= 1.0):
                # heuristic clamp for unknown ranges
                x = 1.0 / (1.0 + math.exp(-x))  # squashing

        # direction
        if not f.higher_is_better:
            x = 1.0 - x

        return max(0.0, min(1.0, x))

    def composite(self, comp: Competitor) -> Tuple[float, Dict[str, float]]:
        """
        Returns (overall_score, per_dimension_scores)
        """
        weights = {f.key: max(0.0, f.weight) for f in self.fields}
        tot_w = sum(weights.values()) or 1.0
        per: Dict[str, float] = {}
        s = 0.0
        for k, w in weights.items():
            raw = comp.get(k, 0.0)
            per[k] = self.normalize(k, raw)
            s += w * per[k]
        return (s / tot_w, per)


# ============================== Repository ===============================

class CompetitorBook:
    """
    Small in-memory store with ranking, filtering, exports, and text SWOT.
    """
    def __init__(self, schema: ScoringSchema):
        self.schema = schema
        self._book: Dict[str, Competitor] = {}

    def upsert(self, c: Competitor) -> None:
        self._book[c.name] = c

    def remove(self, name: str) -> bool:
        return self._book.pop(name, None) is not None

    def get(self, name: str) -> Optional[Competitor]:
        return self._book.get(name)

    def list(self, *, category: Optional[str] = None, region: Optional[str] = None) -> List[Competitor]:
        arr = list(self._book.values())
        if category:
            arr = [c for c in arr if c.category.lower() == category.lower()]
        if region:
            arr = [c for c in arr if (c.region or "").lower() == region.lower()]
        return sorted(arr, key=lambda c: c.name.lower())

    def rank(self, *, top_k: Optional[int] = None, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]: # type: ignore
        comps = self.list(
            category=(filters or {}).get("category"),
            region=(filters or {}).get("region"),
        )
        scored: List[Dict[str, Any]] = []
        for c in comps:
            total, per = self.schema.composite(c)
            scored.append({
                "name": c.name,
                "category": c.category,
                "region": c.region,
                "url": c.url,
                "score": round(total, 4),
                "dimensions": {k: round(v, 4) for k, v in per.items()},
                "strengths": c.strengths,
                "weaknesses": c.weaknesses,
                "pricing_model": c.pricing_model,
            })
        scored.sort(key=lambda x: (-x["score"], x["name"]))
        return scored[:top_k] if top_k else scored

    # ---------- Exports ----------
    def to_json(self) -> str:
        payload = {
            "schema": {
                "name": self.schema.name,
                "fields": [asdict(f) for f in self.schema.fields],
            },
            "competitors": [asdict(_flatten_features(c)) for c in self._book.values()], # type: ignore
        }
        return json.dumps(payload, indent=2)

    @staticmethod
    def from_json(s: str) -> "CompetitorBook":
        o = json.loads(s)
        schema = ScoringSchema([SchemaField(**f) for f in o["schema"]["fields"]], name=o["schema"].get("name","default"))
        book = CompetitorBook(schema)
        for raw in o.get("competitors", []):
            c = _unflatten_features(raw)
            book.upsert(c)
        return book

    def to_csv(self, path: str) -> None:
        rows = []
        for c in self._book.values():
            row = {
                "name": c.name, "category": c.category, "region": c.region,
                "pricing_model": c.pricing_model, "url": c.url or "",
                "strengths": " | ".join(c.strengths), "weaknesses": " | ".join(c.weaknesses),
            }
            for k, fs in c.features.items():
                row[f"feat:{k}"] = fs.value # type: ignore
            rows.append(row)
        keys = sorted({k for r in rows for k in r.keys()})
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    def to_dataframe(self):
        if pd is None:
            raise RuntimeError("pandas not installed")
        ranked = self.rank()
        return pd.json_normalize(ranked)


# ============================== SWOT from text ===============================

_SWOT_LEX = {
    "strengths": [
        "low latency", "low fees", "deep liquidity", "broad coverage", "reliable",
        "mature", "strong api", "good docs", "enterprise", "compliant", "trusted",
    ],
    "weaknesses": [
        "high fees", "limited coverage", "unstable", "slow", "downtime", "laggy",
        "poor docs", "beta", "restricted", "delayed", "complex pricing",
    ],
}

def auto_swot(text: str) -> Tuple[List[str], List[str]]:
    t = " " + " ".join((text or "").lower().split()) + " "
    s = [p for p in _SWOT_LEX["strengths"] if f" {p} " in t]
    w = [p for p in _SWOT_LEX["weaknesses"] if f" {p} " in t]
    return (s, w)


# ============================== Convenience ===============================

def _flatten_features(c: Competitor) -> Dict[str, Any]:
    d = asdict(c)
    d["features"] = {k: {"value": fs.value, "note": fs.note, "meta": fs.meta} for k, fs in c.features.items()}
    return d

def _unflatten_features(d: Dict[str, Any]) -> Competitor:
    feats = {k: FeatureScore(**v) for k, v in d.get("features", {}).items()}
    return Competitor(
        name=d["name"], category=d["category"], region=d.get("region",""),
        pricing_model=d.get("pricing_model",""), features=feats,
        strengths=d.get("strengths", []), weaknesses=d.get("weaknesses", []),
        url=d.get("url")
    )

def build_default_schema(kind: str = "broker") -> ScoringSchema:
    """
    Ready-to-use schemas:
      - broker: execution quality, venue coverage, asset coverage, fees (lower better), reliability
      - data: latency, coverage, freshness, cost (lower better), docs/support
      - fund: sharpe, capacity, drawdown (lower better), fees (lower better), transparency
    """
    kind = (kind or "broker").lower()
    if kind == "data":
        fields = [
            SchemaField("latency_ms", 0.25, higher_is_better=False, min_raw=1, max_raw=500, transform="log"),
            SchemaField("coverage_breadth", 0.25, higher_is_better=True, min_raw=10, max_raw=1000, transform="log"),
            SchemaField("freshness_sec", 0.20, higher_is_better=False, min_raw=0, max_raw=120, transform="sqrt"),
            SchemaField("cost_usd_mo", 0.15, higher_is_better=False, min_raw=0, max_raw=5000, transform="log"),
            SchemaField("docs_quality", 0.15, higher_is_better=True),  # already 0..1
        ]
    elif kind == "fund":
        fields = [
            SchemaField("sharpe", 0.30, higher_is_better=True, min_raw=0.0, max_raw=3.0),
            SchemaField("capacity_usd_mm", 0.20, higher_is_better=True, min_raw=0, max_raw=2000, transform="log"),
            SchemaField("max_dd_pct", 0.20, higher_is_better=False, min_raw=0, max_raw=40),
            SchemaField("mgmt_fee_pct", 0.15, higher_is_better=False, min_raw=0, max_raw=3.0),
            SchemaField("transparency", 0.15, higher_is_better=True),
        ]
    else:  # broker / venue / OMS
        fields = [
            SchemaField("exec_quality", 0.30, higher_is_better=True),              # 0..1 composite you set
            SchemaField("venue_coverage", 0.20, higher_is_better=True, min_raw=1, max_raw=80, transform="log"),
            SchemaField("asset_coverage", 0.15, higher_is_better=True, min_raw=1, max_raw=20),
            SchemaField("fees_bps", 0.20, higher_is_better=False, min_raw=0.0, max_raw=10.0),
            SchemaField("reliability", 0.15, higher_is_better=True),               # 0..1
        ]
    return ScoringSchema(fields, name=f"default:{kind}")


# ============================== Tiny demo ===============================

if __name__ == "__main__":
    # Example: brokers for India/US flow
    schema = build_default_schema("broker")
    book = CompetitorBook(schema)

    # Fill a few (values are examples—replace with your real measurements)
    zerodha = Competitor(
        name="Zerodha", category="broker", region="IN", pricing_model="flat",
        features={
            "exec_quality": FeatureScore(0.78, "Good fills; smart-router basic"),
            "venue_coverage": FeatureScore(3, "NSE, BSE, MCX"),
            "asset_coverage": FeatureScore(4, "Equity, F&O, Commodities, MF (no US cash)"),
            "fees_bps": FeatureScore(0.5),
            "reliability": FeatureScore(0.85),
        },
        strengths=["low fees", "compliant", "popular"],
        weaknesses=["limited international coverage"],
        url="https://zerodha.com",
    )

    ibkr = Competitor(
        name="IBKR", category="broker", region="Global", pricing_model="bps",
        features={
            "exec_quality": FeatureScore(0.86, "SmartRouter+ darks"),
            "venue_coverage": FeatureScore(60, "Global exchanges"),
            "asset_coverage": FeatureScore(12),
            "fees_bps": FeatureScore(0.2),
            "reliability": FeatureScore(0.9),
        },
        strengths=["deep liquidity", "broad coverage"],
        weaknesses=["complex pricing"],
        url="https://interactivebrokers.com",
    )

    book.upsert(zerodha)
    book.upsert(ibkr)

    # Rank
    ranked = book.rank()
    print(json.dumps(ranked, indent=2))

    # Optional DataFrame
    if pd is not None:
        df = book.to_dataframe()
        print(df.head())