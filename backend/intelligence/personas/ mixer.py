# backend/common/mixer.py
"""
Signal & Proposal Mixers (stdlib only)

Use cases
---------
1) Merge multiple signal feeds into one canonical map:
   - linear or rank-based blends
   - z/minmax normalization, winsorize/clip, recency decay
   - per-source weights & per-key overrides
2) (Optional) Aggregate agent proposals into per-symbol intents,
   if you want a fast/offline mixer without the full Coordinator.

No third-party dependencies.

Examples
--------
from backend.common.mixer import SignalMixer, MixerConfig, ProposalMixer
mx = SignalMixer(MixerConfig(
    mode="linear",
    weights={"altdata": 0.5, "tech": 0.3, "nlp": 0.2},
    norm="z",
    winsor=(0.01, 0.99),
    recency_half_life_sec=86_400,  # 1 day
))
out = mx.mix({
    "altdata": {"AAPL.mom": 1.2, "BTC.sent": 0.35},
    "tech":    {"AAPL.mom": 0.9, "BTC.sent": 0.10},
    "nlp":     {"BTC.sent": 0.55},
}, timestamps={"altdata": 1724200000, "tech": 1724190000, "nlp": 1724210000})

# Optional: proposals aggregation (agent outputs from agents/base.py)
from agents.base import Proposal
pmx = ProposalMixer(min_abs_score=0.15)
legs = pmx.aggregate([
    ("crypto", crypto_proposal),
    ("equities", equities_proposal),
])
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable, Any

try:
    import yaml  # optional for load_config, not required at runtime
    _HAVE_YAML = True
except Exception:
    _HAVE_YAML = False


# ------------------------------ Configs --------------------------------

@dataclass
class MixerConfig:
    """
    Global config for SignalMixer.

    mode: "linear" (weighted sum) or "rank" (average of ranks)
    weights: per-source weight (auto-normalized)
    key_overrides: per-key weight overrides: {"AAPL.mom": {"alt":0.7,"tech":0.3}}
    norm: None | "z" | "minmax" (applied per-source before mixing)
    z_eps: numerical floor for std in z-score
    minmax_clip: (lo, hi) bounds used if norm="minmax" and no range known (defaults 1st-99th percentile proxy)
    winsor: optional (lo_p, hi_p) percentiles to trim extremes AFTER normalization (proxy via rank-based clamp)
    clip: optional absolute clip (lo, hi) after winsor
    recency_half_life_sec: optional half-life for source-level decay using provided timestamps
    allowlist / blocklist: optional sets of keys to include/exclude
    """
    mode: str = "linear"
    weights: Dict[str, float] = field(default_factory=dict)
    key_overrides: Dict[str, Dict[str, float]] = field(default_factory=dict)
    norm: Optional[str] = "z"
    z_eps: float = 1e-6
    minmax_clip: Tuple[float, float] = (-3.0, 3.0)
    winsor: Optional[Tuple[float, float]] = (0.01, 0.99)
    clip: Optional[Tuple[float, float]] = (-5.0, 5.0)
    recency_half_life_sec: Optional[float] = None
    allowlist: Optional[Iterable[str]] = None
    blocklist: Optional[Iterable[str]] = None

    @classmethod
    def from_yaml(cls, path: str) -> "MixerConfig":
        if not _HAVE_YAML:
            raise RuntimeError("pyyaml not installed; cannot load yaml config")
        with open(path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        return cls(**doc)


# ------------------------------ Utilities ------------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _normalize_source(values: Dict[str, float], mode: Optional[str], *, z_eps: float, minmax_clip: Tuple[float, float]) -> Dict[str, float]:
    if not values:
        return {}
    xs = list(values.values())
    if mode is None:
        return dict(values)
    if mode == "z":
        mu = statistics.fmean(xs)
        # sample std with ddof=1 where possible
        if len(xs) > 1:
            var = sum((v - mu) ** 2 for v in xs) / (len(xs) - 1)
        else:
            var = 0.0
        sd = math.sqrt(max(z_eps, var))
        return {k: (v - mu) / (sd or z_eps) for k, v in values.items()}
    if mode == "minmax":
        lo = min(xs); hi = max(xs)
        if hi - lo <= 0:
            lo, hi = minmax_clip
        return {k: (2.0 * (v - lo) / max(1e-12, (hi - lo)) - 1.0) for k, v in values.items()}
    raise ValueError(f"unknown norm mode '{mode}'")

def _winsorize(values: Dict[str, float], lo_p: float, hi_p: float) -> Dict[str, float]:
    if not values:
        return {}
    xs = sorted(values.values())
    n = len(xs)
    def q(p: float) -> float:
        if n == 1:
            return xs[0]
        idx = _clamp(p * (n - 1), 0.0, n - 1)
        i0, i1 = int(math.floor(idx)), int(math.ceil(idx))
        if i0 == i1:
            return xs[i0]
        t = idx - i0
        return xs[i0] * (1 - t) + xs[i1] * t
    lo_v = q(lo_p); hi_v = q(hi_p)
    return {k: _clamp(v, lo_v, hi_v) for k, v in values.items()}

def _apply_clip(values: Dict[str, float], bounds: Tuple[float, float]) -> Dict[str, float]:
    lo, hi = bounds
    return {k: _clamp(v, lo, hi) for k, v in values.items()}

def _decay_weight(ts: Optional[float], now_ts: Optional[float], half_life: float) -> float:
    if ts is None or now_ts is None or half_life <= 0:
        return 1.0
    dt = max(0.0, now_ts - ts)
    return 0.5 ** (dt / half_life)

def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    # zero-out negatives, renormalize; if empty, returns {}
    ww = {k: max(0.0, float(v)) for k, v in w.items()}
    s = sum(ww.values())
    return {k: (v / s if s > 0 else 0.0) for k, v in ww.items()}


# ------------------------------ SignalMixer ----------------------------

class SignalMixer:
    """
    Blend signals from multiple sources into a single key -> score map.

    mix(sources, timestamps=None, now_ts=None) -> Dict[key, score]

    - sources: {"src_name": {"key": float, ...}, ...}
    - timestamps: optional {"src_name": epoch_seconds} for recency decay
    - now_ts: optional epoch; required if using recency_half_life_sec
    """

    def __init__(self, cfg: Optional[MixerConfig] = None):
        self.cfg = cfg or MixerConfig()

    def mix(
        self,
        sources: Dict[str, Dict[str, float]],
        *,
        timestamps: Optional[Dict[str, float]] = None,
        now_ts: Optional[float] = None
    ) -> Dict[str, float]:
        if not sources:
            return {}

        cfg = self.cfg
        base_w = _normalize_weights(cfg.weights)
        key_over = {k: _normalize_weights(v) for k, v in (cfg.key_overrides or {}).items()}

        # Build full key universe (respect allow/block lists)
        keys: set[str] = set()
        for src, kv in sources.items():
            if not isinstance(kv, dict):
                continue
            keys.update(kv.keys())
        if cfg.allowlist is not None:
            allow = set(cfg.allowlist)
            keys = {k for k in keys if k in allow}
        if cfg.blocklist is not None:
            block = set(cfg.blocklist)
            keys = {k for k in keys if k not in block}

        # Normalize each source first (if cfg.norm)
        normed: Dict[str, Dict[str, float]] = {}
        for src, kv in sources.items():
            if not isinstance(kv, dict) or not kv:
                continue
            normed[src] = _normalize_source(kv, cfg.norm, z_eps=cfg.z_eps, minmax_clip=cfg.minmax_clip)

        # Optionally winsorize/clip per-source
        if cfg.winsor:
            lo_p, hi_p = cfg.winsor
            for src in list(normed.keys()):
                normed[src] = _winsorize(normed[src], lo_p, hi_p)
        if cfg.clip:
            for src in list(normed.keys()):
                normed[src] = _apply_clip(normed[src], cfg.clip)

        # Recency-decay per source
        decay: Dict[str, float] = {}
        if cfg.recency_half_life_sec:
            for src in normed.keys():
                ts = (timestamps or {}).get(src)
                decay[src] = _decay_weight(ts, now_ts, cfg.recency_half_life_sec)
        else:
            for src in normed.keys():
                decay[src] = 1.0

        # Mix per key
        out: Dict[str, float] = {}
        for k in keys:
            # pick weights: per-key overrides else global
            w = key_over.get(k, base_w)
            if not w:
                # if no weights provided, uniform over sources that have this key
                present = [s for s in normed.keys() if k in normed[s]]
                if not present:
                    continue
                w = {s: 1.0 / len(present) for s in present}
            # ensure we only use sources that actually provide k
            usable = {s: w.get(s, 0.0) for s in normed.keys() if k in normed[s]}
            usable = {s: v for s, v in usable.items() if v > 0}
            if not usable:
                continue

            # incorporate decay
            if decay:
                usable = {s: usable[s] * decay.get(s, 1.0) for s in usable}
                usable = _normalize_weights(usable)

            if self.cfg.mode == "linear":
                v = sum(normed[s][k] * usable.get(s, 0.0) for s in usable.keys())
            elif self.cfg.mode == "rank":
                # average of fractional ranks across sources (higher -> stronger)
                ranks: List[float] = []
                for s in usable.keys():
                    vals = normed[s]
                    # rank of k within source s
                    sorted_items = sorted(vals.items(), key=lambda kv: kv[1])
                    # fractional rank in [0,1]
                    idx = next(i for i, (kk, _) in enumerate(sorted_items) if kk == k)
                    r = idx / max(1, len(sorted_items) - 1)
                    ranks.append(r)
                # convert rank (0=low) to score (centered around 0)
                if ranks:
                    v = (sum(ranks) / len(ranks)) * 2.0 - 1.0
                else:
                    v = 0.0
            else:
                raise ValueError(f"unknown mode '{self.cfg.mode}'")

            out[k] = float(v)

        # Optional final clip (safety)
        if cfg.clip:
            out = _apply_clip(out, cfg.clip)
        return out


# ------------------------------ ProposalMixer --------------------------

class ProposalMixer:
    """
    Lightweight aggregation of multiple agents' Proposals into per-symbol legs.

    Input: List[Tuple[agent_name, Proposal-like]]
      - Proposal-like must have: .orders (list), .score (float), .confidence (float), .thesis (str)
      - Order has: symbol, side ("BUY"/"SELL"), qty (float), venue (opt), meta (dict with optional "score")

    Output: List[Dict] legs with:
      {symbol, side, qty, venue, net_score, contributors, rationale}

    Notes:
    - This is a minimal alternative to Coordinator._negotiate() for quick tests.
    - Quantity = median(agent_qtys) * scale(|net_score|)
    """

    def __init__(
        self,
        *,
        min_abs_score: float = 0.20,
        qty_scale_min: float = 0.75,
        qty_scale_max: float = 1.75
    ) -> None:
        self.min_abs_score = float(min_abs_score)
        self.qty_scale_min = float(qty_scale_min)
        self.qty_scale_max = float(qty_scale_max)

    def aggregate(self, items: List[Tuple[str, Any]]) -> List[Dict[str, Any]]:
        # symbol bucket
        buckets: Dict[str, Dict[str, Any]] = {}
        for name, prop in items:
            if not getattr(prop, "orders", None):
                continue
            score = float(getattr(prop, "score", 0.0) or 0.0)
            conf = float(getattr(prop, "confidence", 0.5) or 0.5)
            contrib = score * conf
            for o in getattr(prop, "orders"):
                side = (getattr(o, "side", "") or "").upper()
                sym = getattr(o, "symbol", "")
                qty = float(getattr(o, "qty", 0.0) or 0.0)
                venue = getattr(o, "venue", None)
                # prefer per-leg meta score if present
                o_meta = getattr(o, "meta", {}) or {}
                leg_sc = float(o_meta.get("score", score))
                c = buckets.setdefault(sym, {"sum": 0.0, "qtys": [], "venues": [], "contributors": []})
                sgn = +1.0 if side == "BUY" else -1.0
                c["sum"] += sgn * leg_sc * conf
                c["qtys"].append(qty)
                if venue:
                    c["venues"].append(venue)
                c["contributors"].append(name)

        legs: List[Dict[str, Any]] = []
        for sym, b in buckets.items():
            net = float(b["sum"])
            if abs(net) < self.min_abs_score:
                continue
            base_qty = _median(b["qtys"]) if b["qtys"] else 0.0
            scale = self._scale_from_net(abs(net))
            side = "BUY" if net > 0 else "SELL"
            venue = _plurality(b["venues"]) if b["venues"] else None
            legs.append({
                "symbol": sym,
                "side": side,
                "qty": max(0.0, base_qty * scale),
                "venue": venue,
                "net_score": net,
                "contributors": sorted(set(b["contributors"])),
                "rationale": f"net={net:.2f} scale={scale:.2f} base≈{base_qty:g}",
            })
        # sort by |net_score| desc
        legs.sort(key=lambda d: abs(d.get("net_score", 0.0)), reverse=True)
        return legs

    def _scale_from_net(self, net_abs: float) -> float:
        x0, x1 = self.min_abs_score, 1.0
        y0, y1 = self.qty_scale_min, self.qty_scale_max
        x = _clamp((net_abs - x0) / max(1e-9, (x1 - x0)), 0.0, 1.0)
        return y0 + x * (y1 - y0)


# ------------------------------ Small helpers --------------------------

def _median(xs: Iterable[float]) -> float:
    arr = sorted(float(x) for x in xs if x is not None)
    n = len(arr)
    if n == 0:
        return 0.0
    m = n // 2
    return arr[m] if n % 2 == 1 else 0.5 * (arr[m - 1] + arr[m])

def _plurality(items: Iterable[str]) -> Optional[str]:
    counts: Dict[str, int] = {}
    for it in items:
        counts[it] = counts.get(it, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


# ------------------------------ Tiny demo ------------------------------

if __name__ == "__main__":
    # Quick smoke for SignalMixer
    cfg = MixerConfig(
        mode="linear",
        weights={"alt": 0.6, "tech": 0.3, "nlp": 0.1},
        norm="z",
        winsor=(0.05, 0.95),
        clip=(-3, 3),
        recency_half_life_sec=24*3600,
    )
    mx = SignalMixer(cfg)
    mixed = mx.mix(
        {
            "alt": {"AAPL.mom": 1.2, "BTC.sent": 0.5, "EURUSD.carry": 0.02},
            "tech": {"AAPL.mom": 0.8, "BTC.sent": 0.2},
            "nlp": {"BTC.sent": 0.7},
        },
        timestamps={"alt": 1_724_180_000, "tech": 1_724_160_000, "nlp": 1_724_200_000},
        now_ts=1_724_210_000,
    )
    print("[mixer] mixed:", mixed)