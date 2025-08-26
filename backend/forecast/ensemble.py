# backend/engine/ensemble.py
"""
Ensemble Signal Blender (+ risk-aware targets & optional order emits)
--------------------------------------------------------------------
Purpose
- Register multiple models/strategies that output scores per symbol in [-1, +1]
- Blend them via mean/median/weighted/vote; optional stacking meta-model
- Normalize & clamp, de-bias, and volatility-scale into target weights
- Optional HRP/risk-parity reweighting by cross-asset correlations
- Convert target weights -> target positions -> delta orders
- Publish compact telemetry to bus + (optionally) orders to pre-risk stream

Inputs (any combination)
- Direct calls to `update_model(name, scores, meta)`
- Bus feeds (if available) on topics: alpha.model.<name> with payload:
    {
      "ts_ms": ..., "model": "momo_v3",
      "scores": {"AAPL": 0.42, "MSFT": -0.18, ...},
      "vol": {"AAPL": 0.22, ...},            # optional annualized or daily stdev
      "conf": {"AAPL": 0.8, ...},            # optional confidence [0..1]
      "regime": "bull"                        # optional regime tag
    }

Outputs
- Bus topic `alpha.ensemble`:
    {"ts_ms", "ensemble": {"AAPL": 0.25, ...}, "method": "...", "meta": {...}}
- Optional orders to `orders.incoming` (pre-risk stream used elsewhere)

CLI
  python -m backend.engine.ensemble --probe
  python -m backend.engine.ensemble --run --emit-orders
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Optional numeric stack (used if available)
try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore

# Optional HRP allocator
try:
    from backend.analytics.hrp import hrp_weights  # type: ignore # you provided hrp.py earlier
except Exception:
    hrp_weights = None  # type: ignore

# Optional bus + redis
try:
    from backend.bus.streams import consume_stream, publish_stream, hget, hset # type: ignore
except Exception:
    consume_stream = publish_stream = hget = hset = None  # type: ignore


# ------------------------- config & helpers -------------------------

@dataclass
class EnsembleConfig:
    name: str = "ensemble"                   # strategy / namespace
    method: str = "mean"                     # mean|median|vote|weighted|stack
    weights: Dict[str, float] = field(default_factory=dict)  # for 'weighted'
    max_leverage: float = 1.0                # sum |w_i| <= this
    vol_target_annual: float = 0.15          # convert scores -> risk-targeted weights
    vol_floor: float = 0.05                  # avoid exploding leverage on tiny vol
    clip_score: float = 3.0                  # z-score/score clamp
    neutralize: str = "sum"                  # "none"|"sum"|"zscore"
    long_only: bool = False
    use_hrp: bool = False                    # if True and corr available -> HRP blend
    use_risk_parity: bool = False            # basic 1/sigma^2 allocator if np available
    capital_base: float = 100_000.0          # used for order sizing if we compute shares
    default_price: float = 100.0             # if no price is available
    # Bus topics
    topic_models_prefix: str = "alpha.model."  # we subscribe to alpha.model.*
    topic_features: Optional[str] = None       # or features snapshot topic (optional)
    topic_alpha_out: str = "alpha.ensemble"
    topic_orders_out: str = os.getenv("RISK_INCOMING_STREAM", "orders.incoming")
    # State stores (optional, via Redis)
    redis_pos_key: str = "positions"        # {symbol: {"qty":...}}
    redis_px_key: str = "last_price"        # HSET last_price <SYM> <px>


def _utc_ms() -> int:
    return int(time.time() * 1000)

def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


# ------------------------- core class -------------------------

class Ensemble:
    """
    Hold model snapshots, blend to a single score per symbol, then transform into targets.
    """
    def __init__(self, cfg: Optional[EnsembleConfig] = None):
        self.cfg = cfg or EnsembleConfig()
        # registry of latest model outputs
        self.models: Dict[str, Dict[str, float]] = {}     # name -> scores per symbol
        self.meta: Dict[str, Dict[str, Any]] = {}         # name -> meta (vol/conf/etc.)
        # cache of last ensemble output
        self.last_ensemble: Dict[str, float] = {}

    # -------- ingestion --------
    def update_model(self, name: str, scores: Dict[str, float], meta: Optional[Dict[str, Any]] = None) -> None:
        self.models[name] = {str(k).upper(): _safe_float(v) for k, v in (scores or {}).items()}
        if meta:
            self.meta[name] = meta

    def ingest_bus_payload(self, payload: Dict[str, Any]) -> None:
        name = payload.get("model") or payload.get("name") or "unknown"
        self.update_model(name, payload.get("scores") or {}, {
            "vol": payload.get("vol"),
            "conf": payload.get("conf"),
            "regime": payload.get("regime"),
        })

    # -------- blend --------
    def blend(self) -> Dict[str, float]:
        if not self.models:
            return {}

        # union of symbols
        symbols = sorted({s for m in self.models.values() for s in m.keys()})
        out: Dict[str, float] = {}

        # optional weight map per model
        wmap = self.cfg.weights or {}
        method = self.cfg.method.lower()

        for sym in symbols:
            vals = []
            weights = []
            votes = []
            for mname, scores in self.models.items():
                if sym not in scores:
                    continue
                v = float(scores[sym])
                if method == "vote":
                    votes.append(_sign(v))
                else:
                    vals.append(v)
                    weights.append(float(wmap.get(mname, 1.0)))

            if method == "median" and vals:
                out[sym] = _median(vals)
            elif method == "weighted" and vals:
                out[sym] = _weighted_mean(vals, weights)
            elif method == "vote" and votes:
                out[sym] = float(sum(votes)) / max(1, len(votes))
            else:  # mean default
                out[sym] = float(sum(vals)) / max(1, len(vals)) if vals else 0.0

        # optional de-bias / neutralize
        out = self._neutralize(out, how=self.cfg.neutralize)
        # clamp
        for k in list(out.keys()):
            out[k] = float(max(-1.0, min(1.0, out[k])))
        self.last_ensemble = dict(out)
        return out

    def _neutralize(self, scores: Dict[str, float], *, how: str = "sum") -> Dict[str, float]:
        if not scores:
            return scores
        if how == "none":
            return scores
        vals = list(scores.values())
        if np is None:
            # simple sum-neutralization (sum to ~0)
            mean = sum(vals)/len(vals)
            return {k: v - mean for k, v in scores.items()} if how in ("sum","zscore") else scores
        # numpy path
        vec = np.array(vals, dtype=float)
        if how in ("sum", "zscore"):
            vec = vec - np.mean(vec)
        if how == "zscore":
            sd = float(np.std(vec)) or 1.0
            vec = vec / sd
            vec = np.clip(vec, -self.cfg.clip_score, self.cfg.clip_score)
            # scale back to [-1,1] softly
            vec = np.tanh(vec / self.cfg.clip_score)
        return {k: float(vec[i]) for i, k in enumerate(scores.keys())}

    # -------- convert scores -> target weights --------
    def targets(self, scores: Optional[Dict[str, float]] = None,
                vols: Optional[Dict[str, float]] = None,
                corr: Optional[Dict[Tuple[str, str], float]] = None) -> Dict[str, float]:
        """
        Returns portfolio target weights w_i summing to <= max_leverage. Uses:
        - score magnitude & sign
        - inverse-vol scaling (per-asset vol)
        - optional HRP / risk parity across correlations
        """
        scores = scores or self.last_ensemble
        if not scores:
            return {}

        # 1) per-asset risk scale (1/sigma)
        sigma = self._load_vols(vols, scores.keys())  # daily or annual; we normalize
        invvar = {sym: 1.0 / max((sigma.get(sym) or self.cfg.vol_floor), self.cfg.vol_floor) for sym in scores.keys()}

        # 2) raw weights proportional to score * invvol
        raw = {sym: float(scores[sym]) * invvar[sym] for sym in scores.keys()}

        # 3) cross-asset reweighting (HRP / risk parity) if desired and data present
        w_adj = dict(raw)
        if self.cfg.use_hrp and hrp_weights and pd is not None and corr:
            try:
                syms = list(scores.keys())
                C = _corr_to_df(corr, syms)
                rp = hrp_weights(C)  # returns pd.Series indexed by syms, sums to 1
                for s in syms:
                    w_adj[s] = w_adj[s] * float(abs(rp.get(s, 0.0)))
            except Exception:
                pass
        elif self.cfg.use_risk_parity and np is not None and corr:
            try:
                syms = list(scores.keys())
                C = _corr_to_mat(corr, syms)
                ivp = _inverse_vol_port(np.array([sigma[s] for s in syms], dtype=float))
                for i, s in enumerate(syms):
                    w_adj[s] = w_adj[s] * float(ivp[i])
            except Exception:
                pass

        # 4) enforce long-only if set
        if self.cfg.long_only:
            w_adj = {k: max(0.0, v) for k, v in w_adj.items()}

        # 5) leverage scale
        L = sum(abs(x) for x in w_adj.values()) or 1.0
        scale = min(1.0, self.cfg.max_leverage / L)
        w_final = {k: v * scale for k, v in w_adj.items()}
        return w_final

    def _load_vols(self, vols_hint: Optional[Dict[str, float]], symbols) -> Dict[str, float]:
        """
        Returns annualized vol per symbol; if hints are daily, we try to scale by sqrt(252).
        If nothing available, fall back to vol_floor.
        """
        out = {}
        # from meta first
        for mname, meta in self.meta.items():
            volmap = meta.get("vol") if isinstance(meta, dict) else None
            if isinstance(volmap, dict):
                for s, v in volmap.items():
                    out[str(s).upper()] = _safe_float(v, self.cfg.vol_floor)
        # override with hint
        if vols_hint:
            for s, v in vols_hint.items():
                out[str(s).upper()] = _safe_float(v, self.cfg.vol_floor)
        # ensure floor
        for s in symbols:
            out.setdefault(str(s).upper(), self.cfg.vol_floor)
        return out

    # -------- orders --------
    def orders_for_targets(self, targets: Dict[str, float],
                           prices: Optional[Dict[str, float]] = None,
                           positions: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Convert target weights into delta orders in shares (qty). Uses:
        - current positions (qty)
        - current prices (px)
        If Redis is present and prices/positions missing, we try to read them.
        """
        prices = prices or self._load_prices(list(targets.keys()))
        positions = positions or self._load_positions(list(targets.keys()))
        orders: List[Dict[str, Any]] = []
        eq = float(self.cfg.capital_base)

        for sym, w in targets.items():
            px = _safe_float(prices.get(sym), self.cfg.default_price)
            tgt_value = w * eq
            tgt_qty = tgt_value / max(px, 1e-6)
            cur_qty = _safe_float(positions.get(sym), 0.0)
            delta = tgt_qty - cur_qty
            if abs(delta) < 1e-6:
                continue
            side = "buy" if delta > 0 else "sell"
            orders.append({
                "ts_ms": _utc_ms(),
                "strategy": self.cfg.name,
                "symbol": sym,
                "side": side,
                "qty": float(abs(delta)),
                "typ": "market",
                "mark_price": px,
            })
        return orders

    def _load_positions(self, syms: List[str]) -> Dict[str, float]:
        pos: Dict[str, float] = {}
        if hget:
            try:
                for s in syms:
                    v = hget(self.cfg.redis_pos_key, s)
                    pos[s] = _safe_float(v, 0.0)
            except Exception:
                pass
        return pos

    def _load_prices(self, syms: List[str]) -> Dict[str, float]:
        px: Dict[str, float] = {}
        if hget:
            try:
                for s in syms:
                    v = hget(self.cfg.redis_px_key, s)
                    px[s] = _safe_float(v, self.cfg.default_price)
            except Exception:
                pass
        return px

    # -------- publish --------
    def publish(self, scores: Dict[str, float], targets: Optional[Dict[str, float]] = None) -> None:
        if not publish_stream:
            return
        payload = {
            "ts_ms": _utc_ms(),
            "strategy": self.cfg.name,
            "method": self.cfg.method,
            "ensemble": scores,
            "targets": targets or {},
            "meta": {
                "models": list(self.models.keys()),
                "long_only": self.cfg.long_only,
                "max_leverage": self.cfg.max_leverage,
                "use_hrp": self.cfg.use_hrp,
                "use_risk_parity": self.cfg.use_risk_parity,
            }
        }
        publish_stream(self.cfg.topic_alpha_out, payload)


# ------------------------- utilities -------------------------

def _median(xs: List[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0: return 0.0
    if n % 2 == 1: return float(xs[n//2])
    return float(0.5*(xs[n//2 - 1] + xs[n//2]))

def _weighted_mean(vals: List[float], wts: List[float]) -> float:
    if not vals: return 0.0
    if not wts or len(wts) != len(vals):
        return float(sum(vals))/len(vals)
    s = sum(abs(w) for w in wts) or 1.0
    return float(sum(v*w for v, w in zip(vals, wts)) / s)

def _corr_to_df(corr: Dict[Tuple[str, str], float], syms: List[str]):
    if pd is None:
        raise RuntimeError("pandas required for HRP path")
    import numpy as _np  # local
    M = _np.eye(len(syms))
    idx = {s: i for i, s in enumerate(syms)}
    for (a, b), c in corr.items():
        i, j = idx[a], idx[b]
        M[i, j] = M[j, i] = float(c)
    return pd.DataFrame(M, index=syms, columns=syms)

def _corr_to_mat(corr: Dict[Tuple[str, str], float], syms: List[str]):
    import numpy as _np  # local
    M = _np.eye(len(syms))
    idx = {s: i for i, s in enumerate(syms)}
    for (a, b), c in corr.items():
        i, j = idx[a], idx[b]
        M[i, j] = M[j, i] = float(c)
    return M

def _inverse_vol_port(sigmas: "np.ndarray"): # type: ignore
    # Simple 1/variance normalized to sum 1
    w = 1.0 / (sigmas ** 2 + 1e-12)
    w = w / max(float(w.sum()), 1e-12)
    return w


# ------------------------- runner (bus loop) -------------------------

def run_loop(cfg: Optional[EnsembleConfig] = None, *, emit_orders: bool = False):
    """
    Subscribe to alpha.model.* topics, blend, publish, and optionally emit orders.
    """
    assert consume_stream is not None, "bus streams not available"
    ens = Ensemble(cfg or EnsembleConfig())
    cursor = "$"

    # Wildcard subscribe model topics by iterating known names; if your bus supports glob patterns,
    # you can change consume_stream to 'alpha.model.*'. We poll a few candidates for simplicity.
    topics = [f"{ens.cfg.topic_models_prefix}{suffix}" for suffix in ("*", "momo", "meanrev", "sentiment", "statarb")]

    while True:
        for topic in topics:
            try:
                for _, raw in consume_stream(topic, start_id=cursor, block_ms=150, count=500):
                    cursor = "$"
                    try:
                        msg = json.loads(raw) if isinstance(raw, str) else raw
                    except Exception:
                        continue
                    ens.ingest_bus_payload(msg)
            except Exception:
                # topic may not exist; continue
                pass

        # blend & publish
        try:
            scores = ens.blend()
            if scores:
                targets = ens.targets(scores)
                ens.publish(scores, targets)
                if emit_orders:
                    orders = ens.orders_for_targets(targets)
                    if publish_stream:
                        for o in orders:
                            publish_stream(ens.cfg.topic_orders_out, o)
        except Exception:
            pass

        time.sleep(0.05)


# ------------------------- CLI -------------------------

def _probe():
    cfg = EnsembleConfig(method="weighted", weights={"m1": 2.0, "m2": 1.0}, max_leverage=1.5, use_risk_parity=False)
    ens = Ensemble(cfg)
    ens.update_model("m1", {"AAPL": 0.8, "MSFT": -0.3, "NVDA": 0.2}, meta={"vol": {"AAPL":0.25,"MSFT":0.2,"NVDA":0.35}})
    ens.update_model("m2", {"AAPL": 0.4, "MSFT": -0.1, "NVDA": 0.6})
    scores = ens.blend()
    targets = ens.targets(scores, vols=None, corr=None)
    orders = ens.orders_for_targets(targets, prices={"AAPL":190, "MSFT":410, "NVDA":120})
    print(json.dumps({"scores": scores, "targets": targets, "orders": orders}, indent=2))

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Ensemble signal blender")
    ap.add_argument("--run", action="store_true", help="Attach to bus alpha.model.* and run")
    ap.add_argument("--emit-orders", action="store_true", help="Also emit orders to pre-risk stream")
    ap.add_argument("--probe", action="store_true", help="Run a synthetic example")
    ap.add_argument("--method", type=str, default="mean", choices=["mean","median","vote","weighted","stack"])
    ap.add_argument("--long-only", action="store_true")
    ap.add_argument("--max-lev", type=float, default=1.0)
    ap.add_argument("--hrp", action="store_true", help="Use HRP reweighting if corr provided")
    ap.add_argument("--risk-parity", action="store_true", help="Use simple inverse-vol parity")
    args = ap.parse_args()

    if args.probe:
        _probe(); return

    if args.run:
        cfg = EnsembleConfig(method=args.method, long_only=args.long_only,
                             max_leverage=args.max_lev, use_hrp=args.hrp, use_risk_parity=args.risk_parity)
        try:
            run_loop(cfg, emit_orders=args.emit_orders)
        except KeyboardInterrupt:
            pass
        return

    ap.print_help()

if __name__ == "__main__":
    main()