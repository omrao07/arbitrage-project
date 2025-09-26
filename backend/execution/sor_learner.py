# backend/router/sor_learner.py
"""
Smart Order Routing (SOR) Learner
---------------------------------
Online contextual-bandit learner for dark/alt venues (or lit), producing
weights per venue for the next child slice.

Core ideas
  • For each venue, learn reward(x) ≈ θᵀx  (LinUCB with ridge regularization)
  • Maintain a Beta(a,b) fill-rate posterior per venue
  • Score = UCB_reward * expected_fill_prob - fee_bps - penalty_bps
  • Allocate weights ∝ max(score, 0) with risk/coverage constraints

What counts as "reward"?
  • Signed effective spread capture (bps): side*(mid - px)/mid*1e4
  • Minus realized/expected markout bps (if provided)
  • Minus taker fees (bps)

Inputs (from fills stream)
  Envelope fields expected (see darkpool_sim or live exec):
    {
      "ts": <ms>, "symbol": "...", "venue": "DP1", "side": "BUY|SELL",
      "qty": 1234, "px": 100.12, "px_mid": 100.15, "fee": 0.12, ...,
      # Optional extras (if you have them):
      "markout_bps": { "1s": -2.3, "5s": -4.1 }   # realized markouts
      "features": { ... }                         # context used for learning (optional)
    }

Features
  You can pass a dict of features to suggest_allocation(); if omitted, a
  minimal set is derived (log qty, 1, spread_bps, time_sin/cos, etc).
  The FeatureEncoder handles dimensionality + online scaling.

Streams
  • Subscribes to fills (sim or live) to learn online
  • Publishes decisions to STREAM_ROUTER_DECISIONS

Persistence
  • save(path) / load(path) persist model state (JSON + np arrays)

Dependencies
  • numpy
"""

from __future__ import annotations

import json
import math
import os
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------- Bus hooks (safe stubs if missing) ----------------

try:
    from backend.bus.streams import publish_stream, consume_stream # type: ignore
except Exception:
    def publish_stream(stream: str, payload):
        head = {k: payload.get(k) for k in ("ts","kind","symbol","decision_id")}
        print(f"[stub publish_stream] {stream} <- {json.dumps(head, separators=(',',':'))[:200]}")
    def consume_stream(stream: str, handler):
        print(f"[stub consume_stream] {stream} -> {handler.__name__} (no-op)")

# ---------------- Config ----------------

@dataclass
class SORConfig:
    venues: Sequence[str]
    d: int = 12                         # feature dimension (after encoding)
    ridge_lambda: float = 1.0           # L2 prior for LinUCB
    alpha_ucb: float = 1.5              # exploration strength
    decay: float = 0.995                # exponential forgetting per update
    min_prob_fill: float = 0.05         # drop venues with low fill-odds
    fee_bps: Dict[str, float] = None    # type: ignore # per-venue fee bps override (optional)
    penalty_bps: Dict[str, float] = None# type: ignore # per-venue constant penalty (toxicity)
    stream_fills: str = "STREAM_SIM_FILLS"      # or "STREAM_EXEC_FILLS"
    stream_decisions: str = "STREAM_ROUTER_DECISIONS"
    ledger_path: Optional[str] = None
    persistence_path: Optional[str] = "data/sor_learner_state.json"
    seed: Optional[int] = 7

# ---------------- Feature Encoder ----------------

class FeatureEncoder:
    """
    Online feature encoder with basic scaling. You can customize _build_features().
    Maintains mean/var for standardization (per-dim), updated with exponential decay.
    """
    def __init__(self, d: int, decay: float = 0.995):
        self.d = int(d)
        self.decay = float(decay)
        self.mu = np.zeros(d, dtype=float)
        self.var = np.ones(d, dtype=float)

    def encode(self, side: str, qty: int, px_mid: Optional[float], spread_bps: Optional[float],
               t_ms: Optional[int], extra: Optional[Dict[str, Any]] = None) -> np.ndarray:
        x = self._build_features(side, qty, px_mid, spread_bps, t_ms, extra)
        # Standardize with running stats
        self._update_stats(x)
        z = (x - self.mu) / np.sqrt(self.var + 1e-8)
        return z

    def _update_stats(self, x: np.ndarray) -> None:
        # EW mean/var update
        self.mu = self.decay * self.mu + (1 - self.decay) * x
        diff = x - self.mu
        self.var = self.decay * self.var + (1 - self.decay) * (diff * diff)
        self.var = np.maximum(self.var, 1e-8)

    def _build_features(self, side: str, qty: int, px_mid: Optional[float],
                        spread_bps: Optional[float], t_ms: Optional[int],
                        extra: Optional[Dict[str, Any]]) -> np.ndarray:
        side_sign = 1.0 if str(side).upper() == "BUY" else -1.0
        q = max(1.0, float(qty))
        log_q = math.log(q)

        spr = float(spread_bps) if (spread_bps is not None) else 10.0
        spr = max(1e-3, spr)
        spr_ln = math.log(spr)

        # time of day features
        if t_ms is None:
            t_ms = int(time.time() * 1000)
        t_s = (t_ms // 1000) % (24 * 3600)
        tod = 2 * math.pi * (t_s / (24 * 3600))
        sin_t = math.sin(tod)
        cos_t = math.cos(tod)

        # Optional extras
        extra = extra or {}
        vol_bps = float(extra.get("vol_bps", 20.0))
        vol_ln = math.log(max(1e-3, vol_bps))
        tox = float(extra.get("toxicity_beta", 0.5))
        min_qty = float(extra.get("venue_min_qty", 200.0))
        match_rate = float(extra.get("venue_match_rate", 0.5))

        # Bias + features; pad/truncate to d
        feats = np.array([
            1.0,
            side_sign,
            log_q,
            spr_ln,
            sin_t, cos_t,
            vol_ln,
            tox,
            math.log(max(1.0, min_qty)),
            match_rate,
            (log_q * spr_ln),
            (side_sign * spr_ln),
            # ... (extend here if you set d>12)
        ], dtype=float)

        if feats.size < self.d:
            feats = np.pad(feats, (0, self.d - feats.size), mode='constant', constant_values=0.0)
        elif feats.size > self.d:
            feats = feats[:self.d]
        return feats

# ---------------- LinUCB Arm ----------------

class LinUCBArm:
    """Ridge-regression UCB: θ = A^{-1} b, score = xᵀθ + α sqrt(xᵀ A^{-1} x)"""
    def __init__(self, d: int, ridge_lambda: float, alpha_ucb: float, decay: float):
        self.d = int(d)
        self.alpha = float(alpha_ucb)
        self.decay = float(decay)
        self.lmb = float(ridge_lambda)
        self.A = np.eye(d, dtype=float) * self.lmb
        self.b = np.zeros((d, 1), dtype=float)
        self.A_inv = np.linalg.inv(self.A)  # keep cached
        # Fill-rate Beta prior
        self.fr_a = 1.0
        self.fr_b = 1.0

    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        x = x.reshape(-1, 1)
        theta = self.A_inv @ self.b
        mean = float((x.T @ theta)[0, 0])
        var = float((x.T @ self.A_inv @ x)[0, 0])
        ucb = mean + self.alpha * math.sqrt(max(0.0, var))
        return mean, ucb

    def update(self, x: np.ndarray, reward_bps: float, *, filled: bool) -> None:
        # Exponential forgetting on A, b
        self.A = self.decay * self.A + (1 - self.decay) * np.eye(self.d) * self.lmb
        self.b = self.decay * self.b
        x = x.reshape(-1, 1)
        self.A += x @ x.T
        self.b += float(reward_bps) * x
        # Recompute inverse (small d, OK)
        self.A_inv = np.linalg.inv(self.A)
        # Update fill Beta posterior
        if filled:
            self.fr_a = 0.99 * self.fr_a + 1.0
            self.fr_b = 0.99 * self.fr_b + 0.0
        else:
            self.fr_a = 0.99 * self.fr_a + 0.0
            self.fr_b = 0.99 * self.fr_b + 1.0

    def expected_fill_prob(self) -> float:
        a, b = max(1e-6, self.fr_a), max(1e-6, self.fr_b)
        return float(a / (a + b))

# ---------------- SOR Learner ----------------

class SORLearner:
    def __init__(self, cfg: SORConfig):
        self.cfg = cfg
        self.venues = list(cfg.venues)
        if self.cfg.fee_bps is None: self.cfg.fee_bps = {}
        if self.cfg.penalty_bps is None: self.cfg.penalty_bps = {}
        self.enc = FeatureEncoder(cfg.d, decay=cfg.decay)
        self.arms: Dict[str, LinUCBArm] = {v: LinUCBArm(cfg.d, cfg.ridge_lambda, cfg.alpha_ucb, cfg.decay) for v in self.venues}
        if cfg.seed is not None:
            np.random.seed(cfg.seed)

    # ---------- Online hooks ----------

    def start_learning_from_stream(self) -> None:
        """Subscribe to fills stream to update models online."""
        consume_stream(self.cfg.stream_fills, self._on_fill_event)

    def _on_fill_event(self, env: Dict[str, Any]) -> None:
        try:
            venue = env.get("venue")
            if not venue:
                return
            if venue not in self.arms:
                # Auto-register new venue
                self.venues.append(venue)
                self.arms[venue] = LinUCBArm(self.cfg.d, self.cfg.ridge_lambda, self.cfg.alpha_ucb, self.cfg.decay)

            side = str(env.get("side", "BUY")).upper()
            qty = int(env.get("qty", 0))
            px = float(env.get("px")) # type: ignore
            px_mid = float(env.get("px_mid", px))
            fee = float(env.get("fee", 0.0))
            # Optional extras
            features = env.get("features", {}) or {}
            spread_bps = features.get("spread_bps")
            t_ms = int(env.get("ts", int(time.time() * 1000)))
            # Reward (bps): effective spread capture - fee_bps - optional markout
            side_sign = 1.0 if side == "BUY" else -1.0
            eff_bps = side_sign * (px_mid - px) / max(1e-9, px_mid) * 1e4
            fee_bps = 1e4 * fee / max(1e-9, (qty * px))
            mo_bps = 0.0
            if isinstance(env.get("markout_bps"), dict):
                mo_bps = float(env["markout_bps"].get("5s", 0.0))
            reward_bps = float(eff_bps - fee_bps - mo_bps)

            x = self.enc.encode(side, qty, px_mid, spread_bps, t_ms, features)
            self.arms[venue].update(x, reward_bps, filled=(qty > 0))
        except Exception as e:
            print(f"[sor_learner fill error] {e}")

    # ---------- Decision API ----------

    def suggest_allocation(
        self,
        *,
        symbol: str,
        side: str,
        child_qty: int,
        spread_bps: Optional[float] = None,
        features_by_venue: Optional[Dict[str, Dict[str, Any]]] = None,
        decision_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Produce venue weights for a child order.
        features_by_venue: optional dict of per-venue extra features (toxicity, match_rate, min_qty, vol_bps, etc.)
        Returns an audit envelope with the chosen weights and scores.
        """
        now_ms = int(time.time() * 1000)
        side_u = str(side).upper()
        feats_by_v = features_by_venue or {}
        scores: Dict[str, float] = {}
        means: Dict[str, float] = {}
        ucbs: Dict[str, float] = {}
        pfill: Dict[str, float] = {}
        cost_bps: Dict[str, float] = {}

        for v in list(self.venues):
            arm = self.arms.get(v)
            if arm is None:
                continue
            extra = feats_by_v.get(v, {})
            # Build a representative feature vector for *this* decision (qty unknown to learner -> use child_qty)
            x = self.enc.encode(side_u, child_qty, None, spread_bps, now_ms, extra)
            mean, ucb = arm.predict(x)
            pf = arm.expected_fill_prob()
            # Apply fee & penalty adjustments
            fee_bps = float(self.cfg.fee_bps.get(v, 0.0))
            pen_bps = float(self.cfg.penalty_bps.get(v, 0.0))
            score = max(-1e6, ucb * pf - fee_bps - pen_bps)
            scores[v] = float(score)
            means[v] = float(mean)
            ucbs[v] = float(ucb)
            pfill[v] = float(pf)
            cost_bps[v] = float(fee_bps + pen_bps)

        # Filter low-prob venues
        usable = [v for v in self.venues if pfill.get(v, 0.0) >= self.cfg.min_prob_fill]
        if not usable:
            # fall back to best pfill regardless
            usable = sorted(self.venues, key=lambda v: pfill.get(v, 0.0), reverse=True)[:1]

        # Turn scores into weights (clipped at 0)
        pos = np.array([max(0.0, scores[v]) for v in usable], dtype=float)
        if pos.sum() <= 0:
            # If all non-positive, allocate everything to argmax(score)
            best = usable[int(np.argmax([scores[v] for v in usable]))]
            weights = {v: (1.0 if v == best else 0.0) for v in usable}
        else:
            w = pos / pos.sum()
            weights = {v: float(w[i]) for i, v in enumerate(usable)}
        # Include zero weights for others for clarity
        for v in self.venues:
            if v not in weights:
                weights[v] = 0.0

        env = {
            "ts": now_ms,
            "kind": "sor_decision",
            "decision_id": decision_id or _hash_str(f"{symbol}|{side_u}|{now_ms}|{child_qty}"),
            "symbol": symbol,
            "side": side_u,
            "child_qty": int(child_qty),
            "min_prob_fill": float(self.cfg.min_prob_fill),
            "weights": weights,
            "scores": scores,
            "ucb": ucbs,
            "mean": means,
            "pfill": pfill,
            "cost_bps": cost_bps,
            "version": 1,
        }
        env["hash"] = _hash_json(env)
        publish_stream(self.cfg.stream_decisions, env)
        _ledger_append(env, self.cfg.ledger_path)
        return env

    # ---------- Persistence ----------

    def save(self, path: Optional[str] = None) -> None:
        path = path or self.cfg.persistence_path
        os.makedirs(os.path.dirname(path), exist_ok=True) # type: ignore
        state = {
            "cfg": asdict(self.cfg),
            "enc": {"mu": self.enc.mu.tolist(), "var": self.enc.var.tolist()},
            "venues": self.venues,
            "arms": {
                v: {
                    "A": self.arms[v].A.tolist(),
                    "b": self.arms[v].b.flatten().tolist(),
                    "A_inv": self.arms[v].A_inv.tolist(),
                    "fr_a": self.arms[v].fr_a,
                    "fr_b": self.arms[v].fr_b,
                } for v in self.venues
            },
        }
        with open(path, "w", encoding="utf-8") as f: # type: ignore
            json.dump(state, f, separators=(",", ":"))
        print(f"[sor_learner] state saved to {path}")

    @classmethod
    def load(cls, path: str) -> "SORLearner":
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        cfg = SORConfig(**state["cfg"])
        obj = cls(cfg)
        obj.venues = state["venues"]
        obj.enc.mu = np.array(state["enc"]["mu"], dtype=float)
        obj.enc.var = np.array(state["enc"]["var"], dtype=float)
        obj.arms = {}
        for v in obj.venues:
            arm = LinUCBArm(cfg.d, cfg.ridge_lambda, cfg.alpha_ucb, cfg.decay)
            s = state["arms"][v]
            arm.A = np.array(s["A"], dtype=float)
            arm.b = np.array(s["b"], dtype=float).reshape(-1, 1)
            arm.A_inv = np.array(s["A_inv"], dtype=float)
            arm.fr_a = float(s["fr_a"]); arm.fr_b = float(s["fr_b"])
            obj.arms[v] = arm
        print(f"[sor_learner] state loaded from {path}")
        return obj

# ---------------- Helpers ----------------

def _hash_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode()).hexdigest()

def _hash_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

def _ledger_append(payload: Dict[str, Any], ledger_path: Optional[str]) -> None:
    if not ledger_path:
        return
    try:
        from backend.audit.merkle_ledger import MerkleLedger
        MerkleLedger(ledger_path).append({"type": "sor_decision", "payload": payload})
    except Exception:
        pass

# ---------------- Quick demo ----------------

if __name__ == "__main__":
    # Example venues & config
    cfg = SORConfig(
        venues=["DP1","DP2","DP3"],
        d=12,
        ridge_lambda=2.0,
        alpha_ucb=1.2,
        decay=0.995,
        min_prob_fill=0.05,
        fee_bps={"DP1":0.2,"DP2":0.0,"DP3":0.1},
        penalty_bps={"DP3":0.3},
        stream_fills="STREAM_SIM_FILLS",
        stream_decisions="STREAM_ROUTER_DECISIONS",
        persistence_path="data/sor_learner_state.json",
        seed=42
    )
    sor = SORLearner(cfg)

    # (Optional) start online learning from stream
    # sor.start_learning_from_stream()

    # Simulate a few fills to train
    def sim_fill(venue, side, qty, px, mid, fee, tox=0.6, spr=8.0, ts=None):
        return {
            "ts": ts or int(time.time()*1000),
            "symbol": "AAPL",
            "venue": venue,
            "side": side,
            "qty": qty,
            "px": px,
            "px_mid": mid,
            "fee": fee,
            "features": {"toxicity_beta": tox, "spread_bps": spr, "vol_bps": 20.0}
        }

    rng = np.random.default_rng(0)
    for _ in range(200):
        v = rng.choice(cfg.venues)
        mid = 100.0 + rng.normal(0, 0.1)
        # midpoint cross with tiny improvement
        px = mid - (rng.random()*0.5 - 0.25) * 0.01  # +/- 0.25 bp
        q = int(rng.lognormal(7.0, 0.6))
        fee = (cfg.fee_bps.get(v, 0.0) / 1e4) * q * px
        sor._on_fill_event(sim_fill(v, "BUY", q, px, mid, fee, tox=rng.uniform(0.2,0.9)))

    # Ask for an allocation
    decision = sor.suggest_allocation(
        symbol="AAPL",
        side="BUY",
        child_qty=1500,
        spread_bps=8.0,
        features_by_venue={"DP1":{"toxicity_beta":0.5,"match_rate":0.6},
                           "DP2":{"toxicity_beta":0.3,"match_rate":0.4},
                           "DP3":{"toxicity_beta":0.8,"match_rate":0.8}}
    )
    print(json.dumps({"weights": decision["weights"], "scores": decision["scores"]}, indent=2))
    # Save state
    sor.save()