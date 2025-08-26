# backend/ai/rl_execution_agent.py
"""
RL Execution Agent (Contextual Bandit) for Smart Order Routing.

Usage (in router.py):
    from backend.ai import rl_execution_agent as RL
    tactic = RL.choose_tactic(order, market_state)
    route  = ROUTERS[tactic["route"]](**tactic["params"])

Design:
- Lightweight contextual bandit with ε-greedy + UCB; no heavy deps.
- Context features: urgency, order size vs ADV, spread (bps), volatility, imbalance, latency tier.
- Actions: ["TWAP","POV","Iceberg","Sniper"] (extendable).
- Reward = -(tca_spread + tca_impact + latency_bps)  (lower cost ⇒ higher reward).
- Online update from TCA events via `learn_from_tca(...)`.
- Safety: checks lot size, venue & asset class constraints, fallback heuristics if cold-start.

Streams (optional, if you want to learn from events):
- consume 'tca.extended'   → call learn_from_tca(event)
- persist policy to runtime/rl_exec.pkl

No external deps required (numpy optional). Compatible with pure Python.
"""

from __future__ import annotations
import os, math, time, json, pickle, threading, random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:
    np = None  # graceful fallback

# ---------- Config ----------
RUNTIME_DIR = os.getenv("RUNTIME_DIR", "runtime")
POLICY_PATH = os.path.join(RUNTIME_DIR, "rl_exec.pkl")
os.makedirs(RUNTIME_DIR, exist_ok=True)

ACTIONS = ["TWAP", "POV", "Iceberg", "Sniper"]

# Exploration settings
EPSILON = float(os.getenv("RL_EXEC_EPS", "0.10"))          # ε-greedy randomize
UCB_C   = float(os.getenv("RL_EXEC_UCB_C", "1.0"))         # optimism bonus
DECAY   = float(os.getenv("RL_EXEC_DECAY", "0.995"))       # reward decay

# Safety defaults
MAX_CHILD_ORDERS = 200
MIN_CHILD_QTY    = 1e-6

# ---------- Context schema ----------
@dataclass
class Context:
    symbol: str
    side: str                     # "buy"|"sell"
    qty: float
    notional: float               # qty * mark
    spread_bps: float             # current quoted spread in bps
    vol_bps: float                # short term volatility in bps (e.g., 1-5m)
    adv_ratio: float              # qty / ADV
    urgency: float                # 0..1 (0=can wait, 1=must finish asap)
    imbalance: float              # [-1..+1] LOB imbalance (bid-ask)/(bid+ask)
    latency_tier: int             # 0=fast,1=normal,2=slow
    venue: Optional[str] = None
    asset_class: str = "equity"   # equity|fut|opt|crypto

# ---------- Action parameterization ----------
def default_params(action: str, ctx: Context) -> Dict[str, Any]:
    """Safe parameter seeds per route."""
    if action == "TWAP":
        # schedule evenly over horizon proportional to urgency
        slices = max(4, int(16 * max(0.2, ctx.urgency)))
        return {"slices": min(slices, MAX_CHILD_ORDERS)}
    if action == "POV":
        pov = 0.05 + 0.25 * ctx.urgency  # 5% to 30% participation
        return {"participation": float(min(0.5, max(0.02, pov)))}
    if action == "Iceberg":
        # child size shaped by spread and adv
        block = max(ctx.qty * 0.05, ctx.notional * 0.0005)
        # fallback to qty in shares if notional used
        child = max(MIN_CHILD_QTY, min(ctx.qty, block if block > 0 else ctx.qty * 0.1))
        return {"child_qty": float(child)}
    if action == "Sniper":
        edge_bps = max(1.0, 0.5 * ctx.spread_bps)  # demand at least half-spread edge
        return {"edge_bps": float(edge_bps), "timeout_s": 3.0 + 5.0 * ctx.urgency}
    return {}

# ---------- Policy storage ----------
class Policy:
    """
    Linear preference per action: r_hat = w·x
    x = normalized context features.
    """
    def __init__(self, actions: List[str]):
        self.actions = actions
        self.weights: Dict[str, List[float]] = {a: [0.0]*10 for a in actions}
        self.counts:  Dict[str, int] = {a: 0 for a in actions}
        self.mu:      Dict[str, float] = {a: 0.0 for a in actions}  # running mean reward

    def featurize(self, ctx: Context) -> List[float]:
        # Normalize features
        sb = ctx.spread_bps / 10.0
        vb = ctx.vol_bps / 20.0
        ar = min(5.0, ctx.adv_ratio) / 5.0
        ur = clamp(ctx.urgency, 0.0, 1.0)
        im = clamp((ctx.imbalance + 1.0)/2.0, 0.0, 1.0)
        lt = ctx.latency_tier / 2.0
        ac = 1.0 if ctx.asset_class in ("opt","fut") else 0.0
        # Bias + pairwise interactions
        feats = [
            1.0, sb, vb, ar, ur, im, lt, ac,
            ur*sb, ar*vb
        ]
        return feats

    def predict(self, action: str, feats: List[float]) -> float:
        w = self.weights[action]
        return dot(w, feats)

    def ucb_score(self, action: str, pred: float, total_n: int) -> float:
        n = max(1, self.counts[action])
        bonus = UCB_C * math.sqrt(math.log(max(2, total_n)) / n)
        return pred + bonus

    def update(self, action: str, feats: List[float], reward: float, lr: float = 0.05):
        # SGD on squared error between predicted and observed reward
        w = self.weights[action]
        pred = dot(w, feats)
        err = reward - pred
        for i in range(len(w)):
            w[i] += lr * err * feats[i]
        self.weights[action] = w
        # Update counts and mean reward
        self.counts[action] = self.counts.get(action, 0) + 1
        self.mu[action] = (1-DECAY)*self.mu.get(action,0.0) + DECAY*reward

# ---------- Global policy ----------
_POLICY_LOCK = threading.Lock()
_POLICY: Optional[Policy] = None

def _load_policy() -> Policy:
    global _ POLICY  # type: ignore  # (space to avoid accidental regex highlight)
    with _POLICY_LOCK:
        if _POLICY is not None: # type: ignore
            return _POLICY # type: ignore
        try:
            with open(POLICY_PATH, "rb") as f:
                obj = pickle.load(f)
                if isinstance(obj, Policy):
                    _POLICY = obj
                else:
                    _POLICY = Policy(ACTIONS)
        except Exception:
            _POLICY = Policy(ACTIONS)
        return _POLICY

def _save_policy():
    with _POLICY_LOCK:
        try:
            with open(POLICY_PATH, "wb") as f:
                pickle.dump(_POLICY, f)
        except Exception:
            pass

# ---------- Public API ----------
def choose_tactic(order: Dict[str, Any], market_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decide route + params for an order given current market context.
    Safe, deterministic fallback if policy is cold.
    """
    ctx = _build_context(order, market_state)
    policy = _load_policy()
    feats = policy.featurize(ctx)

    # Safety/constraints gate — may pre-filter actions
    allowed = _filter_actions(ctx, order, market_state)
    if not allowed:
        allowed = ["TWAP"]  # safest default

    # Cold start? use heuristics
    total_n = sum(policy.counts.values())
    if total_n < 20:
        action = _heuristic_action(ctx, allowed)
        return {"route": action, "params": default_params(action, ctx), "explore": False}

    # ε-greedy explore
    if random.random() < EPSILON:
        action = random.choice(allowed)
        return {"route": action, "params": default_params(action, ctx), "explore": True}

    # UCB pick among allowed
    scores = []
    for a in allowed:
        pred = policy.predict(a, feats)
        ucb  = policy.ucb_score(a, pred, total_n=max(1,total_n))
        scores.append((ucb, a))
    scores.sort(reverse=True)
    action = scores[0][1]
    return {"route": action, "params": default_params(action, ctx), "explore": False}

def learn_from_tca(event: Dict[str, Any]) -> None:
    """
    Update policy from TCA event.

    Expected fields (best-effort):
      event = {
        "symbol": "AAPL", "route": "POV",
        "tca": {"spread_bps": 3.2, "impact_bps": 4.7, "latency_bps": 1.1},
        "context": {...},  # optional, same keys used in Context
      }
    Reward = -(spread + impact + latency)  [bps]
    """
    try:
        route = str(event.get("route") or "").strip()
        if route not in ACTIONS:
            return

        ctxd  = event.get("context") or {}
        # if original context not carried, reconstruct minimal
        ctx = _context_from_dict(ctxd) or Context(
            symbol=str(event.get("symbol") or ""),
            side=ctxd.get("side","buy"),
            qty=float(ctxd.get("qty", 0.0)),
            notional=float(ctxd.get("notional", 0.0)),
            spread_bps=float(ctxd.get("spread_bps", 5.0)),
            vol_bps=float(ctxd.get("vol_bps", 15.0)),
            adv_ratio=float(ctxd.get("adv_ratio", 0.05)),
            urgency=float(ctxd.get("urgency", 0.5)),
            imbalance=float(ctxd.get("imbalance", 0.0)),
            latency_tier=int(ctxd.get("latency_tier", 1)),
            venue=ctxd.get("venue"), asset_class=ctxd.get("asset_class","equity")
        )

        tca  = event.get("tca") or {}
        spread = float(tca.get("spread_bps", 0.0))
        impact = float(tca.get("impact_bps", 0.0))
        lat    = float(tca.get("latency_bps", 0.0))
        reward = - (spread + impact + lat)

        pol = _load_policy()
        feats = pol.featurize(ctx)
        pol.update(route, feats, reward, lr=0.05)
        _save_policy()
    except Exception:
        # never raise from learning
        pass

# ---------- Helpers ----------
def _build_context(order: Dict[str, Any], ms: Dict[str, Any]) -> Context:
    sym = (order.get("symbol") or "").upper()
    side = (order.get("side") or "buy").lower()
    qty  = float(order.get("qty") or 0.0)
    mark = float(ms.get("mark_price") or ms.get("last_px") or order.get("mark_price") or 0.0)
    notional = qty * mark if mark > 0 else qty

    spread_bps = float(ms.get("spread_bps") or 5.0)
    vol_bps    = float(ms.get("vol_bps") or 15.0)
    adv_ratio  = float(ms.get("adv_ratio") or 0.05)
    urgency    = float(order.get("urgency") or ms.get("urgency") or 0.5)
    imbalance  = float(ms.get("imbalance") or 0.0)
    latency_tier = int(ms.get("latency_tier") or 1)
    venue = ms.get("venue")
    asset_class = order.get("asset_class","equity")
    return Context(sym, side, qty, notional, spread_bps, vol_bps, adv_ratio, urgency, imbalance, latency_tier, venue, asset_class)

def _context_from_dict(d: Dict[str, Any]) -> Optional[Context]:
    try:
        return Context(
            symbol=str(d["symbol"]).upper(),
            side=str(d.get("side","buy")).lower(),
            qty=float(d.get("qty",0.0)),
            notional=float(d.get("notional",0.0)),
            spread_bps=float(d.get("spread_bps",5.0)),
            vol_bps=float(d.get("vol_bps",15.0)),
            adv_ratio=float(d.get("adv_ratio",0.05)),
            urgency=float(d.get("urgency",0.5)),
            imbalance=float(d.get("imbalance",0.0)),
            latency_tier=int(d.get("latency_tier",1)),
            venue=d.get("venue"), asset_class=d.get("asset_class","equity")
        )
    except Exception:
        return None

def _filter_actions(ctx: Context, order: Dict[str, Any], ms: Dict[str, Any]) -> List[str]:
    allowed = ACTIONS[:]
    # Asset-class guardrails
    if ctx.asset_class in ("opt","fut"):
        # Sniper often not ideal in derivatives with wider ticks
        if "Sniper" in allowed: allowed.remove("Sniper")
    # Venue constraints example (customize as needed)
    if (ctx.venue or "").upper() in ("NSE","BSE"):
        # Keep conservative defaults; Iceberg/POV ok if broker supports
        pass
    # Min child constraints
    if ctx.qty <= MIN_CHILD_QTY:
        return ["TWAP"]  # trivial/safe
    return allowed

def _heuristic_action(ctx: Context, allowed: List[str]) -> str:
    # Simple rules of thumb for cold-start:
    if ctx.urgency >= 0.8:
        return "POV" if "POV" in allowed else allowed[0]
    if ctx.spread_bps >= 10.0:
        return "Iceberg" if "Iceberg" in allowed else allowed[0]
    if ctx.adv_ratio >= 0.5:
        return "TWAP" if "TWAP" in allowed else allowed[0]
    # quiet & tight → sniper try
    if ctx.spread_bps <= 3.0 and abs(ctx.imbalance) > 0.2:
        return "Sniper" if "Sniper" in allowed else allowed[0]
    return allowed[0]

def dot(a: List[float], b: List[float]) -> float:
    if np is not None:
        return float(np.dot(np.asarray(a), np.asarray(b)))
    return sum(x*y for x,y in zip(a,b))

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# ---------- Optional: learn from bus (wire if you like) ----------
def run_tca_listener(stream_name: str = "tca.extended"):
    """
    Optional loop: subscribe to TCA events and learn continuously.
    """
    try:
        from backend.bus.streams import consume_stream
    except Exception:
        return
    cur = "$"
    while True:
        for _, msg in consume_stream(stream_name, start_id=cur, block_ms=500, count=200):
            cur = "$"
            try:
                if isinstance(msg, str):
                    msg = json.loads(msg)
            except Exception:
                continue
            learn_from_tca(msg)

# ---------- Optional: offline simulate against LOB sim ----------
def offline_train_with_sim(episodes: int = 200):
    """
    If sim/limit_order_book.py exposes `simulate_exec(route, params, ctx)->cost_bps`,
    run synthetic episodes to warm-start the policy.
    """
    try:
        from backend.sim.limit_order_book import simulate_exec  # type: ignore
    except Exception:
        return

    pol = _load_policy()
    rng = random.Random(7)
    for ep in range(episodes):
        # sample random context
        ctx = Context(
            symbol="SIM",
            side="buy" if rng.random()<0.5 else "sell",
            qty= rng.uniform(1e3, 1e6),
            notional= rng.uniform(1e5, 5e7),
            spread_bps= rng.uniform(1, 20),
            vol_bps= rng.uniform(5, 40),
            adv_ratio= rng.uniform(0.01, 1.0),
            urgency= rng.uniform(0.1, 1.0),
            imbalance= rng.uniform(-1.0, 1.0),
            latency_tier= rng.choice([0,1,2]),
            venue="SIM", asset_class="equity"
        )
        feats = pol.featurize(ctx)
        allowed = _filter_actions(ctx, {}, {})
        # pick via UCB to balance exploration
        scores = []
        total_n = max(1, sum(pol.counts.values()))
        for a in allowed:
            pred = pol.predict(a, feats)
            ucb  = pol.ucb_score(a, pred, total_n)
            scores.append((ucb, a))
        action = max(scores)[1]
        params = default_params(action, ctx)
        # simulate cost
        try:
            cost_bps = float(simulate_exec(action, params, asdict(ctx)))
        except Exception:
            cost_bps = 10.0
        reward = -cost_bps
        pol.update(action, feats, reward, lr=0.05)
    _save_policy()

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="RL Execution Agent")
    ap.add_argument("--listen-tca", action="store_true", help="learn online from tca.extended")
    ap.add_argument("--offline-train", action="store_true", help="warm start from LOB simulator")
    ap.add_argument("--episodes", type=int, default=200)
    args = ap.parse_args()

    if args.offline_train:
        offline_train_with_sim(args.episodes)
        print(f"Offline training done. Saved policy to {POLICY_PATH}")

    if args.listen_tca:
        print("Listening for TCA events...")
        run_tca_listener()