# backend/ai/agents/concrete/rl_execution_agent.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal, Tuple

# ============================================================
# Framework shims (no external deps)
# ============================================================

try:
    from ..core.base_agent import BaseAgent  # type: ignore
except Exception:
    class BaseAgent:
        name: str = "rl_execution_agent"
        def plan(self, *a, **k): ...
        def act(self, *a, **k): ...
        def explain(self, *a, **k): ...
        def heartbeat(self, *a, **k): return {"ok": True}

# Market inputs
try:
    # expected: get_orderbook(symbol, depth=5) -> {"bids":[{"px","qty"}], "asks":[...], "ts":ms}
    from ..skills.market.orderbook import get_orderbook  # type: ignore
except Exception:
    def get_orderbook(symbol: str, depth: int = 5) -> Dict[str, Any]:
        mid = 100.0
        spread = 0.02
        bids = [{"px": round(mid - spread/2 - i*0.01, 4), "qty": 1000 - i*50} for i in range(depth)]
        asks = [{"px": round(mid + spread/2 + i*0.01, 4), "qty": 1000 - i*50} for i in range(depth)]
        return {"bids": bids, "asks": asks, "ts": int(time.time()*1000)}

try:
    # expected: get_candles(symbol, interval="1m", lookback=N) -> [{ts,o,h,l,c,v}]
    from ..skills.market.quotes import get_candles  # type: ignore
except Exception:
    def get_candles(symbol: str, *, interval: str = "1m", lookback: int = 30) -> List[Dict[str, Any]]:
        now = int(time.time()*1000); px = 100.0
        out = []
        for i in range(lookback):
            px *= (1 + (0.0003 if (i % 5) else -0.001))
            out.append({"ts": now - (lookback - i)*60_000, "o": px, "h": px*1.001, "l": px*0.999, "c": px, "v": 1_000_00})
        return out

# Broker / TCA / Risk
try:
    from ..skills.trading.broker_interface import submit_order  # type: ignore
except Exception:
    def submit_order(symbol: str, side: str, qty: float, order_type: str = "market",
                     limit_price: Optional[float] = None, tag: Optional[str] = None) -> str:
        oid = f"SIM-{int(time.time()*1e6)}"
        print(f"[SIM] {side} {qty} {symbol} {order_type} {'' if limit_price is None else f'@{limit_price:.4f}'} tag={tag} -> {oid}")
        return oid

try:
    from ..skills.trading.tca import estimate_cost_bps  # type: ignore
except Exception:
    def estimate_cost_bps(symbol: str, side: str, qty: float, px: float, venue: Optional[str] = None) -> float:
        return 2.0 + min(25.0, 10.0 * math.sqrt(max(0.0, qty) / 1_000_000.0))

try:
    from ..policies.rules.risk_policies import check_gates  # type: ignore
except Exception:
    def check_gates(symbol: str, side: str, qty: float, px: float, *, context: Dict[str, Any]) -> Tuple[bool, str]:
        return (False, "Kill-switch active") if context.get("kill_switch") else (True, "OK")

# Optional RL policy (your learned model)
try:
    # expected contract:
    # class RLPolicy: def act(self, state: Dict[str,float]) -> Dict[str, Any]
    from ..policies.rl.execution_agent import RLPolicy  # type: ignore
except Exception:
    RLPolicy = None  # fallback handled below

# ============================================================
# Data models
# ============================================================

Side = Literal["buy", "sell"]
Mode = Literal["passive", "aggressive"]
AlgoTag = Literal["RL", "RL-PASSIVE", "RL-AGGR"]

@dataclass
class RLExecTarget:
    symbol: str
    side: Side
    qty: float
    tag: Optional[str] = None

@dataclass
class RLExecSchedule:
    horizon_min: int = 10        # total time budget
    step_ms: int = 15_000        # decision cadence
    max_participation: float = 0.15
    venue: Optional[str] = None
    limit_cross_bps: float = 8.0   # how far we allow crossing for limit
    sliding_limit: bool = True

@dataclass
class RLExecRequest:
    target: RLExecTarget
    schedule: RLExecSchedule = field(default_factory=RLExecSchedule)
    dry_run: bool = False
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Decision:
    t_ms: int
    mode: Mode
    px_ref: float
    limit_offset_bps: float
    qty: float
    participation_cap_qty: float
    venue: Optional[str]
    reason: str

@dataclass
class RLExecReport:
    ok: bool
    symbol: str
    side: Side
    requested_qty: float
    filled_qty: float
    est_cost_bps: float
    decisions: List[Decision]
    messages: List[str]

# ============================================================
# Feature extraction
# ============================================================

def _mid_spread(ob: Dict[str, Any]) -> Tuple[float, float]:
    b0 = float(ob["bids"][0]["px"]) if ob.get("bids") else float("nan")
    a0 = float(ob["asks"][0]["px"]) if ob.get("asks") else float("nan")
    mid = (b0 + a0) / 2.0
    spread = max(0.0, a0 - b0)
    return mid, spread

def _imbalance(ob: Dict[str, Any], levels: int = 3) -> float:
    bids = ob.get("bids", [])[:levels]; asks = ob.get("asks", [])[:levels]
    qb = sum(float(x["qty"]) for x in bids); qa = sum(float(x["qty"]) for x in asks)
    if qb + qa <= 0: return 0.0
    return (qb - qa) / (qb + qa)

def _vol_est(symbol: str) -> float:
    cs = get_candles(symbol, interval="1m", lookback=30)
    if len(cs) < 2: return 0.0
    closes = [float(c["c"]) for c in cs]
    rets = [closes[i]/max(1e-9, closes[i-1]) - 1 for i in range(1, len(closes))]
    m = sum(rets) / len(rets)
    var = sum((x - m)**2 for x in rets) / max(1, len(rets) - 1)
    return math.sqrt(max(0.0, var))

# ============================================================
# Fallback RL policy (epsilon-greedy on linear score)
# ============================================================

class _FallbackPolicy:
    def __init__(self, epsilon: float = 0.1):
        self.eps = epsilon
        # simple weights for score = w_spread*spread_bps + w_imb*imb + w_vol*vol
        self.w_spread = -0.6   # wider spread → prefer passive
        self.w_imb     =  0.8  # imbalance to our side → be more aggressive
        self.w_vol     =  0.5  # higher vol → execute faster

    def act(self, state: Dict[str, float]) -> Dict[str, Any]:
        # actions: choose mode ∈ {passive, aggressive} and target participation ∈ (0, pmax)
        spread_bps = state.get("spread_bps", 5.0)
        imb        = state.get("imbalance", 0.0)
        vol        = state.get("vol", 0.0)
        score = self.w_spread*spread_bps + self.w_imb*imb + self.w_vol*vol

        # epsilon-greedy exploration
        rnd = (state.get("rand", 0.34159) + time.time()*1e-6) % 1.0
        explore = rnd < self.eps
        mode: Mode = "aggressive" if (score > 0 or explore) else "passive"

        # participation target (0..1), mapped from features
        base_part = 0.08 + 0.25 * max(0.0, vol) + 0.10 * max(0.0, imb if mode=="aggressive" else 0.0)
        part = max(0.02, min(0.6, base_part))
        limit_off = 3.0 if mode == "passive" else 10.0  # bps
        return {"mode": mode, "target_participation": part, "limit_offset_bps": limit_off}

# ============================================================
# RL Execution Agent
# ============================================================

class RLExecutionAgent(BaseAgent): # type: ignore
    """
    RL-driven microstructure executor:
      • Observes orderbook state (mid, spread, imbalance, vol)
      • Calls RL policy (your model) or a fallback epsilon-greedy policy
      • Decides passive vs aggressive child orders, participation cap, and limit offsets
      • Applies risk gates and submits orders via broker_interface
    """

    name = "rl_execution_agent"

    def __init__(self):
        super().__init__()
        self.policy = RLPolicy() if RLPolicy else _FallbackPolicy()

    # -------- API --------

    def plan(self, req: RLExecRequest | Dict[str, Any]) -> RLExecRequest:
        if isinstance(req, RLExecRequest):
            return req
        t = req.get("target", {})
        s = req.get("schedule", {})
        return RLExecRequest(
            target=RLExecTarget(symbol=t.get("symbol"), side=t.get("side", "buy"), qty=float(t.get("qty", 0)), tag=t.get("tag")),
            schedule=RLExecSchedule(
                horizon_min=int(s.get("horizon_min", 10)),
                step_ms=int(s.get("step_ms", 15_000)),
                max_participation=float(s.get("max_participation", 0.15)),
                venue=s.get("venue"),
                limit_cross_bps=float(s.get("limit_cross_bps", 8.0)),
                sliding_limit=bool(s.get("sliding_limit", True)),
            ),
            dry_run=bool(req.get("dry_run", False)),
            context=dict(req.get("context", {})),
        )

    def act(self, request: RLExecRequest | Dict[str, Any]) -> RLExecReport:
        req = self.plan(request)
        sym, side, qty_total = req.target.symbol, req.target.side, float(req.target.qty)
        sched = req.schedule

        decisions: List[Decision] = []
        messages: List[str] = []
        filled = 0.0

        # Pre-compute volatility to stabilize features across steps
        vol = _vol_est(sym)

        # Time loop
        steps = max(1, int((sched.horizon_min * 60_000) / max(5_000, sched.step_ms)))
        qty_rem = qty_total

        # Participation cap per step (approx ADV per step via orderbook top qty proxy)
        for i in range(steps):
            if qty_rem <= 0:
                break

            ob = get_orderbook(sym, depth=5)
            mid, spr = _mid_spread(ob)
            spr_bps = (spr / max(1e-9, mid)) * 1e4
            imb = _imbalance(ob, levels=3)

            # build state
            state = {
                "mid": mid,
                "spread_bps": spr_bps,
                "imbalance": imb if side == "buy" else -imb,  # flip sign for sells
                "vol": vol,
                "step": i,
                "rand": (i * 0.123457) % 1.0,
            }

            # policy action
            act = self.policy.act(state) or {}
            mode: Mode = act.get("mode", "passive")
            part = float(max(0.0, min(1.0, act.get("target_participation", 0.1))))
            limit_off = float(act.get("limit_offset_bps", 5.0))

            # rough per-step volume proxy: sum of top-of-book quantities
            tob_qty = (ob["bids"][0]["qty"] if ob.get("bids") else 1_000) + (ob["asks"][0]["qty"] if ob.get("asks") else 1_000)
            adv_step = max(1.0, 0.5 * float(tob_qty))  # very coarse proxy

            cap_qty = max(0.0, sched.max_participation) * adv_step
            slice_qty = min(qty_rem, max(1.0, part * adv_step))
            slice_qty = min(slice_qty, cap_qty)

            # Risk gate on slice
            ok, msg = check_gates(sym, side, slice_qty, mid, context=req.context)
            messages.append(msg)
            if not ok:
                messages.append("blocked by risk policy; stopping.")
                break

            # decide order params
            order_type = "market" if mode == "aggressive" else "limit"
            limit_px = None
            if order_type == "limit":
                bump = (limit_off / 1e4) * mid
                limit_px = round(mid + (bump if side == "buy" else -bump), 4)
            elif order_type == "market" and sched.limit_cross_bps > 0 and sched.sliding_limit:
                # guardrail: convert to protective limit a few bps across the spread
                bump = (sched.limit_cross_bps / 1e4) * mid
                limit_px = round(mid + (bump if side == "buy" else -bump), 4)
                order_type = "limit"

            # execute (or dry-run)
            if req.dry_run:
                messages.append(f"DRY {mode} {side} {slice_qty} {sym} @{limit_px or mid:.4f}")
            else:
                oid = submit_order(sym, side, slice_qty, order_type=order_type, limit_price=limit_px, tag=req.target.tag or "RL")
                messages.append(f"sent {oid}: {mode} {side} {slice_qty} {sym} {order_type} {'' if limit_px is None else f'@{limit_px:.4f}'}")

            decisions.append(Decision(
                t_ms=int(time.time()*1000),
                mode=mode,
                px_ref=mid,
                limit_offset_bps=limit_off,
                qty=slice_qty,
                participation_cap_qty=cap_qty,
                venue=sched.venue,
                reason=f"spread={spr_bps:.1f}bps, imb={imb:.2f}, vol~{vol:.4%}"
            ))

            filled += slice_qty
            qty_rem -= slice_qty

            # sleep between steps only in non-dry-run demo mode (commented out)
            # time.sleep(sched.step_ms / 1000.0)

        est_bps = estimate_cost_bps(sym, side, max(0.0, qty_total), decisions[-1].px_ref if decisions else 0.0, sched.venue)

        return RLExecReport(
            ok=(filled > 0),
            symbol=sym,
            side=side,
            requested_qty=qty_total,
            filled_qty=filled,
            est_cost_bps=est_bps,
            decisions=decisions,
            messages=messages,
        )

    # -------- Docs / Health --------

    def explain(self) -> str:
        return (
            "RLExecutionAgent observes microstructure state (mid, spread, imbalance, short-term vol) "
            "and queries an RL policy for actions (passive vs aggressive, target participation, limit offsets). "
            "It enforces risk gates, caps participation, and submits child orders, returning a full decision trace."
        )

    def heartbeat(self) -> Dict[str, Any]:
        return {"ok": True, "agent": self.name, "ts": int(time.time())}


# ------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    agent = RLExecutionAgent()
    req = RLExecRequest(
        target=RLExecTarget(symbol="AAPL", side="buy", qty=50_000, tag="RL-DEMO"),
        schedule=RLExecSchedule(horizon_min=2, step_ms=15_000, max_participation=0.12, limit_cross_bps=6.0),
        dry_run=True,
        context={"book": "demo"}
    )
    rep = agent.act(req)
    print("OK:", rep.ok, "filled:", rep.filled_qty, "est cost (bps):", rep.est_cost_bps)
    for d in rep.decisions[:5]:
        print(f"[{d.t_ms}] {d.mode} qty={d.qty:.0f} cap={d.participation_cap_qty:.0f} px~{d.px_ref:.4f} reason={d.reason}")
    if rep.messages:
        print("messages:", rep.messages[:5])