# selector/policy_router.py
"""
PolicyRouter
------------
A rule-based symbol -> arm router with pluggable policies.

Goals
- Compose multiple routing rules: allow/deny lists, symbol groups, schedules,
  weighted splits, hash split fallback, epsilon-greedy exploitation, and
  state-aware constraints (e.g., leverage cap, volatility blocks).
- Deterministic and testable; stdlib only.

Key concepts
- PolicyRouter takes:
    * arms: list[str]                  # e.g., ["A","B"]
    * rules: list[Rule]                # evaluated in order; first decisive wins
    * default: BaseRouter              # fallback when no rule decides (e.g., HashSplit)
- Rule types included:
    * AllowDenyRule: if symbol in allow/deny, pick specific arm or block
    * GroupRule: route predefined symbol groups to fixed arms
    * ScheduleRule: time-window based arm preference (cron-like light)
    * WeightedRule: weighted round-robin per symbol namespace
    * HashFallbackRule: convenience wrapper (rarely needed if default provided)
    * StateGuardRule: consult a StateProvider (leverage, exposure, etc.) to freeze/redirect
    * BanditRule: epsilon-greedy overlay (delegates to EpsilonGreedyRouter logic)
- StateProvider (optional):
    * Any object that can expose getters like:
        - leverage() -> float
        - gross_exposure() -> float
        - telemetry() -> dict   # free-form

Usage
-----
from selector.ab_tests import HashSplitRouter, EpsilonGreedyRouter
from selector.policy_router import PolicyRouter, AllowDenyRule, GroupRule, ScheduleRule, StateGuardRule, WeightedRule

arms = ["A","B"]
default = HashSplitRouter(arms)

router = PolicyRouter(
    arms=arms,
    rules=[
        AllowDenyRule(allow={"TSLA":"B"}, deny={"GME": None}),     # deny means: no decision; falls through
        GroupRule(mapping={"PSU_BANKS": (["SBIN","BOB","PNB"], "A")}),
        ScheduleRule(prefer={"A": [("09:15","12:00")]}, else_arm="B", tz_offset_minutes=0),
        StateGuardRule(max_leverage=4.0, on_violation_arm="A"),     # if lev>4 -> A
        WeightedRule(weights={"A":0.7,"B":0.3}),
    ],
    default=default,
)

owner = router.owner("SBIN")  # -> "A"
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

# ---------- Minimal SymbolRouter protocol (mirrors selector/ab_tests.py) -----

class SymbolRouter:
    def arms(self) -> List[str]:
        raise NotImplementedError
    def owner(self, symbol: str) -> str:
        raise NotImplementedError
    def feedback(self, symbol: str, arm: str, trade_pnl: float) -> None:
        pass

# Optional: import bandit/hash routers if available for composition
try:
    from .ab_tests import HashSplitRouter, EpsilonGreedyRouter  # type: ignore
except Exception:  # fallback stubs to keep this file standalone-friendly
    class HashSplitRouter(SymbolRouter):  # type: ignore
        def __init__(self, arms: Iterable[str], share_map: Optional[Dict[str,float]]=None):
            self._arms=list(arms) or ["A"]
            shares = share_map or {a:1.0/len(self._arms) for a in self._arms}
            s=sum(shares.values()) or 1.0
            acc=0.0; self._cum=[]
            for a in self._arms:
                acc += shares.get(a,0.0)/s; self._cum.append((acc,a))
        def arms(self)->List[str]: return self._arms
        def owner(self, symbol:str)->str:
            h = abs(hash(symbol))%10_000/10_000.0
            for c,a in self._cum:
                if h<=c: return a
            return self._arms[-1]
    class EpsilonGreedyRouter(SymbolRouter):  # type: ignore
        def __init__(self, arms: Iterable[str], eps: float=0.1, decay: float=0.1):
            self._arms=list(arms) or ["A"]; self.eps=max(0,min(1,eps)); self.decay=max(0,min(1,decay)); self._q={}
        def arms(self)->List[str]: return self._arms
        def owner(self, symbol:str)->str:
            import random
            if random.random()<self.eps: return random.choice(self._arms)
            best,arm=-1e18,self._arms[0]
            for a in self._arms:
                q=self._q.get((symbol,a),0.0)
                if q>best: best,arm=q,a
            return arm
        def feedback(self, symbol:str, arm:str, trade_pnl:float)->None:
            q=self._q.get((symbol,arm),0.0); self._q[(symbol,arm)]=(1-self.decay)*q + self.decay*trade_pnl

# ------------------------------ State Provider --------------------------------

class StateProvider:
    """
    Optional adapter to expose live state to rules (e.g., execution agent).
    Implement any subset; missing values are treated as None.
    """
    def leverage(self) -> Optional[float]: return None
    def gross_exposure(self) -> Optional[float]: return None
    def telemetry(self) -> Dict: return {}

# ------------------------------ Rule base -------------------------------------

class Rule:
    """
    Contract: decide(symbol, now_ts) -> (decided: bool, arm: Optional[str])
    - decided=False => fall through to next rule
    - decided=True & arm is not None => route to arm
    - decided=True & arm is None    => explicitly undecided (block/skip), fall through
    """
    def decide(self, symbol: str, now_ts: float) -> Tuple[bool, Optional[str]]:
        raise NotImplementedError
    def feedback(self, symbol: str, arm: str, trade_pnl: float) -> None:
        """Optional learning feedback (e.g., bandit)."""
        pass

# ------------------------------ Concrete rules --------------------------------

@dataclass
class AllowDenyRule(Rule):
    """
    Hard allow/deny lists.
    - allow: {symbol: "A"/"B"} routes immediately.
    - deny: {symbol: None} indicates rule matched but no decision; falls through.
    """
    allow: Dict[str, str] = field(default_factory=dict)
    deny: Dict[str, Optional[str]] = field(default_factory=dict)
    def decide(self, symbol: str, now_ts: float) -> Tuple[bool, Optional[str]]:
        if symbol in self.allow:
            return True, self.allow[symbol]
        if symbol in self.deny:
            return True, None  # explicit no-decision -> fall through
        return False, None

@dataclass
class GroupRule(Rule):
    """
    Map groups of symbols to fixed arms.
    mapping: { group_name: ([symbols...], "A"/"B") }
    """
    mapping: Dict[str, Tuple[List[str], str]] = field(default_factory=dict)
    _index: Dict[str, str] = field(default_factory=dict, init=False)
    def __post_init__(self):
        for _, (syms, arm) in self.mapping.items():
            for s in syms:
                self._index[s] = arm
    def decide(self, symbol: str, now_ts: float) -> Tuple[bool, Optional[str]]:
        if symbol in self._index:
            return True, self._index[symbol]
        return False, None

@dataclass
class ScheduleRule(Rule):
    """
    Prefer an arm within certain local-time windows.
    prefer: {"A": [("09:15","12:00"), ("13:00","15:30")], "B":[...]}
    else_arm: used if symbol not matched by preferred windows (optional -> fall through)
    tz_offset_minutes: apply offset to UTC timestamp to derive local HH:MM.
    """
    prefer: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)
    else_arm: Optional[str] = None
    tz_offset_minutes: int = 0
    def decide(self, symbol: str, now_ts: float) -> Tuple[bool, Optional[str]]:
        # convert to local minutes since midnight
        t_local = now_ts + self.tz_offset_minutes * 60
        local_minutes = int((t_local % 86400) // 60)
        for arm, windows in self.prefer.items():
            for hhmm_start, hhmm_end in windows:
                s = _hhmm_to_minutes(hhmm_start)
                e = _hhmm_to_minutes(hhmm_end)
                if _in_window(local_minutes, s, e):
                    return True, arm
        if self.else_arm:
            return True, self.else_arm
        return False, None

@dataclass
class WeightedRule(Rule):
    """
    Weighted round-robin per symbol namespace.
    Example: weights={"A":0.7,"B":0.3} -> choose A 7x, B 3x in cycle.
    Deterministic per symbol (cycles independently).
    """
    weights: Dict[str, float]
    _cycles: Dict[str, itertools.cycle] = field(default_factory=dict, init=False)

    def _cycle_for(self, symbol: str):
        cyc = self._cycles.get(symbol)
        if cyc:
            return cyc
        seq = []
        wsum = sum(max(0.0, w) for w in self.weights.values()) or 1.0
        norm = {a: max(0.0, w)/wsum for a,w in self.weights.items()}
        # build a 100-step discrete wheel
        steps = []
        for arm, p in norm.items():
            k = max(1, int(round(p*100)))
            steps.extend([arm]*k)
        cyc = itertools.cycle(steps)
        self._cycles[symbol] = cyc
        return cyc

    def decide(self, symbol: str, now_ts: float) -> Tuple[bool, Optional[str]]:
        return True, next(self._cycle_for(symbol))

@dataclass
class HashFallbackRule(Rule):
    """Explicit hash choice; mostly redundant if PolicyRouter has default=HashSplitRouter."""
    arms: List[str]
    shares: Optional[Dict[str,float]] = None
    _h: Optional[HashSplitRouter] = None
    def __post_init__(self):
        self._h = HashSplitRouter(self.arms, self.shares)
    def decide(self, symbol: str, now_ts: float) -> Tuple[bool, Optional[str]]:
        return True, self._h.owner(symbol)  # type: ignore

@dataclass
class StateGuardRule(Rule):
    """
    Guardrails from live state (via StateProvider).
    - If leverage > max_leverage -> redirect to on_violation_arm
    - If gross_exposure > max_gross  -> redirect
    """
    state: StateProvider = field(default_factory=StateProvider)
    max_leverage: Optional[float] = None
    max_gross: Optional[float] = None
    on_violation_arm: Optional[str] = None
    def decide(self, symbol: str, now_ts: float) -> Tuple[bool, Optional[str]]:
        lev = _safe_float(self.state.leverage())
        gross = _safe_float(self.state.gross_exposure())
        if self.max_leverage is not None and lev is not None and lev > self.max_leverage:
            return True, self.on_violation_arm
        if self.max_gross is not None and gross is not None and gross > self.max_gross:
            return True, self.on_violation_arm
        return False, None

@dataclass
class BanditRule(Rule):
    """
    Epsilon-greedy overlay: learn best arm per symbol using feedback().
    Place it late in the rule order so earlier hard rules take priority.
    """
    arms: List[str]
    eps: float = 0.1
    decay: float = 0.1
    _bandit: Optional[EpsilonGreedyRouter] = None
    def __post_init__(self):
        self._bandit = EpsilonGreedyRouter(self.arms, eps=self.eps, decay=self.decay)
    def decide(self, symbol: str, now_ts: float) -> Tuple[bool, Optional[str]]:
        return True, self._bandit.owner(symbol)  # type: ignore
    def feedback(self, symbol: str, arm: str, trade_pnl: float) -> None:
        try:
            self._bandit.feedback(symbol, arm, trade_pnl)  # type: ignore
        except Exception:
            pass

# ------------------------------ Policy Router ---------------------------------

class PolicyRouter(SymbolRouter):
    """
    Evaluate rules in order; the first decisive rule returns the owner.
    If no rule decides, use the default router (e.g., HashSplitRouter).
    """
    def __init__(self, arms: Iterable[str], rules: List[Rule], default: Optional[SymbolRouter] = None):
        self._arms = list(arms) or ["A"]
        self._rules = list(rules) or []
        self._default = default or HashSplitRouter(self._arms)

    def arms(self) -> List[str]:
        return self._arms

    def owner(self, symbol: str) -> str:
        now_ts = time.time()
        for r in self._rules:
            decided, arm = r.decide(symbol, now_ts)
            if not decided:
                continue
            if arm is not None and arm in self._arms:
                return arm
            # decided but None -> fall through
        return self._default.owner(symbol)

    def feedback(self, symbol: str, arm: str, trade_pnl: float) -> None:
        # propagate to rules that care (e.g., BanditRule)
        for r in self._rules:
            try:
                r.feedback(symbol, arm, trade_pnl)
            except Exception:
                pass

# ------------------------------ Utilities -------------------------------------

def _hhmm_to_minutes(hhmm: str) -> int:
    hh, mm = hhmm.split(":")
    return int(hh)*60 + int(mm)

def _in_window(mins: int, start: int, end: int) -> bool:
    # supports wrap-around (e.g., 22:00 -> 01:00)
    if start <= end:
        return start <= mins <= end
    return mins >= start or mins <= end

def _safe_float(x) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None

# ------------------------------ YAML loader (optional) ------------------------

# If you want to build PolicyRouter from a YAML dict:
def build_from_config(cfg: Dict, state: Optional[StateProvider] = None) -> PolicyRouter:
    """
    cfg example (mirrors selector/config.yaml style):
    {
      "arms": ["A","B"],
      "default": {"kind": "hash", "shares": {"A":0.5,"B":0.5}},
      "rules": [
        {"kind":"allow_deny","allow":{"TSLA":"B"},"deny":{"GME":null}},
        {"kind":"group","mapping":{"PSU_BANKS": { "symbols": ["SBIN","BOB","PNB"], "arm": "A"}}},
        {"kind":"schedule","prefer":{"A":[["09:15","12:00"]]},"else_arm":"B","tz_offset_minutes":330},
        {"kind":"state_guard","max_leverage":4.0,"on_violation_arm":"A"},
        {"kind":"weighted","weights":{"A":0.7,"B":0.3}},
        {"kind":"bandit","eps":0.1,"decay":0.1}
      ]
    }
    """
    arms = list(cfg.get("arms") or ["A"])
    # default router
    dft = cfg.get("default", {"kind": "hash"})
    if dft.get("kind") == "hash":
        default = HashSplitRouter(arms, share_map=dft.get("shares"))
    else:
        default = HashSplitRouter(arms)

    rules: List[Rule] = []
    for r in cfg.get("rules", []):
        kind = (r.get("kind") or "").lower()
        if kind == "allow_deny":
            rules.append(AllowDenyRule(allow=r.get("allow", {}), deny=r.get("deny", {})))
        elif kind == "group":
            mapping = {}
            raw = r.get("mapping", {})
            for gname, spec in raw.items():
                syms = spec.get("symbols") or spec.get("syms") or []
                arm = spec.get("arm") or arms[0]
                mapping[gname] = (list(syms), arm)
            rules.append(GroupRule(mapping=mapping))
        elif kind == "schedule":
            rules.append(ScheduleRule(
                prefer={k: [tuple(t) for t in v] for k, v in (r.get("prefer") or {}).items()},
                else_arm=r.get("else_arm"),
                tz_offset_minutes=int(r.get("tz_offset_minutes") or 0),
            ))
        elif kind == "state_guard":
            rules.append(StateGuardRule(
                state=state or StateProvider(),
                max_leverage=r.get("max_leverage"),
                max_gross=r.get("max_gross"),
                on_violation_arm=r.get("on_violation_arm"),
            ))
        elif kind == "weighted":
            rules.append(WeightedRule(weights=r.get("weights", {})))
        elif kind == "bandit":
            rules.append(BanditRule(arms=arms, eps=float(r.get("eps", 0.1)), decay=float(r.get("decay", 0.1))))
        elif kind == "hash":
            rules.append(HashFallbackRule(arms=arms, shares=r.get("shares")))
        else:
            # unknown kind -> ignore
            pass

    return PolicyRouter(arms=arms, rules=rules, default=default)


# ------------------------------ Smoke test ------------------------------------

if __name__ == "__main__":
    # Minimal self-check
    pr = PolicyRouter(
        arms=["A","B"],
        rules=[
            AllowDenyRule(allow={"TSLA":"B"}, deny={"GME": None}),
            GroupRule(mapping={"PSU": (["SBIN","BOB","PNB"], "A")}),
            ScheduleRule(prefer={"A":[("09:15","12:00")]}, else_arm="B", tz_offset_minutes=330),
            StateGuardRule(max_leverage=4.0, on_violation_arm="A"),
            WeightedRule(weights={"A":0.7,"B":0.3}),
            BanditRule(arms=["A","B"], eps=0.2, decay=0.1),
        ],
        default=HashSplitRouter(["A","B"])
    )

    for s in ["TSLA","GME","SBIN","AAPL","MSFT","PNB"]:
        print(s, "->", pr.owner(s))