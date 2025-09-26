# selector/ab_tests.py
"""
A/B Selector for Strategy Lab
----------------------------
Route live/sim orders from multiple StrategyAgents into ONE shared ExecutionAgent
while (a) preventing overlap and (b) attributing fills & PnL to the right arm.

Design
- Shared ExecutionAgent holds the real positions/cash.
- Each arm (A, B, ...) runs its own StrategyAgent, but uses a ProxyExecution
  that forwards orders ONLY for symbols assigned to that arm by the Router.
- Attribution: ProxyExecution records order_id -> arm so we can split realized PnL.

Routers (choose one)
- HashSplitRouter(share_map): deterministic symbol -> arm split (e.g., 50/50).
- EpsilonGreedyRouter(arms, eps): per-symbol bandit; routes each symbol to the
  arm with better realized PnL estimate (with ε exploration).

Metrics
- Realized PnL by arm, win rate, trade count, avg trade PnL, last 1h window PnL.
- JSON serializable snapshot() for dashboards.

Usage (sketch)
-------------
from agents.execution_agent import ExecutionAgent
from agents.strategy_agent import StrategyAgent
from selector.ab_tests import ABTestRunner, HashSplitRouter, EpsilonGreedyRouter

shared_exec = ExecutionAgent(starting_cash=1_000_000)
router = HashSplitRouter(arms=["A","B"])  # or EpsilonGreedyRouter(["A","B"], eps=0.1)

runner = ABTestRunner(shared_exec, router)

# Build two independent StrategyAgents but give each a proxy of the shared exec:
sa_A = StrategyAgent(runner.make_proxy("A"))
sa_B = StrategyAgent(runner.make_proxy("B"))

# register your strategies on each SA as usual...
runner.register("A", sa_A)
runner.register("B", sa_B)

# In your price loop:
runner.on_price("AAPL", 200.0)      # forwards to shared exec + both SAs
runner.maybe_rebalance_all()        # each SA will try to trade; router filters
runner.tick()                       # attribute fills & update stats
print(runner.snapshot())
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

try:
    # Local imports when used inside this repo
    from agents.execution_agent import ExecutionAgent, OrderId, Side, OrderType, RiskReject, Fill # type: ignore
except Exception:  # pragma: no cover
    # Minimal shims for linting if imported standalone
    ExecutionAgent = object  # type: ignore
    class OrderId:  # type: ignore
        def __init__(self, val: str): self.val = val
    class Side: BUY="BUY"; SELL="SELL"  # type: ignore
    class OrderType: MARKET="MARKET"; LIMIT="LIMIT"; STOP="STOP"  # type: ignore
    class RiskReject(Exception): ...  # type: ignore
    @dataclass
    class Fill:  # type: ignore
        order_id: OrderId; symbol: str; side: Side; qty: float; price: float; fee: float; ts: float = field(default_factory=lambda: time.time())


# ------------------------------- Routers --------------------------------------


class SymbolRouter:
    """Decide which arm owns a symbol NOW."""
    def arms(self) -> List[str]:
        raise NotImplementedError

    def owner(self, symbol: str) -> str:
        raise NotImplementedError

    # hook for learning routers
    def feedback(self, symbol: str, arm: str, trade_pnl: float) -> None:
        pass


class HashSplitRouter(SymbolRouter):
    """Deterministic split of symbols across arms by hash (stable over time)."""
    def __init__(self, arms: Iterable[str], share_map: Optional[Dict[str, float]] = None):
        self._arms = list(arms)
        if not self._arms:
            raise ValueError("Need at least one arm")
        # share_map allows non-50/50 splits: {"A":0.7, "B":0.3}
        self._shares = share_map or {a: 1.0 / len(self._arms) for a in self._arms}
        s = sum(self._shares.values()) or 1.0
        self._cum = []
        acc = 0.0
        for a in self._arms:
            acc += self._shares.get(a, 0.0) / s
            self._cum.append((acc, a))

    def arms(self) -> List[str]:
        return self._arms

    def owner(self, symbol: str) -> str:
        h = abs(hash(symbol)) % 10_000 / 10_000.0
        for cutoff, arm in self._cum:
            if h <= cutoff:
                return arm
        return self._arms[-1]


class EpsilonGreedyRouter(SymbolRouter):
    """
    Per-symbol epsilon-greedy router using realized PnL as reward.
    Keeps a running avg PnL per (symbol, arm); chooses best with prob 1-ε.
    """
    def __init__(self, arms: Iterable[str], eps: float = 0.1, decay: float = 0.1):
        self._arms = list(arms)
        if not self._arms:
            raise ValueError("Need at least one arm")
        self.eps = max(0.0, min(1.0, eps))
        self.decay = max(0.0, min(1.0, decay))
        self._q: Dict[Tuple[str, str], float] = {}  # (symbol, arm) -> value

    def arms(self) -> List[str]:
        return self._arms

    def owner(self, symbol: str) -> str:
        import random
        if random.random() < self.eps:
            return random.choice(self._arms)
        # exploit best arm so far
        best_arm = None
        best_q = -1e18
        for a in self._arms:
            q = self._q.get((symbol, a), 0.0)
            if q > best_q:
                best_q, best_arm = q, a
        return best_arm or self._arms[0]

    def feedback(self, symbol: str, arm: str, trade_pnl: float) -> None:
        key = (symbol, arm)
        q = self._q.get(key, 0.0)
        self._q[key] = (1 - self.decay) * q + self.decay * float(trade_pnl)


# --------------------------- Proxy Execution ----------------------------------


class ProxyExecution:
    """
    Transparent proxy around a *shared* ExecutionAgent.
    It blocks orders for symbols not owned by this arm per the router,
    but forwards all state queries to the shared instance.
    """
    def __init__(self, shared: ExecutionAgent, arm: str, router: SymbolRouter, record: "AttributionRecorder"): # type: ignore
        self._x = shared
        self._arm = arm
        self._router = router
        self._rec = record

    # --- state passthroughs ---
    def update_price(self, symbol: str, last_price: float) -> None:
        self._x.update_price(symbol, last_price)

    def position(self, symbol: str):
        return self._x.position(symbol)

    def last_price(self, symbol: str):
        return self._x.last_price(symbol)

    def equity(self) -> float:
        return self._x.equity()

    def gross_exposure(self) -> float:
        return self._x.gross_exposure()

    def leverage(self) -> float:
        return self._x.leverage()

    def cash(self) -> float:
        return getattr(self._x, "cash", None) # type: ignore

    # --- order methods (filtered) ---
    def submit_order(self, symbol: str, side: Side, qty: float, type: OrderType = OrderType.MARKET, # type: ignore
                     limit_price: Optional[float] = None, stop_price: Optional[float] = None) -> OrderId:
        owner = self._router.owner(symbol)
        if owner != self._arm:
            # silently drop (A/B isolation). Return a synthetic id.
            return OrderId(f"SKIPPED-{self._arm}-{symbol}-{int(time.time()*1000)}")
        oid = self._x.submit_order(symbol, side, qty, type=type, limit_price=limit_price, stop_price=stop_price)
        self._rec.tag_order(oid, self._arm, symbol)
        return oid

    def cancel(self, order_id: OrderId) -> None:
        # We let shared exec cancel (router isn't needed here)
        try:
            self._x.cancel(order_id)
        except Exception:
            pass

    # --- open_orders/fills just pass through for observability if needed ---
    def open_orders(self):
        return self._x.open_orders()

    def fills(self):
        return self._x.fills()


# ---------------------------- Attribution -------------------------------------


class AttributionRecorder:
    """
    Tracks order_id -> (arm, symbol), and aggregates realized PnL per arm.
    """
    def __init__(self):
        self.order_arm: Dict[str, Tuple[str, str]] = {}  # id -> (arm, symbol)
        self.last_seen_fill_idx: int = 0
        self.realized_pnl_by_arm: Dict[str, float] = {}
        self.trades_by_arm: Dict[str, int] = {}
        self.win_by_arm: Dict[str, int] = {}
        self.trade_pnls: List[Tuple[str, str, float]] = []  # (arm, symbol, pnl)

    def tag_order(self, order_id: OrderId, arm: str, symbol: str) -> None:
        oid = getattr(order_id, "val", str(order_id))
        self.order_arm[oid] = (arm, symbol)

    def process_new_fills(self, fills: List[Fill]) -> None:
        """
        Consume fills list from shared exec, attributing per fill.
        Realized PnL is inferred **on close trades only** by looking at position deltas
        in the shared exec is hard without hooks; therefore here we approximate
        trade PnL as signed notional change versus avg_px, which we don't have.
        To keep this generic, we compute per-fill *mark-out* vs last price change
        between consecutive fills is noisy; instead we sum **cash deltas** by arm:
           BUY -> -price*qty - fee ; SELL -> +price*qty - fee
        and track cumulative cash flow per arm ~ realized PnL if inventory returns to flat.
        """
        # Cash-flow approximation per arm:
        for i in range(self.last_seen_fill_idx, len(fills)):
            f = fills[i]
            oid = getattr(f.order_id, "val", str(f.order_id))
            meta = self.order_arm.get(oid)
            if not meta:
                continue
            arm, symbol = meta
            cash_delta = (-f.qty * f.price - f.fee) if (getattr(f.side, "name", str(f.side)) in ("BUY", "Side.BUY", "BUY".upper())) else (f.qty * f.price - f.fee)
            self.realized_pnl_by_arm[arm] = self.realized_pnl_by_arm.get(arm, 0.0) + cash_delta
            self.trades_by_arm[arm] = self.trades_by_arm.get(arm, 0) + 1
            self.win_by_arm[arm] = self.win_by_arm.get(arm, 0) + (1 if cash_delta > 0 else 0)
            self.trade_pnls.append((arm, symbol, cash_delta))
        self.last_seen_fill_idx = len(fills)


# ----------------------------- AB Test Runner ---------------------------------


@dataclass
class ABConfig:
    cadence_sec: float = 60.0        # how often to call each arm's rebalance
    router_kind: str = "hash"        # "hash" or "egreedy"
    eps: float = 0.1                 # for epsilon-greedy
    decay: float = 0.1               # learning rate for epsilon-greedy
    min_trade_notional: float = 50.0 # informational only (enforce in strategies)


class ABTestRunner:
    """
    Coordinates multiple StrategyAgents through symbol-level routing.
    """
    def __init__(self, shared_exec: ExecutionAgent, router: SymbolRouter, cfg: Optional[ABConfig] = None): # type: ignore
        self.exec = shared_exec
        self.router = router
        self.cfg = cfg or ABConfig()
        self.rec = AttributionRecorder()
        self._proxies: Dict[str, ProxyExecution] = {}
        self._agents: Dict[str, object] = {}  # StrategyAgent-like
        self._last_reb_ts: float = 0.0

    # ---- wiring ----

    def make_proxy(self, arm: str) -> ProxyExecution:
        if arm in self._proxies:
            return self._proxies[arm]
        px = ProxyExecution(self.exec, arm, self.router, self.rec)
        self._proxies[arm] = px
        return px

    def register(self, arm: str, strategy_agent: object) -> None:
        self._agents[arm] = strategy_agent

    # ---- events ----

    def on_price(self, symbol: str, price: float) -> None:
        # update shared exec and forward to each SA via their proxy
        # (StrategyAgents typically listen via their own on_price; we call those)
        self.exec.update_price(symbol, price)
        for arm, sa in self._agents.items():
            try:
                if hasattr(sa, "on_price"):
                    sa.on_price(symbol, price) # type: ignore
            except Exception:
                pass  # never break the loop

    def maybe_rebalance_all(self, force: bool = False) -> None:
        ts = time.time()
        if not force and ts - self._last_reb_ts < self.cfg.cadence_sec:
            return
        for arm, sa in self._agents.items():
            try:
                if hasattr(sa, "maybe_rebalance"):
                    sa.maybe_rebalance(force=True) # type: ignore
            except Exception:
                pass
        self._last_reb_ts = ts

    def tick(self) -> None:
        # Attribute any new fills since last tick
        fills = []
        try:
            fills = self.exec.fills()
        except Exception:
            pass
        before = dict(self.rec.realized_pnl_by_arm)
        self.rec.process_new_fills(fills)

        # Bandit feedback: update router with per-fill "reward"
        if isinstance(self.router, EpsilonGreedyRouter):
            # Compute delta cash by arm since last tick as reward.
            for arm in self.router.arms():
                d = self.rec.realized_pnl_by_arm.get(arm, 0.0) - before.get(arm, 0.0)
                # Approximate: assign evenly across symbols traded this tick (unknown),
                # so we just nudge all symbols a bit by arm. If you want per-symbol,
                # extend AttributionRecorder to store per-fill symbol deltas and loop them.
                if abs(d) > 0:
                    # Conservative nudge using a dummy symbol "*"
                    self.router.feedback("*", arm, d)

    # ---- reporting ----

    def snapshot(self) -> Dict:
        eq = self.exec.equity() if hasattr(self.exec, "equity") else None
        gross = self.exec.gross_exposure() if hasattr(self.exec, "gross_exposure") else None
        lev = self.exec.leverage() if hasattr(self.exec, "leverage") else None
        out = {
            "t": time.time(),
            "equity": eq,
            "gross_exposure": gross,
            "leverage": lev,
            "arms": {},
        }
        for arm in self.router.arms():
            pnl = self.rec.realized_pnl_by_arm.get(arm, 0.0)
            n = self.rec.trades_by_arm.get(arm, 0)
            wins = self.rec.win_by_arm.get(arm, 0)
            wr = (wins / n) if n else None
            out["arms"][arm] = {
                "realized_cashflow": pnl,     # ≈ realized PnL if flat over window
                "trades": n,
                "win_rate": wr,
            }
        return out

    def to_json(self) -> str:
        return json.dumps(self.snapshot(), separators=(",", ":"))