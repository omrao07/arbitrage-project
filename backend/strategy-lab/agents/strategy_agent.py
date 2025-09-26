# agents/strategy_agent.py
"""
StrategyAgent
-------------
A small coordinator that:
- Hosts multiple Strategy objects (signal generators).
- Converts signals to target portfolio weights (per-strategy and combined).
- Enforces simple caps: gross, per-symbol, long/short, max positions.
- Rebalances on a cadence and sends delta orders to ExecutionAgent.
- Keeps a cash buffer and uses latest prices from ExecutionAgent.
- Hands through on_price/on_fill events to strategies.

No external deps. Works with agents.execution_agent.ExecutionAgent.

Quick start
-----------
from agents.execution_agent import ExecutionAgent, Side, OrderType
from agents.strategy_agent import StrategyAgent, StrategyBase, RebalanceConfig

exec_agent = ExecutionAgent(starting_cash=1_000_000)
sa = StrategyAgent(exec_agent, RebalanceConfig())
sa.register(MyStrategy(...), weight=1.0)
# in your loop:
sa.on_price("AAPL", 200.0)
sa.maybe_rebalance()  # will place orders if time/cadence criteria met
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable
import math
import time

try:
    # Local import within this repo layout
    from .execution_agent import ExecutionAgent, Side, OrderType, RiskReject # type: ignore
except Exception:  # pragma: no cover
    # Fallback names for linting if imported standalone
    ExecutionAgent = object  # type: ignore
    class Side: BUY="BUY"; SELL="SELL"  # type: ignore
    class OrderType: MARKET="MARKET"    # type: ignore
    class RiskReject(Exception): ...    # type: ignore


# ------------------------------- Strategy API --------------------------------

class StrategyBase:
    """
    Minimal contract for a strategy.
    Implement generate_signals() to return {symbol: score}. Higher => more long.
    Optional hooks: on_price, on_fill.
    """
    name: str = "strategy"
    warmup_bars: int = 0
    max_positions: Optional[int] = None          # per strategy
    gross_target: float = 0.5                    # per strategy gross weight target (0..1)
    long_only: bool = False

    def universe(self) -> Iterable[str]:
        """Return the symbols this strategy cares about (optional override)."""
        return []

    def generate_signals(self, now_ts: float) -> Dict[str, float]:
        """Return cross-sectional scores at current time."""
        raise NotImplementedError

    # Optional life-cycle hooks
    def on_price(self, symbol: str, price: float) -> None:
        pass

    def on_fill(self, order_id: str, symbol: str, qty: float, price: float) -> None:
        pass


# ------------------------------- Config models --------------------------------

@dataclass
class RebalanceConfig:
    cadence_sec: float = 60.0             # try to rebalance at most once per N seconds
    cash_buffer: float = 0.02             # keep 2% in cash (reduces target gross)
    global_gross_cap: float = 1.0         # combined gross exposure cap (|w| sum)
    per_symbol_cap: float = 0.10          # abs weight cap per name
    long_cap: float = 0.60                # sum of positive target weights capped
    short_cap: float = 0.60               # sum of negative abs target weights capped
    min_trade_notional: float = 50.0      # skip tiny trades
    min_shares: float = 1.0               # round to whole shares by default
    use_market_orders: bool = True        # true => MARKET; false => LIMIT at last px
    limit_price_bps: float = 1.0          # if LIMIT, set at ±1bp from last as buffer


# ------------------------------ Strategy Agent --------------------------------

class StrategyAgent:
    def __init__(self, execution: ExecutionAgent, cfg: Optional[RebalanceConfig] = None): # type: ignore
        self.x = execution
        self.cfg = cfg or RebalanceConfig()
        self._strategies: List[Tuple[StrategyBase, float]] = []  # (strategy, blend_weight)
        self._last_rebalance_ts: float = 0.0

    # ---------- registration ----------

    def register(self, strategy: StrategyBase, weight: float = 1.0) -> None:
        """
        Add a strategy with a blending weight (relative importance in combine step).
        """
        self._strategies.append((strategy, float(weight)))

    # ---------- event bridge ----------

    def on_price(self, symbol: str, price: float) -> None:
        """Forward price to ExecutionAgent and strategies."""
        self.x.update_price(symbol, price)
        for s, _ in self._strategies:
            try:
                s.on_price(symbol, price)
            except Exception:
                pass  # strategies should not break the loop

    def on_fill(self, order_id: str, symbol: str, qty: float, price: float) -> None:
        for s, _ in self._strategies:
            try:
                s.on_fill(order_id, symbol, qty, price)
            except Exception:
                pass

    # ---------- rebalance entrypoint ----------

    def maybe_rebalance(self, force: bool = False) -> Optional[Dict[str, float]]:
        ts = time.time()
        if not force and ts - self._last_rebalance_ts < self.cfg.cadence_sec:
            return None

        targets = self._compute_combined_targets(ts)
        self._place_delta_orders(targets)
        self._last_rebalance_ts = ts
        return targets

    # ---------- core steps ----------

    def _compute_combined_targets(self, ts: float) -> Dict[str, float]:
        """
        1) Ask each strategy for scores.
        2) Convert scores -> weights per strategy.
        3) Blend strategy weights by their registration weights.
        4) Enforce global caps & cash buffer.
        """
        per_strat_weights: List[Tuple[Dict[str, float], float]] = []
        for strat, w in self._strategies:
            try:
                scores = strat.generate_signals(ts) or {}
            except Exception:
                scores = {}

            weights = self._scores_to_weights(scores,
                                              gross_target=max(0.0, min(1.0, strat.gross_target)),
                                              max_positions=strat.max_positions,
                                              long_only=strat.long_only)
            per_strat_weights.append((weights, w))

        # Blend
        combined: Dict[str, float] = {}
        total_w = sum(max(0.0, w) for _, w in per_strat_weights) or 1.0
        for weights, w in per_strat_weights:
            bw = max(0.0, w) / total_w
            for sym, tw in weights.items():
                combined[sym] = combined.get(sym, 0.0) + bw * tw

        # Caps & cash buffer
        combined = self._apply_caps(combined)
        combined = self._apply_cash_buffer(combined)

        return combined

    # ---------- helpers: signals -> weights ----------

    def _scores_to_weights(
        self,
        scores: Dict[str, float],
        gross_target: float,
        max_positions: Optional[int],
        long_only: bool,
    ) -> Dict[str, float]:
        """
        Convert arbitrary scores to portfolio weights by:
        - z-scoring (mean/std) -> continuous tilt
        - clipping to +/- per_symbol_cap
        - normalizing to target gross
        - optionally zeroing shorts for long-only
        - optionally limiting number of names by absolute score
        """
        if not scores:
            return {}

        # Rank by absolute score if we need to limit positions
        names = list(scores.keys())
        if max_positions and max_positions < len(names):
            names = [s for s, _ in sorted(scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:max_positions]]
            scores = {s: scores[s] for s in names}

        # z-score (stdlib)
        mu = sum(scores.values()) / len(scores)
        var = sum((v - mu) ** 2 for v in scores.values()) / max(1, (len(scores) - 1))
        std = math.sqrt(var) if var > 0 else 0.0

        weights: Dict[str, float] = {}
        for s, v in scores.items():
            z = (v - mu) / std if std > 1e-12 else (1.0 if v >= mu else -1.0)
            w = z
            if long_only and w < 0:
                w = 0.0
            # per-symbol clip
            w = max(-self.cfg.per_symbol_cap, min(self.cfg.per_symbol_cap, w))
            weights[s] = w

        # Normalize to target gross
        gross = sum(abs(w) for w in weights.values()) or 1.0
        scale = gross_target / gross
        for s in list(weights.keys()):
            weights[s] *= scale

        # Enforce long/short caps
        weights = self._apply_long_short_caps(weights)
        return weights

    def _apply_long_short_caps(self, weights: Dict[str, float]) -> Dict[str, float]:
        pos = {s: w for s, w in weights.items() if w > 0}
        neg = {s: -w for s, w in weights.items() if w < 0}
        sum_long = sum(pos.values())
        sum_short = sum(neg.values())

        # scale down if exceeding caps
        if sum_long > self.cfg.long_cap and sum_long > 0:
            k = self.cfg.long_cap / sum_long
            for s in pos:
                weights[s] *= k
        if sum_short > self.cfg.short_cap and sum_short > 0:
            k = self.cfg.short_cap / sum_short
            for s in neg:
                weights[s] *= k
        return weights

    def _apply_caps(self, weights: Dict[str, float]) -> Dict[str, float]:
        # per-symbol cap already applied; also enforce global gross cap
        gross = sum(abs(w) for w in weights.values()) or 1.0
        if gross > self.cfg.global_gross_cap:
            k = self.cfg.global_gross_cap / gross
            for s in list(weights.keys()):
                weights[s] *= k
        return weights

    def _apply_cash_buffer(self, weights: Dict[str, float]) -> Dict[str, float]:
        if self.cfg.cash_buffer <= 0:
            return weights
        k = max(0.0, 1.0 - self.cfg.cash_buffer)
        for s in list(weights.keys()):
            weights[s] *= k
        return weights

    # ---------- generate delta orders ----------

    def _place_delta_orders(self, target_weights: Dict[str, float]) -> None:
        equity = self.x.equity()
        if equity <= 0:
            return

        last = {s: self.x.last_price(s) for s in target_weights.keys()}
        # derive current weights from positions
        cur_w: Dict[str, float] = {}
        for s, px in last.items():
            if not px or px <= 0:
                continue
            pos = self.x.position(s)
            cur_w[s] = (pos.qty * px) / max(1e-12, equity)

        # union of symbols (include any currently held but not targeted -> target 0)
        for s in list(self._held_symbols()):
            if s not in target_weights:
                target_weights[s] = 0.0
                last.setdefault(s, self.x.last_price(s))

        for s, tw in target_weights.items():
            px = last.get(s)
            if px is None or px <= 0:
                continue
            cw = cur_w.get(s, 0.0)
            dw = tw - cw
            notional = abs(dw) * equity
            if notional < self.cfg.min_trade_notional:
                continue

            target_shares = (tw * equity) / px
            cur_shares = (cw * equity) / px
            delta_shares = target_shares - cur_shares

            # Round shares
            step = max(1.0, self.cfg.min_shares)
            # preserve sign
            if delta_shares >= 0:
                delta_shares = math.floor(delta_shares / step) * step
            else:
                delta_shares = -math.floor(abs(delta_shares) / step) * step

            if abs(delta_shares) < self.cfg.min_shares:
                continue

            side = Side.BUY if delta_shares > 0 else Side.SELL
            qty = abs(delta_shares)

            try:
                if self.cfg.use_market_orders:
                    self.x.submit_order(s, side, qty, type=OrderType.MARKET)
                else:
                    # Small buffer on limit (±bps)
                    bps = self.cfg.limit_price_bps / 1e4
                    limit = px * (1 + bps) if side == Side.BUY else px * (1 - bps)
                    self.x.submit_order(s, side, qty, type=OrderType.LIMIT, limit_price=limit) # type: ignore
            except RiskReject as e:
                # Skip this symbol if risk rejects; continue others
                # (You can add logging here.)
                continue

    def _held_symbols(self) -> Iterable[str]:
        # All symbols with non-zero positions
        for sym, pos in list(getattr(self.x, "_positions", {}).items()):
            if pos.qty != 0:
                yield sym