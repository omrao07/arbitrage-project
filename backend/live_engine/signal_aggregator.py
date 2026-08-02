# backend/live_engine/signal_aggregator.py
"""
Aggregates signals from all active strategies into a single allocation vector.
Supports equal-weight, vol-weighted, and signal-strength-weighted modes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Literal


@dataclass
class StrategySignal:
    name: str
    score: float        # [-1, +1]
    vol: float          # annualized return vol (for vol-weighting)
    drawdown: float     # current drawdown fraction
    ts_ms: int          # timestamp of last update


class SignalAggregator:
    """
    Collects per-strategy signals and combines them into portfolio weights.

    Modes:
      "equal"   — equal weight across active strategies (score > 0 = long)
      "vol"     — 1/vol weighting (risk-parity style)
      "score"   — raw score weighting (|score| as weight)
    """

    def __init__(
        self,
        mode: Literal["equal", "vol", "score"] = "vol",
        max_signal_age_ms: int = 30_000,     # stale after 30s
        min_score_threshold: float = 0.05,   # ignore near-zero signals
    ):
        self.mode = mode
        self.max_signal_age_ms = max_signal_age_ms
        self.min_score = min_score_threshold
        self._signals: Dict[str, StrategySignal] = {}

    def update(
        self,
        name: str,
        score: float,
        vol: float = 0.2,
        drawdown: float = 0.0,
    ) -> None:
        self._signals[name] = StrategySignal(
            name=name,
            score=max(-1.0, min(1.0, score)),
            vol=max(0.001, vol),
            drawdown=max(0.0, drawdown),
            ts_ms=int(time.time() * 1000),
        )

    def _active_signals(self) -> Dict[str, StrategySignal]:
        now = int(time.time() * 1000)
        return {
            k: v for k, v in self._signals.items()
            if (now - v.ts_ms) <= self.max_signal_age_ms
            and abs(v.score) >= self.min_score
        }

    def aggregate(self) -> Dict[str, float]:
        """
        Returns normalized weight per strategy name in [-1, +1].
        Positive = long allocation, negative = short.
        """
        active = self._active_signals()
        if not active:
            return {}

        if self.mode == "equal":
            w = {k: 1.0 for k in active}
        elif self.mode == "vol":
            w = {k: 1.0 / s.vol for k, s in active.items()}
        else:  # score
            w = {k: abs(s.score) for k, s in active.items()}

        total = sum(w.values()) or 1.0
        # Apply signal direction
        return {
            k: (w[k] / total) * active[k].score
            for k in active
        }

    def combined_score(self) -> float:
        """Single scalar [-1, +1] representing net portfolio direction."""
        agg = self.aggregate()
        return max(-1.0, min(1.0, sum(agg.values())))

    def summary(self) -> Dict[str, object]:
        active = self._active_signals()
        return {
            "n_active": len(active),
            "n_stale": len(self._signals) - len(active),
            "mode": self.mode,
            "combined_score": self.combined_score(),
            "strategies": {
                k: {"score": v.score, "vol": v.vol, "dd": v.drawdown}
                for k, v in active.items()
            },
        }
