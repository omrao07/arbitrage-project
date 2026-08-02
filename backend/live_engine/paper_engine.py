# backend/live_engine/paper_engine.py
"""
LiveEngine: orchestrates multiple StreamStrategyRunners, the SignalAggregator,
RiskGates, and a monitor loop for the generic / paper-trading path.

This is the lightweight orchestrator (Redis-stream driven, PaperBroker-backed).
The institutional NSE/BSE path is driven separately by EngineState +
LiveEngineScheduler (engine_state.py / scheduler.py / __main__.py).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict, Optional

from backend.engine.strategy_base import Strategy

from .risk_gates import RiskGates
from .signal_aggregator import SignalAggregator
from .stream_runner import StreamStrategyRunner

try:
    from backend.execution.brokers.paper import OrderRequest, OrderResult, PaperBroker
    _HAVE_PAPER = True
except Exception:
    _HAVE_PAPER = False
    PaperBroker = None  # type: ignore[assignment,misc]

log = logging.getLogger(__name__)


class LiveEngine:
    """
    Central orchestrator for live trading.
    Set paper_mode=True (default from PAPER_MODE env var) to use the
    PaperBroker for all order routing instead of live venues.

    Usage:
        engine = LiveEngine(capital=1_000_000, paper_mode=True)
        engine.add_strategy(MyStrategy(), stream="ticks.equities.us")
        engine.start()
        ...
        engine.stop()
    """

    def __init__(
        self,
        capital: float = 1_000_000.0,
        aggregator_mode: str = "vol",
        signal_age_ms: int = 30_000,
        paper_mode: Optional[bool] = None,
        fee_bps: float = 5.0,
        slippage_bps: float = 5.0,
    ):
        self.capital = capital
        self.aggregator = SignalAggregator(mode=aggregator_mode, max_signal_age_ms=signal_age_ms)
        self.risk = RiskGates(capital=capital)
        self._runners: Dict[str, StreamStrategyRunner] = {}
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False

        if paper_mode is None:
            paper_mode = os.getenv("PAPER_MODE", "true").lower() in ("1", "true", "yes")
        self.paper_mode = paper_mode
        self.broker: Optional[PaperBroker] = None
        if paper_mode and _HAVE_PAPER:
            self.broker = PaperBroker(
                starting_cash=capital,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
            )
            log.info("LiveEngine: paper broker initialised (capital=%.0f)", capital)

    def add_strategy(
        self,
        strategy: Strategy,
        stream: str,
        start_id: str = "$",
    ) -> None:
        name = strategy.ctx.name
        runner = StreamStrategyRunner(
            strategy=strategy,
            stream=stream,
            aggregator=self.aggregator,
            start_id=start_id,
            on_error=self._on_strategy_error,
        )
        with self._lock:
            if name in self._runners:
                log.warning("strategy %s already registered; replacing", name)
                self._runners[name].stop()
            self._runners[name] = runner

    def remove_strategy(self, name: str) -> None:
        with self._lock:
            runner = self._runners.pop(name, None)
        if runner:
            runner.stop()

    def start(self) -> None:
        self._running = True
        with self._lock:
            for runner in self._runners.values():
                if not runner.is_alive():
                    runner.start()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="live-engine-monitor"
        )
        self._monitor_thread.start()
        log.info("LiveEngine started with %d strategies", len(self._runners))

    def stop(self) -> None:
        self._running = False
        with self._lock:
            for runner in self._runners.values():
                runner.stop()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        log.info("LiveEngine stopped")

    def submit_order(self, req: "OrderRequest") -> Optional["OrderResult"]:  # type: ignore[name-defined]
        """Route an order through the paper broker (no-op if live mode or broker unavailable)."""
        if self.broker is None:
            log.warning("submit_order: no broker available (live mode or import failed)")
            return None
        result = self.broker.submit_order(req)
        if result.status == "filled":
            self.risk.update_pnl(0.0)  # positions update handled by broker
        return result

    def status(self) -> Dict:
        sig_summary = self.aggregator.summary()
        gate1_ok, g1 = self.risk.gate1_daily_loss()
        gate2_ok, g2 = self.risk.gate2_drawdown()
        out: Dict = {
            "running": self._running,
            "paper_mode": self.paper_mode,
            "n_strategies": len(self._runners),
            "signal": sig_summary,
            "risk": {
                "daily_pnl": self.risk._daily_pnl,
                "gate1_daily_loss": "ok" if gate1_ok else g1,
                "gate2_drawdown": "ok" if gate2_ok else g2,
            },
        }
        if self.broker is not None:
            out["broker"] = self.broker.account()
        return out

    def _on_strategy_error(self, name: str, exc: Exception) -> None:
        log.error("strategy %s error: %s", name, exc)

    def _monitor_loop(self) -> None:
        """Periodically log engine health and check daily loss gate."""
        while self._running:
            time.sleep(60)
            ok, reason = self.risk.gate1_daily_loss()
            if not ok:
                log.critical("DAILY LOSS GATE TRIGGERED: %s — halting all strategies", reason)
                self.stop()
                break
            ok2, reason2 = self.risk.gate2_drawdown()
            if not ok2:
                log.critical("DRAWDOWN GATE TRIGGERED: %s — halting all strategies", reason2)
                self.stop()
                break
            log.info("LiveEngine heartbeat: %d strategies alive", len(self._runners))
            try:
                self.risk.to_redis()
            except Exception:
                pass
