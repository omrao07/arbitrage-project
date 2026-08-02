"""
D-Strategies Live Engine — the single Python live-trading engine.

Two orchestration paths live side by side on top of the ``backend.engine``
foundation:

* **Generic / paper path** — the lightweight, Redis-stream-driven ``LiveEngine``
  (``paper_engine``) with ``RiskGates``, ``SignalAggregator`` and per-strategy
  ``StreamStrategyRunner`` threads, backed by the ``PaperBroker``.
* **Institutional NSE/BSE path** — ``EngineState`` + ``LiveEngineScheduler``
  (APScheduler jobs), ``OrderRouter``, ``PnLTracker``, ``MarketDataService`` and
  Telegram alerts, launched via ``python -m backend.live_engine``.

Only the lightweight generic classes are re-exported here. The institutional
modules (``engine_state``, ``scheduler``, ``order_router`` …) are imported
on-demand by their consumers so that importing this package never opens a Redis
connection or other resources.
"""

from .crash_recovery import (
    install_signal_handlers,
    load_checkpoint,
    save_checkpoint,
)
from .paper_engine import LiveEngine
from .risk_gates import RiskGates
from .signal_aggregator import SignalAggregator, StrategySignal
from .stream_runner import StreamStrategyRunner

__all__ = [
    "LiveEngine",
    "RiskGates",
    "SignalAggregator",
    "StrategySignal",
    "StreamStrategyRunner",
    "save_checkpoint",
    "load_checkpoint",
    "install_signal_handlers",
]
