"""
Strategy adapter — makes heterogeneous strategy files executable by one runner.

THE PROBLEM
-----------
An AST survey of `backend/strategies` (373 files) found four incompatible shapes:

    82  class with on_bar/on_tick   <- already satisfies the engine contract
   162  module-level run()          <- standalone CLI research script
   107  class without on_bar        <- some other interface entirely
    14  module-level generate_signals()
     7  module-level main() only

Rewriting 300 files is not the job. One adapter is. A strategy that cannot be
expressed through this contract without heroics is usually one that was written
against a lookahead-biased research frame — the friction is diagnostic, so
adapter failures are recorded, not smoothed over.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does not invent data, and it does not fabricate returns for a strategy it
cannot run. `AdapterResult.ok` is False with a stated reason. A strategy that
cannot be executed has not "passed with no signal" — it has not been tested.
"""

from __future__ import annotations

import ast
import importlib
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


class StrategyShape(str, Enum):
    ON_BAR_CLASS = "class_on_bar"
    PLAIN_CLASS = "class_plain"
    RUN_FN = "fn_run"
    GENERATE_SIGNALS_FN = "fn_generate_signals"
    MAIN_FN = "fn_main"
    UNKNOWN = "unknown"


@dataclass
class AdapterResult:
    """Outcome of trying to make one strategy runnable."""

    strategy_id: str
    shape: StrategyShape
    ok: bool
    reason: str = ""
    entrypoint: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "shape": self.shape.value,
            "ok": self.ok,
            "reason": self.reason,
            "entrypoint": self.entrypoint,
        }


def detect_shape(path: Path) -> tuple[StrategyShape, Optional[str]]:
    """
    Classify a strategy file by static analysis.

    Static, not import-based, because importing 373 research scripts executes
    arbitrary module-level code — including file reads and network calls.
    Classification must be safe to run over the whole tree.
    """
    try:
        tree = ast.parse(path.read_text(errors="ignore"))
    except (SyntaxError, OSError) as exc:
        log.debug("cannot parse %s: %s", path, exc)
        return StrategyShape.UNKNOWN, None

    functions = {
        n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]

    for cls in classes:
        for body_item in cls.body:
            if isinstance(body_item, (ast.FunctionDef, ast.AsyncFunctionDef)) and body_item.name in (
                "on_bar",
                "on_tick",
            ):
                return StrategyShape.ON_BAR_CLASS, cls.name

    if "generate_signals" in functions:
        return StrategyShape.GENERATE_SIGNALS_FN, "generate_signals"
    if "run" in functions:
        return StrategyShape.RUN_FN, "run"
    if classes:
        return StrategyShape.PLAIN_CLASS, classes[0].name
    if "main" in functions:
        return StrategyShape.MAIN_FN, "main"
    return StrategyShape.UNKNOWN, None


def try_import(module_path: str) -> tuple[Optional[Any], str]:
    """
    Import a strategy module, converting any failure into a reason string.

    Broad except is deliberate: these are third-party-ish research scripts that
    fail at import time in every imaginable way (missing optional deps, reading
    a CSV at module scope, calling an API). A crash here must degrade to a
    recorded diagnostic, never take down a batch run over 371 files.
    """
    try:
        return importlib.import_module(module_path), ""
    except Exception as exc:  # noqa: BLE001 - see docstring
        return None, f"{type(exc).__name__}: {exc}"


class StrategyAdapter:
    """
    Wraps one strategy into a uniform `generate_returns(prices) -> np.ndarray`.

    The contract is intentionally narrow: given a price series, produce a
    per-period return series. That is the only thing the gauntlet needs, and
    narrowing to it is what lets 371 unlike files be compared on equal terms.
    """

    def __init__(self, strategy_id: str, module_path: str, file_path: Path):
        self.strategy_id = strategy_id
        self.module_path = module_path
        self.file_path = file_path
        self.shape, self.entrypoint = detect_shape(file_path)
        self._module: Optional[Any] = None
        self._callable: Optional[Callable] = None
        self._harness: Optional[Any] = None

    # ── Loading ─────────────────────────────────────────────────────────────

    def load(self) -> AdapterResult:
        """Import and resolve the entrypoint. Does not execute the strategy."""
        if self.shape is StrategyShape.UNKNOWN:
            return AdapterResult(
                self.strategy_id, self.shape, False, "no recognisable entrypoint"
            )

        module, err = try_import(self.module_path)
        if module is None:
            return AdapterResult(self.strategy_id, self.shape, False, f"import failed: {err}")
        self._module = module

        if self.entrypoint is None:
            return AdapterResult(self.strategy_id, self.shape, False, "no entrypoint name")

        target = getattr(module, self.entrypoint, None)
        if target is None:
            return AdapterResult(
                self.strategy_id,
                self.shape,
                False,
                f"entrypoint {self.entrypoint!r} not found after import",
            )

        self._callable = target
        return AdapterResult(
            self.strategy_id, self.shape, True, "loaded", entrypoint=self.entrypoint
        )

    # ── Execution ───────────────────────────────────────────────────────────

    def generate_returns(self, prices: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Run the strategy over `prices` and return its per-period returns.

        Raises rather than returning an empty or zero-filled array: a strategy
        that could not run has not produced a flat P&L, it has produced nothing,
        and the difference decides whether it enters the gauntlet.
        """
        if self._callable is None:
            raise RuntimeError(f"{self.strategy_id}: load() must succeed before running")

        prices = np.asarray(prices, dtype=float)
        if prices.ndim != 1 or prices.size < 2:
            raise ValueError(f"{self.strategy_id}: need a 1-D price series of length >= 2")
        if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
            raise ValueError(f"{self.strategy_id}: price series has non-finite/non-positive values")

        if self.shape is StrategyShape.ON_BAR_CLASS:
            return self._run_on_bar(prices, **kwargs)
        if self.shape is StrategyShape.RUN_FN:
            from backend.validation.script_adapter import run_script_strategy

            return run_script_strategy(
                self._module, self.file_path, prices, symbol=kwargs.get("symbol", "SYNTH")
            )
        raise NotImplementedError(
            f"{self.strategy_id}: shape {self.shape.value} has no adapter yet "
            "(its entrypoint is neither on_bar/on_tick nor run(cfg))."
        )

    def _run_on_bar(self, prices: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Drive an on_bar/on_tick class bar-by-bar and read its positions.

        These strategies signal by calling `self.emit_signal(score)`, not by
        returning a value — `Strategy.emit_signal` writes to Redis unless a
        `_collector` is attached, which is precisely the backtest hook the base
        class provides. We attach `BacktestCollector` so nothing touches Redis
        and every emission is captured in-memory. A handful of strategies do
        return a value instead, so both channels are read, with an emitted
        signal taking precedence.
        """
        from backend.backtester.backtest_engine import BacktestCollector
        from backend.validation.replay import ReplayHarness, install

        symbol = kwargs.get("symbol", "SYNTH")
        harness = ReplayHarness(symbol=symbol)
        # Strategies fetch their inputs from a module-level Redis handle rather
        # than receiving them; feed that handle instead of a live server.
        install(self._module, harness)
        self._harness = harness  # exposed for missing-key diagnostics

        cls = self._callable
        try:
            instance = cls(**kwargs) if kwargs else cls()
        except Exception as exc:  # noqa: BLE001 - constructor signatures vary wildly
            raise RuntimeError(f"{self.strategy_id}: constructor failed: {exc}") from exc

        handler = _resolve_overridden_handler(instance)
        if handler is None:
            raise RuntimeError(
                f"{self.strategy_id}: neither on_bar nor on_tick is overridden "
                "(only the base-class no-op is present)"
            )

        collector = BacktestCollector(self.strategy_id)
        instance._collector = collector  # the documented backtest intercept

        # Give the strategy a clock that moves with the bars, not the CPU.
        clock = _SimulatedClock(step_seconds=float(kwargs.get("bar_seconds", 86400.0)))
        real_time_fn = getattr(getattr(self._module, "time", None), "time", None)
        patched = hasattr(self._module, "time") and real_time_fn is not None
        if patched:
            self._module.time.time = clock.time

        positions: List[float] = []
        emitted = 0
        try:
            for i, px in enumerate(prices):
                clock.advance()
                harness.advance(float(px), i)
                bar = {
                    "symbol": kwargs.get("symbol", "SYNTH"),
                    "close": float(px),
                    "open": float(px),
                    "high": float(px),
                    "low": float(px),
                    "price": float(px),
                    "ltp": float(px),
                    "volume": 0.0,
                    "ts": i,
                }
                before = collector.signal
                try:
                    returned = handler(bar)
                except Exception as exc:  # noqa: BLE001 - strategy bodies are arbitrary
                    raise RuntimeError(
                        f"{self.strategy_id}: handler failed at bar {i}: {exc}"
                    ) from exc

                if collector.signal != before:
                    emitted += 1
                    positions.append(float(np.clip(collector.signal, -1.0, 1.0)))
                else:
                    positions.append(
                        _coerce_position(returned)
                        if returned is not None
                        else float(np.clip(collector.signal, -1.0, 1.0))
                    )
        finally:
            if patched:
                self._module.time.time = real_time_fn

        if emitted == 0 and not any(positions):
            # Never dress this up as a flat P&L: the strategy did not trade
            # because it never received the inputs it reads, not because it
            # chose to. Report what it asked for and could not get.
            wanted = ", ".join(sorted(harness.redis.access_log)[:6]) or "nothing"
            raise RuntimeError(
                f"{self.strategy_id}: produced no signal over {len(prices)} bars. "
                f"It read: [{wanted}] — its inputs come from external state a bare "
                "price series does not supply. Not a result; not tested."
            )

        pos = np.asarray(positions, dtype=float)
        px_returns = np.diff(prices) / prices[:-1]
        # Position from bar i is held into bar i+1 — never same-bar, which
        # would be lookahead.
        return pos[:-1] * px_returns


def _resolve_overridden_handler(instance: Any) -> Optional[Callable]:
    """
    Pick the handler the CONCRETE strategy actually implements.

    `Strategy` (the base ABC) defines a no-op `on_bar`, so a naive
    `getattr(obj, "on_bar") or getattr(obj, "on_tick")` silently binds the
    base-class stub and every strategy looks like it emits nothing. Walk the MRO
    below the base and take whichever of the two the subclass overrode.
    """
    cls = type(instance)
    for name in ("on_bar", "on_tick"):
        defining = next(
            (k for k in cls.__mro__ if name in vars(k)),
            None,
        )
        if defining is not None and defining.__name__ not in ("Strategy", "ABC", "object"):
            return getattr(instance, name)
    # Fall back to anything callable, so a strategy inheriting differently is
    # still attempted rather than silently skipped.
    return getattr(instance, "on_tick", None) or getattr(instance, "on_bar", None)


class _SimulatedClock:
    """
    Advances wall-clock time in step with bars.

    Strategies throttle themselves with `if time.time() - self._last <
    RECHECK_SECS: return` (RECHECK_SECS is typically 2-60s). That is correct
    live, where bars arrive a minute apart, and fatal in a backtest, where 800
    bars execute inside a millisecond and the strategy evaluates exactly once.
    Patching `time.time` in the strategy's own module makes simulated time move
    with the data instead of the CPU. It changes only pacing, never values.
    """

    def __init__(self, step_seconds: float = 60.0):
        self.step = step_seconds
        self.now = 1_700_000_000.0

    def advance(self) -> None:
        self.now += self.step

    def time(self) -> float:
        return self.now


def _coerce_position(signal: Any) -> float:
    """Map a strategy's return value onto a position in [-1, 1]. None -> flat."""
    if signal is None:
        return 0.0
    if isinstance(signal, (int, float, np.floating)):
        return float(np.clip(float(signal), -1.0, 1.0))
    if isinstance(signal, dict):
        for key in ("position", "signal", "weight", "qty", "side"):
            if key in signal:
                return _coerce_position(signal[key])
        return 0.0
    if isinstance(signal, str):
        return {"buy": 1.0, "long": 1.0, "sell": -1.0, "short": -1.0}.get(signal.lower(), 0.0)
    return 0.0


def audit_tree(root: Path | str = "backend/strategies") -> List[AdapterResult]:
    """
    Static shape census over the whole tree — safe, no imports, no execution.

    This is the honest denominator for "how many strategies do I have": a file
    with no recognisable entrypoint is not a strategy the pipeline can test.
    """
    root_path = Path(root)
    results: List[AdapterResult] = []
    skip = {"__pycache__", "registry", "utils", "configs", "examples"}
    for py in sorted(root_path.rglob("*.py")):
        if py.name.startswith("__") or any(p in skip for p in py.parts):
            continue
        shape, entry = detect_shape(py)
        results.append(
            AdapterResult(
                strategy_id=f"{py.parent.name}.{py.stem}",
                shape=shape,
                ok=shape is not StrategyShape.UNKNOWN,
                reason="" if shape is not StrategyShape.UNKNOWN else "no recognisable entrypoint",
                entrypoint=entry,
            )
        )
    return results
