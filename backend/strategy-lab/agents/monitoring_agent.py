# agents/monitoring_agent.py
"""
MonitoringAgent
---------------
Lightweight, stdlib-only observability for trading sims/backtests/live.

What it does
- Tracks rolling metrics: PnL (realized/unrealized), exposure, leverage, cash.
- Measures order latency (submit -> first fill), fill rate, slippage estimate.
- Drawdown monitor (peak-to-trough on equity), anomaly thresholds.
- Rule-based alerting with throttling (print/log/callback).
- JSON log records (one-line, parseable).
- Optional heartbeat thread for periodic invariant checks.

How to use
1) Create with references to your ExecutionAgent-like object (optional).
2) Wire the hooks:
    - on_price(symbol, price)
    - on_order_submitted(order_id, symbol, side, qty, type, limit_price=None, stop_price=None)
    - on_fill(order_id, symbol, side, qty, price, fee)
3) Optionally call `tick()` or start the heartbeat to run checks every interval.
4) Read metrics via `snapshot()`.

No external deps. Ready for unit tests.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple


# ----------------------------- Helpers ---------------------------------------


def now() -> float:
    return time.time()


class RollingEWMA:
    """Exponentially weighted moving average/variance for online stats."""
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.mean = 0.0
        self.var = 0.0
        self._init = False

    def update(self, x: float) -> None:
        if not self._init:
            self.mean = x
            self.var = 0.0
            self._init = True
            return
        a = self.alpha
        prev_mean = self.mean
        self.mean = a * x + (1 - a) * self.mean
        # EW variance (Welford-like, approximate)
        self.var = (1 - a) * (self.var + a * (x - prev_mean) ** 2)

    @property
    def std(self) -> float:
        return self.var ** 0.5


class RollingWindow:
    """Fixed-size rolling buffer with quick sum/avg/min/max."""
    def __init__(self, size: int):
        self.size = max(1, size)
        self.buf: Deque[float] = deque(maxlen=self.size)

    def push(self, x: float) -> None:
        self.buf.append(float(x))

    def sum(self) -> float:
        return sum(self.buf)

    def avg(self) -> float:
        return (sum(self.buf) / len(self.buf)) if self.buf else 0.0

    def min(self) -> float:
        return min(self.buf) if self.buf else 0.0

    def max(self) -> float:
        return max(self.buf) if self.buf else 0.0

    def last(self) -> float:
        return self.buf[-1] if self.buf else 0.0

    def __len__(self) -> int:
        return len(self.buf)


# ----------------------------- Alerting --------------------------------------


@dataclass
class AlertRule:
    name: str
    # fn returns (triggered: bool, details: dict)
    fn: Callable[["MonitoringAgent"], Tuple[bool, Dict]]
    min_interval_sec: float = 60.0
    last_sent_ts: float = field(default=0.0)


class AlertSink:
    """
    Where alerts go. Default prints a JSON line.
    You can pass a custom function: fn(level, title, payload_dict).
    """
    def __init__(self, fn: Optional[Callable[[str, str, Dict], None]] = None):
        self._fn = fn

    def emit(self, level: str, title: str, payload: Dict) -> None:
        if self._fn:
            self._fn(level, title, payload)
        else:
            record = {"t": now(), "level": level, "title": title, **payload}
            print(json.dumps(record, separators=(",", ":")))


# ----------------------------- Monitoring Agent ------------------------------


class MonitoringAgent:
    """
    Passive observer and rule runner. Keep this file stdlib-only.
    Wire your trading loop/sim to call the hooks below.
    """

    def __init__(
        self,
        # Optional handle to your execution agent (needs small API: equity(), gross_exposure(), leverage(), cash, total_realized_pnl(), gross_unrealized_pnl())
        execution_agent: Optional[object] = None,
        alert_sink: Optional[AlertSink] = None,
        heartbeat_sec: Optional[float] = None,  # if set, starts a background thread on start()
        max_orders_tracked: int = 10_000,
    ):
        self.x = execution_agent
        self.alerts = alert_sink or AlertSink()
        self.max_orders_tracked = max_orders_tracked

        # State
        self.last_price: Dict[str, float] = {}
        self.order_submitted_ts: Dict[str, float] = {}            # order_id -> ts
        self.order_first_fill_ts: Dict[str, float] = {}           # order_id -> ts
        self.order_filled_qty: Dict[str, float] = defaultdict(float)
        self.order_total_qty: Dict[str, float] = {}               # qty at submit time
        self.order_slippage_bps: Dict[str, float] = {}            # simple est

        # Rolling metrics
        self.equity_rw = RollingEWMA(alpha=0.1)
        self.leverage_rw = RollingEWMA(alpha=0.2)
        self.exposure_rw = RollingEWMA(alpha=0.2)
        self.pnl_rw = RollingEWMA(alpha=0.1)                      # realized + unreal
        self.drawdown_peak: float = 0.0
        self.drawdown_cur: float = 0.0
        self.drawdown_max: float = 0.0

        self.fill_latency_ms_rw = RollingEWMA(alpha=0.2)
        self.fill_rate_rw = RollingEWMA(alpha=0.2)
        self.slippage_bps_rw = RollingEWMA(alpha=0.2)

        # Activity windows (last N ticks)
        self.unreal_pnl_win = RollingWindow(300)   # ~ last 5min @1s tick
        self.real_pnl_win = RollingWindow(300)

        # Rules
        self.rules: List[AlertRule] = []
        self._install_default_rules()

        # Heartbeat
        self._hb_sec = heartbeat_sec
        self._hb_thread: Optional[threading.Thread] = None
        self._hb_stop = threading.Event()

    # -------------------- Public hooks (wire these up) ------------------------

    def on_price(self, symbol: str, price: float) -> None:
        self.last_price[symbol] = float(price)

    def on_order_submitted(self, order_id: str, symbol: str, side: str, qty: float, otype: str, limit_price: Optional[float] = None, stop_price: Optional[float] = None) -> None:
        ts = now()
        if len(self.order_submitted_ts) >= self.max_orders_tracked:
            # Drop the oldest
            oldest = sorted(self.order_submitted_ts.items(), key=lambda kv: kv[1])[0][0]
            self._forget_order(oldest)
        self.order_submitted_ts[order_id] = ts
        self.order_total_qty[order_id] = float(qty)

    def on_fill(self, order_id: str, symbol: str, side: str, qty: float, price: float, fee: float, ref_price: Optional[float] = None) -> None:
        ts = now()
        if order_id not in self.order_first_fill_ts and order_id in self.order_submitted_ts:
            self.order_first_fill_ts[order_id] = ts
            # latency ms
            latency_ms = max(0.0, (ts - self.order_submitted_ts[order_id]) * 1000.0)
            self.fill_latency_ms_rw.update(latency_ms)

        # Fill rate
        self.order_filled_qty[order_id] += float(qty)
        total = max(1e-12, self.order_total_qty.get(order_id, float(qty)))
        self.fill_rate_rw.update(self.order_filled_qty[order_id] / total)

        # Slippage bps (simple: compare fill to ref_price if provided)
        if ref_price is not None and ref_price > 0:
            if side.upper() == "BUY":
                bps = ((price - ref_price) / ref_price) * 1e4
            else:
                bps = ((ref_price - price) / ref_price) * 1e4
            self.order_slippage_bps[order_id] = bps
            self.slippage_bps_rw.update(bps)

        # If fully filled, we can forget heavy state
        if self.order_filled_qty[order_id] >= total - 1e-9:
            self._forget_order(order_id)

    def tick(self) -> None:
        """Call this periodically (e.g., once per second) to refresh metrics and run rules."""
        self._update_from_execution_agent()
        self._run_rules()

    # ------------------------- Metrics & Snapshots ----------------------------

    def snapshot(self) -> Dict:
        """Return a compact, JSON-serializable metrics view."""
        equity = self._safe_equity()
        gross_expo = self._safe_gross_exposure()
        leverage = self._safe_leverage()
        realized = self._safe_realized()
        unreal = self._safe_unrealized()

        snap = {
            "t": now(),
            "equity": equity,
            "cash": getattr(self.x, "cash", None) if self.x else None,
            "gross_exposure": gross_expo,
            "leverage": leverage,
            "pnl_realized": realized,
            "pnl_unrealized": unreal,
            "pnl_total": (realized + unreal) if (realized is not None and unreal is not None) else None,
            "drawdown_cur": self.drawdown_cur,
            "drawdown_max": self.drawdown_max,
            "fill_latency_ms_ewma": self.fill_latency_ms_rw.mean,
            "fill_rate_ewma": self.fill_rate_rw.mean,
            "slippage_bps_ewma": self.slippage_bps_rw.mean,
        }
        return snap

    # ------------------------- Rules & Heartbeat ------------------------------

    def add_rule(self, rule: AlertRule) -> None:
        self.rules.append(rule)

    def _install_default_rules(self) -> None:
        # 1) Excess leverage
        def r_leverage(agent: "MonitoringAgent"):
            lev = agent._safe_leverage() or 0.0
            if lev > 5.0:
                return True, {"leverage": lev}
            return False, {}
        self.add_rule(AlertRule("excess_leverage", r_leverage, min_interval_sec=30))

        # 2) Fast drawdown
        def r_drawdown(agent: "MonitoringAgent"):
            if agent.drawdown_cur <= -0.05:  # -5%
                return True, {"drawdown_cur": agent.drawdown_cur, "drawdown_max": agent.drawdown_max}
            return False, {}
        self.add_rule(AlertRule("sharp_drawdown", r_drawdown, min_interval_sec=60))

        # 3) Poor fill rate
        def r_fillrate(agent: "MonitoringAgent"):
            fr = agent.fill_rate_rw.mean
            if len(agent.order_submitted_ts) > 0 and fr < 0.2:
                return True, {"fill_rate_ewma": fr}
            return False, {}
        self.add_rule(AlertRule("low_fill_rate", r_fillrate, min_interval_sec=45))

        # 4) High slippage
        def r_slippage(agent: "MonitoringAgent"):
            sbps = agent.slippage_bps_rw.mean
            if abs(sbps) > 5.0:  # > 5 bps
                return True, {"slippage_bps_ewma": sbps}
            return False, {}
        self.add_rule(AlertRule("high_slippage", r_slippage, min_interval_sec=45))

    def start(self) -> None:
        if self._hb_sec and not self._hb_thread:
            self._hb_stop.clear()
            self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._hb_thread.start()

    def stop(self) -> None:
        if self._hb_thread:
            self._hb_stop.set()
            self._hb_thread.join(timeout=2.0)
            self._hb_thread = None

    # ----------------------------- Internals ----------------------------------

    def _update_from_execution_agent(self) -> None:
        if not self.x:
            return
        equity = self._safe_equity()
        if equity is not None:
            self.equity_rw.update(equity)
            # drawdown tracking on equity
            if equity > self.drawdown_peak:
                self.drawdown_peak = equity
            self.drawdown_cur = (equity - self.drawdown_peak) / self.drawdown_peak if self.drawdown_peak > 0 else 0.0
            self.drawdown_max = min(self.drawdown_max, self.drawdown_cur)

        lev = self._safe_leverage()
        if lev is not None:
            self.leverage_rw.update(lev)

        gross = self._safe_gross_exposure()
        if gross is not None:
            self.exposure_rw.update(gross)

        realized = self._safe_realized() or 0.0
        unreal = self._safe_unrealized() or 0.0
        self.pnl_rw.update(realized + unreal)
        self.real_pnl_win.push(realized)
        self.unreal_pnl_win.push(unreal)

    def _run_rules(self) -> None:
        ts = now()
        for rule in self.rules:
            triggered, details = rule.fn(self)
            if not triggered:
                continue
            if ts - rule.last_sent_ts < rule.min_interval_sec:
                continue  # throttle
            rule.last_sent_ts = ts
            self.alerts.emit("warn", rule.name, {"t": ts, **self.snapshot(), **details})

    def _heartbeat_loop(self) -> None:
        interval = max(0.1, float(self._hb_sec)) # type: ignore
        while not self._hb_stop.is_set():
            try:
                self.tick()
            except Exception as e:
                # Never crash the process from monitoring
                self.alerts.emit("error", "monitor_heartbeat_exception", {"err": str(e)})
            self._hb_stop.wait(interval)

    def _forget_order(self, order_id: str) -> None:
        self.order_submitted_ts.pop(order_id, None)
        self.order_first_fill_ts.pop(order_id, None)
        self.order_filled_qty.pop(order_id, None)
        self.order_total_qty.pop(order_id, None)
        self.order_slippage_bps.pop(order_id, None)

    # Safe getters from execution agent (so tests can pass a stub)
    def _safe_equity(self) -> Optional[float]:
        try:
            if self.x and hasattr(self.x, "equity"):
                val = self.x.equity() if callable(self.x.equity) else getattr(self.x, "equity", None) # type: ignore
                return float(val) if val is not None else None # type: ignore
        except Exception:
            return None
        return None

    def _safe_gross_exposure(self) -> Optional[float]:
        try:
            if self.x and hasattr(self.x, "gross_exposure"):
                val = self.x.gross_exposure() if callable(self.x.gross_exposure) else getattr(self.x, "gross_exposure", None) # type: ignore
                return float(val) if val is not None else None # type: ignore
        except Exception:
            return None
        return None

    def _safe_leverage(self) -> Optional[float]:
        try:
            if self.x and hasattr(self.x, "leverage"):
                val = self.x.leverage() if callable(self.x.leverage) else getattr(self.x, "leverage", None) # type: ignore
                return float(val) if val is not None else None # type: ignore
        except Exception:
            return None
        return None

    def _safe_realized(self) -> Optional[float]:
        try:
            if self.x and hasattr(self.x, "total_realized_pnl"):
                val = self.x.total_realized_pnl() if callable(self.x.total_realized_pnl) else getattr(self.x, "total_realized_pnl", None) # type: ignore
                return float(val) if val is not None else None # type: ignore
        except Exception:
            return None
        return None

    def _safe_unrealized(self) -> Optional[float]:
        try:
            if self.x and hasattr(self.x, "gross_unrealized_pnl"):
                val = self.x.gross_unrealized_pnl() if callable(self.x.gross_unrealized_pnl) else getattr(self.x, "gross_unrealized_pnl", None) # type: ignore
                return float(val) if val is not None else None # type: ignore
        except Exception:
            return None
        return None


# ----------------------------- Quick demo ------------------------------------
if __name__ == "__main__":
    # Minimal stub to prove it runs
    class StubExec:
        def __init__(self):
            self._equity = 1_000_000.0
            self.cash = 1_000_000.0
            self._gross = 0.0
            self._lev = 0.0
            self._real = 0.0
            self._unreal = 0.0

        def equity(self): return self._equity
        def gross_exposure(self): return self._gross
        def leverage(self): return self._lev
        def total_realized_pnl(self): return self._real
        def gross_unrealized_pnl(self): return self._unreal

    x = StubExec()
    mon = MonitoringAgent(execution_agent=x, heartbeat_sec=None)

    # Simulate a few ticks
    for i in range(5):
        x._gross = 200_000 + i * 10_000
        x._lev = x._gross / x._equity
        x._unreal += 500
        mon.on_price("AAPL", 200 + i)
        mon.tick()
        print(json.dumps(mon.snapshot(), separators=(",", ":"), default=float))
        time.sleep(0.1)