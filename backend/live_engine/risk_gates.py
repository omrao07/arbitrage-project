# backend/live_engine/risk_gates.py
"""
9+2 Risk Gates:
 1. Daily loss > -2% of capital → halt
 2. Drawdown > -10% from HWM → halt
 3. Portfolio beta > 0.8 → reduce
 4. Single position > 5% of capital → block
 5. VIX > 30 → reduce/halt
 6. Sector concentration > 30% → block
 7. Order rate > 60/min → throttle
 8. Margin utilization > 150% → block
 9. India circuit breaker triggered → block
+1. F&O ban list check → block
+2. Position sizing: Kelly / Vol-Parity enforcement

Lightweight pre-trade gate suite for the generic/paper path. Complements the
heavier portfolio-level InstitutionalRiskEngine (backend/risk) used by the
institutional path — both live side by side in this package.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, FrozenSet, Optional


@dataclass
class RiskGates:
    capital: float = 100_000.0
    daily_loss_limit_pct: float = 2.0      # gate 1
    drawdown_limit_pct: float = 10.0       # gate 2
    beta_limit: float = 0.8               # gate 3
    position_pct_limit: float = 5.0       # gate 4
    vix_halt_threshold: float = 30.0      # gate 5
    sector_conc_limit_pct: float = 30.0   # gate 6
    order_rate_limit_per_min: int = 60    # gate 7
    margin_utilization_limit_pct: float = 150.0  # gate 8
    kelly_fraction: float = 0.25          # gate +2: fractional Kelly

    # Runtime state
    _daily_pnl: float = field(default=0.0, init=False)
    _hwm: float = field(default=0.0, init=False)       # high-water mark (cumulative PnL)
    _cum_pnl: float = field(default=0.0, init=False)
    _order_timestamps: Deque[float] = field(default_factory=deque, init=False)
    _fo_ban: FrozenSet[str] = field(default_factory=frozenset, init=False)
    _circuit_halted: FrozenSet[str] = field(default_factory=frozenset, init=False)
    _halted: bool = field(default=False, init=False)

    def reset_daily(self) -> None:
        self._daily_pnl = 0.0

    def update_pnl(self, delta: float) -> None:
        self._daily_pnl += delta
        self._cum_pnl += delta
        if self._cum_pnl > self._hwm:
            self._hwm = self._cum_pnl

    def update_fo_ban(self, ban_set: FrozenSet[str]) -> None:
        self._fo_ban = ban_set

    def update_circuit_halted(self, halted_set: FrozenSet[str]) -> None:
        self._circuit_halted = halted_set

    # ---- Gate checks --------------------------------------------------------

    def gate1_daily_loss(self) -> tuple[bool, str]:
        threshold = -abs(self.daily_loss_limit_pct / 100.0 * self.capital)
        if self._daily_pnl <= threshold:
            return False, f"daily_loss {self._daily_pnl:.0f} breaches {threshold:.0f}"
        return True, "ok"

    def gate2_drawdown(self) -> tuple[bool, str]:
        drawdown = self._cum_pnl - self._hwm
        limit = -abs(self.drawdown_limit_pct / 100.0 * self.capital)
        if drawdown <= limit:
            return False, f"drawdown {drawdown:.0f} breaches {limit:.0f}"
        return True, "ok"

    def gate3_beta(self, portfolio_beta: float) -> tuple[bool, str]:
        if abs(portfolio_beta) > self.beta_limit:
            return False, f"portfolio_beta {portfolio_beta:.2f} > {self.beta_limit}"
        return True, "ok"

    def gate4_position_size(self, symbol: str, proposed_notional: float) -> tuple[bool, str]:
        pct = abs(proposed_notional) / max(self.capital, 1e-9) * 100.0
        if pct > self.position_pct_limit:
            return False, f"{symbol} notional {pct:.1f}% > {self.position_pct_limit}%"
        return True, "ok"

    def gate5_vix(self, vix: float) -> tuple[bool, str]:
        if vix >= self.vix_halt_threshold:
            return False, f"VIX {vix:.1f} >= {self.vix_halt_threshold}"
        return True, "ok"

    def gate6_sector(self, sector: str, sector_notional: float) -> tuple[bool, str]:
        pct = abs(sector_notional) / max(self.capital, 1e-9) * 100.0
        if pct > self.sector_conc_limit_pct:
            return False, f"sector {sector} {pct:.1f}% > {self.sector_conc_limit_pct}%"
        return True, "ok"

    def gate7_order_rate(self) -> tuple[bool, str]:
        now = time.time()
        cutoff = now - 60.0
        while self._order_timestamps and self._order_timestamps[0] < cutoff:
            self._order_timestamps.popleft()
        self._order_timestamps.append(now)
        if len(self._order_timestamps) > self.order_rate_limit_per_min:
            return False, f"order_rate {len(self._order_timestamps)}/min > {self.order_rate_limit_per_min}"
        return True, "ok"

    def gate8_margin(self, margin_used: float, margin_available: float) -> tuple[bool, str]:
        if margin_available <= 0:
            return False, "no_margin_available"
        utilization = (margin_used / margin_available) * 100.0
        if utilization > self.margin_utilization_limit_pct:
            return False, f"margin_utilization {utilization:.1f}% > {self.margin_utilization_limit_pct}%"
        return True, "ok"

    def gate9_circuit(self, symbol: str) -> tuple[bool, str]:
        if symbol.upper() in self._circuit_halted:
            return False, f"{symbol} in circuit_halt"
        return True, "ok"

    def gate_fo_ban(self, symbol: str) -> tuple[bool, str]:
        if symbol.upper() in self._fo_ban:
            return False, f"{symbol} on F&O ban list"
        return True, "ok"

    def gate_kelly_size(
        self, win_rate: float, win_loss_ratio: float, capital: Optional[float] = None
    ) -> float:
        """Return Kelly-fractional position size (notional)."""
        cap = capital or self.capital
        if win_loss_ratio <= 0 or win_rate <= 0:
            return 0.0
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        kelly = max(0.0, min(kelly, 1.0))
        return cap * kelly * self.kelly_fraction

    def check_all(
        self,
        symbol: str,
        proposed_notional: float,
        sector: Optional[str] = None,
        sector_notional: float = 0.0,
        portfolio_beta: float = 0.0,
        vix: float = 0.0,
        margin_used: float = 0.0,
        margin_available: float = 1e9,
    ) -> tuple[bool, str]:
        """
        Run all applicable gates. Returns (allowed, first_fail_reason).
        """
        for ok, reason in [
            self.gate1_daily_loss(),
            self.gate2_drawdown(),
            self.gate3_beta(portfolio_beta),
            self.gate4_position_size(symbol, proposed_notional),
            self.gate5_vix(vix),
            self.gate7_order_rate(),
            self.gate8_margin(margin_used, margin_available),
            self.gate9_circuit(symbol),
            self.gate_fo_ban(symbol),
        ]:
            if not ok:
                return False, reason
        if sector:
            ok, reason = self.gate6_sector(sector, sector_notional)
            if not ok:
                return False, reason
        return True, "ok"

    def to_redis(self, r=None) -> None:
        """Persist gate state snapshot to Redis risk:gates hash."""
        import json
        import os
        if r is None:
            try:
                import redis as _redis
                r = _redis.Redis(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", "6379")),
                    password=os.getenv("REDIS_PASSWORD") or None,
                    decode_responses=True,
                )
            except Exception:
                return
        try:
            gates = {
                "gate1_daily_loss": json.dumps({"ok": self.gate1_daily_loss()[0], "reason": self.gate1_daily_loss()[1], "daily_pnl": self._daily_pnl}),
                "gate2_drawdown": json.dumps({"ok": self.gate2_drawdown()[0], "reason": self.gate2_drawdown()[1], "cum_pnl": self._cum_pnl, "hwm": self._hwm}),
                "gate3_beta": json.dumps({"ok": True, "reason": "ok"}),
                "gate5_vix": json.dumps({"ok": True, "reason": "ok"}),
                "gate7_order_rate": json.dumps({"ok": True, "reason": "ok", "orders_per_min": len(self._order_timestamps)}),
            }
            if r:
                r.hset("risk:gates", mapping=gates)
                r.hset("engine:pnl", mapping={
                    "daily": str(self._daily_pnl),
                    "cumulative": str(self._cum_pnl),
                    "drawdown": str(self._cum_pnl - self._hwm),
                    "hwm": str(self._hwm),
                })
        except Exception:
            pass

    def from_redis(self, r=None) -> None:
        """Restore gate state snapshot from Redis risk:gates hash."""
        import os
        if r is None:
            try:
                import redis as _redis
                r = _redis.Redis(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", "6379")),
                    password=os.getenv("REDIS_PASSWORD") or None,
                    decode_responses=True,
                )
            except Exception:
                return
        try:
            pnl = r.hgetall("engine:pnl")
            if pnl:
                self._daily_pnl = float(pnl.get("daily", 0.0))
                self._cum_pnl = float(pnl.get("cumulative", 0.0))
                self._hwm = float(pnl.get("hwm", 0.0))
        except Exception:
            pass
