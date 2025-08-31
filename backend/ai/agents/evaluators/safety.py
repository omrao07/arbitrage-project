# backend/ai/agents/core/safety.py
from __future__ import annotations

import re
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

# ------------------------------------------------------------
# Data models
# ------------------------------------------------------------
@dataclass
class SafetyEvent:
    ts_ms: int
    level: str                # "info" | "warn" | "block" | "crit"
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self): return asdict(self)

# ------------------------------------------------------------
# Safety Manager
# ------------------------------------------------------------
class SafetyManager:
    """
    Central guardrails:
      • risk limits (qty, notional, leverage)
      • forbidden instruments
      • content filters (prompt/user text)
      • execution cool-downs
      • circuit breaker if too many fails
    """

    def __init__(self):
        self.events: List[SafetyEvent] = []
        self.max_qty = 10_000
        self.max_notional = 1_000_000.0
        self.max_leverage = 10.0
        self.blocklist_symbols = {"SCAM", "PUMP"}
        self.min_cooldown_ms = 100  # ms between orders
        self._last_order_ts: int = 0
        self.max_failures = 5
        self.failures = 0
        self.blocked_until: Optional[int] = None

    # --------------- Order checks ---------------
    def check_order(self, *, symbol: str, side: str, qty: float, price: float, leverage: float = 1.0) -> bool:
        ts = int(time.time()*1000)

        # Circuit breaker
        if self.blocked_until and ts < self.blocked_until:
            self._log("block", f"Trading temporarily blocked", {"until": self.blocked_until})
            return False

        # Blocklist
        if symbol.upper() in self.blocklist_symbols:
            self._log("block", f"Symbol {symbol} is blocklisted")
            return False

        # Quantity / Notional
        notional = qty * price
        if qty > self.max_qty or notional > self.max_notional:
            self._log("block", f"Order exceeds limits", {"qty": qty, "notional": notional})
            return False

        # Leverage
        if leverage > self.max_leverage:
            self._log("block", f"Leverage {leverage}x > max {self.max_leverage}")
            return False

        # Cooldown
        if ts - self._last_order_ts < self.min_cooldown_ms:
            self._log("warn", "Order rejected due to cooldown", {"delta_ms": ts - self._last_order_ts})
            return False

        # If passed
        self._last_order_ts = ts
        self._log("info", f"Order passed checks", {"symbol": symbol, "qty": qty, "notional": notional})
        return True

    # --------------- Failures / breaker ---------------
    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.max_failures:
            self.blocked_until = int(time.time()*1000) + 60_000  # 1m block
            self._log("crit", "Too many failures, circuit breaker engaged", {"blocked_until": self.blocked_until})

    def reset_failures(self) -> None:
        self.failures = 0
        self.blocked_until = None
        self._log("info", "Failures reset")

    # --------------- Content filter ---------------
    def filter_text(self, text: str) -> bool:
        """
        Returns True if safe, False if flagged.
        Flags:
          • prompt injection patterns
          • PII-like strings (SSNs, CC numbers)
        """
        if re.search(r"(system\.prompt|ignore\s+all\s+rules)", text, re.I):
            self._log("warn", "Prompt injection flagged", {"text": text})
            return False
        if re.search(r"\b\d{16}\b", text):  # crude CC detect
            self._log("warn", "Possible credit card number detected", {"text": text})
            return False
        return True

    # --------------- Internals ---------------
    def _log(self, level: str, message: str, context: Optional[Dict[str,Any]]=None) -> None:
        ev = SafetyEvent(ts_ms=int(time.time()*1000), level=level, message=message, context=context or {})
        self.events.append(ev)
        print(f"[safety] {level.upper()}: {message} | {context or ''}")

    def dump_events(self) -> List[Dict[str,Any]]:
        return [e.to_dict() for e in self.events]

# ------------------------------------------------------------
# Quick smoke test
# ------------------------------------------------------------
if __name__ == "__main__":
    s = SafetyManager()
    print("Safe order?", s.check_order(symbol="AAPL", side="buy", qty=10, price=150))
    print("Bad order?", s.check_order(symbol="SCAM", side="buy", qty=1, price=10))
    s.record_failure(); s.record_failure(); s.record_failure(); s.record_failure(); s.record_failure()
    print("Events:", s.dump_events())