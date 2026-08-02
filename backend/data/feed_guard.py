"""
FeedGuard — the hard-fail gate on every market-data read.

THE RULE
--------
A missing, stale, or malformed feed RAISES. It never returns a fabricated value.
There is no "fallback data" in a trading system — there is real data, or there
is a halt.

This is the data-layer half of the D-2 fix. The other half (risk metrics that
returned 0.0 rather than admitting ignorance) lives in
`backend/risk/institutional_risk_engine.py`. Both exist because a plausible
wrong number is far more dangerous than an exception: an exception stops the
system, a plausible number sizes a position.

STALENESS IS REGIME-DEPENDENT
-----------------------------
2s of stale data on a dead midday tape is not the same risk as 2s during an
expiry-Tuesday gamma spike. `max_staleness_ms` is therefore per-instrument and
per-regime, not one global constant. Defaults here are deliberately tight;
widen them per instrument with evidence, never globally out of convenience.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import numpy as np

log = logging.getLogger(__name__)


class FeedError(Exception):
    """Base for every data-integrity failure. Always fatal to the current tick."""


class MissingFeedError(FeedError):
    """No tick at all, or a tick with no usable price."""


class StaleDataError(FeedError):
    """A tick arrived but is older than this instrument's tolerance."""


class MalformedTickError(FeedError):
    """A tick is structurally wrong (missing keys, non-numeric, negative price)."""


# Default tolerance; override per instrument via FeedGuard(overrides=...).
DEFAULT_MAX_STALENESS_MS = int(os.getenv("FEED_MAX_STALENESS_MS", "2000"))

# Regime multipliers on the base tolerance. Expiry/high-vol regimes tighten it,
# because that is exactly when a stale price does the most damage.
REGIME_MULTIPLIERS: Dict[str, float] = {
    "normal": 1.0,
    "expiry": 0.5,
    "high_vol": 0.5,
    "illiquid": 2.0,
}


@dataclass
class FeedGuard:
    """
    Wraps every market-data read. Real data or exception. Never fabrication.

    >>> guard = FeedGuard()
    >>> guard.validate_tick({"ltp": 100.0, "exchange_ts_ms": time.time()*1000}, "NIFTY")
    """

    max_staleness_ms: int = DEFAULT_MAX_STALENESS_MS
    overrides: Dict[str, int] = field(default_factory=dict)
    regime: str = "normal"
    # Counters for observability — you want to see degradation before it halts you.
    stats: Dict[str, int] = field(
        default_factory=lambda: {"ok": 0, "missing": 0, "stale": 0, "malformed": 0}
    )

    def tolerance_for(self, symbol: str) -> float:
        base = self.overrides.get(symbol, self.max_staleness_ms)
        return base * REGIME_MULTIPLIERS.get(self.regime, 1.0)

    def validate_tick(
        self,
        tick: Optional[Mapping[str, Any]],
        symbol: str,
        *,
        now_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Validate one tick. Returns the tick on success; raises on any doubt.

        Required keys: `ltp` (positive finite price) and `exchange_ts_ms`.
        The timestamp must be the EXCHANGE's, not local receipt time — local
        time hides feed-handler lag, which is the lag that matters.
        """
        if tick is None:
            self.stats["missing"] += 1
            raise MissingFeedError(f"No tick for {symbol} — HALT, do not fabricate")

        if "exchange_ts_ms" not in tick:
            self.stats["malformed"] += 1
            raise MalformedTickError(
                f"{symbol} tick has no exchange_ts_ms — cannot establish freshness, HALT"
            )

        try:
            ts_ms = float(tick["exchange_ts_ms"])
        except (TypeError, ValueError) as exc:
            self.stats["malformed"] += 1
            raise MalformedTickError(
                f"{symbol} exchange_ts_ms is not numeric: {tick['exchange_ts_ms']!r} — HALT"
            ) from exc

        if not np.isfinite(ts_ms):
            self.stats["malformed"] += 1
            raise MalformedTickError(f"{symbol} exchange_ts_ms is not finite — HALT")

        now = time.time() * 1000.0 if now_ms is None else float(now_ms)
        age_ms = now - ts_ms
        tol = self.tolerance_for(symbol)

        # A timestamp from the future means clock skew between us and the
        # exchange. Trusting it would mask real staleness, so it is fatal.
        if age_ms < -tol:
            self.stats["malformed"] += 1
            raise MalformedTickError(
                f"{symbol} tick is {-age_ms:.0f}ms in the FUTURE — clock skew, HALT"
            )

        if age_ms > tol:
            self.stats["stale"] += 1
            raise StaleDataError(
                f"{symbol} tick is {age_ms:.0f}ms stale (>{tol:.0f}ms, regime={self.regime}) — HALT"
            )

        px = tick.get("ltp")
        if px is None:
            self.stats["missing"] += 1
            raise MissingFeedError(f"{symbol} tick has no ltp — HALT")
        try:
            px_f = float(px)
        except (TypeError, ValueError) as exc:
            self.stats["malformed"] += 1
            raise MalformedTickError(f"{symbol} price is not numeric: {px!r} — HALT") from exc
        if not np.isfinite(px_f) or px_f <= 0:
            self.stats["malformed"] += 1
            raise MissingFeedError(f"{symbol} price invalid: {px_f} — HALT")

        # Bid/ask, when present, must be coherent. A crossed book is bad data,
        # not an arbitrage.
        bid, ask = tick.get("bid"), tick.get("ask")
        if bid is not None and ask is not None:
            try:
                b, a = float(bid), float(ask)
            except (TypeError, ValueError) as exc:
                self.stats["malformed"] += 1
                raise MalformedTickError(f"{symbol} bid/ask not numeric — HALT") from exc
            if np.isfinite(b) and np.isfinite(a) and b > 0 and a > 0 and b > a:
                self.stats["malformed"] += 1
                raise MalformedTickError(
                    f"{symbol} crossed book: bid {b} > ask {a} — bad data, HALT"
                )

        self.stats["ok"] += 1
        return dict(tick)

    def is_healthy(self, min_ok_ratio: float = 0.95) -> bool:
        """Rolling health for the graceful-degradation ladder."""
        total = sum(self.stats.values())
        return True if total == 0 else (self.stats["ok"] / total) >= min_ok_ratio

    def reset_stats(self) -> None:
        for k in self.stats:
            self.stats[k] = 0


# ── Series cleaning ─────────────────────────────────────────────────────────

# Forward-fill is permitted ONLY for slow-moving reference series. Filling a
# live tradeable price is the synthetic-fabrication bug in a nicer costume.
FFILLABLE_REFERENCE_SERIES = frozenset(
    {"prev_day_oi", "lot_size", "expiry", "strike", "contract_meta", "prev_close"}
)


def clean_series(
    series: "Any",
    name: str,
    *,
    allow_ffill: bool = False,
    ffill_limit: int = 1,
) -> "Any":
    """
    The only acceptable cleaning: explicit, logged, never silent invention.

    inf -> NaN always. Forward-fill only for a named reference series, and only
    when explicitly requested. Price series are never filled — they raise
    upstream so the caller decides to halt.
    """
    import pandas as pd

    s = pd.Series(series).replace([np.inf, -np.inf], np.nan)
    n_missing = int(s.isna().sum())
    if not n_missing:
        return s

    if allow_ffill:
        if name not in FFILLABLE_REFERENCE_SERIES:
            raise MalformedTickError(
                f"refusing to forward-fill {name!r}: only slow reference series "
                f"{sorted(FFILLABLE_REFERENCE_SERIES)} may be filled, never a live price"
            )
        log.warning("clean_series: forward-filling %d missing in reference series %s", n_missing, name)
        return s.ffill(limit=ffill_limit)

    # A missing value you know about is fine; one you invented is fatal.
    log.warning("clean_series: %s has %d missing values (left as NaN)", name, n_missing)
    return s
