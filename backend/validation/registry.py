"""
Strategy registry — the single source of truth for what is allowed to trade.

THE RULE
--------
Nothing trades unless it is in `register.yaml` with `status: live` and a
non-zero `capital_cap_inr`, and it only reaches that state by clearing the
gauntlet in order. `can_trade()` is fail-closed: an unknown id, a malformed
entry, or an unreadable registry all mean "no".

STATUS LADDER (forward-only, one rung at a time)
------------------------------------------------
    backtest -> walkforward -> shadow -> live
                                          |
                          retired <-------+   (from any rung)

You may always demote, and demotion to `retired` is available from anywhere —
that is the autopsy path. You may not skip a rung on the way up, because each
rung exists to catch a failure the previous one structurally cannot see:
walk-forward catches in-sample fitting, shadow catches lookahead bias that
survived it, and live catches the gap between simulated and real fills.

`capital_cap_inr` stays 0 until `live`. Promotion to live requires a recorded
DSR >= 0.95 and PBO < 0.5 on the concatenated out-of-sample track.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from backend.validation.dsr import DSR_GATE
from backend.validation.pbo import PBO_GATE

log = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path(
    os.getenv("STRATEGY_REGISTRY", "backend/config/register.yaml")
)

VALID_STATUSES = ("backtest", "walkforward", "shadow", "live", "retired")

# Forward transitions must be one rung at a time; retirement is always allowed.
_NEXT_RUNG: Dict[str, str] = {
    "backtest": "walkforward",
    "walkforward": "shadow",
    "shadow": "live",
}
_RUNG_ORDER = {s: i for i, s in enumerate(("backtest", "walkforward", "shadow", "live"))}


class RegistryError(ValueError):
    """Raised on a malformed registry or an illegal status transition."""


@dataclass
class StrategyEntry:
    id: str
    module: str
    cls: str
    status: str = "backtest"
    capital_cap_inr: float = 0.0
    dsr: Optional[float] = None
    pbo: Optional[float] = None
    shadow_days: int = 0
    universe: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise RegistryError("registry entry has no id")
        if self.status not in VALID_STATUSES:
            raise RegistryError(
                f"{self.id}: status {self.status!r} not one of {VALID_STATUSES}"
            )
        if self.capital_cap_inr < 0:
            raise RegistryError(f"{self.id}: capital_cap_inr may not be negative")
        if self.status != "live" and self.capital_cap_inr > 0:
            raise RegistryError(
                f"{self.id}: capital_cap_inr must be 0 until status is 'live' "
                f"(status={self.status}, cap={self.capital_cap_inr})"
            )

    # ── Gates ────────────────────────────────────────────────────────────────

    def gates_passed(self) -> bool:
        """Has this strategy cleared the statistical gates on its OOS track?"""
        return (
            self.dsr is not None
            and self.pbo is not None
            and self.dsr >= DSR_GATE
            and self.pbo < PBO_GATE
        )

    def blocking_reasons(self) -> List[str]:
        """Why this strategy may not trade. Empty list == cleared to trade."""
        reasons: List[str] = []
        if self.status != "live":
            reasons.append(f"status is {self.status!r}, not 'live'")
        if self.capital_cap_inr <= 0:
            reasons.append("capital_cap_inr is 0")
        if self.dsr is None:
            reasons.append("no recorded DSR")
        elif self.dsr < DSR_GATE:
            reasons.append(f"DSR {self.dsr:.3f} < {DSR_GATE}")
        if self.pbo is None:
            reasons.append("no recorded PBO")
        elif self.pbo >= PBO_GATE:
            reasons.append(f"PBO {self.pbo:.3f} >= {PBO_GATE}")
        return reasons

    def can_trade(self) -> bool:
        return not self.blocking_reasons()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "module": self.module,
            "class": self.cls,
            "status": self.status,
            "capital_cap_inr": self.capital_cap_inr,
            "dsr": self.dsr,
            "pbo": self.pbo,
            "shadow_days": self.shadow_days,
            "universe": self.universe,
            "tags": self.tags,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyEntry":
        missing = [k for k in ("id", "module") if not d.get(k)]
        if missing:
            raise RegistryError(f"registry entry missing required key(s): {missing}: {d!r}")
        return cls(
            id=str(d["id"]),
            module=str(d["module"]),
            cls=str(d.get("class", "")),
            status=str(d.get("status", "backtest")),
            capital_cap_inr=float(d.get("capital_cap_inr", 0.0) or 0.0),
            dsr=None if d.get("dsr") is None else float(d["dsr"]),
            pbo=None if d.get("pbo") is None else float(d["pbo"]),
            shadow_days=int(d.get("shadow_days", 0) or 0),
            universe=list(d.get("universe") or []),
            tags=list(d.get("tags") or []),
            notes=str(d.get("notes", "") or ""),
        )


class StrategyRegistry:
    """Loads, validates and mutates `register.yaml` under the promotion rules."""

    def __init__(self, entries: Optional[Dict[str, StrategyEntry]] = None,
                 path: Optional[Path] = None):
        self._entries: Dict[str, StrategyEntry] = entries or {}
        self.path = path

    # ── IO ───────────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Optional[Path | str] = None) -> "StrategyRegistry":
        p = Path(path) if path else DEFAULT_REGISTRY_PATH
        if not p.exists():
            raise RegistryError(f"registry not found at {p} — nothing may trade")
        raw = yaml.safe_load(p.read_text()) or {}
        rows = raw.get("strategies") or []
        if not isinstance(rows, list):
            raise RegistryError(f"{p}: 'strategies' must be a list, got {type(rows).__name__}")

        entries: Dict[str, StrategyEntry] = {}
        for row in rows:
            entry = StrategyEntry.from_dict(row)
            if entry.id in entries:
                raise RegistryError(f"duplicate strategy id in registry: {entry.id!r}")
            entries[entry.id] = entry
        return cls(entries, path=p)

    def save(self, path: Optional[Path | str] = None) -> None:
        p = Path(path) if path else self.path
        if p is None:
            raise RegistryError("no path to save registry to")
        payload = {
            "# NOTE": "Single source of truth. Nothing trades unless listed here as live.",
            "strategies": [e.to_dict() for e in sorted(self._entries.values(), key=lambda x: x.id)],
        }
        p.write_text(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))

    # ── Access ───────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, strategy_id: str) -> bool:
        return strategy_id in self._entries

    def get(self, strategy_id: str) -> Optional[StrategyEntry]:
        return self._entries.get(strategy_id)

    def all(self) -> List[StrategyEntry]:
        return list(self._entries.values())

    def by_status(self, status: str) -> List[StrategyEntry]:
        if status not in VALID_STATUSES:
            raise RegistryError(f"unknown status {status!r}")
        return [e for e in self._entries.values() if e.status == status]

    def add(self, entry: StrategyEntry) -> None:
        if entry.id in self._entries:
            raise RegistryError(f"strategy {entry.id!r} already registered")
        self._entries[entry.id] = entry

    def can_trade(self, strategy_id: str) -> bool:
        """
        Fail-closed. An unregistered id is not an error to be worked around —
        it is a strategy that has never been validated.
        """
        entry = self._entries.get(strategy_id)
        if entry is None:
            log.warning("can_trade(%r): not in registry — refusing", strategy_id)
            return False
        return entry.can_trade()

    def tradeable(self) -> List[StrategyEntry]:
        return [e for e in self._entries.values() if e.can_trade()]

    def total_capital_committed(self) -> float:
        return sum(e.capital_cap_inr for e in self._entries.values() if e.status == "live")

    # ── Promotion / demotion ─────────────────────────────────────────────────

    def promote(
        self,
        strategy_id: str,
        *,
        dsr: Optional[float] = None,
        pbo: Optional[float] = None,
        shadow_days: Optional[int] = None,
        capital_cap_inr: float = 0.0,
    ) -> StrategyEntry:
        """
        Advance a strategy exactly one rung, enforcing the gate for that rung.
        Raises rather than promoting on unproven evidence.
        """
        entry = self._entries.get(strategy_id)
        if entry is None:
            raise RegistryError(f"cannot promote unknown strategy {strategy_id!r}")
        if entry.status == "retired":
            raise RegistryError(f"{strategy_id}: retired strategies are not promoted, re-register")
        nxt = _NEXT_RUNG.get(entry.status)
        if nxt is None:
            raise RegistryError(f"{strategy_id}: already at 'live', nothing to promote to")

        if dsr is not None:
            entry.dsr = float(dsr)
        if pbo is not None:
            entry.pbo = float(pbo)
        if shadow_days is not None:
            entry.shadow_days = int(shadow_days)

        # Gate for entering 'shadow': the statistics must already be in.
        if nxt == "shadow" and not entry.gates_passed():
            raise RegistryError(
                f"{strategy_id}: cannot enter shadow — needs DSR >= {DSR_GATE} and "
                f"PBO < {PBO_GATE}, have DSR={entry.dsr}, PBO={entry.pbo}"
            )

        # Gate for entering 'live': gauntlet clear AND >= 4 weeks of shadow.
        if nxt == "live":
            if not entry.gates_passed():
                raise RegistryError(
                    f"{strategy_id}: cannot go live — DSR={entry.dsr}, PBO={entry.pbo}"
                )
            if entry.shadow_days < 28:
                raise RegistryError(
                    f"{strategy_id}: cannot go live — needs >= 28 days of shadow, "
                    f"has {entry.shadow_days}"
                )
            if capital_cap_inr <= 0:
                raise RegistryError(
                    f"{strategy_id}: going live requires a positive capital_cap_inr"
                )

        entry.status = nxt
        entry.capital_cap_inr = capital_cap_inr if nxt == "live" else 0.0
        log.info("promoted %s -> %s", strategy_id, nxt)
        return entry

    def demote(self, strategy_id: str, reason: str, to: str = "shadow") -> StrategyEntry:
        """Move a strategy down (or retire it) and record why. Always allowed."""
        entry = self._entries.get(strategy_id)
        if entry is None:
            raise RegistryError(f"cannot demote unknown strategy {strategy_id!r}")
        if to not in VALID_STATUSES:
            raise RegistryError(f"unknown target status {to!r}")
        if to != "retired" and _RUNG_ORDER.get(to, 99) >= _RUNG_ORDER.get(entry.status, 99):
            raise RegistryError(
                f"{strategy_id}: demote target {to!r} is not below current {entry.status!r}"
            )
        prev = entry.status
        entry.status = to
        entry.capital_cap_inr = 0.0   # capital is withdrawn immediately
        entry.notes = f"demoted {prev}->{to}: {reason}"
        log.warning("demoted %s %s -> %s: %s", strategy_id, prev, to, reason)
        return entry

    def retire(self, strategy_id: str, reason: str) -> StrategyEntry:
        return self.demote(strategy_id, reason, to="retired")


def build_from_tree(
    root: Path | str = "backend/strategies",
    skip_dirs: Iterable[str] = ("__pycache__", "registry", "utils", "configs", "examples"),
) -> StrategyRegistry:
    """
    Seed a registry by scanning the strategy tree.

    Everything lands at `status: backtest`, `capital_cap_inr: 0`, no recorded
    DSR/PBO — i.e. nothing can trade. That is the correct starting state: being
    on disk is not evidence of anything.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise RegistryError(f"strategy tree not found: {root_path}")

    skip = set(skip_dirs)
    reg = StrategyRegistry(path=DEFAULT_REGISTRY_PATH)
    for py in sorted(root_path.rglob("*.py")):
        if py.name.startswith("__") or any(part in skip for part in py.parts):
            continue
        category = py.parent.name
        module = ".".join(py.with_suffix("").parts)
        reg.add(
            StrategyEntry(
                id=f"{category}.{py.stem}",
                module=module,
                cls="",
                status="backtest",
                capital_cap_inr=0.0,
                tags=[category],
            )
        )
    return reg
