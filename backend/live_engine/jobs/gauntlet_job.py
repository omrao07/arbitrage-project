"""
Nightly gauntlet job — re-validates every registered strategy, unattended.

Runs after the market data job so it scores against the day's real history.
Writes DSR/PBO back to `register.yaml` and demotes anything whose statistics
have decayed below the gates.

HARD-FAIL SEMANTICS
-------------------
This job refuses to run against anything but real data. If the configured price
history is missing or too short it raises, which the scheduler records as a job
failure and alerts on. It does NOT fall back to a synthetic series — a nightly
job that quietly re-scores the book against noise would rewrite every DSR in
the registry with a fiction, and nobody would notice until capital moved.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

log = logging.getLogger(__name__)

# Real price history for the nightly re-score. No default that could resolve
# to sample data — an unset path is a configuration error, not a reason to guess.
GAUNTLET_PRICES = os.getenv("GAUNTLET_PRICES_FILE", "")
GAUNTLET_COLUMN = os.getenv("GAUNTLET_PRICE_COLUMN", "close")


def run() -> Dict[str, Any]:
    """
    Re-run the validation gauntlet and update the registry.

    Returns a summary dict for the scheduler's job log. Raises on any condition
    that would otherwise produce a dishonest verdict.
    """
    from backend.validation.pipeline import csv_source, run_gauntlet

    if not GAUNTLET_PRICES:
        raise RuntimeError(
            "GAUNTLET_PRICES_FILE is unset — refusing to re-score the registry. "
            "Point it at real price history; there is deliberately no synthetic fallback."
        )
    path = Path(GAUNTLET_PRICES)
    if not path.exists():
        raise RuntimeError(f"GAUNTLET_PRICES_FILE does not exist: {path} — HALT")

    source = csv_source(path, GAUNTLET_COLUMN)   # raises if too few rows
    if not source.is_real:
        raise RuntimeError("gauntlet source is not real data — refusing to commit")

    report = run_gauntlet(source, commit=True)
    log.info(
        "gauntlet: %d/%d ran, %d cleared DSR, PBO=%s",
        report.n_ran, report.n_attempted, len(report.dsr_passed), report.pbo,
    )

    demoted = _demote_decayed(report)

    return {
        "attempted": report.n_attempted,
        "ran": report.n_ran,
        "failed": report.n_failed,
        "dsr_passed": len(report.dsr_passed),
        "pbo": report.pbo,
        "demoted": demoted,
        "committed": report.committed,
    }


def _demote_decayed(report) -> list[str]:
    """
    Withdraw capital from anything that no longer clears its gates.

    Demotion is automatic and immediate. A strategy whose DSR has decayed below
    the gate keeps trading for exactly as long as it takes this job to notice,
    which is the point of running it nightly rather than when someone remembers.
    """
    from backend.validation.dsr import DSR_GATE
    from backend.validation.pbo import PBO_GATE
    from backend.validation.registry import RegistryError, StrategyRegistry

    reg = StrategyRegistry.load()
    demoted: list[str] = []
    for entry in reg.all():
        if entry.status not in ("shadow", "live"):
            continue
        stale_dsr = entry.dsr is not None and entry.dsr < DSR_GATE
        stale_pbo = entry.pbo is not None and entry.pbo >= PBO_GATE
        if not (stale_dsr or stale_pbo):
            continue
        why = []
        if stale_dsr:
            why.append(f"DSR {entry.dsr:.3f} < {DSR_GATE}")
        if stale_pbo:
            why.append(f"PBO {entry.pbo:.3f} >= {PBO_GATE}")
        target = "shadow" if entry.status == "live" else "retired"
        try:
            reg.demote(entry.id, "; ".join(why), to=target)
            demoted.append(entry.id)
        except RegistryError as exc:
            log.warning("could not demote %s: %s", entry.id, exc)
    if demoted:
        reg.save()
        log.warning("gauntlet demoted %d strategies: %s", len(demoted), demoted[:10])
    return demoted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(run())
