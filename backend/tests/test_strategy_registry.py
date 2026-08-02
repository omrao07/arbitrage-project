"""
Registry tests: nothing trades unless it has earned the right.

The failure these prevent is a strategy reaching live capital without a
recorded DSR/PBO or without serving its shadow period — i.e. the gauntlet
existing on paper while capital routes around it.
"""

from __future__ import annotations

import pytest

from backend.validation.registry import (
    RegistryError,
    StrategyEntry,
    StrategyRegistry,
    build_from_tree,
)


def _entry(**kw) -> StrategyEntry:
    base = dict(id="s1", module="backend.strategies.x.y", cls="Strat")
    base.update(kw)
    return StrategyEntry(**base)


@pytest.fixture
def reg():
    r = StrategyRegistry()
    r.add(_entry())
    return r


# ── Schema invariants ───────────────────────────────────────────────────────

def test_new_entry_starts_untradeable():
    e = _entry()
    assert e.status == "backtest"
    assert e.capital_cap_inr == 0.0
    assert not e.can_trade()


def test_capital_cannot_be_assigned_before_live():
    with pytest.raises(RegistryError, match="must be 0 until status is 'live'"):
        _entry(status="shadow", capital_cap_inr=100_000)


def test_negative_capital_rejected():
    with pytest.raises(RegistryError, match="negative"):
        _entry(status="live", capital_cap_inr=-1)


def test_unknown_status_rejected():
    with pytest.raises(RegistryError, match="not one of"):
        _entry(status="probably_fine")


def test_entry_requires_id_and_module():
    with pytest.raises(RegistryError, match="missing required key"):
        StrategyEntry.from_dict({"module": "m"})
    with pytest.raises(RegistryError, match="missing required key"):
        StrategyEntry.from_dict({"id": "x"})


def test_blocking_reasons_are_specific():
    reasons = _entry().blocking_reasons()
    assert any("not 'live'" in r for r in reasons)
    assert any("capital_cap_inr is 0" in r for r in reasons)
    assert any("no recorded DSR" in r for r in reasons)
    assert any("no recorded PBO" in r for r in reasons)


# ── can_trade is fail-closed ────────────────────────────────────────────────

def test_unregistered_strategy_cannot_trade(reg):
    assert reg.can_trade("never-heard-of-it") is False


def test_live_but_failing_gates_cannot_trade():
    e = _entry(status="live", capital_cap_inr=1000, dsr=0.4, pbo=0.2)
    assert not e.can_trade()
    assert any("DSR" in r for r in e.blocking_reasons())


def test_live_with_high_pbo_cannot_trade():
    e = _entry(status="live", capital_cap_inr=1000, dsr=0.99, pbo=0.8)
    assert not e.can_trade()
    assert any("PBO" in r for r in e.blocking_reasons())


def test_fully_cleared_strategy_can_trade():
    e = _entry(status="live", capital_cap_inr=1000, dsr=0.97, pbo=0.2)
    assert e.can_trade()
    assert e.blocking_reasons() == []


# ── Promotion ladder ────────────────────────────────────────────────────────

def test_promotion_is_one_rung_at_a_time(reg):
    assert reg.promote("s1").status == "walkforward"


def test_cannot_enter_shadow_without_recorded_statistics(reg):
    reg.promote("s1")  # -> walkforward
    with pytest.raises(RegistryError, match="cannot enter shadow"):
        reg.promote("s1")


@pytest.mark.parametrize("dsr,pbo", [(0.5, 0.2), (0.99, 0.7), (0.5, 0.7)])
def test_cannot_enter_shadow_on_failing_statistics(reg, dsr, pbo):
    reg.promote("s1")
    with pytest.raises(RegistryError, match="cannot enter shadow"):
        reg.promote("s1", dsr=dsr, pbo=pbo)


def test_cannot_go_live_without_enough_shadow_time(reg):
    reg.promote("s1")
    reg.promote("s1", dsr=0.97, pbo=0.2)          # -> shadow
    with pytest.raises(RegistryError, match=">= 28 days of shadow"):
        reg.promote("s1", shadow_days=10, capital_cap_inr=50_000)


def test_cannot_go_live_without_capital_cap(reg):
    reg.promote("s1")
    reg.promote("s1", dsr=0.97, pbo=0.2)
    with pytest.raises(RegistryError, match="positive capital_cap_inr"):
        reg.promote("s1", shadow_days=30, capital_cap_inr=0)


def test_full_ladder_reaches_live(reg):
    reg.promote("s1")
    reg.promote("s1", dsr=0.97, pbo=0.2)
    e = reg.promote("s1", shadow_days=30, capital_cap_inr=50_000)
    assert e.status == "live"
    assert e.capital_cap_inr == 50_000
    assert reg.can_trade("s1")


def test_cannot_promote_beyond_live(reg):
    reg.promote("s1")
    reg.promote("s1", dsr=0.97, pbo=0.2)
    reg.promote("s1", shadow_days=30, capital_cap_inr=50_000)
    with pytest.raises(RegistryError, match="already at 'live'"):
        reg.promote("s1")


def test_cannot_promote_unknown_strategy(reg):
    with pytest.raises(RegistryError, match="unknown strategy"):
        reg.promote("ghost")


# ── Demotion ────────────────────────────────────────────────────────────────

def test_demotion_withdraws_capital_immediately(reg):
    reg.promote("s1")
    reg.promote("s1", dsr=0.97, pbo=0.2)
    reg.promote("s1", shadow_days=30, capital_cap_inr=50_000)

    e = reg.demote("s1", "live P&L diverged from shadow")
    assert e.status == "shadow"
    assert e.capital_cap_inr == 0.0
    assert not reg.can_trade("s1")
    assert "diverged" in e.notes


def test_retire_is_allowed_from_any_rung(reg):
    e = reg.retire("s1", "regime shift, edge gone")
    assert e.status == "retired"
    assert not e.can_trade()


def test_demote_must_move_downward(reg):
    with pytest.raises(RegistryError, match="not below current"):
        reg.demote("s1", "nonsense", to="live")


def test_retired_strategy_is_not_promoted(reg):
    reg.retire("s1", "dead")
    with pytest.raises(RegistryError, match="retired"):
        reg.promote("s1")


# ── Persistence ─────────────────────────────────────────────────────────────

def test_round_trip_through_yaml(tmp_path, reg):
    reg.promote("s1")
    reg.promote("s1", dsr=0.97, pbo=0.2)
    reg.promote("s1", shadow_days=30, capital_cap_inr=50_000)

    p = tmp_path / "register.yaml"
    reg.save(p)
    back = StrategyRegistry.load(p)

    assert len(back) == 1
    e = back.get("s1")
    assert e.status == "live" and e.capital_cap_inr == 50_000
    assert e.dsr == 0.97 and e.pbo == 0.2
    assert back.can_trade("s1")


def test_duplicate_ids_rejected(tmp_path):
    p = tmp_path / "dup.yaml"
    p.write_text(
        "strategies:\n"
        "  - {id: a, module: m, class: C}\n"
        "  - {id: a, module: m2, class: D}\n"
    )
    with pytest.raises(RegistryError, match="duplicate"):
        StrategyRegistry.load(p)


def test_missing_registry_is_an_error_not_an_empty_pass(tmp_path):
    """An unreadable registry must not silently mean 'no restrictions'."""
    with pytest.raises(RegistryError, match="not found"):
        StrategyRegistry.load(tmp_path / "nope.yaml")


def test_add_rejects_duplicates(reg):
    with pytest.raises(RegistryError, match="already registered"):
        reg.add(_entry())


def test_total_capital_counts_only_live(reg):
    reg.promote("s1")
    reg.promote("s1", dsr=0.97, pbo=0.2)
    assert reg.total_capital_committed() == 0.0
    reg.promote("s1", shadow_days=30, capital_cap_inr=50_000)
    assert reg.total_capital_committed() == 50_000


# ── Tree scan ───────────────────────────────────────────────────────────────

def test_build_from_tree_registers_everything_as_untradeable():
    built = build_from_tree()
    assert len(built) > 100, "expected the real strategy tree"
    assert built.tradeable() == [], "a file on disk is not evidence of an edge"
    assert all(e.status == "backtest" for e in built.all())
    assert built.total_capital_committed() == 0.0
