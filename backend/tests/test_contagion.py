# tests/test_contagion.py
"""
Contagion engine tests

Assumptions (duck-typed, any similar API works):
- Module: backend.risk.contagion_graph (imported via pytest.importorskip)
- One of the following is available:
  A) Class ContagionGraph with methods:
     - add_bank(id, equity, liquid_assets, illiquid_assets, liabilities, **kw)
     - add_exposure(lender, borrower, amount, recovery_rate=0.4, **kw)
     - set_default(bank_id, flag=True)
     - step(recovery_rate: float | None = None) -> bool   # one propagation step
     - run(max_rounds: int = 10, recovery_rate: float | None = None) -> list[dict] | None
     - banks: dict[str, Any]  with per-bank fields: equity, liabilities, defaulted (bool)
  B) Function propagate_defaults(graph_like, recovery_rate: float, rounds: int) -> list[dict]
     where graph_like has .banks dict and .exposures list like the visualizer.

If neither API is present, tests are skipped.
"""

from importlib import import_module
import math
import pytest # type: ignore


cg = pytest.importorskip("backend.risk.contagion_graph", reason="contagion_graph module not found")

# --- Helpers to abstract over API ------------------------------------------------

def _has_class_api():
    return hasattr(cg, "ContagionGraph")

def _mk_graph():
    if _has_class_api():
        return cg.ContagionGraph()
    # Fallback: try a minimal struct if provided
    if hasattr(cg, "Graph"):
        return cg.Graph()
    pytest.skip("No compatible graph class (ContagionGraph/Graph) found")

def _add_bank(g, *, id, equity, liq, illiq, liab, **kw):
    if hasattr(g, "add_bank"):
        return g.add_bank(id=id, equity=equity, liquid_assets=liq, illiquid_assets=illiq, liabilities=liab, **kw)
    # some impls might expose g.banks dict
    if hasattr(g, "banks"):
        g.banks[id] = type("B", (), dict(id=id, name=id, equity=float(equity),
                                         liquid_assets=float(liq), illiquid_assets=float(illiq),
                                         liabilities=float(liab), defaulted=False))()
        return
    pytest.skip("Graph has no add_bank() and no banks dict to populate")

def _add_exposure(g, lender, borrower, amount, rr=0.4):
    if hasattr(g, "add_exposure"):
        return g.add_exposure(lender, borrower, amount, recovery_rate=rr)
    if hasattr(g, "exposures"):
        if isinstance(g.exposures, list):
            g.exposures.append({"lender": lender, "borrower": borrower, "amount": float(amount), "recovery_rate": float(rr)})
            return
    pytest.skip("Graph has no add_exposure() and no exposures list")

def _set_default(g, bank_id, flag=True):
    if hasattr(g, "set_default"):
        return g.set_default(bank_id, flag)
    # direct toggle
    b = _bank(g, bank_id)
    b.defaulted = bool(flag) # type: ignore

def _bank(g, bank_id):
    if hasattr(g, "get_bank"):
        return g.get_bank(bank_id)
    if hasattr(g, "banks"):
        return g.banks[bank_id]
    pytest.skip("Cannot access bank state")

def _step(g, recovery_rate=None):
    if hasattr(g, "step"):
        return g.step(recovery_rate=recovery_rate) if recovery_rate is not None else g.step()
    # Try function API
    if hasattr(cg, "propagate_defaults"):
        frames = cg.propagate_defaults(g, recovery_rate=recovery_rate or 0.4, rounds=1)
        # emulate changed=True if anything moved
        return bool(frames and frames[-1])
    pytest.skip("No step() method or propagate_defaults() function available")

def _run(g, rounds=5, recovery_rate=None):
    if hasattr(g, "run"):
        return g.run(max_rounds=rounds, recovery_rate=recovery_rate)
    if hasattr(cg, "propagate_defaults"):
        return cg.propagate_defaults(g, recovery_rate=recovery_rate or 0.4, rounds=rounds)
    pytest.skip("No run() or propagate_defaults() available")


# --- Fixtures -------------------------------------------------------------------

@pytest.fixture
def tiny_net():
    """
    A -> B -> C -> A ring with varying exposures.
    Start healthy; we will toggle defaults in tests.
    """
    g = _mk_graph()
    _add_bank(g, id="A", equity=100.0, liq=300.0, illiq=700.0, liab=800.0)
    _add_bank(g, id="B", equity=80.0,  liq=200.0, illiq=500.0, liab=620.0)
    _add_bank(g, id="C", equity=60.0,  liq=150.0, illiq=450.0, liab=540.0)

    _add_exposure(g, "A", "B", amount=120.0, rr=0.5)
    _add_exposure(g, "B", "C", amount=100.0, rr=0.4)
    _add_exposure(g, "C", "A", amount=90.0,  rr=0.3)
    return g


# --- Tests ---------------------------------------------------------------------

def test_single_default_hit(tiny_net):
    g = tiny_net
    # Default borrower B; lender A should lose (1-rr)*exposure = 0.5*120 = 60
    _set_default(g, "B", True)
    changed = _step(g)  # one propagation
    assert changed is True

    A = _bank(g, "A")
    assert pytest.approx(A.equity, rel=1e-6) == 100.0 - 60.0 # type: ignore

def test_recovery_rate_applied(tiny_net):
    g = tiny_net
    _set_default(g, "C", True)  # A has exposure 90 to C with rr=0.3 → loss 63
    _step(g)
    A = _bank(g, "A")
    assert pytest.approx(A.equity, rel=1e-6) == 100.0 - (1 - 0.3) * 90.0 # type: ignore

def test_two_round_cascade(tiny_net):
    g = tiny_net
    # Make B fragile so that loss from C default pushes B into default next round
    B = _bank(g, "B")
    B.equity = 30.0  # type: ignore # small buffer
    _set_default(g, "C", True)

    frames = _run(g, rounds=3)  # allow cascade
    # After C default: A loses 63; if that causes A default, C exposure back etc.
    # More importantly, B loses from exposure to C: (1-0.4)*100 = 60 > 30 equity → B should default in cascade.
    B = _bank(g, "B")
    assert getattr(B, "defaulted", False) is True

def test_no_negative_equity(tiny_net):
    g = tiny_net
    # Huge loss to ensure we don't go deeply negative (engine may clamp to >= -1e-9 or set to 0)
    _add_exposure(g, "A", "B", amount=10_000.0, rr=0.0)
    _set_default(g, "B", True)
    _step(g)

    A = _bank(g, "A")
    assert A.equity > -1e-6  # type: ignore # allow tiny numerical negatives; prefer clamp to 0

def test_directionality(tiny_net):
    g = tiny_net
    # Default A; lender is C (edge C->A). B should not be hit directly in first step.
    _set_default(g, "A", True)
    _step(g)

    B = _bank(g, "B")
    # B only lends to C; shouldn't be impacted by A's default in step 1.
    assert getattr(B, "defaulted", False) is False
    assert B.equity == pytest.approx(80.0, rel=1e-6) # type: ignore

def test_idempotent_when_no_new_defaults(tiny_net):
    g = tiny_net
    # One step with no defaults set should be idempotent (no changes)
    changed = _step(g)
    assert changed in (False, None)  # engine may return False/None when nothing propagated