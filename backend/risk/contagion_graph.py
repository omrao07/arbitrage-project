# backend/risk/contagion_graph.py
"""
Compatibility shim for the contagion graph module.

Your full implementation lives in `backend/risk/contagian_graph.py`
(typo kept for backward-compat). This file lets you import the same
classes with the correct spelling:

    from backend.risk.contagion_graph import ContagionGraph, Bank, ShockParams

If the underlying module isn’t found, we raise a clear ImportError
with instructions.
"""

from __future__ import annotations

try:
    # Re-export everything you need from the original file
    from .contagian_graph import ( # type: ignore
        ContagionGraph,
        Bank,
        Exposure,
        ShockParams,
    )

    __all__ = ["ContagionGraph", "Bank", "Exposure", "ShockParams"]

except Exception as _e:
    # Graceful error that tells you exactly what to fix
    class _MissingDep:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Could not import from 'backend/risk/contagian_graph.py'. "
                "Make sure that file exists (with your implementation) "
                "or copy it here and fix imports."
            ) from _e

    class ContagionGraph(_MissingDep): ...
    class Bank(_MissingDep): ...
    class Exposure(_MissingDep): ...
    class ShockParams(_MissingDep): ...

    __all__ = ["ContagionGraph", "Bank", "Exposure", "ShockParams"]