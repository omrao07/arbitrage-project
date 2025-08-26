# backend/execution_plus/factory.py
"""
Factory utilities for wiring up execution stack components.

Provides helpers to build:
- RegistryHub (with adapters loaded from venues.yaml)
- Cost model(s)
- Discovery service
- ArbRouter (pre-wired with discovery + cost model)

Why?
-----
Keeps bootstrapping in one place, instead of scattering:
  HUB + load_adapters_from_yaml + DefaultCostModel + ArbRouter

Usage
-----
from backend.execution_plus.factory import (
    build_hub, build_router, build_cost_model, build_discovery
)

router = build_router()    # ready-to-go ArbRouter
hub    = build_hub()       # registry hub with adapters
"""

from __future__ import annotations

import os
from typing import Optional

from backend.execution_plus.registry import HUB, RegistryHub, load_adapters_from_yaml # type: ignore
from backend.execution_plus.cost_model import DefaultCostModel, get_default_model # type: ignore
from backend.execution_plus.arb_router.discovery import Discovery
from backend.execution_plus.arb_router.router import ArbRouter


# ---------------------------------------------------------------------
# Core builders
# ---------------------------------------------------------------------

def build_hub(venues_yaml: Optional[str] = None) -> RegistryHub:
    """
    Ensure HUB is populated with adapters from venues.yaml.
    If no file found, HUB falls back to mocks from AdapterRegistry.
    """
    path = venues_yaml or os.getenv("VENUES_YAML", "backend/config/venues.yaml")
    try:
        load_adapters_from_yaml(path)
    except Exception:
        # swallow errors if file missing; HUB already has mock defaults
        pass
    return HUB


def build_cost_model() -> DefaultCostModel:
    """
    Return a default cost model with env-tuned params.
    """
    return get_default_model()


def build_discovery(venues_yaml: Optional[str] = None) -> Discovery:
    """
    Create a Discovery service (for venue universe + probes).
    """
    path = venues_yaml or os.getenv("VENUES_YAML", "backend/config/venues.yaml")
    return Discovery(path)


def build_router(
    venues_yaml: Optional[str] = None,
    *,
    cost_model: Optional[DefaultCostModel] = None,
    topk: Optional[int] = None,
    min_child_notional: Optional[float] = None,
    max_slippage_bps: Optional[float] = None,
) -> ArbRouter:
    """
    Construct a fully wired ArbRouter.
    """
    cm = cost_model or build_cost_model()
    path = venues_yaml or os.getenv("VENUES_YAML", "backend/config/venues.yaml")
    return ArbRouter(
        venues_yaml=path,
        cost_model=cm,
        topk=topk,
        min_child_notional=min_child_notional,
        max_slippage_bps=max_slippage_bps,
    )


# ---------------------------------------------------------------------
# Tiny CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    hub = build_hub()
    print("Adapters loaded:", list(hub.adapters.all().keys()))

    router = build_router()
    print("Router ready with topk=", router.topk)