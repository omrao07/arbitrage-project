# backend/engine/region_router.py
"""
Region Router:
- Infers region from symbol/venue
- Applies per-region compliance policy
- Publishes order to region stream + global orders stream

Usage:
    from backend.engine.region_router import route_order
    route_order({"strategy":"pairs","symbol":"AAPL","side":"buy","qty":10})
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from backend.bus.streams import (
    publish_stream,
    publish_pubsub,
    STREAM_ORDERS,
    CHAN_ORDERS,
)

# ---------- Paths ----------
REPO_ROOT = Path(__file__).resolve().parents[2]           # .../arbitrage-project
POLICY_DIR = REPO_ROOT / "backend" / "config" / "policies"

# ---------- Helpers: region inference ----------
_CRYPTO_SUFFIXES = ("USDT", "USDC", "BTC", "ETH")
_FX_PAIR_RE = re.compile(r"^[A-Z]{3}[A-Z]{3}$")  # e.g., EURUSD, USDJPY
_IN_SUFFIXES = (".NS", ".BSE", ".BO")            # NSE/BSE
_JP_SUFFIXES = (".T",)                           # Tokyo
_CN_HK_SUFFIXES = (".HK", ".SS", ".SZ")          # HKEX/Shanghai/Shenzhen
_EU_SUFFIXES = (".PA", ".AS", ".DE", ".FR", ".MI", ".ST")  # Euronext/Xetra/etc.

def infer_region(symbol: str, venue: Optional[str] = None) -> str:
    s = (symbol or "").upper()
    v = (venue or "").upper()

    # Venue hint first
    if v in ("BINANCE", "BYBIT", "COINBASE"):
        return "CRYPTO"
    if v in ("OANDA", "FXCM"):
        return "FX"

    # FX pairs
    if _FX_PAIR_RE.match(s):
        return "FX"

    # Crypto tickers commonly end with stable/major base
    if any(s.endswith(suf) for suf in _CRYPTO_SUFFIXES):
        return "CRYPTO"

    # India
    if any(s.endswith(suf) for suf in _IN_SUFFIXES):
        return "IN"

    # Japan
    if any(s.endswith(suf) for suf in _JP_SUFFIXES):
        return "JP"

    # China/HK
    if any(s.endswith(suf) for suf in _CN_HK_SUFFIXES):
        return "CNHK"

    # Europe (basic)
    if any(s.endswith(suf) for suf in _EU_SUFFIXES):
        return "EU"

    # Default to US if plain ticker
    return "US"

# ---------- Policy loading ----------
# Cache policies in-memory
_POLICIES: Dict[str, Dict[str, Any]] = {}

def _load_policy(region: str) -> Dict[str, Any]:
    r = region.upper()
    if r in _POLICIES:
        return _POLICIES[r]

    path = POLICY_DIR / f"{r}.yaml"
    if not path.exists():
        # safe defaults
        policy = {
            "region": r,
            "restricted_list": [],
            "allow_short": True,
            "market_hours_only": False,
            "max_leverage": 1.0,
        }
        _POLICIES[r] = policy
        return policy

    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh) or {}
    # normalize
    policy = {
        "region": r,
        "restricted_list": cfg.get("restricted_list", []),
        "allow_short": bool(cfg.get("allow_short", True)),
        "market_hours_only": bool(cfg.get("market_hours_only", False)),
        "max_leverage": float(cfg.get("max_leverage", 1.0)),
    }
    _POLICIES[r] = policy
    return policy

# ---------- Compliance checks ----------
def _is_restricted(symbol: str, policy: Dict[str, Any]) -> bool:
    rs = {x.upper() for x in policy.get("restricted_list", [])}
    return symbol.upper() in rs

def _is_short(side: str) -> bool:
    return side.lower() == "sell"

def _market_hours_ok(region: str) -> bool:
    # Placeholder: proper calendar check can be added later
    # For now always allow; if policy enforces market_hours_only, you can plug in calendars here.
    return True

def check_compliance(order: Dict[str, Any], region: str) -> Tuple[bool, Optional[str]]:
    """
    Returns (ok, reason_if_blocked)
    """
    policy = _load_policy(region)
    symbol = str(order.get("symbol", "")).upper()
    side = str(order.get("side", "")).lower()

    if _is_restricted(symbol, policy):
        return False, "restricted_symbol"

    if policy.get("market_hours_only", False) and not _market_hours_ok(region):
        return False, "market_closed"

    if _is_short(side) and not policy.get("allow_short", True):
        return False, "shorting_disallowed"

    # You can add leverage / notional checks here if you track margin usage per region.
    return True, None

# ---------- Routing ----------
def _region_stream_name(region: str) -> str:
    return f"orders.{region.lower()}"

def route_order(order: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrich and route an order:
      - infer region
      - run compliance
      - publish to region stream + global orders stream
      - fan out to UI pubsub (CHAN_ORDERS)

    Returns the enriched order dict (or a rejection event payload if blocked).
    """
    enriched = dict(order)
    symbol = str(enriched.get("symbol", ""))
    venue = enriched.get("venue")  # optional hint
    region = infer_region(symbol, venue)
    enriched["region"] = region

    ok, reason = check_compliance(enriched, region)
    if not ok:
        payload = {"event": "reject", "reason": reason, **enriched}
        publish_pubsub(CHAN_ORDERS, payload)
        return payload

    # Region stream (optional consumers) + Global stream (execution_engine reads this)
    publish_stream(_region_stream_name(region), enriched)
    publish_stream(STREAM_ORDERS, enriched)

    publish_pubsub(CHAN_ORDERS, {"event": "order_routed", **enriched})
    return enriched

# Convenience for strategies:
def submit_order(strategy: str, symbol: str, side: str, qty: float, order_type: str = "market", limit_price: float | None = None, venue: Optional[str] = None) -> Dict[str, Any]:
    order = {
        "strategy": strategy,
        "symbol": symbol,
        "side": side,
        "qty": float(qty),
        "typ": order_type,
        "limit_price": limit_price,
    }
    if venue:
        order["venue"] = venue
    return route_order(order)

if __name__ == "__main__":
    # quick self-test
    print(route_order({"strategy":"test","symbol":"BTCUSDT","side":"buy","qty":0.01,"venue":"BINANCE"}))
    print(route_order({"strategy":"test","symbol":"AAPL","side":"sell","qty":5}))
    print(route_order({"strategy":"test","symbol":"RELIANCE.NS","side":"buy","qty":10}))
    print(route_order({"strategy":"test","symbol":"7203.T","side":"sell","qty":2}))