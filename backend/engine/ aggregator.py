# backend/engine/aggregator.py
"""
Aggregator: consumes all enabled trade streams, rebroadcasts ticks to UI,
updates last prices for OMS, and forwards ticks to the strategy router.

Run from VS Code: "Run Strategy Engine" launch config.
"""

from __future__ import annotations

import json
import logging
import threading
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

import redis

# Local modules
from backend.config.settings import REGISTER_CONFIG, FEEDS_DIR, REDIS_HOST, REDIS_PORT, LOG_LEVEL
from backend.bus.streams import (
    consume_stream,
    publish_pubsub,
    hset,
    set as kv_set,
)

# Optional: strategy router hook (we keep running if it doesn't exist)
try:
    # Expect a function: route_tick(tick: dict) -> List[dict] (orders)
    from backend.engine.strategy_router import route_tick  # type: ignore
    HAS_ROUTER = True
except Exception:
    HAS_ROUTER = False


log = logging.getLogger("aggregator")
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

# Redis (for quick KV updates OMS/API may use)
_r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def _load_region_configs() -> List[Dict[str, Any]]:
    """Load each enabled region feed yaml listed in register.yaml."""
    feeds = REGISTER_CONFIG.get("feeds", [])
    enabled: List[Dict[str, Any]] = []
    for f in feeds:
        if not f.get("enabled", True):
            continue
        cfg_path = Path(FEEDS_DIR) / Path(f["config_file"]).name  # support "feeds/xx.yaml" or just "xx.yaml"
        if not cfg_path.exists():
            log.warning("Region config file missing: %s", cfg_path)
            continue
        with open(cfg_path, "r") as fh:
            rcfg = yaml.safe_load(fh)
            rcfg["_region"] = rcfg.get("region", f.get("region"))
            enabled.append(rcfg)
    return enabled


def _topic_names_for_region(rcfg: Dict[str, Any]) -> Dict[str, str]:
    """Return stream/channel names for a region config."""
    streams = rcfg.get("streams", {}) or {}
    channels = rcfg.get("channels", {}) or {}
    return {
        "trades_stream": streams.get("trades", f"trades.{rcfg.get('_region', 'XX').lower()}"),
        "orders_stream": streams.get("orders", f"orders.{rcfg.get('_region', 'XX').lower()}"),
        "ticks_channel": channels.get("ticks", f"ticks.{rcfg.get('_region', 'XX').lower()}"),
    }


def _update_last_price_and_publish(tick: Dict[str, Any], ticks_channel: str) -> None:
    """
    - Store last price per symbol (for OMS fill simulation).
    - Publish tick to UI via Pub/Sub.
    """
    symbol = tick.get("symbol") or tick.get("s")
    price = tick.get("price") or tick.get("p")
    if symbol and price is not None:
        _r.hset("last_price", symbol, json.dumps({"price": float(price)}))
    publish_pubsub(ticks_channel, tick)


def _consume_loop(region_name: str, trades_stream: str, ticks_channel: str) -> None:
    log.info("Starting consumer for region=%s stream=%s -> channel=%s", region_name, trades_stream, ticks_channel)
    try:
        for _, tick in consume_stream(trades_stream, start_id="$", block_ms=1000, count=200):
            # Normalize minimal fields (for safety across mixed sources)
            if isinstance(tick, str):
                try:
                    tick = json.loads(tick)
                except Exception:
                    tick = {"raw": tick}
            tick.setdefault("source", region_name)

            # 1) Update last price & fan-out to UI
            _update_last_price_and_publish(tick, ticks_channel)

            # 2) Forward to strategy router (optional)
            if HAS_ROUTER:
                try:
                    orders: Optional[List[Dict[str, Any]]] = route_tick(tick)  # type: ignore
                    if orders:
                        # Strategy router should publish to STREAM_ORDERS itself;
                        # but as a safety, we can drop them to a Redis list for audit:
                        _r.lpush("orders.audit", json.dumps(orders))
                except Exception as e:
                    log.exception("Strategy router error on tick %s: %s", tick.get("symbol"), e)
    except Exception as e:
        log.exception("Consumer crashed for region=%s: %s", region_name, e)


def main():
    regions = _load_region_configs()
    if not regions:
        log.error("No enabled region configs found. Check config/register.yaml and config/feeds/*.yaml")
        return

    # A tiny heartbeat so API/health checks can see we're alive
    kv_set("aggregator:alive", {"ts": int(time.time())}) # type: ignore

    threads: List[threading.Thread] = []
    for rcfg in regions:
        names = _topic_names_for_region(rcfg)
        region = rcfg.get("_region", "XX")
        t = threading.Thread(
            target=_consume_loop,
            args=(region, names["trades_stream"], names["ticks_channel"]),
            name=f"consumer-{region}",
            daemon=True,
        )
        t.start()
        threads.append(t)
        log.info("Spawned consumer thread for region=%s", region)

    log.info("Aggregator started. Waiting on %d streams...", len(threads))
    # Keep the main thread alive
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        log.info("Aggregator shutting down...")


if __name__ == "__main__":
    main()