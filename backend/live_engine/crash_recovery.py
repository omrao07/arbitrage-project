# backend/live_engine/crash_recovery.py
"""
Crash recovery: persist engine state to Redis on shutdown signals,
restore on restart.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
from typing import Any, Callable, Dict, Optional

import redis

logger = logging.getLogger("live_engine.crash_recovery")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
RECOVERY_KEY = "engine:crash_recovery:state"
CHECKPOINT_TTL = 86400 * 3  # 3 days

_r: Optional[redis.Redis] = None

def _get_r() -> redis.Redis:
    global _r
    if _r is None:
        _r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT,
                         password=os.getenv("REDIS_PASSWORD") or None,
                         decode_responses=True)
    return _r


def save_checkpoint(state: Dict[str, Any], r=None) -> None:
    """Persist engine state dict to Redis with TTL."""
    rc = r or _get_r()
    rc.set(RECOVERY_KEY, json.dumps(state), ex=CHECKPOINT_TTL)
    logger.info(f"[crash_recovery] checkpoint saved: {list(state.keys())}")


def load_checkpoint(r=None) -> Optional[Dict[str, Any]]:
    """Restore state from Redis. Returns None if no checkpoint exists."""
    rc = r or _get_r()
    raw = rc.get(RECOVERY_KEY)
    if raw is None:
        logger.info("[crash_recovery] no checkpoint found — cold start")
        return None
    state = json.loads(raw)
    logger.info(f"[crash_recovery] checkpoint loaded: {list(state.keys())}")
    return state


def clear_checkpoint(r=None) -> None:
    rc = r or _get_r()
    rc.delete(RECOVERY_KEY)
    logger.info("[crash_recovery] checkpoint cleared")


def install_signal_handlers(state_fn: Callable[[], Dict[str, Any]], r=None) -> None:
    """
    Register SIGTERM/SIGINT handlers that save a checkpoint before exit.
    state_fn: callable that returns the current engine state dict.
    """
    def _handler(signum, frame):
        logger.warning(f"[crash_recovery] signal {signum} received — saving checkpoint")
        try:
            save_checkpoint(state_fn(), r)
        except Exception as e:
            logger.error(f"[crash_recovery] failed to save checkpoint: {e}")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
    logger.info("[crash_recovery] signal handlers installed (SIGTERM, SIGINT)")
