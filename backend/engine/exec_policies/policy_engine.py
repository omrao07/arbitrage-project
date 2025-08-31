# backend/engine/exec_policies/kill_switch.py
from __future__ import annotations

import os
import sys
import time
import json
import signal
import redis
from typing import Dict, Any

from backend.bus.streams import hset, publish_stream

# --- Environment ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# State storage
KILL_KEY   = os.getenv("KILL_SWITCH_KEY", "policy:kill_switch")
ALERTS_KEY = os.getenv("ALERTS_STREAM", "alerts")

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


class KillSwitch:
    """
    A central 'circuit breaker' for the hedge fund OS.
    - When enabled, allocators return zero weights
    - OMS blocks new orders
    - Risk layer can stop marking
    """

    def __init__(self, key: str = KILL_KEY) -> None:
        self.key = key

    # --- core state ---
    def enable(self, reason: str = "manual") -> None:
        state = {"enabled": True, "reason": reason, "ts": int(time.time()*1000)}
        hset(self.key, "enabled", "true")
        hset(self.key, "reason", reason)
        hset(self.key, "ts", state["ts"])
        publish_stream(ALERTS_KEY, {"lvl": "critical", "src": "kill_switch", "msg": "ENABLED", "reason": reason})
        print("[KILL SWITCH] ENABLED:", reason)

    def disable(self) -> None:
        state = {"enabled": False, "reason": "", "ts": int(time.time()*1000)}
        hset(self.key, "enabled", "false")
        hset(self.key, "reason", "")
        hset(self.key, "ts", state["ts"])
        publish_stream(ALERTS_KEY, {"lvl": "info", "src": "kill_switch", "msg": "DISABLED"})
        print("[KILL SWITCH] DISABLED")

    def status(self) -> Dict[str, Any]:
        raw = r.hgetall(self.key) or {}
        return {
            "enabled": str(raw.get("enabled", "false")).lower() in ("1", "true", "yes", "on"),#type:ignore
            "reason": raw.get("reason"),#type: ignore
            "ts": int(raw.get("ts") or 0),#type:ignore
        }


# --- CLI / Runner ---
def _main() -> None:
    ks = KillSwitch()

    def stop_handler(signum, frame):
        print("[KILL SWITCH] exiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    import argparse
    p = argparse.ArgumentParser(description="Kill Switch control")
    p.add_argument("action", choices=["status", "enable", "disable"], help="action")
    p.add_argument("--reason", default="manual", help="reason for enable")
    args = p.parse_args()

    if args.action == "enable":
        ks.enable(args.reason)
    elif args.action == "disable":
        ks.disable()
    else:
        print(ks.status())


if __name__ == "__main__":
    _main()