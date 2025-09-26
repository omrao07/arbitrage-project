#!/usr/bin/env python3
"""
scripts/reset_shm.py
-----------------------------------------------------------
Removes and recreates all shm segments used by the gateway.

Usage:
    python3 scripts/reset_shm.py
    python3 scripts/reset_shm.py /orders_in   # reset only one
"""

import os
import sys

# List of shm segments (must match include/shm_layouts.hpp and configs/gateway.yaml)
SHM_NAMES = ["/risk_wall", "/md_ringbuf", "/orders_in", "/fills_out", "/heartbeat"]

def reset_one(name: str):
    path = f"/dev/shm{name}"
    try:
        os.remove(path)
        print(f"removed {path}")
    except FileNotFoundError:
        print(f"{path} not present")
    except PermissionError:
        print(f"no permission to remove {path}")
    # fresh file will be created on next shm_open by gateway

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Reset a specific shm region
        target = sys.argv[1]
        if not target.startswith("/"):
            target = "/" + target
        reset_one(target)
    else:
        for name in SHM_NAMES:
            reset_one(name)