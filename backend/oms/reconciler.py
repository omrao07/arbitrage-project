# backend/oms/reconciler.py
"""
Reconciler
----------
Keeps OMS state (orders, fills, positions) consistent with broker/exchange.

Responsibilities
- Periodically pull broker positions / open orders
- Compare vs internal order_store + positions
- Detect mismatches (fills missing, qty drift, cancelled vs active)
- Optionally auto-fix (cancel ghost orders, resync positions) or raise alerts
- Publish reconciliation report to bus for dashboards / audit

Assumes:
- broker_interface exposes: fetch_orders(), fetch_positions()
- order_store exposes: list_orders(), list_positions()
- bus.streams: publish_stream("reconciler.alerts", {...})
"""

from __future__ import annotations

import os
import time
import json
from typing import Any, Dict, List, Optional

try:
    from backend.bus.streams import publish_stream
except Exception:
    publish_stream = None

try:
    import backend.oms.order_store as order_store # type: ignore
    import backend.oms.broker_interface as broker_interface # type: ignore
except Exception:
    order_store = None
    broker_interface = None

def _utc_ms() -> int:
    import time
    return int(time.time() * 1000)


class Reconciler:
    def __init__(self, broker: str = "paper", auto_fix: bool = False, poll_s: int = 30):
        self.broker = broker
        self.auto_fix = auto_fix
        self.poll_s = poll_s
        self._running = False

    def reconcile_once(self) -> Dict[str, Any]:
        report = {"ts_ms": _utc_ms(), "broker": self.broker, "mismatches": []}
        try:
            broker_orders = broker_interface.fetch_orders(self.broker) if broker_interface else []
            broker_positions = broker_interface.fetch_positions(self.broker) if broker_interface else []
            local_orders = order_store.list_orders() if order_store else []
            local_positions = order_store.list_positions() if order_store else []
        except Exception as e:
            report["error"] = str(e)
            return report

        # reconcile orders
        mism = []
        broker_map = {o["id"]: o for o in broker_orders}
        local_map = {o["id"]: o for o in local_orders}
        # Orders present in broker but missing locally
        for oid, bo in broker_map.items():
            if oid not in local_map:
                mism.append({"type": "order_missing_local", "order": bo})
                if self.auto_fix:
                    order_store.add_order(bo) # type: ignore
        # Orders present locally but not at broker
        for oid, lo in local_map.items():
            if oid not in broker_map:
                mism.append({"type": "order_missing_broker", "order": lo})
                if self.auto_fix and lo.get("status") == "open":
                    # cancel locally
                    order_store.update_order_status(oid, "cancelled") # type: ignore
        # Qty/status drift
        for oid in set(broker_map.keys()) & set(local_map.keys()):
            bo, lo = broker_map[oid], local_map[oid]
            if bo.get("status") != lo.get("status") or abs(float(bo.get("filled_qty",0))-float(lo.get("filled_qty",0)))>1e-6:
                mism.append({"type":"order_status_drift","broker":bo,"local":lo})
                if self.auto_fix:
                    order_store.update_order(oid, bo) # type: ignore

        # reconcile positions
        broker_pos_map = {p["symbol"]: p for p in broker_positions}
        local_pos_map = {p["symbol"]: p for p in local_positions}
        for sym, bp in broker_pos_map.items():
            lp = local_pos_map.get(sym)
            if not lp:
                mism.append({"type":"position_missing_local","symbol":sym,"broker":bp})
                if self.auto_fix:
                    order_store.add_position(bp) # type: ignore
            else:
                if abs(float(bp.get("qty",0))-float(lp.get("qty",0)))>1e-6:
                    mism.append({"type":"position_qty_drift","symbol":sym,"broker":bp,"local":lp})
                    if self.auto_fix:
                        order_store.update_position(sym, bp) # type: ignore
        for sym, lp in local_pos_map.items():
            if sym not in broker_pos_map:
                mism.append({"type":"position_missing_broker","symbol":sym,"local":lp})
                if self.auto_fix and abs(lp.get("qty",0))>1e-6:
                    # mark closed
                    order_store.update_position(sym, {"qty":0}) # type: ignore

        report["mismatches"]=mism
        if publish_stream:
            try:
                publish_stream("reconciler.reports", report)
                if mism:
                    publish_stream("reconciler.alerts", {"ts_ms":_utc_ms(),"broker":self.broker,"mismatches":mism})
            except Exception:
                pass
        return report

    def run(self):
        self._running=True
        while self._running:
            rep=self.reconcile_once()
            time.sleep(self.poll_s)

    def stop(self):
        self._running=False


# CLI
if __name__=="__main__":
    r=Reconciler(auto_fix=False)
    try:
        r.run()
    except KeyboardInterrupt:
        r.stop()