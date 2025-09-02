# workers/scenario_worker.py
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

from platform import bootstrap # type: ignore
from platform import envelope as env # type: ignore

SERVICE = "scenario-worker"

# Streams & consumer group
IN_STREAM = os.getenv("IN_STREAM", "STREAM_SCENARIO_REQUESTS")
OUT_STREAM = os.getenv("OUT_STREAM", "STREAM_SCENARIO_RESULTS")
DLQ_STREAM = os.getenv("DLQ_STREAM", "STREAM_DLQ")
GROUP = os.getenv("GROUP", "scenario_v1")
CONSUMER = os.getenv("CONSUMER", f"scenario-{int(time.time())}")

# Idempotency TTL (seconds)
IDEMP_TTL = int(os.getenv("IDEMP_TTL", "86400"))  # 1 day

# --------------------------------------------------------------------------------------
# Domain: tiny scenario engine (you can swap with research/risk/scenarios.py later)
# --------------------------------------------------------------------------------------
def _apply_pct_shock(prices: Dict[str, float], shock_pct: float) -> Dict[str, float]:
    """Return shocked prices: px * (1 + shock_pct); shock_pct is e.g. -0.1 for -10%."""
    return {sym: px * (1.0 + shock_pct) for sym, px in prices.items()}


def _portfolio_pnl(positions: Dict[str, float], prices_before: Dict[str, float], prices_after: Dict[str, float]) -> float:
    """Sum position * (px_after - px_before). Positions in shares; prices in currency."""
    pnl = 0.0
    for sym, qty in positions.items():
        b = prices_before.get(sym)
        a = prices_after.get(sym)
        if b is None or a is None:
            continue
        pnl += qty * (a - b)
    return pnl


def run_scenario(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a scenario. Expected payload:
      {
        "scenario": {"type": "shock_pct", "value": -0.1},  # or +0.05 etc.
        "region": "US",
        "positions": {"AAPL": 100, "MSFT": -50},
        "ref_prices": {"AAPL": 200.0, "MSFT": 120.0},
      }
    """
    scenario = payload.get("scenario") or {}
    sc_type = str(scenario.get("type", "shock_pct"))
    value = float(scenario.get("value", 0.0))

    positions = {str(k): float(v) for k, v in (payload.get("positions") or {}).items()}
    ref_prices = {str(k): float(v) for k, v in (payload.get("ref_prices") or {}).items()}

    if sc_type != "shock_pct":
        raise ValueError(f"unsupported_scenario_type {sc_type}")

    shocked = _apply_pct_shock(ref_prices, value)
    pnl = _portfolio_pnl(positions, ref_prices, shocked)

    # Per-symbol contribution
    contrib = {
        sym: qty * (shocked.get(sym, ref_prices.get(sym, 0.0)) - ref_prices.get(sym, 0.0))
        for sym, qty in positions.items()
    }

    return {
        "scenario": {"type": sc_type, "value": value},
        "pnl": pnl,
        "contrib": contrib,
        "prices_after": shocked,
    }


# --------------------------------------------------------------------------------------
# Main worker
# --------------------------------------------------------------------------------------
def _ensure_group(r, stream: str, group: str) -> None:
    # Create consumer group if it doesn't exist
    try:
        r.xgroup_create(stream, group, id="$", mkstream=True)
    except Exception:
        # Group probably exists
        pass


def _publish(r, result_env: env.Envelope) -> None:
    r.xadd(OUT_STREAM, result_env.flatten_for_stream())


def main() -> None:
    ctx = bootstrap.init(SERVICE)
    tracer = ctx["tracer"]
    METRICS = ctx["metrics"]
    r = ctx["redis"]
    ent = ctx["ent"]
    calendars = ctx["cal"]
    audit = ctx["audit"]
    dlq = ctx["dlq"]  # DeadLetterQueue

    _ensure_group(r, IN_STREAM, GROUP)

    while True:
        # Block up to 5s waiting for messages
        resp = r.xreadgroup(GROUP, CONSUMER, {IN_STREAM: ">"}, count=50, block=5000) or []
        if not resp:
            continue

        for stream, messages in resp:
            for msg_id, fields in messages:
                try:
                    raw = fields.get("payload") or "{}"
                    e = env.parse(raw)  # validate shape/version
                    topic, corr = env.correlation_tuple(e)

                    # Idempotency
                    if env.is_duplicate(r, e.id, ttl_seconds=IDEMP_TTL):
                        r.xack(stream, GROUP, msg_id)
                        continue

                    # Basic required fields
                    e.require(["scenario", "positions", "ref_prices", "region"])
                    region = str(e.payload.get("region", "US"))

                    # Region open check (optional guard)
                    # If you want scenarios only during trading hours, uncomment:
                    # if not calendars.is_open(region, dt.datetime.now(dt.timezone.utc)):
                    #     raise ValueError(f"market_closed region={region}")

                    # Entitlements (if resource path provided)
                    resource = f"scenarios/{e.payload.get('scenario', {}).get('type','unknown')}"
                    if not ent.allow(e.producer.get("roles", []) if isinstance(e.producer, dict) else [], resource, action="read", region=region):
                        raise PermissionError(f"entitlement_denied resource={resource} region={region}")

                    with tracer.start_as_current_span("scenario.evaluate", attributes={"corr_id": corr, "region": region, "topic": topic}):
                        with METRICS.latency_timer("scenario_evaluate"):
                            result = run_scenario(e.payload)

                    # Audit
                    audit.record(
                        action="scenario_evaluate",
                        resource=resource,
                        user=(e.producer or {}).get("user"),
                        corr_id=corr,
                        region=region,
                        policy_hash=os.getenv("POLICY_HASH"),
                        details={"result_preview": {"pnl": result["pnl"], "n_symbols": len(result["contrib"])}},
                        input_for_hash=e.payload,
                    )

                    # Publish result
                    out = env.new(
                        schema_name="scenarios.result",
                        payload={
                            "request_id": e.id,
                            "corr_id": e.corr_id,
                            "region": region,
                            "result": result,
                        },
                        corr_id=e.corr_id,
                        producer={"svc": SERVICE},
                    )
                    _publish(r, out)

                    r.xack(stream, GROUP, msg_id)

                except PermissionError as pe:
                    METRICS.inc_task("scenario_evaluate", error=True)
                    dlq.push(payload=fields, reason=str(pe), corr_id=fields.get("corr_id"), retries=int(fields.get("retries", "0")))
                    r.xack(stream, GROUP, msg_id)

                except Exception as ex:
                    # Push to DLQ and ack so we don't spin
                    METRICS.inc_task("scenario_evaluate", error=True)
                    dlq.push(payload=fields, reason=str(ex), corr_id=fields.get("corr_id"), retries=int(fields.get("retries", "0")))
                    r.xack(stream, GROUP, msg_id)


if __name__ == "__main__":
    main()