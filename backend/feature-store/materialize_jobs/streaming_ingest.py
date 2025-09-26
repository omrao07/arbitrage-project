# materialize_jobs/streaming_ingest.py
"""
Real-time streaming ingest -> Feast online store.

Transports
---------
- Kafka (aiokafka)
- NATS (asyncio-nats)

What it does
------------
1) Consumes events (JSON) from Kafka topics or NATS subjects
2) Normalizes -> maps into Feast (entity keys + feature columns)
3) Batches and writes to Feast online store (write_to_online_store)
4) Idempotency via event_id de-dup within batch window
5) Backpressure-aware batching (size or time)

Examples
--------
# NATS (FX carry + vol surface)
python streaming_ingest.py \
  --repo ./feature-store \
  --nats "nats://localhost:4222" \
  --subject fx.carry --subject options.volsurface \
  --flush-interval 1.0 --max-batch 500

# Kafka (equity returns ticks)
python streaming_ingest.py \
  --repo ./feature-store \
  --kafka "localhost:9092" \
  --topic equities.returns \
  --flush-interval 1.0 --max-batch 1000

Notes
-----
- Events must be JSON. Minimal examples are shown in the transformers below.
- Install deps as needed:
    pip install feast aiokafka nats-py orjson pandas
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import time
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

try:
    import orjson as jsonlib  # type: ignore # fast path
    def _json_loads(b: bytes) -> Any: return jsonlib.loads(b)
except Exception:
    def _json_loads(b: bytes) -> Any: return json.loads(b.decode("utf-8"))

from feast import FeatureStore # type: ignore

# ------------------------- Config / Mappings -------------------------

@dataclass
class Route:
    """
    A route maps an incoming topic/subject to a Feast feature view,
    with functions to (a) extract entity keys and (b) feature fields.
    """
    name: str                     # route name for logs
    feature_view: str             # Feast FeatureView name
    entity_keys: List[str]        # Feast entity join keys (order matters)
    transformer: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]
    # Optional: per-route timestamp column (not required for online store)
    ts_field: Optional[str] = None

@dataclass
class Settings:
    repo_path: str
    kafka_bootstrap: Optional[str] = None
    kafka_topics: List[str] = field(default_factory=list)
    nats_url: Optional[str] = None
    nats_subjects: List[str] = field(default_factory=list)
    flush_interval: float = 1.0
    max_batch: int = 1000
    max_inflight: int = 5
    log_every: int = 1000


# ------------------------- Route Transformers ------------------------

def _hash_event_id(e: Dict[str, Any]) -> str:
    """Simple event id for de-dup in batch window."""
    h = hashlib.sha1()
    h.update(json.dumps(e, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()

# Example: equities.returns -> eq_returns_1d (online enrichment)
# Expect event:
# {
#   "ts": "2025-09-15T10:30:00Z",
#   "ticker": "AAPL",
#   "ret_1d": 0.0042, "ret_5d": 0.012, "ret_20d": 0.035,
#   "vol_20d": 0.22, "mcap_usd": 3.7e12, "sector": "TECH"
# }
def transform_equities_returns(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if "ticker" not in event:
        return None
    out = {
        "ticker": str(event["ticker"]),
        "ret_1d": float(event.get("ret_1d", 0.0)),
        "ret_5d": float(event.get("ret_5d", 0.0)),
        "ret_20d": float(event.get("ret_20d", 0.0)),
        "vol_20d": float(event.get("vol_20d", 0.0)),
        "mcap_usd": float(event.get("mcap_usd", 0.0)),
        "sector": str(event.get("sector", "")),
    }
    return out

# Example: fx.carry -> fx_carry_signals
# {
#   "ts":"2025-09-15","pair":"EURUSD","base":"EUR","quote":"USD",
#   "ir_base_1y":3.2,"ir_quote_1y":5.3,"carry_1y":-2.1,
#   "ir_base_3m":3.0,"ir_quote_3m":5.1,"carry_3m":-2.1,
#   "spot_ret_20d":-0.9,"vol_20d":5.2
# }
def transform_fx_carry(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    pair = event.get("pair")
    if not pair:
        return None
    return {
        "pair": str(pair),
        "base": str(event.get("base", "")),
        "quote": str(event.get("quote", "")),
        "ir_base_3m": float(event.get("ir_base_3m", 0.0)),
        "ir_quote_3m": float(event.get("ir_quote_3m", 0.0)),
        "ir_base_1y": float(event.get("ir_base_1y", 0.0)),
        "ir_quote_1y": float(event.get("ir_quote_1y", 0.0)),
        "carry_3m": float(event.get("carry_3m", 0.0)),
        "carry_1y": float(event.get("carry_1y", 0.0)),
        "spot_ret_20d": float(event.get("spot_ret_20d", 0.0)),
        "vol_20d": float(event.get("vol_20d", 0.0)),
    }

# Example: options.volsurface -> vol_surface
# {
#   "ts":"2025-09-15","symbol":"AAPL",
#   "iv30":34.2,"iv60":32.0,"iv90":31.4,"rv20":28.1,
#   "rr25":-2.1,"bf25":2.6,"skew_25d":-7.8
# }
def transform_vol_surface(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sym = event.get("symbol")
    if not sym:
        return None
    return {
        "symbol": str(sym),
        "iv30": float(event.get("iv30", 0.0)),
        "iv60": float(event.get("iv60", 0.0)),
        "iv90": float(event.get("iv90", 0.0)),
        "rv20": float(event.get("rv20", 0.0)),
        "rr25": float(event.get("rr25", 0.0)),
        "bf25": float(event.get("bf25", 0.0)),
        "skew_25d": float(event.get("skew_25d", 0.0)),
        "vega_notional": float(event.get("vega_notional", 0.0)) if "vega_notional" in event else None,
        "dvol": float(event.get("dvol", 0.0)) if "dvol" in event else None,
    }

# Example: credit.spreads -> credit_spreads
# {
#   "ts":"2025-09-15","issuer_id":"US123456789","sector":"TECH",
#   "oas_bps":145,"z_spread_bps":160,"duration_yrs":5.3,"rating_num":4,
#   "spread_5d_change_bps":-8,"spread_20d_change_bps":-15
# }
def transform_credit_spreads(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    issuer = event.get("issuer_id")
    if not issuer:
        return None
    return {
        "issuer_id": str(issuer),
        "sector": str(event.get("sector", "")),
        "oas_bps": float(event.get("oas_bps", 0.0)),
        "z_spread_bps": float(event.get("z_spread_bps", 0.0)),
        "duration_yrs": float(event.get("duration_yrs", 0.0)),
        "rating_num": int(event.get("rating_num", 0)),
        "spread_5d_change_bps": float(event.get("spread_5d_change_bps", 0.0)),
        "spread_20d_change_bps": float(event.get("spread_20d_change_bps", 0.0)),
    }


# ------------------------- Route Table (edit me) ---------------------

ROUTES: Dict[str, Route] = {
    # Kafka topic or NATS subject name -> Route
    "equities.returns": Route(
        name="eq_returns_stream",
        feature_view="eq_returns_1d",
        entity_keys=["ticker"],
        transformer=transform_equities_returns,
        ts_field="ts",
    ),
    "fx.carry": Route(
        name="fx_carry_stream",
        feature_view="fx_carry_signals",
        entity_keys=["pair"],
        transformer=transform_fx_carry,
        ts_field="ts",
    ),
    "options.volsurface": Route(
        name="vol_surface_stream",
        feature_view="vol_surface",
        entity_keys=["symbol"],
        transformer=transform_vol_surface,
        ts_field="ts",
    ),
    "credit.spreads": Route(
        name="credit_spreads_stream",
        feature_view="credit_spreads",
        entity_keys=["issuer_id"],
        transformer=transform_credit_spreads,
        ts_field="ts",
    ),
}

# ------------------------- Batch Writer (Feast) ----------------------

class FeastBatchWriter:
    def __init__(self, repo_path: str, max_batch: int = 1000, flush_interval: float = 1.0, log_every: int = 1000):
        self.store = FeatureStore(repo_path=repo_path)
        self.max_batch = max_batch
        self.flush_interval = flush_interval
        self.log_every = log_every

        # route -> list of rows (dict)
        self.buffers: Dict[str, List[Dict[str, Any]]] = {k: [] for k in ROUTES.keys()}
        self.last_flush = time.time()
        self._seen_ids: Dict[str, set] = {k: set() for k in ROUTES.keys()}  # simple per-buffer de-dup

        self._rows_written = 0

    def add(self, route_key: str, row: Dict[str, Any], event_id: Optional[str] = None):
        if route_key not in self.buffers:
            self.buffers[route_key] = []
            self._seen_ids[route_key] = set()

        # de-dup within batch window
        if event_id:
            s = self._seen_ids[route_key]
            if event_id in s:
                return
            s.add(event_id)

        self.buffers[route_key].append(row)

    def _should_flush(self) -> bool:
        if sum(len(b) for b in self.buffers.values()) >= self.max_batch:
            return True
        if (time.time() - self.last_flush) >= self.flush_interval:
            return True
        return False

    async def maybe_flush(self):
        if not self._should_flush():
            return
        await self.flush()

    async def flush(self):
        total = 0
        for topic, rows in list(self.buffers.items()):
            if not rows:
                continue
            route = ROUTES.get(topic)
            if not route:
                continue

            # Feast write_to_online_store expects a DataFrame with entity key columns + feature columns
            try:
                df = pd.DataFrame(rows)
                if df.empty:
                    continue

                # Write per feature view
                self.store.write_to_online_store(route.feature_view, df)
                n = len(df)
                total += n
                self._rows_written += n
                self.buffers[topic].clear()
                self._seen_ids[topic].clear()
            except Exception as e:
                # Do not drop the buffer; log and keep so we can retry on next flush
                print(f"[ERROR] write_to_online_store failed for {route.feature_view}: {e}")

        self.last_flush = time.time()
        if total and (self._rows_written % self.log_every == 0 or total >= self.log_every):
            print(f"[INGEST] wrote {total} rows (cumulative {self._rows_written})")

    async def close(self):
        await self.flush()


# ------------------------- Consumers (Kafka/NATS) --------------------

async def run_kafka(settings: Settings, writer: FeastBatchWriter):
    try:
        from aiokafka import AIOKafkaConsumer # type: ignore
    except Exception:
        raise SystemExit("Kafka selected but aiokafka not installed. pip install aiokafka")

    consumer = AIOKafkaConsumer(
        *settings.kafka_topics,
        bootstrap_servers=settings.kafka_bootstrap, # type: ignore
        enable_auto_commit=True,
        auto_offset_reset="latest",
        value_deserializer=lambda v: v,  # bytes in, decode below with orjson
        key_deserializer=lambda v: v,
        max_poll_records=settings.max_batch,
    )
    await consumer.start()
    print(f"[KAFKA] connected -> {settings.kafka_bootstrap} topics={settings.kafka_topics}")
    try:
        while True:
            msgs = await consumer.getmany(timeout_ms=int(settings.flush_interval * 1000))
            for tp, batch in msgs.items():
                topic = tp.topic
                route = ROUTES.get(topic)
                if not route:
                    continue
                for msg in batch:
                    try:
                        event = _json_loads(msg.value if isinstance(msg.value, (bytes, bytearray)) else bytes(msg.value)) # type: ignore
                        row = route.transformer(event)
                        if row is None:
                            continue
                        event_id = event.get("event_id") or _hash_event_id(event)
                        writer.add(topic, row, event_id=event_id)
                    except Exception as e:
                        print(f"[KAFKA][{topic}] parse/transform error: {e}")
            await writer.maybe_flush()
    except asyncio.CancelledError:
        pass
    finally:
        await writer.close()
        await consumer.stop()
        print("[KAFKA] stopped.")


async def run_nats(settings: Settings, writer: FeastBatchWriter):
    try:
        import nats
    except Exception:
        raise SystemExit("NATS selected but nats-py not installed. pip install nats-py")

    nc = await nats.connect(settings.nats_url) # type: ignore
    print(f"[NATS] connected -> {settings.nats_url} subjects={settings.nats_subjects}")

    async def handler(msg):
        subject = msg.subject
        route = ROUTES.get(subject)
        if not route:
            return
        try:
            event = _json_loads(msg.data)
            row = route.transformer(event)
            if row is None:
                return
            event_id = event.get("event_id") or _hash_event_id(event)
            writer.add(subject, row, event_id=event_id)
        except Exception as e:
            print(f"[NATS][{subject}] parse/transform error: {e}")

    subs = []
    for subj in settings.nats_subjects:
        subs.append(await nc.subscribe(subj, cb=handler))

    try:
        while True:
            await asyncio.sleep(settings.flush_interval / 2)
            await writer.maybe_flush()
    except asyncio.CancelledError:
        pass
    finally:
        for s in subs:
            await s.unsubscribe()
        await writer.close()
        await nc.drain()
        print("[NATS] stopped.")


# ------------------------------ CLI ---------------------------------

def parse_args() -> Settings:
    import argparse
    p = argparse.ArgumentParser("Streaming ingest -> Feast online store")
    p.add_argument("--repo", dest="repo_path", required=True, help="Path to Feast repo (contains repo.yaml)")
    p.add_argument("--kafka", dest="kafka_bootstrap", help="Kafka bootstrap servers host:port")
    p.add_argument("--topic", dest="kafka_topics", action="append", default=[], help="Kafka topic (repeatable)")
    p.add_argument("--nats", dest="nats_url", help="NATS URL, e.g., nats://localhost:4222")
    p.add_argument("--subject", dest="nats_subjects", action="append", default=[], help="NATS subject (repeatable)")
    p.add_argument("--flush-interval", type=float, default=1.0, help="Seconds between flushes")
    p.add_argument("--max-batch", type=int, default=1000, help="Max rows buffered before force flush")
    p.add_argument("--max-inflight", type=int, default=5, help="(Reserved) inflight batches")
    p.add_argument("--log-every", type=int, default=1000, help="Log every N rows written")
    args = p.parse_args()

    if not args.kafka_bootstrap and not args.nats_url:
        raise SystemExit("Select a transport: --kafka ... or --nats ...")
    if args.kafka_bootstrap and not args.kafka_topics:
        raise SystemExit("Kafka selected but no --topic provided.")
    if args.nats_url and not args.nats_subjects:
        raise SystemExit("NATS selected but no --subject provided.")

    return Settings(
        repo_path=args.repo_path,
        kafka_bootstrap=args.kafka_bootstrap,
        kafka_topics=args.kafka_topics,
        nats_url=args.nats_url,
        nats_subjects=args.nats_subjects,
        flush_interval=args.flush_interval,
        max_batch=args.max_batch,
        max_inflight=args.max_inflight,
        log_every=args.log_every,
    )

# ----------------------------- Main ---------------------------------

async def main_async():
    settings = parse_args()
    writer = FeastBatchWriter(
        repo_path=settings.repo_path,
        max_batch=settings.max_batch,
        flush_interval=settings.flush_interval,
        log_every=settings.log_every,
    )

    stop = asyncio.Event()

    def _graceful(*_):
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _graceful)
        except NotImplementedError:
            pass

    if settings.kafka_bootstrap:
        task = asyncio.create_task(run_kafka(settings, writer))
    else:
        task = asyncio.create_task(run_nats(settings, writer))

    await stop.wait()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()