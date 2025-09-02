# tests/test_worker_message_flow.py
from __future__ import annotations

import json
import time
from typing import Dict, Any, List

import pytest # type: ignore

try:
    import fakeredis  # pip install fakeredis
except ImportError as e:
    raise SystemExit("Please `pip install fakeredis` to run this test") from e


# --- Try to use your real envelope; fall back to a tiny local stub ----------
try:
    from platform import envelope as env  # type: ignore # your repo's platform/envelope.py
except Exception:
    # Minimal stand-in so the test still runs without your module
    class _Env:
        def new(self, schema_name: str, payload: Dict[str, Any], producer: Dict[str, Any]):
            body = {
                "schema": schema_name,
                "payload": payload,
                "producer": producer,
                "ts": int(time.time() * 1000),
            }
            return _Envelope(body)

    class _Envelope:
        def __init__(self, body: Dict[str, Any]):
            self.body = body

        def flatten_for_stream(self) -> Dict[str, str]:
            # flatten into string fields for XADD
            return {k: json.dumps(v) if not isinstance(v, str) else v for k, v in self.body.items()}

        @staticmethod
        def parse_from_stream(fields: Dict[bytes, bytes]) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for k, v in fields.items():
                ks = k.decode()
                vs = v.decode()
                try:
                    out[ks] = json.loads(vs)
                except Exception:
                    out[ks] = vs
            return out

    env = _Env()  # type: ignore


# ----------------------------- Test helpers --------------------------------

IN_STREAM = "STREAM_SENTIMENT_REQUESTS"
OUT_STREAM = "STREAM_SENTIMENT_RESULTS"
DLQ_STREAM = "STREAM_DLQ"
GROUP = "sentiment_worker_test"
CONSUMER = "c1"


def publish_requests(r, texts: List[str], svc="test-publisher") -> List[str]:
    ids = []
    e = env.new(
        schema_name="sentiment.request",
        payload={"texts": texts, "language": "en", "meta": {"source": "unit-test"}},
        producer={"svc": svc, "roles": ["research"]},
    )
    for t in texts:
        body = dict(e.body)  # shallow copy
        body["payload"] = dict(e.body["payload"])
        body["payload"]["texts"] = [t]
        msg = _Envelope(body) if "_Envelope" in globals() else e  # local or real
        mid = r.xadd(IN_STREAM, msg.flatten_for_stream())
        ids.append(mid)
    return ids


def _parse_stream_msg(fields: Dict[bytes, bytes]) -> Dict[str, Any]:
    if hasattr(env, "parse_from_stream"):  # if your env exposes one
        try:
            return env.parse_from_stream(fields)  # type: ignore[attr-defined]
        except Exception:
            pass
    # fallback parser (works with the local stub)
    return _Envelope.parse_from_stream(fields)  # type: ignore[name-defined]


def worker_once(r) -> int:
    """
    Mimic a sentiment worker iteration:
      - XREADGROUP 1 message
      - parse envelope
      - "process" by computing a toy sentiment score
      - write a result envelope to OUT_STREAM
      - XACK original
    Returns number of messages processed.
    """
    resp = r.xreadgroup(GROUP, CONSUMER, {IN_STREAM: ">"}, count=1, block=10)
    if not resp:
        return 0

    _, messages = resp[0]
    processed = 0
    for mid, fields in messages:
        try:
            data = _parse_stream_msg(fields)
            payload = data.get("payload", {})
            texts: List[str] = payload.get("texts", [])
            text = texts[0] if texts else ""

            # Toy "sentiment": positive if contains 'up', negative if 'down', else 0
            score = 1.0 if "up" in text.lower() else (-1.0 if "down" in text.lower() else 0.0)

            out_env = env.new(
                schema_name="sentiment.result",
                payload={"text": text, "score": score, "lang": payload.get("language", "en")},
                producer={"svc": "sentiment-worker-test", "roles": ["research"]},
            )
            r.xadd(OUT_STREAM, out_env.flatten_for_stream())
            r.xack(IN_STREAM, GROUP, mid)
            processed += 1

        except Exception as ex:
            # Push to DLQ and ack original
            dlq_env = env.new(
                schema_name="dlq.error",
                payload={"error": str(ex)},
                producer={"svc": "sentiment-worker-test"},
            )
            r.xadd(DLQ_STREAM, dlq_env.flatten_for_stream())
            r.xack(IN_STREAM, GROUP, mid)
    return processed


# --------------------------------- Tests -----------------------------------

def test_message_flow_happy_path():
    r = fakeredis.FakeRedis()
    # Create stream & group (MKSTREAM=True to create if missing)
    try:
        r.xgroup_create(IN_STREAM, GROUP, id="$", mkstream=True)
    except Exception:
        pass  # group may already exist

    texts = ["Stock surges up on earnings", "Guidance cut sends stock down", "Board announces dividend"]
    publish_requests(r, texts)

    # Process all messages
    total = 0
    while total < len(texts):
        total += worker_once(r)

    # Verify OUT stream has 3 results and DLQ is empty
    out_len = r.xlen(OUT_STREAM)
    dlq_len = r.xlen(DLQ_STREAM)
    assert out_len == len(texts), f"Expected {len(texts)} results, got {out_len}"
    assert dlq_len == 0

    # Inspect last result for structure
    last_id, last_fields = r.xrevrange(OUT_STREAM, count=1)[0]#type:ignore
    parsed = _parse_stream_msg(last_fields)
    assert parsed["schema"] in ("sentiment.result", "sentiment.result".encode() if isinstance(parsed["schema"], bytes) else "sentiment.result")
    assert "payload" in parsed and "score" in parsed["payload"]


def test_message_flow_dlq_on_error():
    r = fakeredis.FakeRedis()
    try:
        r.xgroup_create(IN_STREAM, GROUP, id="$", mkstream=True)
    except Exception:
        pass

    # Publish one "bad" message with malformed payload (texts not a list)
    bad_env = env.new(
        schema_name="sentiment.request",
        payload={"texts": "NOT A LIST", "language": "en"},
        producer={"svc": "test-publisher"},
    )
    r.xadd(IN_STREAM, bad_env.flatten_for_stream())#type:ignore

    # Process once; should go to DLQ and ack the original
    processed = worker_once(r)
    assert processed == 0  # processing failed → goes to DLQ, not counted as success
    assert r.xlen(DLQ_STREAM) == 1

    # No pending entries in the group (acked)
    pending = r.xpending_range(IN_STREAM, GROUP, min="-", max="+", count=10)
    assert len(pending) == 0 # type: ignore


def test_idempotency_ack_and_no_duplicates():
    r = fakeredis.FakeRedis()
    try:
        r.xgroup_create(IN_STREAM, GROUP, id="$", mkstream=True)
    except Exception:
        pass

    publish_requests(r, ["up", "down"])

    # Simulate two workers pulling the same idle message then only one acking
    resp = r.xreadgroup(GROUP, "w1", {IN_STREAM: ">"}, count=1)
    assert resp
    mid1 = resp[0][1][0][0]#type:ignore

    # Another consumer tries to read new messages
    resp2 = r.xreadgroup(GROUP, "w2", {IN_STREAM: ">"}, count=10)
    # It should get the second, not the first one (since it's already delivered to w1)
    ids2 = [m[0] for _, msgs in resp2 for m in msgs] # pyright: ignore[reportGeneralTypeIssues]
    assert mid1 not in ids2

    # Finish processing properly
    # put the first message back via worker_once (it will read the second if available)
    total = 0
    while total < 2:
        total += worker_once(r)

    # After processing both, no pending
    assert r.xpending(IN_STREAM, GROUP)["pending"] == 0#type:ignore
    assert r.xlen(OUT_STREAM) >= 2#type:ignore
    