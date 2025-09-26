# tets dlq.py
# A self-contained DLQ (Dead-Letter Queue) test suite with a tiny in-memory
# implementation so you can run it without Kafka/SQS/RabbitMQ.
#
# Supports:
#  - Retry with exponential backoff (simulated clock)
#  - Max attempts -> move to DLQ
#  - Visibility timeout (in-flight lock)
#  - Idempotency via message-id de-dup
#  - Poison-pill detection (non-retryable errors)
#
# How to run:
#   pytest -q "tets dlq.py"
#
# If you want to plug in your own queue/consumer, replace InMemoryQueue/Consumer
# with adapters that satisfy the same method signatures and keep the tests.

from __future__ import annotations

import time
import uuid
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import unittest


# =========================
# Minimal in-memory queue
# =========================
@dataclass
class Message:
    id: str
    body: Any
    attributes: Dict[str, Any] = field(default_factory=dict)


class Clock:
    """Injectable time source for deterministic tests."""
    def __init__(self, start: float = 0.0):
        self._t = start

    def time(self) -> float:
        return self._t

    def sleep(self, dt: float) -> None:
        self._t += dt


class InMemoryQueue:
    """
    SQS/Rabbit-like semantics:
      - send(message)
      - receive(max_messages, visibility_timeout)
      - delete(receipt_handle)
      - change_visibility(receipt_handle, timeout)
    Messages become invisible while "in flight" and reappear when visibility expires.
    """
    def __init__(self, clock: Clock):
        self.clock = clock
        self._store: Dict[str, Dict[str, Any]] = {}  # msg_id -> record
        self._inflight: Dict[str, Dict[str, Any]] = {}  # receipt -> record
        self._order: List[str] = []  # approximate FIFO by id order

    def send(self, body: Any, attributes: Optional[Dict[str, Any]] = None) -> str:
        msg_id = str(uuid.uuid4())
        self._store[msg_id] = {
            "msg": Message(id=msg_id, body=body, attributes=attributes or {}),
            "visible_at": self.clock.time(),
            "attempts": 0,
        }
        self._order.append(msg_id)
        return msg_id

    def _visible_ids(self) -> List[str]:
        t = self.clock.time()
        return [mid for mid in self._order if mid in self._store and self._store[mid]["visible_at"] <= t]

    def receive(self, max_messages: int = 1, visibility_timeout: float = 30.0) -> List[Tuple[str, Message]]:
        out: List[Tuple[str, Message]] = []
        for mid in list(self._visible_ids())[:max_messages]:
            rec = self._store[mid]
            rec["attempts"] += 1
            rec["visible_at"] = self.clock.time() + visibility_timeout
            receipt = str(uuid.uuid4())
            self._inflight[receipt] = rec
            out.append((receipt, rec["msg"]))
        return out

    def delete(self, receipt: str) -> None:
        rec = self._inflight.pop(receipt, None)
        if not rec:
            return
        mid = rec["msg"].id
        self._store.pop(mid, None)
        if mid in self._order:
            self._order.remove(mid)

    def change_visibility(self, receipt: str, timeout: float) -> None:
        rec = self._inflight.get(receipt)
        if rec:
            rec["visible_at"] = self.clock.time() + timeout

    # Introspection helpers for tests
    def size(self) -> int:
        return len(self._store)

    def attempts(self, msg_id: str) -> int:
        return self._store[msg_id]["attempts"] if msg_id in self._store else 0

    def contains(self, msg_id: str) -> bool:
        return msg_id in self._store


# =========================
# Dead Letter Queue
# =========================
class DeadLetterQueue:
    def __init__(self):
        self.messages: List[Message] = []

    def send(self, msg: Message, reason: str, attrs_extra: Optional[Dict[str, Any]] = None) -> None:
        meta = dict(msg.attributes)
        meta["dlq_reason"] = reason
        if attrs_extra:
            meta.update(attrs_extra)
        self.messages.append(Message(id=msg.id, body=msg.body, attributes=meta))

    def size(self) -> int:
        return len(self.messages)


# =========================
# Consumer with retries
# =========================
class NonRetryableError(Exception):
    """Poison pill; do not retry."""


@dataclass
class RetryPolicy:
    max_attempts: int = 5              # total deliveries (1st + retries)
    base_visibility: float = 5.0       # seconds
    backoff_base: float = 2.0          # exponential backoff multiplier
    backoff_jitter: float = 0.0        # deterministic here; add randomness in prod
    max_visibility: float = 60.0       # cap

    def compute_visibility(self, attempt: int) -> float:
        # attempt is 1-based
        timeout = self.base_visibility * (self.backoff_base ** (attempt - 1))
        return min(self.max_visibility, timeout + self.backoff_jitter)


class Consumer:
    def __init__(
        self,
        queue: InMemoryQueue,
        dlq: DeadLetterQueue,
        handler: Callable[[Message], None],
        retry: RetryPolicy,
        clock: Clock,
        dedupe_ttl_sec: float = 600,
    ):
        self.q = queue
        self.dlq = dlq
        self.handler = handler
        self.retry = retry
        self.clock = clock
        self._seen: Dict[str, float] = {}  # msg_id -> last_seen_ts
        self._dedupe_ttl = dedupe_ttl_sec

    def _clean_seen(self):
        now = self.clock.time()
        expired = [mid for mid, ts in self._seen.items() if now - ts > self._dedupe_ttl]
        for mid in expired:
            self._seen.pop(mid, None)

    def poll_once(self, batch: int = 10):
        self._clean_seen()
        deliveries = self.q.receive(max_messages=batch, visibility_timeout=self.retry.base_visibility)
        for receipt, msg in deliveries:
            try:
                # Idempotency gate: skip if recently processed
                if msg.id in self._seen:
                    # Already processed successfully; ack duplicate delivery
                    self.q.delete(receipt)
                    continue

                self.handler(msg)  # may raise

                # success -> ack and mark seen
                self.q.delete(receipt)
                self._seen[msg.id] = self.clock.time()

            except NonRetryableError as e:
                # Poison: straight to DLQ
                self.dlq.send(msg, reason="non_retryable", attrs_extra={"error": str(e)})
                self.q.delete(receipt)
            except Exception as e:
                # retryable
                # Look up current attempts; if exceed max -> DLQ
                # We can't read attempts directly via receipt; consult queue store
                # (In real broker, attempts delivered via header)
                # Here: the queue increments attempts at receive().
                # Try to find the record by msg.id
                attempts = self.q.attempts(msg.id)
                if attempts >= self.retry.max_attempts:
                    self.dlq.send(msg, reason="max_attempts_exceeded", attrs_extra={"error": str(e), "attempts": attempts})
                    self.q.delete(receipt)
                else:
                    # push back by changing visibility per backoff
                    next_vis = self.retry.compute_visibility(attempts)
                    self.q.change_visibility(receipt, next_vis)


# =========================
# Tests
# =========================
class TestDLQ(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.queue = InMemoryQueue(self.clock)
        self.dlq = DeadLetterQueue()

    def test_success_path_ack_and_dedupe(self):
        # Handler that succeeds
        seen: List[str] = []
        def ok_handler(m: Message):
            seen.append(m.id)

        msg_id = self.queue.send({"x": 1})
        consumer = Consumer(self.queue, self.dlq, ok_handler, RetryPolicy(max_attempts=3, base_visibility=10), self.clock)

        # First poll processes and deletes
        consumer.poll_once()
        self.assertFalse(self.queue.contains(msg_id))
        self.assertEqual(self.dlq.size(), 0)
        self.assertEqual(len(seen), 1)

        # Simulate accidental redelivery (message re-sent with same id)
        # In this in-memory queue we can't push with same id via send(),
        # so simulate by re-adding the message in store directly.
        self.queue._store[msg_id] = {
            "msg": Message(id=msg_id, body={"x": 1}),
            "visible_at": self.clock.time(),
            "attempts": 0,
        }
        self.queue._order.append(msg_id)

        consumer.poll_once()
        # Consumer should dedupe and delete it immediately
        self.assertFalse(self.queue.contains(msg_id))
        self.assertEqual(len(seen), 1)

    def test_retry_then_dlq_after_max_attempts(self):
        # Handler that always fails retryably
        def bad_handler(_m: Message):
            raise RuntimeError("boom")

        msg_id = self.queue.send("hello")
        retry = RetryPolicy(max_attempts=4, base_visibility=5, backoff_base=2, max_visibility=40)
        consumer = Consumer(self.queue, self.dlq, bad_handler, retry, self.clock)

        # Each poll_once will receive at most once while visible.
        # We simulate time passing to re-deliver after visibility expiration.
        # Attempt 1
        consumer.poll_once()
        self.assertTrue(self.queue.contains(msg_id))
        self.assertEqual(self.dlq.size(), 0)

        # Visibility for attempt 1 is base_visibility (5s) -> next delivery after 5s
        self.clock.sleep(5.01)
        consumer.poll_once()  # Attempt 2
        self.clock.sleep(retry.compute_visibility(2) + 0.01)  # backoff for attempt 2
        consumer.poll_once()  # Attempt 3
        self.clock.sleep(retry.compute_visibility(3) + 0.01)
        consumer.poll_once()  # Attempt 4 -> should exceed max and go to DLQ this cycle or next

        # After attempt 4 failure, we either DLQ immediately (>= max) or schedule another vis then DLQ.
        # Our implementation DLQs immediately when attempts >= max_attempts.
        self.assertFalse(self.queue.contains(msg_id))
        self.assertEqual(self.dlq.size(), 1)
        self.assertEqual(self.dlq.messages[0].id, msg_id)
        self.assertEqual(self.dlq.messages[0].attributes.get("dlq_reason"), "max_attempts_exceeded")

    def test_poison_pill_non_retryable(self):
        def poison(_m: Message):
            raise NonRetryableError("validation error")

        mid = self.queue.send({"order_id": 1})
        consumer = Consumer(self.queue, self.dlq, poison, RetryPolicy(max_attempts=5, base_visibility=3), self.clock)
        consumer.poll_once()

        self.assertFalse(self.queue.contains(mid))
        self.assertEqual(self.dlq.size(), 1)
        self.assertEqual(self.dlq.messages[0].attributes["dlq_reason"], "non_retryable")
        self.assertIn("validation error", self.dlq.messages[0].attributes.get("error", ""))

    def test_visibility_timeout_requeues_when_not_deleted(self):
        # Handler that times out (simulated by doing nothing and not deleting)
        # We emulate time-based redelivery by NOT calling delete and letting visibility expire.
        # Our Consumer always acks or changes visibility; to test the queue behavior, call queue directly.
        msg_id = self.queue.send("slow job")
        # receive with small visibility
        recs = self.queue.receive(max_messages=1, visibility_timeout=2.0)
        self.assertEqual(len(recs), 1)
        receipt, msg = recs[0]
        self.assertEqual(msg.id, msg_id)

        # Not deleting; advance time -> message should be visible again
        self.clock.sleep(2.01)
        recs2 = self.queue.receive(max_messages=1, visibility_timeout=2.0)
        self.assertEqual(len(recs2), 1)
        self.assertEqual(recs2[0][1].id, msg_id)

    def test_backoff_grows_and_caps(self):
        rp = RetryPolicy(max_attempts=10, base_visibility=3, backoff_base=3, max_visibility=20)
        vis = [rp.compute_visibility(i) for i in range(1, 7)]
        # 3, 9, 27->cap 20, 20, 20, 20
        self.assertEqual(vis[0], 3)
        self.assertEqual(vis[1], 9)
        self.assertEqual(vis[2], 20)
        self.assertTrue(all(v <= 20 for v in vis))

    def test_idempotency_window_eviction(self):
        # Prove that dedupe cache expires after ttl, allowing reprocessing
        processed: List[str] = []
        def ok(m: Message):
            processed.append(m.id)

        mid = self.queue.send("X")
        consumer = Consumer(self.queue, self.dlq, ok, RetryPolicy(max_attempts=2, base_visibility=2), self.clock, dedupe_ttl_sec=10)

        consumer.poll_once()  # process -> seen
        self.assertEqual(processed.count(mid), 1)

        # reinsert same message id (simulate replay)
        self.queue._store[mid] = {"msg": Message(id=mid, body="X"), "visible_at": self.clock.time(), "attempts": 0}
        self.queue._order.append(mid)

        # still within ttl -> dedupe drop
        consumer.poll_once()
        self.assertEqual(processed.count(mid), 1)

        # advance beyond TTL -> should process again
        self.clock.sleep(10.01)
        # reinsert again
        self.queue._store[mid] = {"msg": Message(id=mid, body="X"), "visible_at": self.clock.time(), "attempts": 0}
        self.queue._order.append(mid)
        consumer.poll_once()
        self.assertEqual(processed.count(mid), 2)


# PyTest bridge (so `pytest -q "tets dlq.py"` works cleanly)
def test_pytest_bridge():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestDLQ)
    res = unittest.TextTestRunner(verbosity=0).run(suite)
    assert res.wasSuccessful(), "DLQ tests failed"