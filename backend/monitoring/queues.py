# backend/bus/queues.py
from __future__ import annotations

import os, json, time, uuid, heapq, random, threading
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, Tuple, List, Protocol, runtime_checkable

# -------- optional Redis (graceful fallback) ---------------------------------
HAVE_REDIS = True
try:
    from redis import Redis  # type: ignore
except Exception:
    HAVE_REDIS = False
    Redis = None  # type: ignore

# -------- env / defaults -----------------------------------------------------
REDIS_URL          = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DEFAULT_LEASE_MS   = int(os.getenv("QUEUE_LEASE_MS", "30000"))    # visibility timeout
DEFAULT_MAX_ATTEMPTS = int(os.getenv("QUEUE_MAX_ATTEMPTS", "5"))
DEFAULT_BACKOFF_MS = int(os.getenv("QUEUE_BACKOFF_MS", "1000"))
DEFAULT_BACKOFF_MAX_MS = int(os.getenv("QUEUE_BACKOFF_MAX_MS", "120000"))
IDEMP_TTL_SEC      = int(os.getenv("QUEUE_IDEMP_TTL_SEC", "86400"))

# Priority lanes: high -> normal -> low
PRIORITY_LANES = ("h", "n", "l")

def now_ms() -> int: return int(time.time() * 1000)

# -------- models -------------------------------------------------------------
@dataclass
class Job:
    queue: str
    id: str
    payload: Dict[str, Any]
    ts_ms: int = field(default_factory=now_ms)
    attempt: int = 0
    not_before_ms: int = 0              # for delayed jobs
    priority: str = "n"                 # "h" | "n" | "l"
    idem_key: Optional[str] = None      # dedupe window
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def from_json(s: str) -> "Job":
        d = json.loads(s)
        return Job(**d)

# -------- protocol -----------------------------------------------------------
@runtime_checkable
class QueueBackend(Protocol):
    def enqueue(self, payload: Dict[str, Any], *, priority: str = "n",
                delay_ms: int = 0, idem_key: Optional[str] = None,
                meta: Optional[Dict[str, Any]] = None) -> Job: ...
    def poll(self, *, lease_ms: int = DEFAULT_LEASE_MS) -> Optional[Tuple[Job, str]]:
        """Return (job, raw_token_for_ack) or None."""
    def ack(self, token: str) -> None: ...
    def nack(self, token: str, *, requeue: bool = True) -> None: ...
    def size(self) -> Dict[str, int]: ...
    def dead_letter(self, job: Job, reason: str) -> None: ...
    def purge(self) -> None: ...

# =============================================================================
# In-memory implementation (good for tests, dev, unit runners)
# =============================================================================
class InMemoryQueue(QueueBackend):
    """
    Priority queue with delay & visibility timeout, single-process.
    """
    def __init__(self, name: str, *, max_attempts: int = DEFAULT_MAX_ATTEMPTS):
        self.name = name
        self.max_attempts = max_attempts

        # ready lanes: high/normal/low → lists of (start order, Job)
        self._ready: Dict[str, List[Tuple[int, Job]]] = {p: [] for p in PRIORITY_LANES}
        self._seq = 0

        # delayed min-heap of (ready_at_ms, seq, Job)
        self._delayed: List[Tuple[int, int, Job]] = []

        # inflight: token -> (deadline_ms, Job)
        self._inflight: Dict[str, Tuple[int, Job]] = {}

        # idempotency window
        self._idem: Dict[str, int] = {}

        # dead-letter
        self._dead: List[Job] = []

        # lock
        self._lock = threading.Lock()

    # ---- helpers
    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _release_due(self):
        now = now_ms()
        while self._delayed and self._delayed[0][0] <= now:
            _, _, job = heapq.heappop(self._delayed)
            heap = self._ready[job.priority]
            heap.append((self._next_seq(), job))

    def _requeue_expired(self):
        now = now_ms()
        # copy keys to avoid mutation during iteration
        for token, (deadline, job) in list(self._inflight.items()):
            if deadline <= now:
                # visibility timeout expired -> requeue (increment attempt)
                job.attempt += 1
                self._inflight.pop(token, None)
                if job.attempt >= self.max_attempts:
                    self._dead.append(job)
                else:
                    backoff = self._backoff_ms(job.attempt)
                    heapq.heappush(self._delayed, (now + backoff, self._next_seq(), job))

        # purge expired idempotency keys
        for k, t in list(self._idem.items()):
            if t <= now:
                self._idem.pop(k, None)

    def _backoff_ms(self, attempt: int) -> int:
        # exponential backoff with jitter
        base = min(DEFAULT_BACKOFF_MAX_MS, DEFAULT_BACKOFF_MS * (2 ** (attempt - 1)))
        jitter = int(base * random.uniform(0.1, 0.25))
        return base + jitter

    # ---- public API
    def enqueue(self, payload: Dict[str, Any], *, priority: str = "n",
                delay_ms: int = 0, idem_key: Optional[str] = None,
                meta: Optional[Dict[str, Any]] = None) -> Job:
        with self._lock:
            if idem_key:
                ttl_at = self._idem.get(idem_key)
                if ttl_at and ttl_at > now_ms():
                    # duplicate ignored; return a shadow job (no enqueue)
                    return Job(queue=self.name, id=f"dup-{uuid.uuid4().hex[:8]}", payload=payload,
                               priority=priority, idem_key=idem_key, meta={"ignored": True})
                self._idem[idem_key] = now_ms() + IDEMP_TTL_SEC * 1000

            job = Job(queue=self.name, id=uuid.uuid4().hex, payload=payload,
                      not_before_ms=now_ms() + max(0, delay_ms), priority=priority, idem_key=idem_key,
                      meta=meta or {})
            if delay_ms > 0:
                heapq.heappush(self._delayed, (job.not_before_ms, self._next_seq(), job))
            else:
                self._ready[priority].append((self._next_seq(), job))
            return job

    def poll(self, *, lease_ms: int = DEFAULT_LEASE_MS) -> Optional[Tuple[Job, str]]:
        with self._lock:
            self._release_due()
            self._requeue_expired()
            # priority order
            for lane in ("h", "n", "l"):
                if self._ready[lane]:
                    _, job = self._ready[lane].pop(0)
                    token = f"{job.id}:{uuid.uuid4().hex[:6]}"
                    self._inflight[token] = (now_ms() + max(100, lease_ms), job)
                    return job, token
            return None

    def ack(self, token: str) -> None:
        with self._lock:
            self._inflight.pop(token, None)

    def nack(self, token: str, *, requeue: bool = True) -> None:
        with self._lock:
            t = self._inflight.pop(token, None)
            if not t:
                return
            _, job = t
            job.attempt += 1
            if not requeue or job.attempt >= self.max_attempts:
                self._dead.append(job)
                return
            backoff = self._backoff_ms(job.attempt)
            heapq.heappush(self._delayed, (now_ms() + backoff, self._next_seq(), job))

    def size(self) -> Dict[str, int]:
        with self._lock:
            ready = sum(len(v) for v in self._ready.values())
            return {"ready": ready, "delayed": len(self._delayed), "inflight": len(self._inflight), "dead": len(self._dead)}

    def dead_letter(self, job: Job, reason: str) -> None:
        with self._lock:
            job.meta["dead_reason"] = reason
            self._dead.append(job)

    def purge(self) -> None:
        with self._lock:
            for k in self._ready: self._ready[k].clear()
            self._delayed.clear()
            self._inflight.clear()
            self._dead.clear()
            self._idem.clear()

# =============================================================================
# Redis implementation (multi-process, simple and robust)
# Ready lists per priority; delayed & inflight tracked via ZSETs for deadlines.
# =============================================================================
class RedisQueue(QueueBackend):
    """
    Uses:
      - LPUSH/BRPOP-ready lists per priority: q:<name>:ready:<lane>
      - RPOPLPUSH to claim atomically into q:<name>:inflight (a list)
      - ZSET q:<name>:delayed (score=ready_at_ms)
      - ZSET q:<name>:leases (score=deadline_ms, member=raw json)
      - SETNX q:<name>:idem:<key> for idempotency (TTL)
      - LIST q:<name>:dead for dead-letter
    """
    def __init__(self, name: str, *, redis_url: Optional[str] = None,
                 max_attempts: int = DEFAULT_MAX_ATTEMPTS):
        if not HAVE_REDIS:
            raise RuntimeError("Redis not available")
        self.name = name
        self.max_attempts = max_attempts
        self.r = Redis.from_url(redis_url or REDIS_URL, decode_responses=True)  # type: ignore
        # keys
        self.k_ready = {p: f"q:{name}:ready:{p}" for p in PRIORITY_LANES}
        self.k_inflight = f"q:{name}:inflight"
        self.k_delayed = f"q:{name}:delayed"
        self.k_leases = f"q:{name}:leases"
        self.k_dead = f"q:{name}:dead"
        self.k_idem_pref = f"q:{name}:idem:"
        # small lock for local sequencing of maintenance
        self._local_lock = threading.Lock()

    # ---- helpers
    def _backoff_ms(self, attempt: int) -> int:
        base = min(DEFAULT_BACKOFF_MAX_MS, DEFAULT_BACKOFF_MS * (2 ** (attempt - 1)))
        jitter = int(base * random.uniform(0.1, 0.25))
        return base + jitter

    def _release_due(self):
        now = now_ms()
        # fetch due delayed jobs
        due = self.r.zrangebyscore(self.k_delayed, 0, now, start=0, num=200)  # type: ignore
        if not due: return
        pipe = self.r.pipeline()
        for raw in due: # type: ignore
            try:
                job = Job.from_json(raw)
            except Exception:
                # skip bad record
                pipe.zrem(self.k_delayed, raw)
                continue
            pipe.lpush(self.k_ready[job.priority], raw)
            pipe.zrem(self.k_delayed, raw)
        pipe.execute()

    def _requeue_expired(self):
        now = now_ms()
        expired = self.r.zrangebyscore(self.k_leases, 0, now, start=0, num=200)  # type: ignore
        if not expired: return
        pipe = self.r.pipeline()
        for raw in expired: # type: ignore
            try:
                job = Job.from_json(raw)
            except Exception:
                pipe.zrem(self.k_leases, raw)
                pipe.lrem(self.k_inflight, 0, raw)
                continue
            # increment attempt inside raw
            try:
                d = json.loads(raw); d["attempt"] = int(d.get("attempt", 0)) + 1
                new_raw = json.dumps(d, separators=(",", ":"), ensure_ascii=False)
            except Exception:
                new_raw = raw
            pipe.lrem(self.k_inflight, 0, raw)
            if d.get("attempt", 1) >= self.max_attempts:
                pipe.lpush(self.k_dead, new_raw)
                pipe.zrem(self.k_leases, raw)
            else:
                backoff = self._backoff_ms(int(d.get("attempt", 1)))
                pipe.zadd(self.k_delayed, {new_raw: now + backoff})
                pipe.zrem(self.k_leases, raw)
        pipe.execute()

    # ---- public API
    def enqueue(self, payload: Dict[str, Any], *, priority: str = "n",
                delay_ms: int = 0, idem_key: Optional[str] = None,
                meta: Optional[Dict[str, Any]] = None) -> Job:
        # idempotency
        if idem_key:
            ok = self.r.set(self.k_idem_pref + idem_key, "1", nx=True, ex=IDEMP_TTL_SEC)  # type: ignore
            if not ok:
                return Job(queue=self.name, id=f"dup-{uuid.uuid4().hex[:8]}", payload=payload,
                           priority=priority, idem_key=idem_key, meta={"ignored": True})

        job = Job(queue=self.name, id=uuid.uuid4().hex, payload=payload,
                  not_before_ms=now_ms() + max(0, delay_ms), priority=priority,
                  idem_key=idem_key, meta=meta or {})

        raw = job.to_json()
        if delay_ms > 0:
            self.r.zadd(self.k_delayed, {raw: job.not_before_ms})  # type: ignore
        else:
            self.r.lpush(self.k_ready[priority], raw)  # type: ignore
        return job

    def poll(self, *, lease_ms: int = DEFAULT_LEASE_MS) -> Optional[Tuple[Job, str]]:
        with self._local_lock:
            self._release_due()
            self._requeue_expired()

            # claim by priority (RPOPLPUSH → inflight)
            raw = None
            for lane in ("h", "n", "l"):
                raw = self.r.rpoplpush(self.k_ready[lane], self.k_inflight)  # type: ignore
                if raw:
                    break
            if not raw:
                return None
            try:
                job = Job.from_json(raw) # type: ignore
            except Exception:
                # drop bad payload
                self.r.lrem(self.k_inflight, 0, raw)  # type: ignore
                return None

            # mark lease deadline
            self.r.zadd(self.k_leases, {raw: now_ms() + max(100, lease_ms)})  # type: ignore

            # token is the raw itself (unique per attempt since contain attempt counter mutable later)
            token = raw
            return job, token # type: ignore

    def ack(self, token: str) -> None:
        pipe = self.r.pipeline()
        pipe.lrem(self.k_inflight, 0, token)
        pipe.zrem(self.k_leases, token)
        pipe.execute()

    def nack(self, token: str, *, requeue: bool = True) -> None:
        # remove from inflight and leases
        self.r.lrem(self.k_inflight, 0, token)  # type: ignore
        self.r.zrem(self.k_leases, token)  # type: ignore
        if not requeue:
            self.r.lpush(self.k_dead, token)  # type: ignore
            return
        # bump attempt and requeue with backoff
        try:
            d = json.loads(token); d["attempt"] = int(d.get("attempt", 0)) + 1
        except Exception:
            d = {"attempt": 1}
        if int(d.get("attempt", 1)) >= self.max_attempts:
            self.r.lpush(self.k_dead, json.dumps(d))  # type: ignore
            return
        backoff = self._backoff_ms(int(d.get("attempt", 1)))
        raw2 = json.dumps(d, separators=(",", ":"), ensure_ascii=False)
        self.r.zadd(self.k_delayed, {raw2: now_ms() + backoff})  # type: ignore

    def size(self) -> Dict[str, int]:
        pipe = self.r.pipeline()
        for p in PRIORITY_LANES:
            pipe.llen(self.k_ready[p])
        pipe.zcard(self.k_delayed)
        pipe.llen(self.k_inflight)
        pipe.llen(self.k_dead)
        out = pipe.execute()
        return {"ready": sum(out[:len(PRIORITY_LANES)]), "delayed": out[len(PRIORITY_LANES)],
                "inflight": out[len(PRIORITY_LANES)+1], "dead": out[len(PRIORITY_LANES)+2]}

    def dead_letter(self, job: Job, reason: str) -> None:
        d = asdict(job); d["dead_reason"] = reason
        self.r.lpush(self.k_dead, json.dumps(d))  # type: ignore

    def purge(self) -> None:
        pipe = self.r.pipeline()
        for p in PRIORITY_LANES:
            pipe.delete(self.k_ready[p])
        pipe.delete(self.k_delayed, self.k_inflight, self.k_leases, self.k_dead)
        pipe.execute()

# =============================================================================
# Factory & convenience wrapper
# =============================================================================
class Queue:
    """
    Thin façade that chooses Redis if available (and REDIS_URL set),
    otherwise falls back to in-memory queue.
    """
    def __init__(self, name: str, *, force_memory: bool = False, max_attempts: int = DEFAULT_MAX_ATTEMPTS):
        self.name = name
        if not force_memory and HAVE_REDIS:
            try:
                self.backend: QueueBackend = RedisQueue(name, redis_url=REDIS_URL, max_attempts=max_attempts)
            except Exception:
                self.backend = InMemoryQueue(name, max_attempts=max_attempts)
        else:
            self.backend = InMemoryQueue(name, max_attempts=max_attempts)

    # proxies
    def enqueue(self, *a, **k): return self.backend.enqueue(*a, **k)
    def poll(self, *a, **k):    return self.backend.poll(*a, **k)
    def ack(self, *a, **k):     return self.backend.ack(*a, **k)
    def nack(self, *a, **k):    return self.backend.nack(*a, **k)
    def size(self):             return self.backend.size()
    def purge(self):            return self.backend.purge()
    def dead_letter(self, *a, **k): return self.backend.dead_letter(*a, **k)

# =============================================================================
# Simple Rate Limiter (token bucket) – optional helper
# =============================================================================
class RateLimiter:
    """
    Per-key token bucket limiter. Works in-memory; for Redis, a single-process
    limiter is typically sufficient at gateway level. Extend as needed.
    """
    def __init__(self, rate_per_sec: float, burst: int):
        self.rate = float(rate_per_sec)
        self.burst = int(burst)
        self.tokens: Dict[str, float] = {}
        self.ts: Dict[str, float] = {}

    def allow(self, key: str, n: int = 1) -> bool:
        now = time.time()
        last = self.ts.get(key, now)
        avail = min(self.burst, self.tokens.get(key, self.burst) + (now - last) * self.rate)
        if avail >= n:
            self.tokens[key] = avail - n
            self.ts[key] = now
            return True
        self.tokens[key] = avail
        self.ts[key] = now
        return False

# =============================================================================
# CLI for quick ops / testing
# =============================================================================
def _cli():
    import argparse
    ap = argparse.ArgumentParser("queues")
    sub = ap.add_subparsers(dest="cmd", required=True)

    enq = sub.add_parser("enqueue")
    enq.add_argument("--queue", required=True)
    enq.add_argument("--json", required=True, help='payload JSON')
    enq.add_argument("--prio", default="n", choices=list(PRIORITY_LANES))
    enq.add_argument("--delay-ms", type=int, default=0)
    enq.add_argument("--idem", default=None)

    poll = sub.add_parser("poll")
    poll.add_argument("--queue", required=True)
    poll.add_argument("--lease-ms", type=int, default=DEFAULT_LEASE_MS)

    ack = sub.add_parser("ack")
    ack.add_argument("--queue", required=True)
    ack.add_argument("--token", required=True)

    nack = sub.add_parser("nack")
    nack.add_argument("--queue", required=True)
    nack.add_argument("--token", required=True)
    nack.add_argument("--requeue", action="store_true")

    size = sub.add_parser("size")
    size.add_argument("--queue", required=True)

    purge = sub.add_parser("purge")
    purge.add_argument("--queue", required=True)

    args = ap.parse_args()
    q = Queue(args.queue)

    if args.cmd == "enqueue":
        job = q.enqueue(json.loads(args.json), priority=args.prio, delay_ms=args.delay_ms, idem_key=args.idem)
        print(job.to_json())
    elif args.cmd == "poll":
        res = q.poll(lease_ms=args.lease_ms)
        if not res:
            print("{}")
        else:
            job, token = res
            print(json.dumps({"job": asdict(job), "token": token}, ensure_ascii=False))
    elif args.cmd == "ack":
        q.ack(args.token); print("OK")
    elif args.cmd == "nack":
        q.nack(args.token, requeue=args.requeue); print("OK")
    elif args.cmd == "size":
        print(json.dumps(q.size()))
    elif args.cmd == "purge":
        q.purge(); print("OK")

if __name__ == "__main__":
    _cli()