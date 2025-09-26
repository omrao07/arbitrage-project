# security/mpc_signing/coordinator.py
"""
MPC Signing Coordinator (async, transport-agnostic)

This module orchestrates a multi-party signing round with N participants and a
threshold t (t <= N). It provides:
  - Round lifecycle: keygen -> presign -> partial-sign -> aggregate
  - Timeouts, retries, quorum checks
  - Pluggable protocol logic (via Strategy) and transport

⚠️ Crypto notice
This file does NOT implement cryptography. Use a vetted protocol (e.g., FROST
for Schnorr/Ed25519 or GG18 for ECDSA) and wire it via the Strategy interface.

Typical usage
-------------
# 1) Implement a Transport for your environment (HTTP/JSON, gRPC, Redis, etc.)
# 2) Implement a Strategy that wraps your MPC library or service API.
# 3) Create Coordinator(...).sign(message) to run a signing round.

See the LocalTransport and DummyStrategy examples at the bottom for smoke tests.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple


# ========================= Protocol & Transport Interfaces =========================

class Transport(Protocol):
    """Abstract transport to talk to a participant. Implement one of: HTTP, gRPC, Redis, local."""

    async def request(self, participant_id: str, method: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        """
        Send a method + payload to a participant and await a JSON-like dict reply.
        Should raise on timeout / connection errors.
        """
        ...


class Strategy(Protocol):
    """
    Strategy plugs in the MPC cryptography & message shapes for each phase.
    Each method should return payloads to send and validate/aggregate responses.
    """

    # ---- key material ----
    async def keygen_request(self, pid: str) -> Dict[str, Any]: ...
    async def keygen_accept(self, pid: str, resp: Dict[str, Any]) -> None: ...
    async def public_key(self) -> Dict[str, Any]: ...

    # ---- presign (nonces, commitments, etc.) ----
    async def presign_request(self, pid: str) -> Dict[str, Any]: ...
    async def presign_accept(self, pid: str, resp: Dict[str, Any]) -> None: ...

    # ---- signing ----
    async def partial_sign_request(self, pid: str, message: bytes) -> Dict[str, Any]: ...
    async def partial_sign_accept(self, pid: str, resp: Dict[str, Any]) -> None: ...

    # ---- aggregation & verification ----
    async def aggregate(self, participants: List[str], message: bytes) -> Dict[str, Any]:
        """
        Returns {"signature": <bytes-like or hex>, "ok": bool, "error": str|None}
        """
        ...


# ========================= Data Models =========================

@dataclass
class Participant:
    id: str
    # optional metadata (uri, labels, weight, etc.)
    meta: Dict[str, Any] = field(default_factory=dict)
    healthy: bool = True


@dataclass
class RoundConfig:
    threshold: int
    timeout_s: float = 10.0     # per-request timeout
    quorum_timeout_s: float = 20.0
    max_retries: int = 1


@dataclass
class RoundResult:
    ok: bool
    signature: Optional[bytes] = None
    participants: List[str] = field(default_factory=list)
    public_key: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    latency_ms: float = 0.0


# ========================= Coordinator =========================

class Coordinator:
    """
    Orchestrates MPC signing rounds with N parties and threshold t.
    Transport & Strategy are injected so the coordinator stays protocol-agnostic.
    """

    def __init__(
        self,
        *,
        participants: Iterable[Participant],
        transport: Transport,
        strategy: Strategy,
        cfg: RoundConfig,
        logger: Optional[Any] = None,   # accepts stdlib or your utils.logger
    ):
        self.participants: Dict[str, Participant] = {p.id: p for p in participants}
        self.transport = transport
        self.strategy = strategy
        self.cfg = cfg
        self.log = logger or _StdLogger()

        if self.cfg.threshold < 1 or self.cfg.threshold > len(self.participants):
            raise ValueError("threshold must be in [1, N]")

    # -------- public API --------

    async def ensure_key_material(self) -> Dict[str, Any]:
        """
        Ensure key material (public key, shares) exists by running keygen across participants.
        Idempotent if your Strategy is implemented to no-op when keys exist.
        """
        self.log.info("keygen:start", extra={"N": len(self.participants)})
        await self._fanout("keygen", self.strategy.keygen_request, self.strategy.keygen_accept)
        pub = await self.strategy.public_key()
        self.log.info("keygen:complete", extra={"public_key": _safe_json(pub)})
        return pub

    async def presign(self) -> None:
        """Run the presign (nonce/commitments) phase across a quorum."""
        self.log.info("presign:start")
        await self._fanout("presign", self.strategy.presign_request, self.strategy.presign_accept, need_quorum=True)
        self.log.info("presign:complete")

    async def sign(self, message: bytes) -> RoundResult:
        """
        Execute a full signing round: (optional) presign -> partial-sign -> aggregate.
        Returns a RoundResult with the aggregated signature from at least `threshold` participants.
        """
        started = time.perf_counter()
        # presign might be done ahead of time; running it here is safe
        try:
            await self.presign()
        except Exception as e:
            self.log.error("presign:error", extra={"error": str(e)})
            return RoundResult(ok=False, error=f"presign failed: {e}")

        # partial signing
        self.log.info("sign:start", extra={"len_msg": len(message)})
        ok_participants: List[str] = []
        try:
            ok_participants = await self._fanout(
                "partial_sign",
                lambda pid: self.strategy.partial_sign_request(pid, message),
                self.strategy.partial_sign_accept,
                need_quorum=True,
            )
        except Exception as e:
            self.log.error("partial_sign:error", extra={"error": str(e)})
            return RoundResult(ok=False, error=f"partial sign failed: {e}")

        # aggregate
        agg = await self.strategy.aggregate(ok_participants, message)
        elapsed = (time.perf_counter() - started) * 1000.0
        if agg.get("ok"):
            sig = _coerce_bytes(agg.get("signature"))
            self.log.info("sign:complete", extra={"participants": ok_participants, "latency_ms": round(elapsed, 2)})
            pub = await self.strategy.public_key()
            return RoundResult(
                ok=True,
                signature=sig,
                participants=ok_participants,
                public_key=pub,
                latency_ms=elapsed,
            )
        else:
            err = str(agg.get("error") or "aggregate failed")
            self.log.error("aggregate:error", extra={"error": err})
            return RoundResult(ok=False, error=err, participants=ok_participants, latency_ms=elapsed)

    # -------- internals --------

    async def _fanout(
        self,
        phase: str,
        make_payload,               # async fn(pid) -> dict
        accept_response,            # async fn(pid, resp)
        *,
        need_quorum: bool = False,
    ) -> List[str]:
        """
        Fan out phase requests to all participants, handle retries/timeouts,
        and call `accept_response` for each successful response.
        Returns the list of successful participant IDs (quorum first if requested).
        """
        N = len(self.participants)
        t = self.cfg.threshold
        quorum = t if need_quorum else 1

        # run first attempt
        successes: Dict[str, Dict[str, Any]] = {}
        failures: Dict[str, str] = {}

        async def _one(pid: str) -> Tuple[str, Optional[Dict[str, Any]], Optional[Exception]]:
            try:
                payload = await make_payload(pid)
                resp = await self.transport.request(pid, method=phase, payload=payload, timeout=self.cfg.timeout_s)
                return (pid, resp, None)
            except Exception as e:
                return (pid, None, e)

        attempt = 0
        remaining = set(self.participants.keys())
        while remaining:
            self.log.info(f"{phase}:attempt", extra={"attempt": attempt + 1, "remaining": list(remaining)})

            # batch request concurrently
            results = await asyncio.gather(*[_one(pid) for pid in remaining], return_exceptions=False)
            remaining.clear()

            # process outcomes
            for pid, resp, err in results:
                if err is None and isinstance(resp, dict):
                    try:
                        await accept_response(pid, resp)
                        successes[pid] = resp
                    except Exception as e:
                        failures[pid] = f"accept failed: {e}"
                else:
                    failures[pid] = f"{type(err).__name__ if err else 'UnknownError'}: {err}"

            # check quorum
            if len(successes) >= quorum:
                break

            # retry failed (if allowed)
            attempt += 1
            if attempt > self.cfg.max_retries:
                break
            remaining = set(failures.keys())
            failures.clear()  # retry them

        # mark health
        for pid in self.participants:
            self.participants[pid].healthy = pid in successes

        ok_ids = list(successes.keys())
        if need_quorum and len(ok_ids) < quorum:
            raise RuntimeError(f"{phase}: quorum not met (got {len(ok_ids)}, need {quorum})")
        return ok_ids


# ========================= Simple Std Logger =========================

class _StdLogger:
    def info(self, msg, extra: Optional[Dict[str, Any]] = None):
        print(f"[INFO] {msg} {json.dumps(extra or {})}")

    def error(self, msg, extra: Optional[Dict[str, Any]] = None):
        print(f"[ERROR] {msg} {json.dumps(extra or {})}")


# ========================= Utilities =========================

def _safe_json(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)

def _coerce_bytes(val: Any) -> Optional[bytes]:
    if val is None:
        return None
    if isinstance(val, bytes):
        return val
    if isinstance(val, str):
        try:
            # try hex first
            return bytes.fromhex(val.replace("0x", ""))
        except Exception:
            return val.encode("utf-8")
    return None


# ========================= Example: Local Transport & Dummy Strategy =========================
# These are for smoke-testing the coordinator plumbing only. Replace with real impls.

class LocalTransport:
    """
    Minimal in-process transport for testing. It calls handler coroutines
    registered per participant ID.
    """

    def __init__(self, handlers: Dict[str, Dict[str, Any]]):
        """
        handlers: {pid: {"keygen": coro, "presign": coro, "partial_sign": coro}}
        Each handler is an async function (payload: dict) -> dict
        """
        self.handlers = handlers

    async def request(self, participant_id: str, method: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        if participant_id not in self.handlers or method not in self.handlers[participant_id]:
            raise RuntimeError(f"no handler for {participant_id}.{method}")
        h = self.handlers[participant_id][method]

        async def _call():
            return await h(payload)

        return await asyncio.wait_for(_call(), timeout=timeout)


class DummyStrategy:
    """
    Toy strategy that simulates phases and returns a fake signature.
    DO NOT use for real signing.
    """

    def __init__(self, threshold: int):
        self.t = threshold
        self._pub = {"scheme": "dummy", "pk": "deadbeef"}

    async def keygen_request(self, pid: str) -> Dict[str, Any]:
        return {"pid": pid}

    async def keygen_accept(self, pid: str, resp: Dict[str, Any]) -> None:
        # accept any
        return None

    async def public_key(self) -> Dict[str, Any]:
        return self._pub

    async def presign_request(self, pid: str) -> Dict[str, Any]:
        return {"nonce_req": True}

    async def presign_accept(self, pid: str, resp: Dict[str, Any]) -> None:
        return None

    async def partial_sign_request(self, pid: str, message: bytes) -> Dict[str, Any]:
        return {"msg_len": len(message)}

    async def partial_sign_accept(self, pid: str, resp: Dict[str, Any]) -> None:
        return None

    async def aggregate(self, participants: List[str], message: bytes) -> Dict[str, Any]:
        if len(participants) < self.t:
            return {"ok": False, "error": "not enough partials"}
        # pretend the signature is sha256(participant_ids || message_len)
        import hashlib
        h = hashlib.sha256()
        h.update(",".join(sorted(participants)).encode())
        h.update(str(len(message)).encode())
        return {"ok": True, "signature": h.hexdigest()}


# ========================= Smoke Test (optional) =========================
# Run this file directly to see the round orchestration with dummy pieces.

async def _demo():
    async def ok_handler(_payload):
        await asyncio.sleep(0.01)
        return {"ok": True}

    handlers = {
        "A": {"keygen": ok_handler, "presign": ok_handler, "partial_sign": ok_handler},
        "B": {"keygen": ok_handler, "presign": ok_handler, "partial_sign": ok_handler},
        "C": {"keygen": ok_handler, "presign": ok_handler, "partial_sign": ok_handler},
    }

    participants = [Participant(id=i) for i in handlers.keys()]
    transport = LocalTransport(handlers)
    strategy = DummyStrategy(threshold=2)
    cfg = RoundConfig(threshold=2, timeout_s=1.0, max_retries=1)

    coord = Coordinator(participants=participants, transport=transport, strategy=strategy, cfg=cfg)
    await coord.ensure_key_material()
    res = await coord.sign(b"hello world")
    print("RoundResult:", res)

if __name__ == "__main__":
    asyncio.run(_demo())