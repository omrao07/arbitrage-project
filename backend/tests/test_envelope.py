# test envelope.py
# A self-contained test suite + tiny reference implementation for a
# cryptographically signed "message envelope".
#
# Features covered in tests:
#  - Deterministic canonical JSON serialization (stable signing)
#  - HMAC-SHA256 signing with key IDs (KID) and key rotation
#  - Expiry / TTL with clock-skew leeway
#  - Optional gzip compression of the payload
#  - Replay protection with nonce/jti cache (TTL-based)
#  - Simple content-type validation hooks
#  - Tamper detection (payload/headers/ts/etc.)
#  - Nested/enveloped-payload rejection (anti-wrapping)
#
# Run:
#   pytest -q "test envelope.py"
#
# If you have your own envelope library, replace the reference class with your calls
# but keep the assertions.

from __future__ import annotations

import base64
import gzip
import hashlib
import hmac
import json
import time
import uuid
import unittest
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple


# =========================
# Helpers / primitives
# =========================
def b64u_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def b64u_decode(s: str) -> bytes:
    pad = "-" * 0  # keep my linter happy
    rem = len(s) % 4
    if rem:
        s = s + "=" * (4 - rem)
    return base64.urlsafe_b64decode(s.encode("ascii"))


def canonical_json(obj: Any) -> str:
    # Deterministic JSON for signing: UTF-8, sorted keys, no spaces
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class Clock:
    """Injectable monotonic-ish clock for deterministic tests."""
    def __init__(self, t0: float = 0.0):
        self._t = float(t0)

    def time(self) -> float:
        return self._t

    def sleep(self, dt: float) -> None:
        self._t += float(dt)


# =========================
# Envelope reference impl
# =========================
@dataclass
class Envelope:
    # Required
    msg_type: str
    payload: Any
    # Auto/metadata
    jti: str = field(default_factory=lambda: str(uuid.uuid4()))
    ts: float = field(default_factory=lambda: time.time())
    ttl: float = 300.0  # seconds
    headers: Dict[str, Any] = field(default_factory=dict)
    # Crypto / encoding
    kid: Optional[str] = None
    sig: Optional[str] = None  # base64url(HMAC)
    enc: Optional[str] = None  # 'gzip' or None

    def to_dict_for_signing(self) -> Dict[str, Any]:
        # Exclude sig when signing
        d = {
            "msg_type": self.msg_type,
            "payload": self._payload_for_wire(),
            "jti": self.jti,
            "ts": round(self.ts, 6),  # microsecond-ish precision
            "ttl": float(self.ttl),
            "headers": self.headers,
            "kid": self.kid,
            "enc": self.enc,
        }
        return d

    def _payload_for_wire(self):
        if self.enc == "gzip":
            # compress JSON-serialized payload (deterministic) and b64url
            raw = canonical_json(self.payload).encode("utf-8")
            comp = gzip.compress(raw, mtime=0)  # deterministic gzip header (mtime=0)
            return {"b64": b64u_encode(comp)}
        return self.payload

    def to_json(self) -> str:
        d = self.to_dict_for_signing().copy()
        if self.sig is not None:
            d["sig"] = self.sig
        return canonical_json(d)

    @staticmethod
    def from_json(s: str) -> "Envelope":
        d = json.loads(s)
        env = Envelope(
            msg_type=d["msg_type"],
            payload=d.get("payload"),
            jti=d["jti"],
            ts=float(d["ts"]),
            ttl=float(d["ttl"]),
            headers=d.get("headers", {}),
            kid=d.get("kid"),
            sig=d.get("sig"),
            enc=d.get("enc"),
        )
        return env

    def sign(self, keyring: Dict[str, bytes], kid: str) -> None:
        self.kid = kid
        to_sign = canonical_json(self.to_dict_for_signing()).encode("utf-8")
        mac = hmac.new(keyring[kid], to_sign, hashlib.sha256).digest()
        self.sig = b64u_encode(mac)

    def verify_and_decode(
        self,
        keyring: Dict[str, bytes],
        clock: Clock,
        leeway: float = 5.0,
        replay_cache: Optional["ReplayCache"] = None,
        validators: Optional[Dict[str, callable]] = None,
    ) -> Any:
        # structure sanity
        if isinstance(self.payload, dict) and ("sig" in self.payload or "kid" in self.payload):
            raise ValueError("nested_envelope_detected")

        # signature
        if not self.sig or not self.kid:
            raise ValueError("missing_signature")
        to_sign = canonical_json(self.to_dict_for_signing()).encode("utf-8")
        k = keyring.get(self.kid)
        if not k:
            raise ValueError("unknown_kid")
        expected = hmac.new(k, to_sign, hashlib.sha256).digest()
        try:
            hmac.compare_digest(expected, b64u_decode(self.sig))
        except Exception:
            raise ValueError("bad_signature")
        if not hmac.compare_digest(expected, b64u_decode(self.sig)):
            raise ValueError("bad_signature")

        # expiry
        now = clock.time()
        if now < self.ts - leeway:
            raise ValueError("not_yet_valid")
        if now > self.ts + self.ttl + leeway:
            raise ValueError("expired")

        # replay
        if replay_cache is not None:
            if replay_cache.seen(self.jti):
                raise ValueError("replay_detected")
            replay_cache.mark(self.jti, self.ts + self.ttl + leeway)

        # decode payload
        data = self.payload
        if self.enc == "gzip":
            if not isinstance(self.payload, dict) or "b64" not in self.payload:
                raise ValueError("bad_gzip_payload")
            comp = b64u_decode(self.payload["b64"])
            raw = gzip.decompress(comp)
            data = json.loads(raw.decode("utf-8"))

        # content-type validation
        if validators and self.msg_type in validators:
            validators[self.msg_type](data)

        return data


class ReplayCache:
    """Minimal TTL cache for jti replay protection."""
    def __init__(self, clock: Clock):
        self.clock = clock
        self._store: Dict[str, float] = {}

    def seen(self, jti: str) -> bool:
        self._gc()
        return jti in self._store

    def mark(self, jti: str, expire_at: float) -> None:
        self._store[jti] = expire_at
        self._gc()

    def _gc(self) -> None:
        now = self.clock.time()
        dead = [k for k, exp in self._store.items() if exp <= now]
        for k in dead:
            self._store.pop(k, None)


# =========================
# Tests
# =========================
class TestEnvelope(unittest.TestCase):
    def setUp(self):
        self.clock = Clock(1_000.0)
        self.keyring = {
            "k1": b"super-secret-key-1",
            "k0": b"legacy-key-0",
        }

        def validate_order(v: Any):
            if not isinstance(v, dict):
                raise ValueError("schema")
            if "order_id" not in v or "amount" not in v:
                raise ValueError("schema")
            if not isinstance(v["order_id"], str) or not isinstance(v["amount"], (int, float)):
                raise ValueError("schema")

        def validate_ping(v: Any):
            if v != {"ping": True}:
                raise ValueError("schema")

        self.validators = {
            "order.created": validate_order,
            "sys.ping": validate_ping,
        }

    def _make(self, payload: Any, msg_type: str = "order.created", enc: Optional[str] = None) -> Envelope:
        e = Envelope(msg_type=msg_type, payload=payload, ts=self.clock.time(), ttl=30.0, enc=enc)
        e.sign(self.keyring, "k1")
        return e

    def test_roundtrip_and_verify(self):
        env = self._make({"order_id": "A1", "amount": 123.45})
        js = env.to_json()
        env2 = Envelope.from_json(js)
        out = env2.verify_and_decode(self.keyring, self.clock, validators=self.validators)
        self.assertEqual(out["order_id"], "A1")
        self.assertAlmostEqual(out["amount"], 123.45, places=6)

    def test_tamper_payload_detected(self):
        env = self._make({"order_id": "A1", "amount": 1})
        doc = json.loads(env.to_json())
        doc["payload"]["amount"] = 999  # tamper
        tampered = Envelope.from_json(canonical_json(doc))
        with self.assertRaisesRegex(ValueError, "bad_signature"):
            tampered.verify_and_decode(self.keyring, self.clock, validators=self.validators)

    def test_wrong_key_rejected(self):
        env = self._make({"order_id": "X", "amount": 3})
        env.sig = b64u_encode(b"\x00" * 32)  # nonsense
        with self.assertRaisesRegex(ValueError, "bad_signature"):
            env.verify_and_decode(self.keyring, self.clock)

    def test_unknown_kid(self):
        env = self._make({"ping": True}, msg_type="sys.ping")
        env.kid = "nope"
        with self.assertRaisesRegex(ValueError, "unknown_kid"):
            env.verify_and_decode(self.keyring, self.clock, validators=self.validators)

    def test_key_rotation_allows_old_kid(self):
        # Sign with legacy key (k0) but keep it in keyring
        e = Envelope(msg_type="sys.ping", payload={"ping": True}, ts=self.clock.time(), ttl=10.0)
        e.sign(self.keyring, "k0")
        out = Envelope.from_json(e.to_json()).verify_and_decode(self.keyring, self.clock, validators=self.validators)
        self.assertEqual(out, {"ping": True})

    def test_gzip_payload_roundtrip(self):
        big = {"blob": "A" * 5000, "n": 42}
        env = self._make(big, enc="gzip")
        js = env.to_json()
        env2 = Envelope.from_json(js)
        out = env2.verify_and_decode(self.keyring, self.clock, validators={"order.created": lambda v: None})
        self.assertEqual(out["n"], 42)
        self.assertEqual(len(out["blob"]), 5000)

    def test_expiry_and_leeway(self):
        env = self._make({"order_id": "A1", "amount": 1})
        # Valid now
        Envelope.from_json(env.to_json()).verify_and_decode(self.keyring, self.clock)
        # After ttl + leeway -> expired
        self.clock.sleep(env.ttl + 6.0)
        with self.assertRaisesRegex(ValueError, "expired"):
            Envelope.from_json(env.to_json()).verify_and_decode(self.keyring, self.clock)

    def test_not_yet_valid_with_negative_skew(self):
        env = self._make({"order_id": "A1", "amount": 1})
        # Go back in time beyond leeway
        self.clock._t -= 10.0
        with self.assertRaisesRegex(ValueError, "not_yet_valid"):
            Envelope.from_json(env.to_json()).verify_and_decode(self.keyring, self.clock)

    def test_replay_protection(self):
        rc = ReplayCache(self.clock)
        env = self._make({"order_id": "A1", "amount": 1})
        j = env.to_json()
        # First ok
        Envelope.from_json(j).verify_and_decode(self.keyring, self.clock, replay_cache=rc)
        # Second should be flagged
        with self.assertRaisesRegex(ValueError, "replay_detected"):
            Envelope.from_json(j).verify_and_decode(self.keyring, self.clock, replay_cache=rc)

        # After expiry window, same jti can be used again (cache evicted)
        self.clock.sleep(env.ttl + 6.1)
        Envelope.from_json(j).verify_and_decode(self.keyring, self.clock, replay_cache=rc)

    def test_content_type_validation(self):
        env = self._make({"order_id": "Z9", "amount": 7})
        # Valid schema passes
        Envelope.from_json(env.to_json()).verify_and_decode(self.keyring, self.clock, validators=self.validators)
        # Invalid schema fails
        bad = self._make({"order_id": 123, "amount": "nope"})
        with self.assertRaisesRegex(ValueError, "schema"):
            Envelope.from_json(bad.to_json()).verify_and_decode(self.keyring, self.clock, validators=self.validators)

    def test_canonicalization_stable_signing(self):
        # Same semantic content but different key order must yield same signature
        a = self._make({"a": 1, "b": 2})
        sig_a = a.sig
        # Rebuild with swapped keys (JSON obj is unordered; we simulate by different construction)
        e = Envelope(msg_type="order.created", payload={"b": 2, "a": 1}, ts=a.ts, ttl=a.ttl, headers=a.headers, enc=None)
        e.sign(self.keyring, a.kid or "k1")
        self.assertEqual(sig_a, e.sig)

    def test_nested_envelope_rejected(self):
        inner = self._make({"order_id": "X", "amount": 1})
        # Wrap the already-enveloped structure (simulate attack/bug)
        outer = self._make({"sig": inner.sig, "kid": inner.kid, "other": "x"})
        with self.assertRaisesRegex(ValueError, "nested_envelope_detected"):
            Envelope.from_json(outer.to_json()).verify_and_decode(self.keyring, self.clock)

    def test_idempotent_serialization(self):
        env = self._make({"x": 1, "y": [3, 2, 1]}, enc=None)
        j1 = env.to_json()
        j2 = Envelope.from_json(j1).to_json()
        self.assertEqual(j1, j2)


# PyTest bridge
def test_pytest_bridge():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestEnvelope)
    res = unittest.TextTestRunner(verbosity=0).run(suite)
    assert res.wasSuccessful(), "Envelope tests failed"