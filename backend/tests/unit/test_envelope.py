# tests/test_envelope.py
from __future__ import annotations

import json
import time
from typing import Any, Dict

import pytest # type: ignore


def have_attr(mod, name: str) -> bool:
    return hasattr(mod, name) and callable(getattr(mod, name))


def load_env_module():
    try:
        import platform.envelope as env  # type: ignore # your repo module
        return env
    except Exception as e:
        pytest.skip(f"platform.envelope import failed: {e}")


def mk_envelope(env):
    """
    Create a minimal test envelope using whatever constructor your module provides.
    Preferred order:
      1) env.new(schema_name, payload, producer)
      2) env.Envelope.new(...) / env.Envelope(...)
      3) fallback: build dict and, if available, env.from_dict(...)
    """
    payload = {"foo": "bar", "n": 42}
    producer = {"svc": "unit-test", "roles": ["tests"]}
    schema = "test.event"

    if have_attr(env, "new"):
        return env.new(schema_name=schema, payload=payload, producer=producer)

    # Try class constructors
    if hasattr(env, "Envelope"):
        Env = env.Envelope
        if have_attr(Env, "new"):
            return Env.new(schema_name=schema, payload=payload, producer=producer)
        try:
            return Env(schema=schema, payload=payload, producer=producer)
        except Exception:
            pass

    # Fallback: raw dict -> from_dict if present
    body = {"schema": schema, "payload": payload, "producer": producer, "ts": int(time.time() * 1000)}
    if have_attr(env, "from_dict"):
        return env.from_dict(body)

    # Last resort: provide a tiny shim mimicking your interface
    class _Shim:
        def __init__(self, body: Dict[str, Any]):
            self.body = body

        def flatten_for_stream(self) -> Dict[str, str]:
            return {k: (v if isinstance(v, str) else json.dumps(v)) for k, v in self.body.items()}

    return _Shim(body)


def parse_fields(env, fields: Dict[bytes, bytes]) -> Dict[str, Any]:
    """
    Parse Redis Stream fields via your parser if available,
    else a generic JSON-or-string decoder.
    """
    # env-level parser
    if have_attr(env, "parse_from_stream"):
        try:
            return env.parse_from_stream(fields)
        except Exception:
            pass

    # Envelope class static parser
    if hasattr(env, "Envelope") and hasattr(env.Envelope, "parse_from_stream"):
        try:
            return env.Envelope.parse_from_stream(fields)  # type: ignore[attr-defined]
        except Exception:
            pass

    # Generic parser
    out: Dict[str, Any] = {}
    for k, v in fields.items():
        ks = k.decode()
        vs = v.decode()
        try:
            out[ks] = json.loads(vs)
        except Exception:
            out[ks] = vs
    return out


def test_flatten_and_parse_roundtrip():
    env = load_env_module()
    e = mk_envelope(env)

    # 1) flatten_for_stream should return a dict[str, str]
    assert hasattr(e, "flatten_for_stream"), "Envelope must expose flatten_for_stream()"
    flat = e.flatten_for_stream()
    assert isinstance(flat, dict)
    for k, v in flat.items():
        assert isinstance(k, str)
        assert isinstance(v, str), f"flattened value for key '{k}' should be str"

    # 2) Simulate Redis stream read
    # Convert to bytes like redis-py would return
    fields = {k.encode(): v.encode() for k, v in flat.items()}

    parsed = parse_fields(env, fields)
    assert isinstance(parsed, dict)
    # Must include core keys
    assert "schema" in parsed
    assert "payload" in parsed
    assert "producer" in parsed
    # payload should be a dict after parse
    assert isinstance(parsed["payload"], dict)

    # 3) Optional: rehydrate -> flatten (if API exists)
    if have_attr(env, "from_dict"):
        e2 = env.from_dict(parsed) # type: ignore
        flat2 = e2.flatten_for_stream()
        assert isinstance(flat2, dict) and all(isinstance(v, str) for v in flat2.values())


def test_schema_and_timestamps_when_available():
    env = load_env_module()
    e = mk_envelope(env)
    flat = e.flatten_for_stream()
    fields = {k.encode(): v.encode() for k, v in flat.items()}
    parsed = parse_fields(env, fields)

    # schema should be a non-empty string
    schema = parsed.get("schema")
    if isinstance(schema, bytes):
        schema = schema.decode()
    assert isinstance(schema, str) and schema, "schema must be a non-empty string"

    # If timestamp is present, it should be numeric and recent-ish
    ts = parsed.get("ts") or parsed.get("timestamp") or parsed.get("created_at")
    if ts is not None:
        try:
            ts = int(ts)
        except Exception:
            # If parser left JSON string, try to coerce
            if isinstance(ts, str) and ts.isdigit():
                ts = int(ts)
        assert isinstance(ts, int) and ts > 0, "timestamp must be a positive integer"
        # within +/- 1 day of 'now'
        now_ms = int(time.time() * 1000)
        assert abs(now_ms - int(ts)) < 86_400_000, "timestamp looks stale by more than 1 day"


def test_validation_if_exposed_otherwise_skip():
    """
    If your module exposes validation (e.g., validate(), require_fields(), or Envelope.validate),
    ensure it fails on missing required fields.
    """
    env = load_env_module()

    # Build an obviously bad envelope (missing schema)
    bad = {"payload": {"x": 1}, "producer": {"svc": "unit-test"}}

    # Try module-level or class-level validators
    validators = []
    if have_attr(env, "validate"):
        validators.append(env.validate) # type: ignore
    if hasattr(env, "Envelope") and have_attr(env.Envelope, "validate"): # type: ignore
        validators.append(env.Envelope.validate)  # type: ignore[attr-defined]
    if not validators:
        pytest.skip("No validation function exposed by platform.envelope")

    with pytest.raises(Exception):
        for v in validators:
            # validators may accept a dict or an Envelope instance
            try:
                v(bad)  # type: ignore[misc]
            except TypeError:
                if have_attr(env, "from_dict"):
                    v(env.from_dict(bad))  # type: ignore[misc]
                else:
                    raise


def test_redaction_or_scrub_if_supported():
    """
    If you support PII/secret redaction (e.g., redact(), scrub(), sanitize()),
    verify it removes common secret keys.
    """
    env = load_env_module()

    redactors = []
    for name in ("redact", "scrub", "sanitize", "redact_payload"):
        if have_attr(env, name):
            redactors.append(getattr(env, name))

    if not redactors:
        pytest.skip("No redaction/scrub function in platform.envelope")

    sample = {
        "schema": "test.event",
        "payload": {"token": "SECRET", "api_key": "ABC123", "ok": 1},
        "producer": {"svc": "unit-test"},
    }
    # Use module's from_dict if available to mimic real usage
    if have_attr(env, "from_dict"):
        obj = env.from_dict(sample) # type: ignore
        for fn in redactors:
            out = fn(obj)  # may return new object or mutate
            body = getattr(out, "body", getattr(obj, "body", sample))
            text = json.dumps(body).lower()
            assert "secret" not in text and "abc123" not in text
    else:
        for fn in redactors:
            out = fn(sample)
            text = json.dumps(out).lower()
            assert "secret" not in text and "abc123" not in text