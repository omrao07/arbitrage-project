# backend/ai/agents/core/sessions.py
from __future__ import annotations

import base64
import hmac
import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

# -------- Optional Redis mirror (safe fallback) --------
try:
    import redis  # type: ignore
    _R = redis.Redis(host=os.getenv("REDIS_HOST","localhost"), port=int(os.getenv("REDIS_PORT","6379")), decode_responses=True)
except Exception:
    _R = None

# -------- Env / defaults --------
SESSION_TTL_S     = int(os.getenv("SESS_TTL_S", "86400"))          # 24h default
SESSION_ROLL_S    = int(os.getenv("SESS_ROLL_S", "900"))           # touch if >15m idle
SESSION_SECRET    = os.getenv("SESS_SECRET", "")                   # if not set, generated at runtime
CSRF_HEADER       = os.getenv("SESS_CSRF_HEADER", "x-csrf-token")
ISSUER            = os.getenv("SESS_ISSUER", "bolt.local")
MAX_KV_BYTES      = int(os.getenv("SESS_MAX_KV_BYTES", "16384"))   # per session kv payload cap

# -------- Tiny rate limiter (token bucket) --------
class RateLimiter:
    def __init__(self, rate_per_sec: float, burst: float):
        self.rate = float(rate_per_sec)
        self.burst = float(burst)
        self.tokens = burst
        self.updated = time.time()
    def allow(self, cost: float = 1.0) -> bool:
        now = time.time()
        dt = max(0.0, now - self.updated)
        self.updated = now
        self.tokens = min(self.burst, self.tokens + dt * self.rate)
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

# -------- Models --------
@dataclass
class Session:
    sid: str
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    scopes: List[str] = field(default_factory=list)           # e.g., ["read:quotes","trade:paper"]
    kv: Dict[str, Any] = field(default_factory=dict)          # lightweight agent/user state
    created_at: int = field(default_factory=lambda: _now_ms())
    last_seen_at: int = field(default_factory=lambda: _now_ms())
    expires_at: int = field(default_factory=lambda: _now_ms() + SESSION_TTL_S * 1000)
    csrf: str = field(default_factory=lambda: _rand(24))
    device: Optional[str] = None                              # e.g., user-agent hash/fingerprint
    ip: Optional[str] = None
    issuer: str = ISSUER
    # ephemeral controls (not serialized to cookie)
    _limiter: Optional[RateLimiter] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("_limiter", None)
        return d

    def is_expired(self) -> bool:
        return _now_ms() >= self.expires_at

    def should_roll(self) -> bool:
        return (_now_ms() - self.last_seen_at) > (SESSION_ROLL_S * 1000)

# -------- Store --------
class SessionStore:
    """
    Minimal session store with optional Redis mirror.

    Usage:
      store = SessionStore()
      sess = store.create(user_id="om", scopes=["trade:paper","read:news"], ip="1.2.3.4")
      ok, sess = store.validate(sid_signature, headers)  # CSRF + expiry + signature
    """
    def __init__(self):
        self._mem: Dict[str, Session] = {}
        self._secret = (SESSION_SECRET or _rand(32)).encode("utf-8")

    # -------- ID & signature --------
    def new_sid(self) -> str:
        return _rand(20)

    def sign(self, sid: str) -> str:
        sig = hmac.new(self._secret, sid.encode("utf-8"), hashlib.sha256).digest()
        return f"{sid}.{base64.urlsafe_b64encode(sig).decode('ascii').rstrip('=')}"

    def verify(self, sid_sig: str) -> Optional[str]:
        try:
            sid, sig_b64 = sid_sig.split(".", 1)
            sig = base64.urlsafe_b64decode(_pad(sig_b64))
            calc = hmac.new(self._secret, sid.encode("utf-8"), hashlib.sha256).digest()
            if hmac.compare_digest(sig, calc):
                return sid
        except Exception:
            return None
        return None

    # -------- CRUD --------
    def create(self, *, user_id: Optional[str] = None, user_name: Optional[str] = None,
               scopes: Optional[List[str]] = None, ip: Optional[str] = None,
               device: Optional[str] = None, ttl_s: Optional[int] = None) -> Tuple[str, Session]:
        sid = self.new_sid()
        s = Session(
            sid=sid,
            user_id=user_id,
            user_name=user_name,
            scopes=list(scopes or []),
            ip=ip,
            device=device,
        )
        if ttl_s is not None:
            s.expires_at = _now_ms() + int(ttl_s) * 1000
        s._limiter = RateLimiter(rate_per_sec=5.0, burst=10.0)
        self._mem[sid] = s
        self._mirror_write(s)
        return self.sign(sid), s

    def get(self, sid: str) -> Optional[Session]:
        s = self._mem.get(sid)
        if s: return s
        # try redis
        data = self._mirror_read(sid)
        if data:
            s = _from_dict(Session, data)
            s._limiter = RateLimiter(rate_per_sec=5.0, burst=10.0)
            self._mem[sid] = s
        return s

    def touch(self, sid: str, *, roll: bool = True) -> Optional[Session]:
        s = self.get(sid)
        if not s: return None
        s.last_seen_at = _now_ms()
        if roll and s.should_roll():
            s.expires_at = _now_ms() + SESSION_TTL_S * 1000
        self._mirror_write(s)
        return s

    def end(self, sid: str) -> bool:
        self._mem.pop(sid, None)
        self._mirror_del(sid)
        return True

    # -------- Accessors / helpers --------
    def set_kv(self, sid: str, key: str, value: Any) -> bool:
        s = self.get(sid)
        if not s: return False
        s.kv[key] = value
        if _json_size(s.kv) > MAX_KV_BYTES:
            # rudimentary LRU: drop oldest key
            try:
                k0 = next(iter(s.kv.keys()))
                s.kv.pop(k0, None)
            except Exception:
                pass
        self._mirror_write(s)
        return True

    def get_kv(self, sid: str, key: str, default: Any = None) -> Any:
        s = self.get(sid)
        return s.kv.get(key, default) if s else default

    def grant(self, sid: str, *scopes: str) -> bool:
        s = self.get(sid)
        if not s: return False
        for sc in scopes:
            if sc not in s.scopes: s.scopes.append(sc)
        self._mirror_write(s)
        return True

    def revoke(self, sid: str, *scopes: str) -> bool:
        s = self.get(sid)
        if not s: return False
        s.scopes = [x for x in s.scopes if x not in set(scopes)]
        self._mirror_write(s)
        return True

    # -------- Validation pipeline --------
    def validate(self, sid_sig: str, headers: Optional[Dict[str, str]] = None,
                 *, require_csrf: bool = True, need_scopes: Optional[List[str]] = None,
                 ip: Optional[str] = None, device: Optional[str] = None,
                 rate_cost: float = 1.0) -> Tuple[bool, Optional[Session], str]:
        """
        Returns (ok, session, reason)
        """
        sid = self.verify(sid_sig or "")
        if not sid:
            return False, None, "bad_signature"

        s = self.get(sid)
        if not s:
            return False, None, "unknown_session"

        if s.is_expired():
            return False, None, "expired"

        # CSRF (for state-changing methods; supply header)
        if require_csrf:
            token = (headers or {}).get(CSRF_HEADER, "")
            if token != s.csrf:
                return False, None, "csrf_mismatch"

        # IP/device sticky binding (optional)
        if ip and s.ip and s.ip != ip:
            return False, None, "ip_mismatch"
        if device and s.device and s.device != device:
            return False, None, "device_mismatch"

        # Scopes
        if need_scopes:
            need = set(need_scopes)
            have = set(s.scopes)
            if not need.issubset(have):
                return False, s, "insufficient_scopes"

        # Per-session rate limit
        if s._limiter and not s._limiter.allow(rate_cost):
            return False, s, "rate_limited"

        # Rolling touch
        self.touch(sid, roll=True)
        return True, s, "ok"

    # -------- Mirror to Redis --------
    def _mirror_write(self, s: Session) -> None:
        if _R is None: return
        try:
            ttl = max(1, int((s.expires_at - _now_ms()) / 1000))
            _R.setex(f"sess:{s.sid}", ttl, json.dumps(s.to_dict(), ensure_ascii=False))
        except Exception:
            pass

    def _mirror_read(self, sid: str) -> Optional[Dict[str, Any]]:
        if _R is None: return None
        try:
            v = _R.get(f"sess:{sid}")
            return json.loads(v) if v else None # type: ignore
        except Exception:
            return None

    def _mirror_del(self, sid: str) -> None:
        if _R is None: return
        try:
            _R.delete(f"sess:{sid}")
        except Exception:
            pass

# -------- Helpers --------
def _now_ms() -> int:
    return int(time.time() * 1000)

def _rand(n_bytes: int) -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(n_bytes)).decode("ascii").rstrip("=")

def _pad(s: str) -> bytes:
    # base64 urlsafe paddding
    return (s + "===").encode("ascii")

def _from_dict(cls, d: Dict[str, Any]):
    return cls(**d)  # dataclass is compatible with simple dict here

def _json_size(obj: Any) -> int:
    try:
        return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))
    except Exception:
        return 0

# -------- Minimal smoke test --------
if __name__ == "__main__":  # pragma: no cover
    store = SessionStore()
    sid_sig, s = store.create(user_id="om", user_name="Om", scopes=["read:quotes","trade:paper"], ip="127.0.0.1")
    print("cookie:", sid_sig)
    ok, sess, why = store.validate(sid_sig, headers={CSRF_HEADER: s.csrf}, need_scopes=["read:quotes"])
    print("validate:", ok, why, sess.user_name if sess else None)
    store.set_kv(sess.sid, "last_symbol", "AAPL") # type: ignore
    print("kv:", store.get_kv(sess.sid, "last_symbol")) # type: ignore
    print("end:", store.end(sess.sid)) # type: ignore