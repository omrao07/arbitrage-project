# security/jwt_issuer.py
"""
JWT Issuer/Verifier with rotation.

Algs:
  - HS256  (shared secret bytes)
  - RS256  (RSA PEM keys)
  - EdDSA  (Ed25519; requires 'cryptography')

Features:
  - 'kid' header support (rotate verify keys easily)
  - standard claims helpers (iss, aud, sub, iat, nbf, exp)
  - clock skew leeway
  - optional opaque refresh tokens using HMAC-SHA256

Install:
  pip install pyjwt cryptography

Quick start:
  from security.jwt_issuer import JWTIssuer, JWTConfig
  iss = JWTIssuer(JWTConfig(issuer="news-intel", audience="users", alg="EdDSA",
                            ed_priv_b64=os.getenv("ED25519_PRIV_B64"),
                            ed_pub_b64=os.getenv("ED25519_PUB_B64")))
  tok = iss.issue({"sub":"user123"}, ttl_s=3600)
  claims = iss.verify(tok)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# --- third-party (required) ---
try:
    import jwt  # type: ignore # PyJWT
except Exception as e:  # pragma: no cover
    raise SystemExit("jwt_issuer.py requires PyJWT. Install with: pip install pyjwt") from e

# --- cryptography (for RS256/EdDSA objects) ---
try:
    from cryptography.hazmat.primitives import serialization # type: ignore
    from cryptography.hazmat.primitives.asymmetric import rsa, ed25519 # type: ignore
    _HAVE_CRYPTO = True
except Exception:
    _HAVE_CRYPTO = False


# ===================== helpers =====================

def _now() -> int:
    return int(time.time())

def _b64urld(s: str) -> bytes:
    s += "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s.encode("ascii"))

def _b64urle(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")

def _require_crypto(op: str):
    if not _HAVE_CRYPTO:
        raise RuntimeError(f"{op} requires 'cryptography'. pip install cryptography")


# ===================== config =====================

@dataclass
class JWTConfig:
    issuer: str
    audience: Optional[str] = None
    alg: str = "EdDSA"  # "HS256" | "RS256" | "EdDSA"

    # HS256
    hs_secret_b64: Optional[str] = None  # base64url or base64 secret
    # env fallback if not set: HS_SECRET_B64

    # RS256 (PEM bytes)
    private_key_pem: Optional[bytes] = None
    public_keys_pem: Dict[str, bytes] = field(default_factory=dict)  # kid -> PEM

    # EdDSA (raw keys, base64)
    ed_priv_b64: Optional[str] = None  # raw 32 bytes (not PEM)
    ed_pub_b64: Optional[str] = None   # raw 32 bytes (not PEM)

    # global defaults
    default_ttl_s: int = 3600
    leeway_s: int = 60

    # refresh token (opaque HMAC) secret: base64; if None, falls back to HS secret
    refresh_secret_b64: Optional[str] = None


# ===================== main class =====================

class JWTIssuer:
    def __init__(self, cfg: JWTConfig):
        self.cfg = cfg
        self.alg = cfg.alg.upper()
        if self.alg not in ("HS256", "RS256", "EDDSA"):
            raise ValueError("alg must be HS256, RS256, or EdDSA")

        # secrets / keys
        self._hs_secret = self._load_hs()
        self._rs_priv = self._load_rs_private()
        self._rs_pub = dict(cfg.public_keys_pem)  # kid->PEM
        self._ed_priv, self._ed_pub = self._load_ed_keys()

    # ---------- public API ----------

    def issue(
        self,
        claims: Dict[str, Any],
        *,
        ttl_s: Optional[int] = None,
        kid: Optional[str] = None,
        nbf_s: Optional[int] = None,
        headers: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Issue a signed JWT.
        claims must contain at least 'sub' (subject). 'iss','aud','iat','exp','nbf' auto-managed.
        """
        ttl = int(ttl_s or self.cfg.default_ttl_s)
        now = _now()
        payload = dict(claims)
        payload.setdefault("iss", self.cfg.issuer)
        if self.cfg.audience:
            payload.setdefault("aud", self.cfg.audience)
        payload.setdefault("iat", now)
        if nbf_s is not None:
            payload.setdefault("nbf", now + int(nbf_s))
        payload.setdefault("exp", now + ttl)

        h = dict(headers or {})
        key = self._signing_key_and_set_kid(h, kid)

        token = jwt.encode(payload, key, algorithm=self._pyjwt_alg(), headers=h)
        # PyJWT returns str in v2; keep as str
        return token

    def verify(
        self,
        token: str,
        *,
        audience: Optional[str] = None,
        leeway_s: Optional[int] = None,
        require_sub: bool = True,
    ) -> Dict[str, Any]:
        """
        Verify a JWT (signature + reserved claims).
        Returns decoded claims.
        """
        hdr = jwt.get_unverified_header(token)
        kid = hdr.get("kid")

        key = self._verification_key(kid)
        opts = {
            "require": ["iss", "exp", "iat"] + (["sub"] if require_sub else []),
        }
        aud = audience if audience is not None else self.cfg.audience
        return jwt.decode(
            token,
            key=key,
            algorithms=[self._pyjwt_alg()],
            issuer=self.cfg.issuer,
            audience=aud,
            leeway=int(leeway_s if leeway_s is not None else self.cfg.leeway_s),
            options=opts,
        )

    # ---------- refresh tokens (opaque HMAC, optional) ----------

    def mint_refresh(self, sub: str, *, ttl_s: int = 30 * 24 * 3600) -> str:
        """
        Create an opaque refresh token: base64url(header).base64url(payload).base64url(hmac)
        header: {"v":1,"alg":"HS256"}
        payload: {"iss":..., "sub":..., "iat":..., "exp":...}
        """
        secret = self._refresh_secret()
        now = _now()
        pl = {"iss": self.cfg.issuer, "sub": sub, "iat": now, "exp": now + int(ttl_s)}
        hd = {"v": 1, "alg": "HS256"}

        hdb = _b64urle(json.dumps(hd, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        plb = _b64urle(json.dumps(pl, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        msg = f"{hdb}.{plb}".encode("ascii")
        mac = hmac.new(secret, msg, hashlib.sha256).digest()
        return f"{hdb}.{plb}.{_b64urle(mac)}"

    def verify_refresh(self, token: str) -> Dict[str, Any]:
        """
        Verify opaque refresh token; returns payload dict on success.
        Raises ValueError on failure.
        """
        secret = self._refresh_secret()
        try:
            hdb, plb, macb = token.split(".")
        except ValueError:
            raise ValueError("malformed refresh token")
        msg = f"{hdb}.{plb}".encode("ascii")
        mac = _b64urld(macb)
        exp_mac = hmac.new(secret, msg, hashlib.sha256).digest()
        if not hmac.compare_digest(mac, exp_mac):
            raise ValueError("bad refresh token MAC")

        payload = json.loads(_b64urld(plb).decode("utf-8"))
        now = _now()
        if payload.get("iss") != self.cfg.issuer:
            raise ValueError("bad issuer")
        if int(payload.get("exp", 0)) < now - self.cfg.leeway_s:
            raise ValueError("refresh token expired")
        return payload

    # ================== internals ==================

    def _signing_key_and_set_kid(self, headers: Dict[str, Any], kid: Optional[str]):
        if self.alg == "HS256":
            key = self._hs_secret
            if key is None:
                raise RuntimeError("HS256 selected but no secret configured")
            headers.setdefault("kid", kid or "hs")
            return key

    # RS256
        if self.alg == "RS256":
            _require_crypto("RS256")
            if self._rs_priv is None:
                raise RuntimeError("RS256 selected but private_key_pem not configured")
            # choose or derive kid from public key fingerprint
            k = kid or self._rs_kid_from_priv(self._rs_priv)
            headers.setdefault("kid", k)
            return self._rs_priv

    # EdDSA
        if self.alg == "EDDSA":
            _require_crypto("EdDSA")
            if self._ed_priv is None:
                raise RuntimeError("EdDSA selected but ed25519 private key not configured")
            k = kid or self._eddsa_kid(self._ed_pub)
            headers.setdefault("kid", k)
            return self._ed_priv

        raise ValueError("unknown alg")

    def _verification_key(self, kid: Optional[str]):
        if self.alg == "HS256":
            if self._hs_secret is None:
                raise RuntimeError("HS256 verify requested but no secret configured")
            return self._hs_secret

        if self.alg == "RS256":
            _require_crypto("RS256")
            # try exact kid, else derive from priv if present
            if kid and kid in self._rs_pub:
                return self._rs_pub[kid]
            if self._rs_priv is not None:
                # derive pub from private
                priv = serialization.load_pem_private_key(self._rs_priv, password=None)
                pub_pem = priv.public_key().public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
                return pub_pem
            raise RuntimeError("no RS256 public key for verification")

        if self.alg == "EDDSA":
            _require_crypto("EdDSA")
            if self._ed_pub is not None:
                return self._ed_pub
            if self._ed_priv is not None:
                # derive pub from private
                priv = serialization.load_pem_private_key(self._ed_priv, password=None)
                pub_pem = priv.public_key().public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
                return pub_pem
            raise RuntimeError("no Ed25519 public key for verification")

        raise ValueError("unknown alg")

    # ----- loaders -----

    def _load_hs(self) -> Optional[bytes]:
        s = self.cfg.hs_secret_b64 or os.getenv("HS_SECRET_B64")
        if not s:
            return None
        try:
            return _b64urld(s)
        except Exception:
            # allow std base64 too
            return base64.b64decode(s.encode("ascii"))

    def _load_rs_private(self) -> Optional[bytes]:
        if self.cfg.private_key_pem:
            return self.cfg.private_key_pem
        return None

    def _load_ed_keys(self) -> tuple[Optional[bytes], Optional[bytes]]:
        if self.alg != "EDDSA":
            return (None, None)
        _require_crypto("EdDSA")
        priv_b64 = self.cfg.ed_priv_b64 or os.getenv("ED25519_PRIV_B64")
        pub_b64 = self.cfg.ed_pub_b64 or os.getenv("ED25519_PUB_B64")

        if not priv_b64 and not pub_b64:
            # nothing configured (caller might only verify with provided pub in future)
            return (None, None)

        priv_pem: Optional[bytes] = None
        pub_pem: Optional[bytes] = None

        if priv_b64:
            raw = _b64urld(priv_b64)
            if len(raw) != 32:
                raise ValueError("ed_priv_b64 must be raw 32-byte Ed25519 key (base64/base64url)")
            priv = ed25519.Ed25519PrivateKey.from_private_bytes(raw)
            priv_pem = priv.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            pub = priv.public_key()
            pub_pem = pub.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )

        if pub_b64 and not pub_pem:
            raw_pub = _b64urld(pub_b64)
            if len(raw_pub) != 32:
                raise ValueError("ed_pub_b64 must be raw 32-byte Ed25519 pubkey (base64/base64url)")
            pub = ed25519.Ed25519PublicKey.from_public_bytes(raw_pub)
            pub_pem = pub.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )

        return (priv_pem, pub_pem)

    # ----- key IDs -----

    def _rs_kid_from_priv(self, pem: bytes) -> str:
        priv = serialization.load_pem_private_key(pem, password=None)
        pub_pem = priv.public_key().public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        h = hashlib.sha256(pub_pem).digest()
        return "rs-" + _b64urle(h[:8])

    def _eddsa_kid(self, pub_pem: Optional[bytes]) -> str:
        if not pub_pem:
            return "ed-unknown"
        h = hashlib.sha256(pub_pem).digest()
        return "ed-" + _b64urle(h[:8])

    # ----- PyJWT alg mapping -----

    def _pyjwt_alg(self) -> str:
        return {"HS256": "HS256", "RS256": "RS256", "EDDSA": "EdDSA"}[self.alg]

    # ----- refresh secret -----

    def _refresh_secret(self) -> bytes:
        s = self.cfg.refresh_secret_b64 or self.cfg.hs_secret_b64 or os.getenv("REFRESH_SECRET_B64") or os.getenv("HS_SECRET_B64")
        if not s:
            raise RuntimeError("refresh secret not configured (set refresh_secret_b64 or HS_SECRET_B64)")
        try:
            return _b64urld(s)
        except Exception:
            return base64.b64decode(s.encode("ascii"))


# ===================== __main__ (smoke) =====================

if __name__ == "__main__":
    # Minimal self-test using EdDSA with ephemeral key from env.
    # Generate a quick key for testing:
    if _HAVE_CRYPTO and not os.getenv("ED25519_PRIV_B64"):
        raw = os.urandom(32)
        os.environ["ED25519_PRIV_B64"] = _b64urle(raw)
        os.environ["ED25519_PUB_B64"] = _b64urle(ed25519.Ed25519PrivateKey.from_private_bytes(raw).public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw))
    cfg = JWTConfig(issuer="news-intel", audience="users", alg="EdDSA",
                    ed_priv_b64=os.getenv("ED25519_PRIV_B64"),
                    ed_pub_b64=os.getenv("ED25519_PUB_B64"))
    iss = JWTIssuer(cfg)
    tok = iss.issue({"sub": "demo"}, ttl_s=5)
    print("JWT:", tok)
    print("Claims:", iss.verify(tok))
    r = iss.mint_refresh("demo", ttl_s=60)
    print("Refresh:", r)
    print("Refresh payload:", iss.verify_refresh(r))