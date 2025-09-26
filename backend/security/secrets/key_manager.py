# security/secrets/key_manager.py
"""
Key Manager (versioned keys, rotation, envelope encryption).

Features
--------
- Versioned symmetric KEKs (Key Encryption Keys) with IDs like "k-2025-09-16-01"
- Envelope encryption: data key (DEK) generated per message, encrypted under KEK
- Optional signing keys (Ed25519) with verify
- Pluggable backends:
    * Local file keystore (JSON, chmod 600)
    * Environment variables (read-only)
    * (Optional) AWS KMS / HashiCorp Vault fetch if boto3/hvac are available

Dependencies
------------
- Recommended: `pip install cryptography`
- Optional: `pip install boto3 hvac`

Quick start
-----------
from security.secrets.key_manager import KeyManager, KMConfig

km = KeyManager(KMConfig(store_path="var/keystore.json"))
km.ensure_kek()                                # create new KEK if none
ct = km.encrypt(b"hello world")                # envelope encrypt
pt = km.decrypt(ct)                            # -> b"hello world"

# rotation
km.rotate_kek()                                # new active KEK; old remains for decrypt
"""

from __future__ import annotations

import base64
import json
import os
import time
import hmac
import hashlib
import secrets
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

# ---------- optional crypto deps ----------
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey  # type: ignore
    from cryptography.hazmat.primitives import serialization  # type: ignore
    _HAVE_CRYPTO = True
except Exception:
    _HAVE_CRYPTO = False

try:
    import boto3  # type: ignore
except Exception:
    boto3 = None  # type: ignore

try:
    import hvac  # type: ignore
except Exception:
    hvac = None  # type: ignore


# ---------- helpers ----------

def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))

def _now_iso() -> str:
    import datetime as dt
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()

def _rand32() -> bytes:
    return secrets.token_bytes(32)

def _require_crypto(op: str):
    if not _HAVE_CRYPTO:
        raise RuntimeError(f"{op} requires the 'cryptography' package. Install with: pip install cryptography")


# ---------- config & state ----------

@dataclass
class KMConfig:
    # local file keystore (JSON)
    store_path: Optional[str] = None          # e.g., "var/keystore.json"
    chmod_600: bool = True

    # environment variables (read-only fallback)
    env_kek_b64: Optional[str] = None         # set to the env var name holding base64 KEK
    env_key_id: Optional[str] = None          # env var name for key ID

    # optional cloud/vault fetch
    aws_kms_key_id: Optional[str] = None      # if set, you can wrap/unwrap via KMS (envelope)
    aws_region: Optional[str] = None
    vault_addr: Optional[str] = None          # VAULT_ADDR; needs role/token separately
    vault_token: Optional[str] = None
    vault_kv_path: Optional[str] = None       # e.g. "kv/data/newsintel/kek"

    # signing keys (Ed25519) file paths (PEM)
    sign_priv_path: Optional[str] = None
    sign_pub_path: Optional[str] = None

    # policy
    kek_len: int = 32                         # 32 bytes -> AES-256-GCM
    rotate_suffix_len: int = 2                # digits in suffix, e.g. 01..99


@dataclass
class _Keystore:
    # structure persisted to file (JSON)
    active_key_id: Optional[str] = None
    keys: Dict[str, str] = field(default_factory=dict)     # key_id -> b64(kek)
    created_at: Dict[str, str] = field(default_factory=dict)
    # signing key material (optional)
    ed25519_priv_b64: Optional[str] = None
    ed25519_pub_b64: Optional[str] = None


# ---------- KeyManager ----------

class KeyManager:
    """
    Versioned KEKs with envelope encryption (AES-256-GCM).
    Ciphertext format (JSON bytes):
        {
          "v": 1,
          "key_id": "k-2025-09-16-01",
          "nonce": "<b64>",
          "ekey": "<b64-encrypted DEK under KEK>",
          "ct": "<b64-ciphertext using DEK>",
          "aad": "<b64-optional AAD>"
        }
    """

    def __init__(self, cfg: KMConfig):
        self.cfg = cfg
        self.ks = _Keystore()
        self._load_keystore()
        # env fallback
        if not self.ks.active_key_id and cfg.env_kek_b64 and cfg.env_key_id:
            kid = os.getenv(cfg.env_key_id)
            kek_b64 = os.getenv(cfg.env_kek_b64)
            if kid and kek_b64:
                self.ks.active_key_id = kid
                self.ks.keys[kid] = kek_b64

    # ----- public: KEK lifecycle -----

    def ensure_kek(self) -> str:
        """Ensure an active KEK exists; returns key_id."""
        if self.ks.active_key_id and self.ks.active_key_id in self.ks.keys:
            return self.ks.active_key_id
        return self.rotate_kek()

    def rotate_kek(self) -> str:
        """Generate a new KEK, set active, keep older keys for decrypt."""
        kek = secrets.token_bytes(self.cfg.kek_len)
        key_id = self._new_key_id()
        self.ks.keys[key_id] = _b64e(kek)
        self.ks.created_at[key_id] = _now_iso()
        self.ks.active_key_id = key_id
        self._persist()
        return key_id

    def list_keys(self) -> Dict[str, str]:
        """Return mapping of key_id -> created_at (ISO)."""
        return dict(self.ks.created_at)

    def revoke_key(self, key_id: str) -> None:
        """Remove a key (cannot decrypt data encrypted with it afterwards)."""
        if key_id in self.ks.keys:
            del self.ks.keys[key_id]
            self.ks.created_at.pop(key_id, None)
            if self.ks.active_key_id == key_id:
                self.ks.active_key_id = None
            self._persist()

    # ----- public: envelope encryption -----

    def encrypt(self, plaintext: bytes, *, aad: Optional[bytes] = None) -> bytes:
        """
        Envelope encrypt:
          1) generate random DEK (32 bytes)
          2) encrypt plaintext with DEK using AES-256-GCM
          3) encrypt DEK with active KEK using AES-256-GCM
        Returns a compact JSON blob (bytes).
        """
        _require_crypto("encrypt")
        kid = self.ensure_kek()
        kek = _b64d(self.ks.keys[kid])

        dek = _rand32()
        nonce_data = secrets.token_bytes(12)
        nonce_key = secrets.token_bytes(12)

        aes_data = AESGCM(dek)
        ct = aes_data.encrypt(nonce_data, plaintext, aad)

        aes_kek = AESGCM(kek)
        ekey = aes_kek.encrypt(nonce_key, dek, aad)

        blob = {
            "v": 1,
            "key_id": kid,
            "nonce": _b64e(nonce_data),
            "ekey": _b64e(ekey),
            "knonce": _b64e(nonce_key),
            "ct": _b64e(ct),
            "aad": _b64e(aad) if aad else None,
        }
        return json.dumps(blob, separators=(",", ":")).encode("utf-8")

    def decrypt(self, blob: bytes, *, aad: Optional[bytes] = None) -> bytes:
        """
        Reverse envelope encryption. Uses key_id in blob to find the KEK.
        If blob carries its own AAD, we prefer that; else use provided `aad`.
        """
        _require_crypto("decrypt")
        obj = json.loads(blob.decode("utf-8"))
        if obj.get("v") != 1:
            raise ValueError("unsupported blob version")
        kid = obj["key_id"]
        if kid not in self.ks.keys:
            raise KeyError(f"unknown key_id: {kid}")

        kek = _b64d(self.ks.keys[kid])
        ekey = _b64d(obj["ekey"])
        knonce = _b64d(obj["knonce"])
        nonce = _b64d(obj["nonce"])
        ct = _b64d(obj["ct"])
        aad_used = _b64d(obj["aad"]) if obj.get("aad") else aad

        dek = AESGCM(kek).decrypt(knonce, ekey, aad_used)
        pt = AESGCM(dek).decrypt(nonce, ct, aad_used)
        return pt

    # ----- public: HMAC (no external deps) -----

    def hmac(self, message: bytes, *, key_id: Optional[str] = None) -> Tuple[str, str]:
        """
        Compute HMAC-SHA256 over message using KEK (OK for integrity tokens).
        Returns (key_id, base64_mac).
        """
        kid = key_id or self.ensure_kek()
        key = _b64d(self.ks.keys[kid])
        mac = hmac.new(key, message, hashlib.sha256).digest()
        return (kid, _b64e(mac))

    def hmac_verify(self, message: bytes, mac_b64: str, *, key_id: str) -> bool:
        try:
            mac = _b64d(mac_b64)
            key = _b64d(self.ks.keys[key_id])
            exp = hmac.new(key, message, hashlib.sha256).digest()
            return hmac.compare_digest(mac, exp)
        except Exception:
            return False

    # ----- public: Ed25519 signing (optional) -----

    def ensure_ed25519(self) -> None:
        _require_crypto("Ed25519 keys")
        if self.ks.ed25519_priv_b64 and self.ks.ed25519_pub_b64:
            return
        priv = Ed25519PrivateKey.generate()
        pub = priv.public_key()
        self.ks.ed25519_priv_b64 = _b64e(priv.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        ))
        self.ks.ed25519_pub_b64 = _b64e(pub.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        ))
        self._persist()

    def sign(self, message: bytes) -> bytes:
        _require_crypto("sign")
        if not self.ks.ed25519_priv_b64:
            self.ensure_ed25519()
        priv = Ed25519PrivateKey.from_private_bytes(_b64d(self.ks.ed25519_priv_b64)) # type: ignore
        return priv.sign(message)

    def verify(self, message: bytes, signature: bytes) -> bool:
        _require_crypto("verify")
        if not self.ks.ed25519_pub_b64:
            return False
        pub = Ed25519PublicKey.from_public_bytes(_b64d(self.ks.ed25519_pub_b64))
        try:
            pub.verify(signature, message)
            return True
        except Exception:
            return False

    def export_public_signing_key(self) -> Optional[bytes]:
        return _b64d(self.ks.ed25519_pub_b64) if self.ks.ed25519_pub_b64 else None

    # ----- optional: cloud/vault helpers (best-effort) -----

    def fetch_kek_from_aws_kms(self) -> Optional[str]:
        """
        If you manage KEK material via AWS KMS (GenerateRandom or data key),
        you can fetch a raw 32-byte random value here and set it active.
        """
        if not self.cfg.aws_kms_key_id or boto3 is None:
            return None
        kms = boto3.client("kms", region_name=self.cfg.aws_region)
        # GenerateDataKey without plaintext master is not allowed; common pattern is:
        # - Create a DEK via KMS.GenerateDataKey(KeyId=master_kms_key_id, KeySpec='AES_256')
        # - Use Plaintext as KEK and store CiphertextBlob if you want to rehydrate later via KMS
        out = kms.generate_data_key(KeyId=self.cfg.aws_kms_key_id, KeySpec="AES_256")
        kek = out["Plaintext"]
        kid = self._new_key_id(prefix="k-kms")
        self.ks.keys[kid] = _b64e(kek)
        self.ks.created_at[kid] = _now_iso()
        self.ks.active_key_id = kid
        self._persist()
        # zeroize plaintext (best effort)
        return kid

    def fetch_kek_from_vault(self) -> Optional[str]:
        """
        Read KEK material from Vault KV (expects base64 in a field 'kek_b64').
        """
        if not (self.cfg.vault_addr and self.cfg.vault_token and self.cfg.vault_kv_path) or hvac is None:
            return None
        client = hvac.Client(url=self.cfg.vault_addr, token=self.cfg.vault_token)
        resp = client.secrets.kv.v2.read_secret_version(path=self.cfg.vault_kv_path)
        data = resp["data"]["data"]
        kek_b64 = data.get("kek_b64")
        if not kek_b64:
            return None
        kid = self._new_key_id(prefix="k-vault")
        self.ks.keys[kid] = kek_b64
        self.ks.created_at[kid] = _now_iso()
        self.ks.active_key_id = kid
        self._persist()
        return kid

    # ----- internal: keystore I/O -----

    def _load_keystore(self) -> None:
        path = self.cfg.store_path
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            self.ks = _Keystore(
                active_key_id=obj.get("active_key_id"),
                keys=obj.get("keys", {}),
                created_at=obj.get("created_at", {}),
                ed25519_priv_b64=obj.get("ed25519_priv_b64"),
                ed25519_pub_b64=obj.get("ed25519_pub_b64"),
            )
        except Exception:
            # fall back to empty
            self.ks = _Keystore()

    def _persist(self) -> None:
        if not self.cfg.store_path:
            return
        os.makedirs(os.path.dirname(self.cfg.store_path) or ".", exist_ok=True)
        tmp = self.cfg.store_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({
                "active_key_id": self.ks.active_key_id,
                "keys": self.ks.keys,
                "created_at": self.ks.created_at,
                "ed25519_priv_b64": self.ks.ed25519_priv_b64,
                "ed25519_pub_b64": self.ks.ed25519_pub_b64,
            }, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.cfg.store_path)
        if self.cfg.chmod_600:
            try:
                os.chmod(self.cfg.store_path, 0o600)
            except Exception:
                pass

    def _new_key_id(self, *, prefix: str = "k") -> str:
        # k-YYYY-MM-DD-XX (XX is rolling suffix to avoid collisions within the same day)
        day = time.strftime("%Y-%m-%d")
        base = f"{prefix}-{day}-"
        # find first free suffix
        for i in range(1, 100):
            sfx = f"{i:0{self.cfg.rotate_suffix_len}d}"
            kid = base + sfx
            if kid not in self.ks.keys:
                return kid
        # fallback to random
        return f"{prefix}-{day}-{secrets.token_hex(2)}"