# security/secrets/vault_client.py
"""
Thin Vault client with hvac-or-HTTP fallback.

Features
--------
- Auth: token or AppRole (role_id/secret_id)
- KV v2: read/write/list/delete (with JSON helpers)
- Transit (optional): encrypt/decrypt
- Namespaces & retries
- Small in-memory cache with TTL
- Safe defaults & explicit exceptions

Env (defaults)
--------------
VAULT_ADDR=https://127.0.0.1:8200
VAULT_TOKEN=<root-or-approle token>
VAULT_NAMESPACE=<namespace/tenant>       # optional
VAULT_ROLE_ID=<approle role id>          # if using AppRole
VAULT_SECRET_ID=<approle secret id>      # if using AppRole

Usage
-----
vc = VaultClient()                           # picks up env
secret = vc.kv_get("kv/data/newsintel/api")  # for KV v2 mount 'kv'
vc.kv_put("kv/data/newsintel/api", {"key":"abc"})
vc.renew_self()
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Optional deps
try:
    import hvac  # type: ignore
except Exception:
    hvac = None  # type: ignore

try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore


# --------------------------- Errors ---------------------------

class VaultError(RuntimeError):
    pass


# --------------------------- Config ---------------------------

@dataclass
class VaultConfig:
    addr: str = field(default_factory=lambda: os.getenv("VAULT_ADDR", "http://127.0.0.1:8200"))
    token: Optional[str] = field(default_factory=lambda: os.getenv("VAULT_TOKEN"))
    namespace: Optional[str] = field(default_factory=lambda: os.getenv("VAULT_NAMESPACE"))
    role_id: Optional[str] = field(default_factory=lambda: os.getenv("VAULT_ROLE_ID"))
    secret_id: Optional[str] = field(default_factory=lambda: os.getenv("VAULT_SECRET_ID"))
    # HTTP behavior
    timeout_s: float = 8.0
    max_retries: int = 3
    backoff_base_s: float = 0.25
    # cache
    cache_ttl_s: float = 15.0
    # mounts (override if your mounts differ)
    kv_mount: str = "kv"          # KV v2 mount name (e.g., "kv")
    transit_mount: str = "transit"


# --------------------------- Client ---------------------------

class VaultClient:
    def __init__(self, cfg: Optional[VaultConfig] = None):
        self.cfg = cfg or VaultConfig()
        self._token = self.cfg.token
        self._session = None  # requests.Session
        self._hvac = None     # hvac.Client
        self._cache: Dict[str, Tuple[float, Any]] = {}  # key -> (exp_ts, value)

        if hvac is not None:
            # Try hvac first if token/approle present
            try:
                self._hvac = hvac.Client(
                    url=self.cfg.addr,
                    token=self._token,
                    namespace=self.cfg.namespace,
                )
                if not self._token and self.cfg.role_id and self.cfg.secret_id:
                    self._hvac.auth.approle.login(self.cfg.role_id, self.cfg.secret_id)
                    self._token = self._hvac.token
            except Exception as e:
                self._hvac = None
                # fall back to HTTP
        if self._hvac is None:
            if requests is None:
                raise VaultError("Neither 'hvac' nor 'requests' is available; install one to use VaultClient")
            self._session = requests.Session()
            if self._token is None and self.cfg.role_id and self.cfg.secret_id:
                self._token = self._approle_login_http(self.cfg.role_id, self.cfg.secret_id)

    # --------------- Public: Auth / Token ---------------

    @property
    def token(self) -> Optional[str]:
        return self._token

    def renew_self(self) -> Dict[str, Any]:
        """Renew own token (if renewable)."""
        if self._hvac:
            try:
                out = self._hvac.auth.token.renew_self()
                return out or {}
            except Exception as e:
                raise VaultError(f"renew-self failed: {e}") from e
        # HTTP
        return self._http_post("/v1/auth/token/renew-self", {}) or {}

    # --------------- Public: KV v2 ---------------

    def kv_get(self, path: str, *, mount: Optional[str] = None, use_cache: bool = True) -> Dict[str, Any]:
        """
        Read secret data from KV v2.
        Path should include 'data/' segment for v2, e.g., 'kv/data/foo/bar'
        """
        mount = mount or self.cfg.kv_mount
        cache_key = f"kv_get:{path}"
        if use_cache:
            c = self._cache_get(cache_key)
            if c is not None:
                return c

        if self._hvac:
            try:
                # hvac wants: client.secrets.kv.v2.read_secret_version(mount_point=mount, path="foo/bar")
                rel = _strip_kv_data_prefix(path)
                resp = self._hvac.secrets.kv.v2.read_secret_version(mount_point=mount, path=rel)
                data = resp.get("data", {}).get("data", {})
                self._cache_put(cache_key, data)
                return data
            except Exception as e:
                raise VaultError(f"kv_get failed: {e}") from e

        # HTTP
        url = self._kv_url(path, mount)
        obj = self._http_get(url)
        data = ((obj or {}).get("data") or {}).get("data") or {}
        self._cache_put(cache_key, data)
        return data

    def kv_put(self, path: str, data: Dict[str, Any], *, mount: Optional[str] = None) -> None:
        """Write secret to KV v2."""
        mount = mount or self.cfg.kv_mount
        self._cache_invalidate(f"kv_get:{path}")

        if self._hvac:
            try:
                rel = _strip_kv_data_prefix(path)
                self._hvac.secrets.kv.v2.create_or_update_secret(mount_point=mount, path=rel, secret=data)
                return
            except Exception as e:
                raise VaultError(f"kv_put failed: {e}") from e

        url = self._kv_url(path, mount)
        self._http_post(url, {"data": data})

    def kv_delete(self, path: str, *, mount: Optional[str] = None) -> None:
        """Soft-delete a secret version on KV v2 (keeps metadata)."""
        mount = mount or self.cfg.kv_mount
        self._cache_invalidate(f"kv_get:{path}")

        if self._hvac:
            try:
                rel = _strip_kv_data_prefix(path)
                self._hvac.secrets.kv.v2.delete_latest_version_of_secret(mount_point=mount, path=rel)
                return
            except Exception as e:
                raise VaultError(f"kv_delete failed: {e}") from e

        url = self._kv_meta_url(path, mount).replace("/metadata/", "/delete/")
        self._http_post(url, {})  # POST to /delete for v2 delete-latest

    def kv_list(self, path: str, *, mount: Optional[str] = None) -> List[str]:
        """List keys under a path (KV v2). Use 'metadata/' in path for HTTP."""
        mount = mount or self.cfg.kv_mount
        if self._hvac:
            try:
                rel = _strip_kv_data_prefix(path)
                resp = self._hvac.secrets.kv.v2.list_secrets(mount_point=mount, path=rel)
                return (resp.get("data") or {}).get("keys") or []
            except Exception as e:
                raise VaultError(f"kv_list failed: {e}") from e

        url = self._kv_meta_url(path, mount)
        obj = self._http_list(url)
        return ((obj or {}).get("data") or {}).get("keys") or []

    # Convenience helpers (JSON payloads as strings)
    def kv_get_json(self, path: str, *, mount: Optional[str] = None) -> str:
        return json.dumps(self.kv_get(path, mount=mount), ensure_ascii=False)

    def kv_put_json(self, path: str, json_str: str, *, mount: Optional[str] = None) -> None:
        self.kv_put(path, json.loads(json_str), mount=mount)

    # --------------- Public: Transit ---------------

    def transit_encrypt(self, key: str, plaintext_b64: str, *, mount: Optional[str] = None, context_b64: Optional[str] = None) -> str:
        """Encrypt with transit; returns ciphertext (e.g., 'vault:v1:...')."""
        mount = mount or self.cfg.transit_mount
        body = {"plaintext": plaintext_b64}
        if context_b64:
            body["context"] = context_b64
        path = f"/v1/{mount}/encrypt/{key}"
        if self._hvac:
            try:
                resp = self._hvac.secrets.transit.encrypt_data(name=key, mount_point=mount, plaintext=plaintext_b64, context=context_b64)
                return (resp.get("data") or {}).get("ciphertext") or ""
            except Exception as e:
                raise VaultError(f"transit_encrypt failed: {e}") from e
        obj = self._http_post(path, body)
        return ((obj or {}).get("data") or {}).get("ciphertext") or ""

    def transit_decrypt(self, key: str, ciphertext: str, *, mount: Optional[str] = None, context_b64: Optional[str] = None) -> str:
        """Decrypt with transit; returns plaintext base64."""
        mount = mount or self.cfg.transit_mount
        body = {"ciphertext": ciphertext}
        if context_b64:
            body["context"] = context_b64
        path = f"/v1/{mount}/decrypt/{key}"
        if self._hvac:
            try:
                resp = self._hvac.secrets.transit.decrypt_data(name=key, mount_point=mount, ciphertext=ciphertext, context=context_b64)
                return (resp.get("data") or {}).get("plaintext") or ""
            except Exception as e:
                raise VaultError(f"transit_decrypt failed: {e}") from e
        obj = self._http_post(path, body)
        return ((obj or {}).get("data") or {}).get("plaintext") or ""

    # --------------- Internals: HTTP ---------------

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._token:
            h["X-Vault-Token"] = self._token
        if self.cfg.namespace:
            h["X-Vault-Namespace"] = self.cfg.namespace
        return h

    def _approle_login_http(self, role_id: str, secret_id: str) -> str:
        body = {"role_id": role_id, "secret_id": secret_id}
        resp = self._http_post("/v1/auth/approle/login", body)
        tok = (((resp or {}).get("auth") or {}).get("client_token")) if resp else None
        if not tok:
            raise VaultError("AppRole login failed: no token")
        return tok

    def _http_get(self, path_or_url: str) -> Dict[str, Any]:
        return self._http_req("GET", path_or_url, None)

    def _http_post(self, path_or_url: str, body: Dict[str, Any]) -> Dict[str, Any]:
        return self._http_req("POST", path_or_url, body)

    def _http_list(self, path_or_url: str) -> Dict[str, Any]:
        return self._http_req("LIST", path_or_url, None)

    def _http_req(self, method: str, path_or_url: str, body: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        assert self._session is not None, "requests.Session missing"
        url = path_or_url if path_or_url.startswith("http") else (self.cfg.addr.rstrip("/") + path_or_url)
        data = json.dumps(body).encode("utf-8") if body is not None else None

        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = self._session.request(
                    method=method,
                    url=url,
                    headers=self._headers(),
                    data=data,
                    timeout=self.cfg.timeout_s,
                )
                if resp.status_code >= 400:
                    raise VaultError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                if resp.content:
                    return resp.json()
                return {}
            except Exception as e:
                last_err = e
                if attempt >= self.cfg.max_retries:
                    raise VaultError(f"vault http error: {e}") from e
                _sleep(self.cfg.backoff_base_s * (2 ** attempt))
        # not reached
        raise VaultError(f"vault http error: {last_err}")

    # --------------- Internals: KV helpers ---------------

    def _kv_url(self, path: str, mount: str) -> str:
        # ensure /v1/<mount>/data/<path_without_mount_data_prefix>
        rel = _strip_kv_data_prefix(path)
        return f"/v1/{mount}/data/{rel}"

    def _kv_meta_url(self, path: str, mount: str) -> str:
        rel = _strip_kv_data_prefix(path)
        return f"/v1/{mount}/metadata/{rel}"

    # --------------- Cache ---------------

    def _cache_get(self, key: str):
        ttl = self.cfg.cache_ttl_s
        if ttl <= 0:
            return None
        it = self._cache.get(key)
        if not it:
            return None
        exp, val = it
        if time.time() < exp:
            return val
        # expired
        self._cache.pop(key, None)
        return None

    def _cache_put(self, key: str, val: Any):
        ttl = self.cfg.cache_ttl_s
        if ttl <= 0:
            return
        self._cache[key] = (time.time() + ttl, val)

    def _cache_invalidate(self, key: str):
        self._cache.pop(key, None)


# --------------------------- Helpers ---------------------------

def _strip_kv_data_prefix(path: str) -> str:
    """
    Accepts either 'kv/data/foo/bar' or 'foo/bar' and returns 'foo/bar'.
    """
    p = path.lstrip("/")
    if "/data/" in p:
        p = p.split("/data/", 1)[1]
    elif p.startswith("data/"):
        p = p[5:]
    elif p.startswith("kv/"):
        p = p.split("/", 1)[1] if "/" in p else ""
        if p.startswith("data/"):
            p = p[5:]
    return p

def _sleep(s: float) -> None:
    time.sleep(max(0.0, s))