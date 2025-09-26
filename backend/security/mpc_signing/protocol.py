# security/mpc_signing/protocol.py
"""
MPC protocol strategy (request/accept/aggregate scaffolding).

⚠️ This file contains NO cryptography. It only:
  - shapes the JSON payloads sent to participant agents,
  - keeps minimal round state (who responded, nonces seen, partials),
  - validates simple invariants (threshold/quorum),
  - defines a single place to plug a real aggregator.

Plugging a real scheme:
  - FROST (Schnorr/Ed25519/secp256k1): replace `aggregate()` with a call that
    verifies & combines partial signatures into a final Schnorr signature.
  - ECDSA (GG18/CGGMP): same idea; combine partials with the library you use.

Participant agents
  Each participant should expose handlers keyed by the `method` strings:
    - "keygen"       -> returns {"ok": true,  "pubkey": "<hex or base64>", "party_id": "<id>"}
    - "presign"      -> returns {"ok": true,  "nonce": "<opaque>", "party_id": "<id>"}
    - "partial_sign" -> returns {"ok": true,  "partial": "<hex>", "party_id": "<id>"}
  You can extend these shapes as needed; this Strategy passes through extra fields.

Safety
  - DO NOT use this stub in production without wiring a vetted MPC library
    into `aggregate()` (and optionally richer validation in accept_* steps).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Import the Strategy protocol to satisfy the coordinator
try:
    from security.mpc_signing.coordinator import Strategy # type: ignore
except Exception:  # fallback if path differs during local tests
    class Strategy:  # type: ignore
        async def keygen_request(self, pid: str) -> Dict[str, Any]: ...
        async def keygen_accept(self, pid: str, resp: Dict[str, Any]) -> None: ...
        async def public_key(self) -> Dict[str, Any]: ...
        async def presign_request(self, pid: str) -> Dict[str, Any]: ...
        async def presign_accept(self, pid: str, resp: Dict[str, Any]) -> None: ...
        async def partial_sign_request(self, pid: str, message: bytes) -> Dict[str, Any]: ...
        async def partial_sign_accept(self, pid: str, resp: Dict[str, Any]) -> None: ...
        async def aggregate(self, participants: List[str], message: bytes) -> Dict[str, Any]: ...


class ProtocolError(RuntimeError):
    pass


@dataclass
class RoundState:
    threshold: int
    scheme: str = "frost-ed25519"   # free-form tag; e.g., "frost-secp256k1", "gg18-ecdsa"
    context: str = "news-intel"     # domain separation tag for hashing
    pubkey: Optional[str] = None    # hex/base64, as provided by agents
    nonces: Dict[str, Any] = field(default_factory=dict)     # pid -> nonce (opaque)
    partials: Dict[str, str] = field(default_factory=dict)   # pid -> partial signature (hex)
    extra: Dict[str, Any] = field(default_factory=dict)      # free-form extension space


class FrostLikeStrategy(Strategy):
    """
    A strategy that speaks a FROST-like JSON dialect to MPC agents.
    Replace `aggregate()` with a real combiner from your MPC library.
    """

    def __init__(self, threshold: int, *, scheme: str = "frost-ed25519", context: str = "news-intel"):
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        self.state = RoundState(threshold=threshold, scheme=scheme, context=context)

    # ---------- keygen ----------

    async def keygen_request(self, pid: str) -> Dict[str, Any]:
        # Agents may be idempotent: if key exists, they return the same pubkey.
        return {
            "op": "keygen",
            "scheme": self.state.scheme,
            "party_id": pid,
        }

    async def keygen_accept(self, pid: str, resp: Dict[str, Any]) -> None:
        if not resp.get("ok"):
            raise ProtocolError(f"keygen failed for {pid}: {resp}")
        pk = resp.get("pubkey")
        if not isinstance(pk, str) or not pk:
            raise ProtocolError(f"keygen missing pubkey from {pid}")
        # If this is the first pubkey we've seen, fix it; else ensure consistency.
        if self.state.pubkey is None:
            self.state.pubkey = pk
        elif self.state.pubkey != pk:
            raise ProtocolError(f"mismatched pubkey from {pid}")

    async def public_key(self) -> Dict[str, Any]:
        if not self.state.pubkey:
            # return a neutral descriptor to help debugging if keygen not run
            return {"scheme": self.state.scheme, "pk": None}
        return {"scheme": self.state.scheme, "pk": self.state.pubkey}

    # ---------- presign ----------

    async def presign_request(self, pid: str) -> Dict[str, Any]:
        # Many protocols pre-compute nonces/commitments; this asks agent to prep/cache one.
        return {
            "op": "presign",
            "scheme": self.state.scheme,
            "party_id": pid,
        }

    async def presign_accept(self, pid: str, resp: Dict[str, Any]) -> None:
        if not resp.get("ok"):
            raise ProtocolError(f"presign failed for {pid}: {resp}")
        # Store opaque nonce handle or material the agent wants us to echo later (optional).
        nonce = resp.get("nonce")
        if nonce is not None:
            self.state.nonces[pid] = nonce

    # ---------- partial signing ----------

    async def partial_sign_request(self, pid: str, message: bytes) -> Dict[str, Any]:
        # Domain-separated challenge hash is typically computed by the agent,
        # but we provide a helper digest as an input hint (some agents use it).
        digest = _domain_hash(self.state.context, message)
        req: Dict[str, Any] = {
            "op": "partial_sign",
            "scheme": self.state.scheme,
            "party_id": pid,
            "digest": digest.hex(),     # optional; agent may ignore and recompute
        }
        if pid in self.state.nonces:
            req["nonce"] = self.state.nonces[pid]
        return req

    async def partial_sign_accept(self, pid: str, resp: Dict[str, Any]) -> None:
        if not resp.get("ok"):
            raise ProtocolError(f"partial_sign failed for {pid}: {resp}")
        part = resp.get("partial")
        if not isinstance(part, str) or not part:
            raise ProtocolError(f"partial_sign missing 'partial' for {pid}")
        self.state.partials[pid] = part

    # ---------- aggregation ----------

    async def aggregate(self, participants: List[str], message: bytes) -> Dict[str, Any]:
        """
        Combine partial signatures from the given participants into a final signature.

        TODO (replace):
          - Use your MPC library's verify/combine here.
          - Validate each partial against the (party_id, commitments, pubkey).
          - Return {"ok": True, "signature": "<hex>"} on success.

        CURRENT PLACEHOLDER:
          - Returns a deterministic digest over (sorted participant ids, pubkey, message),
            which is NOT a signature. For wiring tests only.
        """
        t = self.state.threshold
        have = [pid for pid in participants if pid in self.state.partials]
        if len(have) < t:
            return {"ok": False, "error": f"need {t} partials, have {len(have)}"}

        # --- Replace below with real aggregator ---
        h = hashlib.sha256()
        h.update(b"NOT-A-SIGNATURE|")
        h.update((self.state.pubkey or "").encode())
        h.update(b"|")
        h.update(",".join(sorted(have)).encode())
        h.update(b"|")
        h.update(message)
        fake = h.hexdigest()
        return {"ok": True, "signature": fake}
        # ------------------------------------------

# ---------------------------- helpers ----------------------------

def _domain_hash(domain: str, msg: bytes) -> bytes:
    """
    Domain-separated hash helper: H( len(domain) || domain || msg ).
    Agents can ignore this and recompute; provided here for convenience.
    """
    d = domain.encode()
    return hashlib.sha256(len(d).to_bytes(2, "big") + d + msg).digest()