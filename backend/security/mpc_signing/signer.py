# security/mpc_signing/signer.py
"""
MPC participant 'signer' agent (dependency-light).

This module exposes a minimal Signer that can serve three operations expected by
the FrostLikeStrategy stub:

  - keygen       -> {"ok": true, "pubkey": "<hex>", "party_id": "<id>"}
  - presign      -> {"ok": true, "nonce": "<opaque>", "party_id": "<id>"}
  - partial_sign -> {"ok": true, "partial": "<hex>", "party_id": "<id>"}

⚠️ Crypto note
The 'public key', 'nonce', and 'partial' returned here are placeholders so you
can wire transports and coordinator flows. Replace the signing logic with calls
into a real MPC library (FROST/GG18/etc.) before production use.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import secrets
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable, Awaitable


# --------------------------- Signer ---------------------------

@dataclass
class SignerConfig:
    party_id: str
    scheme: str = "frost-ed25519"
    context: str = "news-intel"
    # persistence (optional): where to cache a pseudo keypair
    keystore_path: Optional[str] = None


@dataclass
class SignerState:
    pubkey_hex: Optional[str] = None
    # nonce cache (very simple): one outstanding nonce handle per caller
    nonces: Dict[str, str] = field(default_factory=dict)  # caller_id -> nonce_handle


class Signer:
    """
    Minimal participant agent. Handlers are async coroutines taking a payload dict
    and returning a dict response. Thread-safe for basic use via a single asyncio loop.
    """

    def __init__(self, cfg: SignerConfig):
        self.cfg = cfg
        self.state = SignerState()
        # load/create a pseudo key
        self._ensure_key()

    # -------- public handlers (async) --------

    async def keygen(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Idempotent: returns the same public key each call (created on first use).
        """
        party_id = payload.get("party_id") or self.cfg.party_id
        return {"ok": True, "pubkey": self.state.pubkey_hex, "party_id": party_id}

    async def presign(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates and remembers a nonce handle for this party_id (caller).
        In real MPC you'd create commitments/nonces bound to a round.
        """
        party_id = payload.get("party_id") or self.cfg.party_id
        # generate an opaque handle, store it
        handle = secrets.token_hex(16)
        self.state.nonces[party_id] = handle
        return {"ok": True, "nonce": handle, "party_id": party_id}

    async def partial_sign(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce a placeholder 'partial signature'.

        Real MPC would:
          - use the active nonce for this round,
          - compute a partial over the message digest and commitments,
          - return a proof or verification info.

        Here we just hash (party_id | scheme | context | digest | nonce?) for wiring tests.
        """
        party_id = payload.get("party_id") or self.cfg.party_id
        digest_hex = str(payload.get("digest") or "")
        nonce = str(payload.get("nonce") or self.state.nonces.get(party_id, ""))

        h = hashlib.sha256()
        h.update(party_id.encode())
        h.update(b"|")
        h.update(self.cfg.scheme.encode())
        h.update(b"|")
        h.update(self.cfg.context.encode())
        h.update(b"|")
        h.update(digest_hex.encode())
        h.update(b"|")
        h.update(nonce.encode())
        partial_hex = h.hexdigest()

        return {"ok": True, "partial": partial_hex, "party_id": party_id}

    # -------- helpers --------

    def handlers(self) -> Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]]:
        """
        Returns a mapping compatible with LocalTransport in coordinator.py:
            {"keygen": coro, "presign": coro, "partial_sign": coro}
        """
        return {
            "keygen": self.keygen,
            "presign": self.presign,
            "partial_sign": self.partial_sign,
        }

    # -------- internal --------

    def _ensure_key(self) -> None:
        """
        Creates a stable pseudo 'public key' representing this signer.
        For wiring only — not a real key. Persisted if keystore_path is set.
        """
        # try load
        if self.cfg.keystore_path and os.path.exists(self.cfg.keystore_path):
            try:
                with open(self.cfg.keystore_path, "r", encoding="utf-8") as f:
                    pk = f.read().strip()
                if pk:
                    self.state.pubkey_hex = pk
                    return
            except Exception:
                pass
        # create a pseudo key: H("pub" | party_id | random_seed)
        seed = os.urandom(16)
        h = hashlib.sha256()
        h.update(b"pub|")
        h.update(self.cfg.party_id.encode())
        h.update(b"|")
        h.update(seed)
        self.state.pubkey_hex = h.hexdigest()
        # persist if requested
        if self.cfg.keystore_path:
            try:
                os.makedirs(os.path.dirname(self.cfg.keystore_path) or ".", exist_ok=True)
                with open(self.cfg.keystore_path, "w", encoding="utf-8") as f:
                    f.write(self.state.pubkey_hex)
            except Exception:
                pass


# --------------------------- In-process demo wiring ---------------------------

async def demo_local_agents():
    """
    Run a quick in-process smoke test pairing with Coordinator.LocalTransport.
    """
    from .coordinator import Coordinator, Participant, RoundConfig, LocalTransport
    from .protocol import FrostLikeStrategy

    # create three local signers
    A = Signer(SignerConfig("A"))
    B = Signer(SignerConfig("B"))
    C = Signer(SignerConfig("C"))

    transport = LocalTransport({
        "A": A.handlers(),
        "B": B.handlers(),
        "C": C.handlers(),
    })

    participants = [Participant("A"), Participant("B"), Participant("C")]
    strategy = FrostLikeStrategy(threshold=2, scheme="frost-ed25519", context="news-intel")
    cfg = RoundConfig(threshold=2, timeout_s=2.0, max_retries=1)

    coord = Coordinator(participants=participants, transport=transport, strategy=strategy, cfg=cfg)

    await coord.ensure_key_material()
    res = await coord.sign(b"hello world")
    print("Round result:", res)


# --------------------------- Optional FastAPI HTTP app ---------------------------

# If FastAPI is installed, expose a tiny REST server so each signer can be run as a service.
try:
    from fastapi import FastAPI
    from pydantic import BaseModel

    class Req(BaseModel):
        party_id: Optional[str] = None
        scheme: Optional[str] = None
        nonce: Optional[str] = None
        digest: Optional[str] = None

    def make_app(signer: Signer) -> "FastAPI": # type: ignore
        """
        Create a FastAPI app exposing POST /keygen, /presign, /partial_sign
        """
        app = FastAPI(title=f"MPC Signer {signer.cfg.party_id}")

        @app.post("/keygen")
        async def _keygen(req: Req):
            return await signer.keygen(req.model_dump())

        @app.post("/presign")
        async def _presign(req: Req):
            return await signer.presign(req.model_dump())

        @app.post("/partial_sign")
        async def _partial(req: Req):
            return await signer.partial_sign(req.model_dump())

        return app

except Exception:
    # FastAPI not available; ignore.
    def make_app(signer: Signer):  # type: ignore
        raise RuntimeError("FastAPI not installed. `pip install fastapi uvicorn` to run HTTP signer.")


# --------------------------- CLI ---------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run a demo signer or start FastAPI app.")
    ap.add_argument("--party-id", default="A")
    ap.add_argument("--keystore", default=None, help="Path to persist pseudo pubkey")
    ap.add_argument("--http", action="store_true", help="Run FastAPI app (requires fastapi+uvicorn)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    signer = Signer(SignerConfig(args.party_id, keystore_path=args.keystore))

    if args.http:
        try:
            import uvicorn  # type: ignore
        except Exception:
            raise SystemExit("FastAPI mode requires: pip install fastapi uvicorn")
        app = make_app(signer)
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        asyncio.run(demo_local_agents())