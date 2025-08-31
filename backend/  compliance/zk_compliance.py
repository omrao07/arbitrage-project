# backend/compliance/zk_compliance.py
from __future__ import annotations
"""
ZK-Style Compliance Attestations (Commit & Attest)
--------------------------------------------------
Privacy-preserving compliance proofs for trading constraints.

Overview
--------
1) Prover commits to secret fields (e.g., qty, price, notional) using salted hash commitments.
2) A trusted Policy Engine evaluates predicates on the secret witness (off-chain),
   then *signs* the tuple: {commitments, public_inputs, predicate_results, policy_version}.
3) Verifiers check the signature + predicate names/versions match the policy they accept.
   Sensitive values remain undisclosed. You can later reveal a specific value by disclosing
   its salt + value, and anyone can verify the commitment opens correctly.

Backends
--------
- Preferred: Ed25519 signatures via PyNaCl (libsodium).
- Fallback: HMAC-SHA256 (shared secret) if PyNaCl is not available.

This module is self-contained and safe by default:
- No network calls
- No heavy dependencies required
- Canonical JSON for signing
- CLI for make/verify

Typical Use
-----------
Prover:
    engine = PolicyEngine(policy_version="2025.09", signer=PolicySigner.from_env())
    bundle = engine.prove(
        commitments=make_commitments({
            "qty":  1000,
            "px":   192.33,
            "notional": 192330.0
        }),
        public_inputs={
            "max_notional": 500000.0,
            "max_qty": 5000,
            "symbol": "AAPL",
            "allowed_symbols": ["AAPL", "MSFT", "NVDA"],
            "venue": "NASDAQ",
            "venue_whitelist": ["NASDAQ", "BATS"],
            "max_leverage": 5.0,
            "gross_exposure": 1_200_000.0,
            "equity": 400_000.0
        },
        witness={
            "qty":  1000,
            "px":   192.33,
            "notional": 192330.0
        },
        predicates=["notional_leq_limit","qty_leq_limit","symbol_allowlist","venue_whitelist","leverage_cap"]
    )

Verifier:
    ok, msg = verify_bundle(bundle, PolicyVerifier.from_env())
    assert ok, msg

Selective Reveal (Optional):
    # Prover can later reveal "notional" by sending:
    open_ok = verify_opening(bundle["commitments"]["notional"], value=192330.0,
                             salt=bundle["openings"]["notional"]["salt"])
    assert open_ok

Environment
-----------
ZK_SIGN_MODE = 'ed25519' | 'hmac' (default tries ed25519 if available)
ZK_SIGN_PRIV (hex or base64; for ed25519) | ZK_SIGN_SECRET (for HMAC)
ZK_SIGN_PUB  (hex/base64 public key for verifiers; ed25519 only)
"""

import os, json, hmac, hashlib, secrets, base64, time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# -------- Optional Ed25519 via PyNaCl --------
_HAS_NACL = False
try:
    from nacl.signing import SigningKey, VerifyKey  # type: ignore
    _HAS_NACL = True
except Exception:
    _HAS_NACL = False

# ----------------- Helpers -----------------

def _canon(obj: Any) -> str:
    """Deterministic JSON for signing/verifying."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))

def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _to_bytes(x: Any) -> bytes:
    if x is None:
        return b""
    if isinstance(x, bytes):
        return x
    if isinstance(x, str):
        return x.encode("utf-8")
    return str(x).encode("utf-8")

def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))

# ----------------- Commitments -----------------

@dataclass
class Commitment:
    """
    Simple salted hash commitment:
        C = SHA256( "zkc|" + salt || "|" + field || "|" + canonical(value) )
    - binding (collision-resistance)
    - hiding if salt stays secret
    """
    field: str
    digest_hex: str
    # Optional public salt disclosure (usually keep it private unless revealing)
    salt_b64: Optional[str] = None

def make_commitment(field: str, value: Any, *, salt: Optional[bytes] = None, disclose_salt: bool = False) -> Tuple[Commitment, bytes]:
    if salt is None:
        salt = secrets.token_bytes(16)
    payload = b"zkc|" + salt + b"|" + _to_bytes(field) + b"|" + _to_bytes(_canon(value))
    digest_hex = _sha256_hex(payload)
    return Commitment(field=field, digest_hex=digest_hex, salt_b64=_b64(salt) if disclose_salt else None), salt

def make_commitments(values: Dict[str, Any], *, disclose_salts: bool = False) -> Dict[str, Any]:
    commits: Dict[str, Any] = {}
    openings: Dict[str, Any] = {}
    for k, v in values.items():
        c, s = make_commitment(k, v, disclose_salt=disclose_salts)
        commits[k] = asdict(c)
        openings[k] = {"salt": _b64(s)}  # keep private by default; you can drop this when sending to verifiers
    return {"commitments": commits, "openings": openings}

def verify_opening(commit_obj: Dict[str, Any], *, value: Any, salt: str) -> bool:
    """Given a commitment (digest_hex) + claimed value + salt, recompute and compare."""
    try:
        salt_b = _b64d(salt)
        field = commit_obj["field"]
        expect = commit_obj["digest_hex"]
    except Exception:
        return False
    payload = b"zkc|" + salt_b + b"|" + _to_bytes(field) + b"|" + _to_bytes(_canon(value))
    return _sha256_hex(payload) == expect

# ----------------- Signer Backends -----------------

class PolicySigner:
    """Deterministic signer for predicate attestations."""
    def __init__(self, mode: str = "auto", priv: Optional[bytes] = None, secret: Optional[bytes] = None):
        self.mode = mode
        self.priv = priv
        self.secret = secret
        if mode == "auto":
            if _HAS_NACL and priv:
                self.mode = "ed25519"
            elif secret:
                self.mode = "hmac"
            elif _HAS_NACL and os.getenv("ZK_SIGN_PRIV"):
                self.mode = "ed25519"
            elif os.getenv("ZK_SIGN_SECRET"):
                self.mode = "hmac"
            else:
                # default to HMAC with ephemeral secret (process-local)
                self.mode = "hmac"
                self.secret = secrets.token_bytes(32)

    @staticmethod
    def from_env() -> "PolicySigner":
        mode = os.getenv("ZK_SIGN_MODE", "auto")
        priv_b = None
        secret_b = None
        if _HAS_NACL and os.getenv("ZK_SIGN_PRIV"):
            raw = os.getenv("ZK_SIGN_PRIV", "")
            try:
                priv_b = bytes.fromhex(raw)
            except Exception:
                try:
                    priv_b = _b64d(raw)
                except Exception:
                    priv_b = None
        if os.getenv("ZK_SIGN_SECRET"):
            raw = os.getenv("ZK_SIGN_SECRET", "")
            try:
                secret_b = bytes.fromhex(raw)
            except Exception:
                try:
                    secret_b = _b64d(raw)
                except Exception:
                    secret_b = raw.encode("utf-8")
        return PolicySigner(mode=mode, priv=priv_b, secret=secret_b)

    def pubkey(self) -> Optional[str]:
        if self.mode == "ed25519" and _HAS_NACL:
            sk = SigningKey(self.priv) if self.priv else SigningKey.generate()
            vk = sk.verify_key
            return vk.encode().hex()
        return None

    def sign(self, message: bytes) -> Dict[str, str]:
        if self.mode == "ed25519" and _HAS_NACL:
            sk = SigningKey(self.priv) if self.priv else SigningKey.generate()
            sig = sk.sign(message).signature
            vk_hex = sk.verify_key.encode().hex()
            return {"mode": "ed25519", "sig": sig.hex(), "pub": vk_hex}
        # HMAC fallback (shared secret)
        key = self.secret or secrets.token_bytes(32)
        mac = hmac.new(key, message, hashlib.sha256).hexdigest()
        return {"mode": "hmac", "mac": mac, "k_hint": _sha256_hex(key)[:12]}

class PolicyVerifier:
    def __init__(self, mode: str = "auto", pub: Optional[bytes] = None, secret: Optional[bytes] = None):
        self.mode = mode
        self.pub = pub
        self.secret = secret
        if mode == "auto":
            if _HAS_NACL and os.getenv("ZK_SIGN_PUB"):
                self.mode = "ed25519"
                raw = os.getenv("ZK_SIGN_PUB", "")
                try:
                    self.pub = bytes.fromhex(raw)
                except Exception:
                    try:
                        self.pub = _b64d(raw)
                    except Exception:
                        self.pub = None
            elif os.getenv("ZK_SIGN_SECRET"):
                self.mode = "hmac"
                raw = os.getenv("ZK_SIGN_SECRET", "")
                try:
                    self.secret = bytes.fromhex(raw)
                except Exception:
                    try:
                        self.secret = _b64d(raw)
                    except Exception:
                        self.secret = raw.encode("utf-8")
            else:
                self.mode = "hmac"  # permissive local verification

    @staticmethod
    def from_env() -> "PolicyVerifier":
        return PolicyVerifier(mode="auto")

    def verify(self, message: bytes, sig: Dict[str, str]) -> bool:
        mode = sig.get("mode")
        if mode == "ed25519" and _HAS_NACL:
            try:
                pub = self.pub or bytes.fromhex(sig["pub"])
            except Exception:
                try:
                    pub = _b64d(sig["pub"])
                except Exception:
                    return False
            try:
                VerifyKey(pub).verify(message, bytes.fromhex(sig["sig"]))
                return True
            except Exception:
                return False
        if mode == "hmac":
            key = self.secret or os.getenv("ZK_SIGN_SECRET", "").encode("utf-8")
            if not key:
                return False
            mac = hmac.new(key, message, hashlib.sha256).hexdigest()
            return hmac.compare_digest(mac, sig.get("mac",""))
        return False

# ----------------- Predicates -----------------

def _leq(a: float, b: float, eps: float = 1e-9) -> bool:
    try: return float(a) <= float(b) + eps
    except Exception: return False

def _allow(item: str, allowlist: List[str]) -> bool:
    try: return str(item) in set(map(str, allowlist))
    except Exception: return False

# Registry of built-in predicate functions
PREDICATES = {
    "notional_leq_limit": lambda w, p: _leq(w.get("notional", 0.0), p.get("max_notional", 0.0)),
    "qty_leq_limit":      lambda w, p: _leq(w.get("qty", 0.0), p.get("max_qty", 0.0)),
    "symbol_allowlist":   lambda w, p: _allow(p.get("symbol",""), p.get("allowed_symbols", [])),
    "venue_whitelist":    lambda w, p: _allow(p.get("venue",""), p.get("venue_whitelist", [])),
    "leverage_cap":       lambda w, p: _leq(p.get("gross_exposure",0.0) / max(1.0, p.get("equity",1.0)), p.get("max_leverage", 10.0)),
}

# ----------------- Policy Engine -----------------

@dataclass
class ProofBundle:
    """
    What verifiers see. It hides the witness by default; only commitments + public + results + signature are shared.
    'openings' block (salts) should be kept private by default and used only for selective reveal.
    """
    ts_ms: int
    policy_version: str
    commitments: Dict[str, Dict[str, Any]]        # field -> {field,digest_hex[,salt_b64]}
    public_inputs: Dict[str, Any]
    predicate_results: Dict[str, bool]
    signer: Dict[str, str]                         # {mode, sig/mac, pub?}
    openings: Optional[Dict[str, Any]] = None      # private by default; optional
    meta: Optional[Dict[str, Any]] = None

class PolicyEngine:
    def __init__(self, policy_version: str, signer: PolicySigner):
        self.policy_version = policy_version
        self.signer = signer

    def prove(
        self,
        *,
        commitments: Dict[str, Any],      # from make_commitments()['commitments']
        public_inputs: Dict[str, Any],
        witness: Dict[str, Any],          # raw secret values (never shared with verifiers)
        predicates: List[str],
        openings: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        # 1) Evaluate predicates locally (trusted policy process)
        results: Dict[str, bool] = {}
        for name in predicates:
            fn = PREDICATES.get(name)
            if not fn:
                raise ValueError(f"Unknown predicate: {name}")
            results[name] = bool(fn(witness, public_inputs))

        # 2) Canonical payload for signature
        payload = {
            "ts_ms": int(time.time() * 1000),
            "policy_version": self.policy_version,
            "commitments": commitments["commitments"] if "commitments" in commitments else commitments,
            "public_inputs": public_inputs,
            "predicate_results": results,
            "meta": meta or {}
        }
        msg = _canon(payload).encode("utf-8")

        # 3) Sign
        sig = self.signer.sign(msg)

        # 4) Assemble bundle (by default do NOT include openings)
        bundle = {
            **payload,
            "signer": sig
        }
        if openings:
            bundle["openings"] = openings
        return bundle

# ----------------- Verification -----------------

def verify_bundle(bundle: Dict[str, Any], verifier: PolicyVerifier) -> Tuple[bool, str]:
    required = ["ts_ms","policy_version","commitments","public_inputs","predicate_results","signer"]
    for k in required:
        if k not in bundle:
            return False, f"missing field: {k}"
    # rebuild signed message deterministically
    payload = {
        "ts_ms": bundle["ts_ms"],
        "policy_version": bundle["policy_version"],
        "commitments": bundle["commitments"],
        "public_inputs": bundle["public_inputs"],
        "predicate_results": bundle["predicate_results"],
        "meta": bundle.get("meta", {})
    }
    msg = _canon(payload).encode("utf-8")
    ok = verifier.verify(msg, bundle["signer"])
    return (True, "ok") if ok else (False, "signature verification failed")

# ----------------- CLI -----------------

def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _cli_prove(args):
    # inputs: witness.json, public.json
    wit = _read_json(args.witness)
    pub = _read_json(args.public)
    # commitments from witness
    cm = make_commitments(wit, disclose_salts=False)
    signer = PolicySigner.from_env()
    eng = PolicyEngine(policy_version=args.policy, signer=signer)
    preds = args.predicates.split(",") if args.predicates else list(PREDICATES.keys())
    bundle = eng.prove(
        commitments=cm,
        public_inputs=pub,
        witness=wit,
        predicates=preds,
        openings=(cm["openings"] if args.include_openings else None),
        meta={"book": args.book, "session": args.session}
    )
    _write_json(args.out, bundle)
    print(f"Wrote proof bundle to {args.out}")

def _cli_verify(args):
    bundle = _read_json(args.bundle)
    verifier = PolicyVerifier.from_env()
    ok, msg = verify_bundle(bundle, verifier)
    print("OK" if ok else "FAIL", "-", msg)

def _cli_genkeys(args):
    if _HAS_NACL:
        sk = SigningKey.generate()
        pk = sk.verify_key
        print("ZK_SIGN_MODE=ed25519")
        print("ZK_SIGN_PRIV (hex):", sk.encode().hex())
        print("ZK_SIGN_PUB  (hex):", pk.encode().hex())
    else:
        key = secrets.token_bytes(32)
        print("ZK_SIGN_MODE=hmac")
        print("ZK_SIGN_SECRET (hex):", key.hex())

def main():
    import argparse
    ap = argparse.ArgumentParser(description="ZK-Style Compliance (Commit & Attest)")
    sub = ap.add_subparsers(required=True, dest="cmd")

    p = sub.add_parser("prove", help="Create a compliance proof bundle")
    p.add_argument("--witness", required=True, help="witness.json (secret values)")
    p.add_argument("--public", required=True, help="public.json (limits & public params)")
    p.add_argument("--policy", default="2025.09", help="policy version string")
    p.add_argument("--predicates", default="notional_leq_limit,qty_leq_limit,symbol_allowlist,venue_whitelist,leverage_cap")
    p.add_argument("--out", required=True, help="output proof bundle path")
    p.add_argument("--include_openings", action="store_true", help="include salts (for selective reveal)")
    p.add_argument("--book", default=None)
    p.add_argument("--session", default=None)
    p.set_defaults(func=_cli_prove)

    v = sub.add_parser("verify", help="Verify a proof bundle")
    v.add_argument("--bundle", required=True, help="proof bundle JSON path")
    v.set_defaults(func=_cli_verify)

    g = sub.add_parser("genkeys", help="Generate signer keys (ed25519 if available; else HMAC secret)")
    g.set_defaults(func=_cli_genkeys)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":  # pragma: no cover
    main()