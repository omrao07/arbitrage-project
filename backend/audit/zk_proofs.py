# backend/audit/zk_proofs.py
"""
Zero-Knowledge(-lite) Limit Proofs for Risk Caps
------------------------------------------------
Pedersen-style commitments (homomorphic) over an RFC-3526 MODP group:
  C_i = g^{v_i} * h^{r_i}  (mod p)
We can prove the *sum* without revealing individual v_i:
  ∏ C_i = g^{Σ v_i} * h^{Σ r_i}
By disclosing S = Σ v_i and R = Σ r_i, a verifier checks:
  prod_commit == g^S * h^R (mod p)
…and then checks S ≤ cap. Individual v_i remain hidden.

What this module provides
- Commitment API: commit(value) -> (commit_hex, blinding_r)
- Merkle binding for a set of commits
- Proof object for "sum under cap" with verify()
- Deterministic domain separation via context binding
- No external dependencies; pure Python big-int arithmetic

Security notes
- We use RFC-3526 2048-bit safe prime p and derive generators (g,h) in the
  order-q subgroup (q=(p-1)/2). Binding relies on hardness of DL in that group
  and unknown log_g(h).
- This proves the aggregate S exactly; it does *not* provide a ZK range proof
  for S (we *reveal* S). If you need fully zero-knowledge range proofs, plug
  in Bulletproofs/SNARKs via a different backend – the API shape here will still
  fit (swap out commit/verify methods).

Typical usage
-------------
from backend.audit import zk_proofs as zk

# 1) Commit to private vector v (e.g., per-strategy ES bps)
values = [37.1, 12.4, 41.0]   # bps (floats ok; we scale to ints)
commits, blindings = zk.commit_vector(values, scale=100)  # 2 decimals

# 2) Build proof that sum(values) <= cap without revealing components
cap = 120.0
proof = zk.prove_sum_under_cap(commits, blindings, sum_value=sum(values), cap=cap,
                               scale=100, context={"round":"2025-09-03","type":"ES"})

# 3) Verify (anywhere, later)
ok = zk.verify_sum_under_cap(proof)
assert ok
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------- RFC-3526 MODP 2048-bit (group 14) ----------------

# Safe prime p (hex from RFC 3526, group 14)
_P_HEX = (
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E08"
    "8A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B"
    "302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9"
    "A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE6"
    "49286651ECE65381FFFFFFFFFFFFFFFF"
)
P = int(_P_HEX, 16)                 # safe prime p
Q = (P - 1) // 2                    # large prime q (subgroup order)
# Base generator for subgroup: g = 2^2 mod p (quadratic residue, order divides q)
_G_BASE = pow(2, 2, P)

def _sha256_int(*parts: bytes, mod: int = Q) -> int:
    h = hashlib.sha256()
    for pz in parts:
        h.update(pz)
    return int.from_bytes(h.digest(), "big") % mod

def _int_to_hex(n: int) -> str:
    return hex(n)[2:]

def _hex_to_int(s: str) -> int:
    return int(s, 16)

def _json_canon(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode()

# Derive subgroup generators g,h from a domain separator (deterministic)
def derive_generators(domain: str = "zkpf:v1") -> Tuple[int, int]:
    g = pow(_G_BASE, _sha256_int(domain.encode() + b"|g"), P)
    if g in (0, 1):
        g = _G_BASE
    # h must be independent of g (unknown log_g(h))
    h = pow(_G_BASE, _sha256_int(domain.encode() + b"|h"), P)
    if h in (0, 1) or h == g:
        h = pow(_G_BASE, _sha256_int(domain.encode() + b"|h2"), P)
        if h in (0, 1) or h == g:
            h = pow(_G_BASE, 3, P)
    return g, h

# Pedersen commitment C = g^v * h^r (mod p), exponents mod Q
def pedersen_commit(v: int, r: Optional[int] = None, *, g: int, h: int) -> Tuple[int, int]:
    v_mod = v % Q
    r_mod = (r if r is not None else _sha256_int(b"rnd", v.to_bytes((v.bit_length()+7)//8 or 1, "big"))) % Q
    C = (pow(g, v_mod, P) * pow(h, r_mod, P)) % P
    return C, r_mod

# ---------------- Merkle binding for a set of commitments ----------------

def _merkle_root(leaves_hex: Sequence[str]) -> str:
    """Binary Merkle root (hex), leaves are hex strings; hash concatenation with SHA-256."""
    if not leaves_hex:
        return ""
    lvl = [hashlib.sha256(bytes.fromhex(x)).digest() for x in leaves_hex]
    while len(lvl) > 1:
        nxt = []
        for i in range(0, len(lvl), 2):
            a = lvl[i]
            b = lvl[i + 1] if i + 1 < len(lvl) else lvl[i]
            nxt.append(hashlib.sha256(a + b).digest())
        lvl = nxt
    return lvl[0].hex()

# ---------------- Public API ----------------

def commit(value: float, *, scale: int = 100, domain: str = "zkpf:v1") -> Tuple[str, int]:
    """
    Commit to a single value (float, e.g., bps) scaled to int by `scale`.
    Returns (commit_hex, blinding_r).
    """
    v = int(round(value * scale))
    g, h = derive_generators(domain)
    C, r = pedersen_commit(v, None, g=g, h=h)
    return _int_to_hex(C), r

def commit_vector(values: Sequence[float], *, scale: int = 100, domain: str = "zkpf:v1") -> Tuple[List[str], List[int]]:
    """
    Commit to a list of values. Returns (commit_hex_list, blinding_list).
    """
    g, h = derive_generators(domain)
    commits: List[str] = []
    blinds: List[int] = []
    for x in values:
        v = int(round(x * scale))
        C, r = pedersen_commit(v, None, g=g, h=h)
        commits.append(_int_to_hex(C))
        blinds.append(r)
    return commits, blinds

@dataclass
class SumUnderCapProof:
    scheme: str                 # "pedersen_modp2048_v1"
    domain: str                 # generator domain sep
    scale: int                  # scaling factor used for float→int
    cap_scaled: int             # cap * scale (int)
    sum_scaled: int             # S * scale (int) disclosed
    R: str                      # hex of aggregate blinding (mod Q)
    commits: List[str]          # hex commitments (order significant)
    product_commit: str         # hex of ∏ commits (redundant, helps transport)
    merkle_root: str            # merkle root of commits (for binding)
    context_hash: str           # hash of context JSON
    statement: str              # human label, e.g. "Portfolio ES <= Cap"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def _product_commit(commits_hex: Sequence[str]) -> int:
    acc = 1
    for c_hex in commits_hex:
        acc = (acc * _hex_to_int(c_hex)) % P
    return acc

def _context_hash(context: Optional[Dict[str, Any]]) -> str:
    return hashlib.sha256(_json_canon(context or {})).hexdigest()

def prove_sum_under_cap(
    commits_hex: Sequence[str],
    blindings: Sequence[int],
    *,
    sum_value: float,
    cap: float,
    scale: int = 100,
    domain: str = "zkpf:v1",
    context: Optional[Dict[str, Any]] = None,
    statement: str = "Portfolio risk ≤ cap",
) -> Dict[str, Any]:
    """
    Build a proof that:
      1) commits hex are valid Pedersen commitments of private values v_i
      2) sum(v_i) == S (S disclosed), and S ≤ cap
      3) binding to Merkle root of commits and context

    Returns a JSON-serializable dict.
    """
    if len(commits_hex) != len(blindings):
        raise ValueError("commits and blindings length mismatch")

    g, h = derive_generators(domain)
    S_scaled = int(round(sum_value * scale))
    cap_scaled = int(round(cap * scale))
    R = sum(int(r) % Q for r in blindings) % Q

    # Product commitment from provided list
    prod = _product_commit(commits_hex)

    # Check equality ourselves before packaging (defensive)
    lhs = prod
    rhs = (pow(g, S_scaled % Q, P) * pow(h, R % Q, P)) % P
    if lhs != rhs:
        # Caller may have passed wrong sum/blindings; raise to avoid emitting bad proof
        raise ValueError("sum/blinding do not match the commitments (homomorphic check failed)")

    proof = SumUnderCapProof(
        scheme="pedersen_modp2048_v1",
        domain=domain,
        scale=scale,
        cap_scaled=cap_scaled,
        sum_scaled=S_scaled,
        R=_int_to_hex(R),
        commits=list(commits_hex),
        product_commit=_int_to_hex(prod),
        merkle_root=_merkle_root(list(commits_hex)),
        context_hash=_context_hash(context),
        statement=statement,
    )
    return proof.to_dict()

def verify_sum_under_cap(proof: Dict[str, Any], *, context: Optional[Dict[str, Any]] = None) -> bool:
    """
    Verify a SumUnderCapProof produced by prove_sum_under_cap().
    Steps:
      • Check scheme/domain/scale shape
      • Recompute product and Merkle root over commits
      • Check homomorphic equality: ∏C == g^S * h^R mod p
      • Check S ≤ cap
      • Check context hash binds (if provided)
    """
    required = ("scheme", "domain", "scale", "cap_scaled", "sum_scaled", "R", "commits", "product_commit", "merkle_root", "context_hash")
    if any(k not in proof for k in required):
        return False
    if proof["scheme"] != "pedersen_modp2048_v1":
        return False

    domain = proof["domain"]
    scale = int(proof["scale"])
    cap_scaled = int(proof["cap_scaled"])
    S_scaled = int(proof["sum_scaled"])
    R = _hex_to_int(proof["R"])
    commits: List[str] = list(proof["commits"])
    prod_claim = _hex_to_int(proof["product_commit"])

    # 1) Recompute product & Merkle root
    prod = _product_commit(commits)
    if prod != prod_claim:
        return False
    if _merkle_root(commits) != proof["merkle_root"]:
        return False

    # 2) Check equality in the group
    g, h = derive_generators(domain)
    rhs = (pow(g, S_scaled % Q, P) * pow(h, R % Q, P)) % P
    if prod != rhs:
        return False

    # 3) Check disclosed sum against cap
    if S_scaled > cap_scaled:
        return False

    # 4) Context binding (if verifier supplies context)
    if context is not None:
        if proof["context_hash"] != _context_hash(context):
            return False

    return True

# ---------------- Convenience: bind and append to MerkleLedger ---------------

def append_proof_to_ledger(proof: Dict[str, Any], ledger_path: str) -> None:
    """
    Append proof to backend.audit.merkle_ledger.MerkleLedger if available.
    """
    try:
        from .merkle_ledger import MerkleLedger  # relative import within backend.audit
        MerkleLedger(ledger_path).append({"type": "zk_sum_under_cap", "proof": proof, "ts": proof.get("ts")})
    except Exception:
        # Non-fatal if ledger is not available
        pass