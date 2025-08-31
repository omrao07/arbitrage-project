# backend/compliance/compliance_recorder.py
from __future__ import annotations
"""
Compliance Recorder — append-only, hash-chained audit ledger
------------------------------------------------------------
Purpose:
- Capture all compliance-critical events in an immutable JSONL stream.
- Each record is hash-chained (prev_hash → curr_hash) and optionally PII-redacted.
- Per-session rollups: Merkle root + session manifest to anchor integrity.
- Optional dual-write to Redis bus streams.

Zero hard deps. Auto-uses optional modules if present:
- backend.compliance.pii_redactor.PIIRedactor  (optional)
- backend.compliance.zk_compliance (for proof bundle links) (optional)
- backend.compliance.compliance_narrative (for narrative hashes) (optional)
- Redis bus: backend.bus.streams.publish_stream (optional)

Files:
- <root>/ledger/<YYYY-MM>/<session_id>.jsonl     (append-only)
- <root>/ledger/<YYYY-MM>/<session_id>.manifest  (finalized rollup)

CLI:
  python -m backend.compliance.compliance_recorder start --session 2025-08-30-ny-open
  python -m backend.compliance.compliance_recorder record --session 2025-08-30-ny-open --kind order --payload '{"id":"O1","symbol":"AAPL"}'
  python -m backend.compliance.compliance_recorder close --session 2025-08-30-ny-open
  python -m backend.compliance.compliance_recorder verify --session 2025-08-30-ny-open
"""

import os, io, json, time, hashlib, base64, pathlib, uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List, Tuple

# -------- Optional PII redactor ----------
try:
    from backend.compliance.pii_redactor import PIIRedactor  # type: ignore
except Exception:
    PIIRedactor = None  # type: ignore

# -------- Optional bus ----------
try:
    from backend.bus.streams import publish_stream  # type: ignore
except Exception:
    def publish_stream(stream: str, payload: Dict[str, Any]) -> None:
        pass

DEFAULT_ROOT = os.getenv("COMPLIANCE_LEDGER_ROOT", "./audit")
DEFAULT_STREAM = os.getenv("COMPLIANCE_LEDGER_STREAM", "compliance.ledger")
ENV = os.getenv("ENV", os.getenv("ENVIRONMENT", "dev")).lower()

# ----------------- hashing helpers -----------------

def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))

def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _h(obj: Any) -> str:
    return _sha256(_canon(obj).encode("utf-8"))

def _now_ms() -> int:
    return int(time.time() * 1000)

# ----------------- types -----------------

@dataclass
class LedgerHeader:
    session_id: str
    started_ts_ms: int
    env: str
    book: Optional[str] = None
    region: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

@dataclass
class LedgerRecord:
    ts_ms: int
    kind: str             # order|route|risk|fill|tca|best_ex|policy|proof|narrative|replay|alert|note
    payload: Dict[str, Any]
    prev_hash: str
    hash: str
    id: str               # record id (uuid)
    redact: bool = False  # true if PII redaction applied

# ----------------- recorder core -----------------

class ComplianceRecorder:
    def __init__(
        self,
        *,
        root: str = DEFAULT_ROOT,
        pii_mode: Optional[str] = None,        # "redact" | "tokenize" | None
        pii_salt: Optional[str] = None,        # if tokenize
        bus_stream: Optional[str] = DEFAULT_STREAM,
    ):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._pii = None
        if pii_mode and PIIRedactor:
            self._pii = PIIRedactor(action=pii_mode, salt=(pii_salt or os.getenv("PII_SALT")))
        self.bus_stream = bus_stream

    # ---------- session paths ----------
    def _sess_dir(self, session_id: str) -> pathlib.Path:
        # partition by YYYY-MM for filesystem hygiene
        ym = time.strftime("%Y-%m")
        p = self.root / "ledger" / ym
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _sess_paths(self, session_id: str) -> Tuple[pathlib.Path, pathlib.Path]:
        base = self._sess_dir(session_id)
        return base / f"{session_id}.jsonl", base / f"{session_id}.manifest"

    # ---------- lifecycle ----------
    def start(self, header: LedgerHeader) -> None:
        ledger_path, manifest_path = self._sess_paths(header.session_id)
        if ledger_path.exists():
            return  # idempotent
        hdr = asdict(header)
        seed = {
            "ts_ms": header.started_ts_ms,
            "kind": "session_start",
            "payload": hdr,
            "prev_hash": "0"*64,
            "id": str(uuid.uuid4()),
        }
        seed["hash"] = _h(seed)
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(_canon(seed) + "\n")
        # no manifest yet
        if self.bus_stream:
            publish_stream(self.bus_stream, {"ts_ms": _now_ms(), "event": "session_start", "session": header.session_id})

    def record(self, session_id: str, kind: str, payload: Dict[str, Any], *, redact: bool = True) -> LedgerRecord:
        ledger_path, _ = self._sess_paths(session_id)
        if not ledger_path.exists():
            raise FileNotFoundError(f"Session not started: {session_id}")

        # load last hash quickly by reading last non-empty line
        prev_hash = "0"*64
        try:
            with open(ledger_path, "rb") as f:
                f.seek(0, io.SEEK_END)
                pos = f.tell()
                line = b""
                while pos > 0:
                    pos -= 1
                    f.seek(pos, io.SEEK_SET)
                    ch = f.read(1)
                    if ch == b"\n":
                        if line:
                            break
                    else:
                        line = ch + line
                if line:
                    prev_hash = json.loads(line.decode("utf-8")).get("hash", prev_hash)
        except Exception:
            pass

        clean_payload = payload
        applied_redaction = False
        if redact and self._pii:
            try:
                clean_payload = self._pii.clean(payload)
                applied_redaction = True
            except Exception:
                clean_payload = payload

        rec_obj = {
            "ts_ms": _now_ms(),
            "kind": str(kind),
            "payload": clean_payload,
            "prev_hash": prev_hash,
            "id": str(uuid.uuid4()),
            "redact": bool(applied_redaction),
        }
        rec_obj["hash"] = _h(rec_obj)

        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(_canon(rec_obj) + "\n")

        if self.bus_stream:
            try:
                publish_stream(self.bus_stream, {"ts_ms": rec_obj["ts_ms"], "event": "append", "session": session_id, "kind": kind})
            except Exception:
                pass

        return LedgerRecord(**rec_obj)  # type: ignore

    def link_artifact(self, session_id: str, *, narrative_path: Optional[str] = None, replay_path: Optional[str] = None, proof_bundle_path: Optional[str] = None) -> None:
        payload: Dict[str, Any] = {}
        for name, path in (("narrative", narrative_path), ("replay", replay_path), ("proof_bundle", proof_bundle_path)):
            if path and os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    try:
                        blob = f.read()
                        # try json; if md, hash raw
                        try:
                            obj = json.loads(blob)
                            payload[f"{name}_hash"] = _h(obj)
                            payload[f"{name}_meta"] = {"type": "json"}
                        except Exception:
                            payload[f"{name}_hash"] = _sha256(blob.encode("utf-8"))
                            payload[f"{name}_meta"] = {"type": "text"}
                    except Exception:
                        pass
        if payload:
            self.record(session_id, "artifact_link", payload, redact=False)

    def close(self, session_id: str) -> Dict[str, Any]:
        ledger_path, manifest_path = self._sess_paths(session_id)
        if not ledger_path.exists():
            raise FileNotFoundError(f"Session not started: {session_id}")

        # compute Merkle-ish rollup: simple pairwise hashing → root
        hashes: List[str] = []
        with open(ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    h = json.loads(line)["hash"]
                    hashes.append(h)
                except Exception:
                    pass

        if not hashes:
            raise RuntimeError("Empty ledger; cannot close session")

        layer = hashes[:]
        while len(layer) > 1:
            nxt: List[str] = []
            for i in range(0, len(layer), 2):
                a = layer[i]
                b = layer[i+1] if i+1 < len(layer) else a  # duplicate last if odd
                nxt.append(_sha256((a + b).encode("utf-8")))
            layer = nxt
        root = layer[0]

        manifest = {
            "session_id": session_id,
            "closed_ts_ms": _now_ms(),
            "ledger_file": str(ledger_path),
            "records": len(hashes),
            "root": root,
            "first_hash": hashes[0],
            "last_hash": hashes[-1],
            "env": ENV,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        if self.bus_stream:
            publish_stream(self.bus_stream, {"ts_ms": manifest["closed_ts_ms"], "event": "session_close", "session": session_id, "root": root})

        return manifest

    # ---------- verifier ----------
    @staticmethod
    def verify(session_id: str, *, root_dir: str = DEFAULT_ROOT) -> Tuple[bool, str]:
        base = pathlib.Path(root_dir)
        ym = time.strftime("%Y-%m")  # best-effort locate; also scan other months if needed
        candidates = list((base / "ledger").glob(f"*/{session_id}.jsonl"))
        if not candidates:
            return False, "ledger file not found"
        ledger_path = candidates[0]
        manifest_path = ledger_path.with_suffix(".manifest")
        if not manifest_path.exists():
            return False, "manifest not found (session not closed?)"

        with open(manifest_path, "r", encoding="utf-8") as f:
            man = json.load(f)

        # recompute chain & root
        prev = "0"*64
        hashes: List[str] = []
        with open(ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                # chain integrity
                if rec["prev_hash"] != prev:
                    return False, f"chain break at record {rec.get('id')}"
                if _h({k: rec[k] for k in ("ts_ms","kind","payload","prev_hash","id","redact")}) != rec["hash"]:
                    return False, f"hash mismatch at record {rec.get('id')}"
                prev = rec["hash"]
                hashes.append(rec["hash"])

        layer = hashes[:]
        while len(layer) > 1:
            nxt = []
            for i in range(0, len(layer), 2):
                a = layer[i]
                b = layer[i+1] if i+1 < len(layer) else a
                nxt.append(_sha256((a + b).encode("utf-8")))
            layer = nxt
        root = layer[0]

        if root != man.get("root"):
            return False, "root mismatch vs manifest"
        return True, "ok"

# ----------------- CLI -----------------

def _json_or_err(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception as e:
        raise SystemExit(f"invalid JSON payload: {e}")

def _cli_start(args):
    rec = ComplianceRecorder(root=args.root, pii_mode=args.pii, pii_salt=args.pii_salt)
    hdr = LedgerHeader(
        session_id=args.session,
        started_ts_ms=_now_ms(),
        env=ENV,
        book=args.book,
        region=args.region,
        meta=_json_or_err(args.meta) if args.meta else None
    )
    rec.start(hdr)
    print(f"started session {args.session}")

def _cli_record(args):
    rec = ComplianceRecorder(root=args.root, pii_mode=args.pii, pii_salt=args.pii_salt)
    payload = _json_or_err(args.payload)
    r = rec.record(args.session, args.kind, payload, redact=(not args.no_redact))
    print(_canon(asdict(r)))

def _cli_link(args):
    rec = ComplianceRecorder(root=args.root, pii_mode=args.pii, pii_salt=args.pii_salt)
    rec.link_artifact(args.session, narrative_path=args.narrative, replay_path=args.replay, proof_bundle_path=args.proof)
    print("linked artifacts")

def _cli_close(args):
    rec = ComplianceRecorder(root=args.root, pii_mode=args.pii, pii_salt=args.pii_salt)
    man = rec.close(args.session)
    print(json.dumps(man, indent=2))

def _cli_verify(args):
    ok, msg = ComplianceRecorder.verify(args.session, root_dir=args.root)
    print("OK" if ok else "FAIL", "-", msg)

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Compliance Recorder — hash-chained audit ledger")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("start", help="start a session")
    p.add_argument("--session", required=True)
    p.add_argument("--book", default=None)
    p.add_argument("--region", default=None)
    p.add_argument("--meta", default=None, help="JSON string")
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--pii", choices=["redact","tokenize"], default=None)
    p.add_argument("--pii_salt", default=None)
    p.set_defaults(func=_cli_start)

    r = sub.add_parser("record", help="append a record")
    r.add_argument("--session", required=True)
    r.add_argument("--kind", required=True)
    r.add_argument("--payload", required=True, help='JSON string payload')
    r.add_argument("--no_redact", action="store_true")
    r.add_argument("--root", default=DEFAULT_ROOT)
    r.add_argument("--pii", choices=["redact","tokenize"], default=None)
    r.add_argument("--pii_salt", default=None)
    r.set_defaults(func=_cli_record)

    l = sub.add_parser("link", help="link artifacts (narrative/replay/proof)")
    l.add_argument("--session", required=True)
    l.add_argument("--narrative", default=None)
    l.add_argument("--replay", default=None)
    l.add_argument("--proof", default=None)
    l.add_argument("--root", default=DEFAULT_ROOT)
    l.set_defaults(func=_cli_link)

    c = sub.add_parser("close", help="close a session and write rollup")
    c.add_argument("--session", required=True)
    c.add_argument("--root", default=DEFAULT_ROOT)
    c.add_argument("--pii", choices=["redact","tokenize"], default=None)
    c.add_argument("--pii_salt", default=None)
    c.set_defaults(func=_cli_close)

    v = sub.add_parser("verify", help="verify session integrity")
    v.add_argument("--session", required=True)
    v.add_argument("--root", default=DEFAULT_ROOT)
    v.set_defaults(func=_cli_verify)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":  # pragma: no cover
    main()