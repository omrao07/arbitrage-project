# backend/audit/merkle_ledger.py
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------- File lock (best-effort, POSIX/NOP on Win) -------------------

class _FileLock:
    def __init__(self, path: Path):
        self._p = path
        self._fh = None

    def __enter__(self):
        try:
            self._fh = open(self._p, "a+")
            try:
                import fcntl  # type: ignore
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass  # best-effort on non-POSIX
        except Exception:
            self._fh = None
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._fh:
                try:
                    import fcntl  # type: ignore
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
                self._fh.close()
        finally:
            self._fh = None


# ----------------------------- Merkle utils -----------------------------------

def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _canon_json(obj: Any) -> bytes:
    # Canonical, stable JSON encoding for hashing
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")

def _merkle_root(hashes: List[str]) -> Tuple[str, List[List[str]]]:
    """
    Build a binary Merkle tree from a list of hex hashes (strings).
    Returns (root_hash, levels), where levels[0] = leaves, levels[-1] = [root].
    If odd count, last element is duplicated at each level (standard trick).
    """
    if not hashes:
        return "", []
    level = hashes[:]
    levels: List[List[str]] = [level]
    while len(level) > 1:
        nxt: List[str] = []
        for i in range(0, len(level), 2):
            a = level[i]
            b = level[i + 1] if i + 1 < len(level) else level[i]
            nxt.append(_sha256_hex((a + b).encode("utf-8")))
        level = nxt
        levels.append(level)
    return level[0], levels


# ----------------------------- Data model -------------------------------------

@dataclass
class Entry:
    idx: int
    prev_hash: str
    payload: Dict[str, Any]
    hash: str

@dataclass
class Header:
    created_ms: int
    version: int = 1

_DEFAULT_OBJECT = {"header": asdict(Header(created_ms=int(time.time() * 1000))),
                   "entries": [], "root": "", "levels": []}


# ----------------------------- Ledger class -----------------------------------

class MerkleLedger:
    """
    Append-only, tamper-evident ledger with Merkle root.
    Storage format (single JSON file, atomic replace on write):

    {
      "header": {"created_ms": 1690000000000, "version": 1},
      "entries": [
        {"idx":0,"prev_hash":"GENESIS","payload":{...},"hash":"..."},
        ...
      ],
      "root": "<hex>",
      "levels": [ [...leaves...], [...], ["<root>"] ]   # optional, helps audits
    }

    Notes
    -----
    • Writes are atomic via .tmp file and os.replace.
    • Best-effort advisory file lock on POSIX.
    • Rotation supported via rotate(max_entries=..., out_path=...).
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._save(_DEFAULT_OBJECT)

    # ------------------ I/O helpers ------------------

    def _load(self) -> Dict[str, Any]:
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, obj: Dict[str, Any]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, separators=(",", ":"), sort_keys=True, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    # ------------------ Public API -------------------

    def append(self, payload: Dict[str, Any]) -> Entry:
        """
        Append a new payload as an Entry, recompute Merkle root, persist atomically.
        Returns the appended Entry.
        """
        with _FileLock(self.path):
            obj = self._safe_load_or_init()
            entries: List[Dict[str, Any]] = obj.get("entries", [])
            prev_hash = entries[-1]["hash"] if entries else "GENESIS"
            idx = len(entries)

            body = {"idx": idx, "prev_hash": prev_hash, "payload": payload}
            eh = _sha256_hex(_canon_json(body))

            entry = {"idx": idx, "prev_hash": prev_hash, "payload": payload, "hash": eh}
            entries.append(entry)

            # recompute merkle
            leaves = [e["hash"] for e in entries]
            root, levels = _merkle_root(leaves)
            obj["entries"] = entries
            obj["root"] = root
            obj["levels"] = levels

            self._save(obj)
            return Entry(**entry)

    def verify(self) -> bool:
        """
        Verify chain integrity and Merkle root correctness.
        """
        try:
            obj = self._load()
        except Exception:
            return False

        entries: List[Dict[str, Any]] = obj.get("entries", [])
        # Chain check
        prev = "GENESIS"
        for i, e in enumerate(entries):
            if e.get("idx") != i or e.get("prev_hash") != prev:
                return False
            body = {"idx": e["idx"], "prev_hash": e["prev_hash"], "payload": e["payload"]}
            if _sha256_hex(_canon_json(body)) != e.get("hash"):
                return False
            prev = e.get("hash", "")

        # Merkle recompute
        leaves = [e["hash"] for e in entries]
        root, _levels = _merkle_root(leaves)
        return root == obj.get("root", "")

    def head(self, n: int = 5) -> List[Entry]:
        """Return the last n entries as Entry dataclasses."""
        obj = self._load()
        entries: List[Dict[str, Any]] = obj.get("entries", [])
        out = []
        for e in entries[-n:]:
            out.append(Entry(idx=e["idx"], prev_hash=e["prev_hash"], payload=e["payload"], hash=e["hash"]))
        return out

    def snapshot(self) -> Dict[str, Any]:
        """Return a safe copy of the ledger header + root + last idx (no entries)."""
        obj = self._load()
        return {
            "header": obj.get("header", {}),
            "last_idx": len(obj.get("entries", [])) - 1,
            "root": obj.get("root", ""),
        }

    def get_root(self) -> str:
        """Return current Merkle root hex."""
        try:
            return self._load().get("root", "")
        except Exception:
            return ""

    def rotate(self, *, max_entries: int, out_path: Optional[str] = None) -> Optional[str]:
        """
        If entries exceed max_entries, archive current file and start fresh.
        Returns the archive path if rotation happened.
        """
        with _FileLock(self.path):
            obj = self._safe_load_or_init()
            entries: List[Dict[str, Any]] = obj.get("entries", [])
            if len(entries) < max_entries:
                return None

            # Archive current file (include root and timestamp)
            ts = int(time.time() * 1000)
            root = obj.get("root", "")[:16]
            archive_name = f"{self.path.stem}.{ts}.{root}.archive{self.path.suffix}"
            archive_path = Path(out_path) / archive_name if out_path else self.path.with_name(archive_name)
            # atomic copy
            with open(archive_path, "w", encoding="utf-8") as f:
                json.dump(obj, f, separators=(",", ":"), sort_keys=True, ensure_ascii=False)

            # Reset current
            self._save(_DEFAULT_OBJECT | {"header": asdict(Header(created_ms=ts))})
            return str(archive_path)

    # ------------------ Internals -------------------

    def _safe_load_or_init(self) -> Dict[str, Any]:
        if not self.path.exists():
            self._save(_DEFAULT_OBJECT)
        try:
            obj = self._load()
        except Exception:
            # If corrupted, we DO NOT overwrite; we raise to avoid losing evidence.
            raise RuntimeError(f"Ledger {self.path} is unreadable/corrupted. Manual intervention required.")
        # minimal shape repair
        if "entries" not in obj or not isinstance(obj["entries"], list):
            obj["entries"] = []
        if "header" not in obj:
            obj["header"] = asdict(Header(created_ms=int(time.time() * 1000)))
        if "root" not in obj:
            obj["root"] = ""
        if "levels" not in obj:
            obj["levels"] = []
        return obj