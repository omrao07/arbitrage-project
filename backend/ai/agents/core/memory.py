# backend/ai/agents/core/memory.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------------------------------------
# Data models
# ------------------------------------------------------------
@dataclass
class MemoryEntry:
    role: str              # "user" | "assistant" | "system"
    content: str
    ts_ms: int = field(default_factory=lambda: int(time.time()*1000))
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UserProfile:
    user_name: Optional[str] = None
    timezone: Optional[str] = None
    default_symbol: Optional[str] = None
    watchlist: List[str] = field(default_factory=list)
    risk_prefs: Dict[str, Any] = field(default_factory=dict)
    notes: Dict[str, str] = field(default_factory=dict)   # arbitrary notes

# ------------------------------------------------------------
# Core Memory
# ------------------------------------------------------------
class MemoryStore:
    """
    Hybrid memory:
      • short-term = rolling chat buffer
      • long-term  = persistent profile, key/value store
      • optional vector_store for embeddings (pluggable)

    Usage:
      mem = MemoryStore("./.memory.json")
      mem.add("user","Buy AAPL", meta={"qty":100})
      mem.save()
    """

    def __init__(self, persist_path: Optional[str] = None, max_history: int = 200):
        self.persist_path = persist_path
        self.max_history = max_history
        self.history: List[MemoryEntry] = []
        self.profile = UserProfile()
        self.facts: Dict[str, Any] = {}  # key/value long-term store
        self._load()

    # ---------------- Short-term ----------------
    def add(self, role: str, content: str, **meta) -> None:
        self.history.append(MemoryEntry(role=role, content=content, meta=meta))
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_history(self, limit: Optional[int] = None) -> List[MemoryEntry]:
        return self.history if limit is None else self.history[-limit:]

    def clear_history(self) -> None:
        self.history = []

    # ---------------- Long-term ----------------
    def set_fact(self, key: str, value: Any) -> None:
        self.facts[key] = value

    def get_fact(self, key: str, default: Any = None) -> Any:
        return self.facts.get(key, default)

    def del_fact(self, key: str) -> None:
        if key in self.facts:
            del self.facts[key]

    # ---------------- Profile ----------------
    def update_profile(self, **kwargs) -> None:
        for k,v in kwargs.items():
            if hasattr(self.profile, k):
                setattr(self.profile, k, v)

    def get_profile(self) -> Dict[str, Any]:
        return asdict(self.profile)

    # ---------------- Persistence ----------------
    def save(self) -> None:
        if not self.persist_path:
            return
        try:
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump({
                    "history": [asdict(m) for m in self.history],
                    "profile": asdict(self.profile),
                    "facts": self.facts,
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("[memory] save failed:", e)

    def _load(self) -> None:
        if not self.persist_path or not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.history = [MemoryEntry(**m) for m in data.get("history",[])]
            self.profile = UserProfile(**data.get("profile",{}))
            self.facts = data.get("facts",{})
        except Exception as e:
            print("[memory] load failed:", e)

    # ---------------- Export / Import ----------------
    def export_text(self, limit: int = 20) -> str:
        """
        Return last N turns as plain text (for LLM context).
        """
        lines = []
        for m in self.get_history(limit):
            lines.append(f"[{m.role.upper()}] {m.content}")
        return "\n".join(lines)

    def dump(self) -> Dict[str, Any]:
        return {
            "history": [asdict(m) for m in self.history],
            "profile": asdict(self.profile),
            "facts": self.facts,
        }

# ------------------------------------------------------------
# Quick smoke test
# ------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    mem = MemoryStore("./.memory.json")
    mem.add("user","Buy AAPL at 190", qty=100)
    mem.update_profile(user_name="Om", timezone="IST", default_symbol="AAPL")
    mem.set_fact("favorite_strategy","pairs trading")
    mem.save()
    print("History export:\n", mem.export_text())
    print("Profile:", mem.get_profile())
    print("Facts:", mem.facts)