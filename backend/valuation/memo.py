#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
memo.py
-------
Simple memo/note system for this project.

Features:
  - Add a memo with timestamp, author, tags, and freeform text
  - List memos (all or filtered by tag/author)
  - Save to/load from JSON (default) or CSV
  - Usable both as a module and from CLI

Usage (Python):
    from memo import MemoStore
    store = MemoStore("memos.json")
    store.add("Check correlation drift in strategy 1723", tags=["risk","quant"], author="om")
    store.add("Need to revisit beta assumptions", tags=["valuation","research"])
    for m in store.list():
        print(m["timestamp"], m["text"])

Usage (CLI):
    python memo.py add "Run regression on yen carry" --tags macro,fx --author om
    python memo.py list --tags fx --author om
"""

from __future__ import annotations
import os
import sys
import json
import csv
import argparse
import datetime
from typing import Any, Dict, List, Optional


class MemoStore:
    def __init__(self, path: str = "memos.json"):
        self.path = path
        self.memos: List[Dict[str, Any]] = []
        if os.path.exists(path):
            self._load()

    # -------------------------
    # Core
    # -------------------------

    def add(self, text: str, tags: Optional[List[str]] = None, author: str = "anon") -> Dict[str, Any]:
        memo = {
            "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "author": author,
            "tags": tags or [],
            "text": text.strip(),
        }
        self.memos.append(memo)
        self._save()
        return memo

    def list(self, tags: Optional[List[str]] = None, author: Optional[str] = None) -> List[Dict[str, Any]]:
        items = self.memos
        if tags:
            items = [m for m in items if any(t in m["tags"] for t in tags)]
        if author:
            items = [m for m in items if m["author"] == author]
        return sorted(items, key=lambda m: m["timestamp"], reverse=True)

    # -------------------------
    # Persistence
    # -------------------------

    def _save(self) -> None:
        ext = os.path.splitext(self.path)[1].lower()
        if ext == ".json":
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.memos, f, ensure_ascii=False, indent=2)
        elif ext == ".csv":
            with open(self.path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["timestamp","author","tags","text"])
                w.writeheader()
                for m in self.memos:
                    w.writerow({**m, "tags": ",".join(m["tags"])})
        else:
            raise ValueError(f"Unsupported extension: {ext}")

    def _load(self) -> None:
        ext = os.path.splitext(self.path)[1].lower()
        if ext == ".json":
            with open(self.path, "r", encoding="utf-8") as f:
                self.memos = json.load(f)
        elif ext == ".csv":
            with open(self.path, "r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                self.memos = []
                for row in r:
                    self.memos.append({
                        "timestamp": row["timestamp"],
                        "author": row["author"],
                        "tags": row["tags"].split(",") if row.get("tags") else [],
                        "text": row["text"],
                    })
        else:
            raise ValueError(f"Unsupported extension: {ext}")


# -------------------------
# CLI
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Lightweight memo system")
    ap.add_argument("--path", default="memos.json", help="Path to memo store (json/csv)")

    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("add", help="Add a new memo")
    p1.add_argument("text", help="Memo text")
    p1.add_argument("--tags", default="", help="Comma-separated tags")
    p1.add_argument("--author", default="anon", help="Author name")

    p2 = sub.add_parser("list", help="List memos")
    p2.add_argument("--tags", default="", help="Comma-separated filter tags")
    p2.add_argument("--author", default=None, help="Filter by author")

    args = ap.parse_args()

    store = MemoStore(args.path)

    if args.cmd == "add":
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        m = store.add(args.text, tags=tags, author=args.author)
        print(f"✅ Added memo: {m['timestamp']} [{m['author']}] {m['text']} (tags={m['tags']})")
        return

    if args.cmd == "list":
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        items = store.list(tags=tags or None, author=args.author)
        if not items:
            print("No memos found.")
            return
        for m in items:
            print(f"{m['timestamp']} [{m['author']}] {m['text']} (tags={','.join(m['tags'])})")


if __name__ == "__main__":
    main()