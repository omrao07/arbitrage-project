# simulation-farm/artifacts/reports/exporters/local_exporter.py
"""
Local filesystem exporter for simulation reports.

- Writes to a configurable root directory (default: artifacts/reports/out)
- Creates subdirs as needed; preserves relative paths on directory uploads
- Returns file:// URLs for easy linking
- Small helpers to clean/list/open outputs

Usage
-----
exp = LocalExporter(root="artifacts/reports/out", prefix="run_2025_09_16/")
url = exp.upload_file("artifacts/reports/tmp/summary.html", dest="summary.html")
print(url)  # file:///.../artifacts/reports/out/run_2025_09_16/summary.html

# upload bytes
exp.upload_bytes(b"hello", "notes.txt", content_type="text/plain")

# upload a whole directory (recursively)
exp.upload_dir("artifacts/reports/run_123")  # keeps relative layout under prefix
"""

from __future__ import annotations

import mimetypes
import os
import pathlib
import shutil
import sys
import webbrowser
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class LocalExporterConfig:
    root: str = "artifacts/reports/out"   # base output directory
    prefix: str = ""                      # optional sub-prefix inside root (e.g., "run_001/")
    make_parents: bool = True             # create directories automatically


class LocalExporter:
    def __init__(self, root: str = "artifacts/reports/out", prefix: str = "", *, make_parents: bool = True):
        # normalize prefix to "foo/bar/" or ""
        prefix = prefix.strip("/").replace("\\", "/")
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        self.cfg = LocalExporterConfig(root=root, prefix=prefix, make_parents=make_parents)
        self._ensure_dir(self.base_path)

    # ------------- public API -------------

    @property
    def base_path(self) -> pathlib.Path:
        return pathlib.Path(self.cfg.root) / self.cfg.prefix

    def upload_file(self, path: str, dest: Optional[str] = None, *, content_type: Optional[str] = None) -> str:
        """
        Copy a single file to root/prefix/dest (or same basename).
        Returns a file:// URL.
        """
        src = pathlib.Path(path)
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(str(src))
        dest_rel = (dest or src.name).lstrip("/").replace("\\", "/")
        out_path = self.base_path / dest_rel
        self._ensure_dir(out_path.parent)
        shutil.copy2(src, out_path)
        # content_type currently informational only (no metadata store on local FS)
        return self._to_file_url(out_path)

    def upload_bytes(self, data: bytes, dest: str, *, content_type: Optional[str] = None) -> str:
        """
        Write raw bytes to root/prefix/dest. Returns a file:// URL.
        """
        dest_rel = dest.lstrip("/").replace("\\", "/")
        out_path = self.base_path / dest_rel
        self._ensure_dir(out_path.parent)
        with open(out_path, "wb") as f:
            f.write(data)
        return self._to_file_url(out_path)

    def upload_dir(self, directory: str, *, strip_prefix: Optional[str] = None) -> List[str]:
        """
        Recursively copy a directory under root/prefix, preserving relative layout.
        Returns a list of file:// URLs for all copied files.
        """
        src_root = pathlib.Path(directory)
        if not src_root.exists() or not src_root.is_dir():
            raise NotADirectoryError(str(src_root))

        if strip_prefix:
            strip_base = pathlib.Path(strip_prefix)
        else:
            strip_base = src_root

        urls: List[str] = []
        for fp in src_root.rglob("*"):
            if fp.is_file():
                rel = fp.relative_to(strip_base)
                urls.append(self.upload_file(str(fp), dest=str(rel).replace("\\", "/")))
        return urls

    def list_outputs(self) -> List[str]:
        """
        List file:// URLs of all files under root/prefix.
        """
        if not self.base_path.exists():
            return []
        urls: List[str] = []
        for fp in self.base_path.rglob("*"):
            if fp.is_file():
                urls.append(self._to_file_url(fp))
        return urls

    def clean(self) -> None:
        """
        Delete the current prefix directory (if exists).
        """
        base = self.base_path
        if base.exists():
            shutil.rmtree(base, ignore_errors=True)

    def open_in_browser(self, relative_path: str = "index.html") -> None:
        """
        Open a file under root/prefix in the default web browser (best-effort).
        """
        target = (self.base_path / relative_path).resolve()
        if target.exists():
            webbrowser.open(self._to_file_url(target))
        else:
            print(f"[local_exporter] not found: {target}", file=sys.stderr)

    # ------------- internals -------------

    def _ensure_dir(self, path: pathlib.Path) -> None:
        if self.cfg.make_parents:
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(exist_ok=True)

    @staticmethod
    def _to_file_url(p: pathlib.Path) -> str:
        return p.resolve().as_uri()

    @staticmethod
    def guess_content_type(name: str) -> str:
        ct, _ = mimetypes.guess_type(name)
        return ct or "application/octet-stream"


# -------------------------- CLI (optional) --------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Export files/dirs to local artifacts folder")
    ap.add_argument("--root", default="artifacts/reports/out")
    ap.add_argument("--prefix", default="")
    ap.add_argument("paths", nargs="*", help="Files or directories to copy")
    args = ap.parse_args()

    exp = LocalExporter(root=args.root, prefix=args.prefix)
    urls: List[str] = []
    for p in args.paths:
        if os.path.isdir(p):
            urls.extend(exp.upload_dir(p))
        else:
            urls.append(exp.upload_file(p))

    for u in urls:
        print(u)