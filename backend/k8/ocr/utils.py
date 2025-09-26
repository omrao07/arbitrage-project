#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kb/ocr/utils.py
---------------
Shared utilities for the OCR stack:

- Robust image/PDF I/O (OpenCV + optional pdf2image)
- Path helpers, logging, wall-time timer
- Text normalization and confusable-character fixes
- Polygon/box utils: rect, IoU, area, NMS, greedy merge
- Lightweight manifest read/write (CSV/JSONL)
"""

from __future__ import annotations
import os
import io
import sys
import re
import json
import time
import glob
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# Prefer headless OpenCV on servers
try:
    import cv2  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: opencv-python-headless (or opencv-python).") from e

# Optional PDF rasterization
_HAS_PDF = True
try:
    from pdf2image import convert_from_path  # type: ignore
    from PIL import Image  # noqa: F401
except Exception:
    _HAS_PDF = False


# =========================================================
# Logging / Timer
# =========================================================

def get_logger(name: str = "ocr", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(h)
    logger.setLevel(level.upper() if isinstance(level, str) else level)
    return logger

class Timer:
    """Context manager to measure wall time in seconds."""
    def __init__(self, label: str = "", sink=print):
        self.label = label
        self.sink = sink
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.time() - self.start#type:ignore
        if self.label:
            self.sink(f"{self.label}: {self.elapsed:.3f}s")


# =========================================================
# Paths / I/O
# =========================================================

IMG_EXTS = {".png",".jpg",".jpeg",".tif",".tiff",".bmp",".gif",".webp"}
PDF_EXTS = {".pdf"}

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def base_noext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def imread_any(path: str) -> np.ndarray:
    """Robust image read (supports non-ASCII paths)."""
    data = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return data

def imwrite_any(path: str, img: np.ndarray, *, png_compression: int = 3, jpeg_quality: int = 95) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    ext = os.path.splitext(path)[1].lower() or ".png"
    params: List[int] = []
    if ext in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    elif ext == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), int(png_compression)]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        # Fallback to PNG
        ok, buf = cv2.imencode(".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    with open(path, "wb") as f:
        f.write(buf.tobytes())

def rasterize_pdf(pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
    if not _HAS_PDF:
        raise RuntimeError("PDF support requires pdf2image + pillow. Install with `pip install pdf2image pillow`.")
    pages = convert_from_path(pdf_path, dpi=dpi)
    return [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pages]

def gather_paths(target: str, pattern: Optional[str] = None) -> List[str]:
    if os.path.isfile(target):
        return [target]
    if os.path.isdir(target):
        pat = pattern or "**/*.*"
        exts = IMG_EXTS.union(PDF_EXTS)
        return [p for p in glob.glob(os.path.join(target, pat), recursive=True)
                if os.path.isfile(p) and os.path.splitext(p.lower())[1] in exts]
    raise FileNotFoundError(target)


# =========================================================
# Text normalization
# =========================================================

_CONFUSABLES = {
    "0": "O", "1": "l", "5": "S", "8": "B",
    "o": "0", "l": "1", "|": "I", "€": "E", "$": "S"
}

def normalize_text_basic(text: str, *, lowercase: bool = True, uppercase: bool = False,
                         correct_confusable: bool = True, strip: bool = True) -> str:
    t = text or ""
    if strip:
        t = t.strip()
    if correct_confusable:
        t = "".join(_CONFUSABLES.get(c, c) for c in t)
    if lowercase and not uppercase:
        t = t.lower()
    elif uppercase and not lowercase:
        t = t.upper()
    # collapse whitespace
    t = re.sub(r"\s+", " ", t)
    return t


# =========================================================
# Geometry / Boxes
# =========================================================

def rect_from_poly(poly: List[List[int]]) -> Tuple[int,int,int,int]:
    """[[x,y]x4] -> (x1,y1,x2,y2) as ints."""
    xs = [int(p[0]) for p in poly]; ys = [int(p[1]) for p in poly]
    return min(xs), min(ys), max(xs), max(ys)

def area_rect(r: Tuple[int,int,int,int]) -> int:
    x1,y1,x2,y2 = r
    return max(0, x2 - x1) * max(0, y2 - y1)

def iou_poly(a: List[List[int]], b: List[List[int]]) -> float:
    x1,y1,x2,y2 = rect_from_poly(a)
    X1,Y1,X2,Y2 = rect_from_poly(b)
    ix1, iy1 = max(x1, X1), max(y1, Y1)
    ix2, iy2 = min(x2, X2), min(y2, Y2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = (x2 - x1) * (y2 - y1)
    a2 = (X2 - X1) * (Y2 - Y1)
    return inter / float(a1 + a2 - inter + 1e-9)

def nms_polys(polys: List[List[List[int]]], scores: List[float], iou_thresh: float = 0.5) -> List[int]:
    """Return indices kept by Non-Max Suppression on polygons (axis-aligned via bbox)."""
    if not polys:
        return []
    idxs = list(range(len(polys)))
    idxs.sort(key=lambda i: scores[i], reverse=True)
    kept: List[int] = []
    while idxs:
        i = idxs.pop(0)
        kept.append(i)
        idxs = [j for j in idxs if iou_poly(polys[i], polys[j]) < iou_thresh]
    return kept

def merge_overlapping_boxes(blocks: List[Dict[str, Any]], iou_thresh: float = 0.5, avg_conf: bool = True) -> List[Dict[str, Any]]:
    """
    Greedy merge: if boxes overlap strongly and text matches (case/space-insensitive),
    keep first box; average confidence if requested.
    """
    if not blocks:
        return []
    norm = lambda s: re.sub(r"\s+", " ", (s or "").strip().lower())
    out: List[Dict[str, Any]] = []
    used = [False] * len(blocks)
    for i, bi in enumerate(blocks):
        if used[i]:
            continue
        group = [i]
        for j in range(i + 1, len(blocks)):
            if used[j]:
                continue
            if norm(blocks[j].get("text")) == norm(bi.get("text")) and iou_poly(blocks[i]["box"], blocks[j]["box"]) >= iou_thresh:
                used[j] = True
                group.append(j)
        if len(group) == 1:
            out.append(bi); continue
        # merge
        merged = dict(bi)
        if avg_conf:
            confs = [blocks[g].get("confidence") for g in group if isinstance(blocks[g].get("confidence"), (int,float))]
            if confs:
                merged["confidence"] = float(sum(confs) / len(confs))#type:ignore
        out.append(merged)
    return out


# =========================================================
# Manifest helpers
# =========================================================

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows