#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kb/ocr/preprocess.py
--------------------
Document image preprocessing pipeline for OCR:

- Load images (png/jpg/tiff/webp) and PDFs (optional via pdf2image)
- Grayscale, denoise, binarize, deskew, resize, pad, contrast boost
- Optional morphology (open/close) to clean speckles or join characters
- Optional page splitting for multi-page TIFF/PDF (suffix _pNNN)
- Save cleaned images and a manifest (CSV or JSONL)

Examples:
  # Simple: standardize a folder (deskew+binarize+resize)
  python kb/ocr/preprocess.py --in data/scans --out data/clean --resize-long 1600 --deskew --binarize

  # With aggressive cleanup + manifest
  python kb/ocr/preprocess.py --in docs --out out/clean --deskew --binarize --denoise \
      --contrast 1.2 --morph open:3 --pad 10 --manifest out/manifest.csv

  # PDFs (requires pdf2image + poppler installed on your system)
  python kb/ocr/preprocess.py --in statements.pdf --out out/clean --pdf-dpi 300 --deskew --binarize
"""

from __future__ import annotations
import os
import sys
import io
import re
import glob
import json
import math
import argparse
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# OpenCV (prefer headless)
try:
    import cv2  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: opencv-python-headless (or opencv-python).") from e

# Optional: PDF rasterization
_HAS_PDF = True
try:
    from pdf2image import convert_from_path  # type: ignore
    from PIL import Image  # noqa: F401
except Exception:
    _HAS_PDF = False


# =========================
# I/O helpers
# =========================

IMG_EXTS = {".png",".jpg",".jpeg",".tif",".tiff",".bmp",".gif",".webp"}

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def imread_any(path: str) -> np.ndarray:
    """Robust image read (supports non-ASCII paths)."""
    data = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return data

def imwrite_any(path: str, img: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    ext = os.path.splitext(path)[1].lower()
    params = []
    if ext in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    elif ext in {".png"}:
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    ok = cv2.imencode(ext if ext else ".png", img, params)[0]
    if not ok:
        # fallback to PNG
        cv2.imencode(".png", img)
    with open(path, "wb") as f:
        f.write(cv2.imencode(ext if ext else ".png", img, params)[1].tobytes())

def rasterize_pdf(pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
    if not _HAS_PDF:
        raise RuntimeError("PDF support requires pdf2image + pillow. `pip install pdf2image pillow`")
    pages = convert_from_path(pdf_path, dpi=dpi)
    return [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pages]


# =========================
# Processing ops
# =========================

def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def denoise(img: np.ndarray, strength: int = 3) -> np.ndarray:
    k = max(1, strength | 1)
    return cv2.medianBlur(img, k)

def binarize(img: np.ndarray, method: str = "adaptive") -> np.ndarray:
    g = to_gray(img)
    if method == "otsu":
        _, thr = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thr
    # adaptive (more robust for uneven lighting)
    g = cv2.GaussianBlur(g, (3,3), 0)
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 5)

def deskew(img: np.ndarray) -> np.ndarray:
    g = to_gray(img)
    thr = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if coords.size == 0:
        return img
    rect = cv2.minAreaRect(coords.astype(np.float32))
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def resize_long(img: np.ndarray, target_long: int) -> np.ndarray:
    if target_long <= 0:
        return img
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side == target_long:
        return img
    scale = target_long / float(long_side)
    nh, nw = max(1, int(round(h*scale))), max(1, int(round(w*scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)

def pad(img: np.ndarray, px: int) -> np.ndarray:
    if px <= 0: return img
    if img.ndim == 2:
        return cv2.copyMakeBorder(img, px, px, px, px, cv2.BORDER_CONSTANT, value=255)
    return cv2.copyMakeBorder(img, px, px, px, px, cv2.BORDER_CONSTANT, value=(255,255,255))

def boost_contrast(img: np.ndarray, alpha: float = 1.2, beta: int = 0) -> np.ndarray:
    """alpha >1 increases contrast; beta shifts brightness."""
    out = cv2.convertScaleAbs(img, alpha=max(0.5, float(alpha)), beta=int(beta))
    return out

def morphology(img: np.ndarray, spec: Optional[str]) -> np.ndarray:
    """
    spec examples:
      'open:3'  -> remove small dots
      'close:3' -> join small gaps
      'erode:2'/'dilate:2'
    """
    if not spec:
        return img
    op, _, size = spec.partition(":")
    k = max(1, int(size) if size else 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    m = op.lower().strip()
    if m == "open":
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    if m == "close":
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    if m == "erode":
        return cv2.erode(img, kernel, iterations=1)
    if m == "dilate":
        return cv2.dilate(img, kernel, iterations=1)
    return img


# =========================
# Config + pipeline
# =========================

@dataclass
class ProcConfig:
    grayscale: bool = True
    denoise_strength: int = 0         # 0=off, else kernel size (odd)
    binarize: bool = True
    bin_method: str = "adaptive"      # adaptive|otsu
    deskew: bool = True
    resize_long: int = 1600
    pad: int = 10
    contrast: float = 1.0             # 1.0 = no change
    brightness: int = 0               # -50..50 typical
    morph: Optional[str] = None       # open:3 | close:3 | erode:2 | dilate:2
    pdf_dpi: int = 300
    out_ext: str = ".png"             # output image extension

def process_image(img: np.ndarray, cfg: ProcConfig) -> np.ndarray:
    out = img.copy()
    if cfg.grayscale:
        out = to_gray(out)
    if cfg.denoise_strength and cfg.denoise_strength > 0:
        out = denoise(out, cfg.denoise_strength)
    if cfg.binarize:
        out = binarize(out, cfg.bin_method)
    if cfg.deskew:
        out = deskew(out)
    if cfg.resize_long and cfg.resize_long > 0:
        out = resize_long(out, cfg.resize_long)
    if cfg.pad and cfg.pad > 0:
        out = pad(out, cfg.pad)
    if cfg.contrast and abs(cfg.contrast - 1.0) > 1e-6:
        out = boost_contrast(out, cfg.contrast, cfg.brightness)
    if cfg.morph:
        out = morphology(out, cfg.morph)
    return out


# =========================
# Runners
# =========================

def gather_inputs(target: str, pattern: Optional[str]) -> List[str]:
    if os.path.isfile(target):
        return [target]
    if os.path.isdir(target):
        pat = pattern or "**/*.*"
        files = [p for p in glob.glob(os.path.join(target, pat), recursive=True)
                 if os.path.isfile(p) and os.path.splitext(p.lower())[1] in IMG_EXTS.union({".pdf"})]
        return files
    raise FileNotFoundError(target)

def write_manifest(rows: List[Dict[str, Any]], path: Optional[str]) -> None:
    if not path:
        return
    ensure_dir(os.path.dirname(path) or ".")
    if path.lower().endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        pd.DataFrame(rows).to_csv(path, index=False)

def output_name(in_path: str, out_dir: str, page: Optional[int], out_ext: str) -> str:
    base = os.path.splitext(os.path.basename(in_path))[0]
    suffix = f"_p{page:03d}" if page is not None else ""
    return os.path.join(out_dir, f"{base}{suffix}{out_ext}")

def process_path(path: str, out_dir: str, cfg: ProcConfig) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if path.lower().endswith(".pdf"):
        imgs = rasterize_pdf(path, dpi=cfg.pdf_dpi)
        for i, img in enumerate(imgs, start=1):
            proc = process_image(img, cfg)
            outp = output_name(path, out_dir, i, cfg.out_ext)
            imwrite_any(outp, proc)
            h, w = proc.shape[:2]
            rows.append({
                "source": f"{path}#page={i}",
                "output": outp,
                "dims": [h, w],
                "ops": asdict(cfg)
            })
        return rows

    # Image path
    img = imread_any(path)
    proc = process_image(img, cfg)
    outp = output_name(path, out_dir, None, cfg.out_ext)
    imwrite_any(outp, proc)
    h, w = proc.shape[:2]
    rows.append({
        "source": path,
        "output": outp,
        "dims": [h, w],
        "ops": asdict(cfg)
    })
    return rows


# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser(description="Preprocess documents for OCR (grayscale/denoise/binarize/deskew/resize/pad)")
    ap.add_argument("--in", dest="inp", required=True, help="Input file or directory")
    ap.add_argument("--glob", default=None, help='Glob inside directory (default="**/*.*")')
    ap.add_argument("--out", required=True, help="Output directory for cleaned images")
    # ops
    ap.add_argument("--no-gray", dest="grayscale", action="store_false", help="Disable grayscale")
    ap.add_argument("--denoise", type=int, default=0, help="Median denoise kernel (odd, 0=off)")
    ap.add_argument("--binarize", action="store_true", help="Enable binarization (adaptive by default)")
    ap.add_argument("--bin-method", default="adaptive", choices=["adaptive","otsu"], help="Binarization method")
    ap.add_argument("--deskew", action="store_true", help="Enable deskew")
    ap.add_argument("--resize-long", type=int, default=1600, help="Resize longest side (0=off)")
    ap.add_argument("--pad", type=int, default=10, help="Pad pixels around border")
    ap.add_argument("--contrast", type=float, default=1.0, help="Contrast multiplier (e.g., 1.2)")
    ap.add_argument("--brightness", type=int, default=0, help="Brightness shift (-100..100)")
    ap.add_argument("--morph", default=None, help="Morphology op: open:3 | close:3 | erode:2 | dilate:2")
    ap.add_argument("--pdf-dpi", type=int, default=300, help="Rasterization DPI for PDFs")
    ap.add_argument("--out-ext", default=".png", choices=[".png",".jpg",".jpeg",".tif",".tiff"], help="Output image extension")
    # outputs
    ap.add_argument("--manifest", default=None, help="Write CSV/JSONL manifest (infer by extension)")
    args = ap.parse_args()

    cfg = ProcConfig(
        grayscale=args.grayscale,
        denoise_strength=args.denoise,
        binarize=args.binarize,
        bin_method=args.bin_method,
        deskew=args.deskew,
        resize_long=args.resize_long,
        pad=args.pad,
        contrast=args.contrast,
        brightness=args.brightness,
        morph=args.morph,
        pdf_dpi=args.pdf_dpi,
        out_ext=args.out_ext
    )

    ensure_dir(args.out)
    inpaths = gather_inputs(args.inp, args.glob)

    all_rows: List[Dict[str, Any]] = []
    for p in inpaths:
        try:
            rows = process_path(p, args.out, cfg)
            all_rows.extend(rows)
            print(f"✓ {p} → {len(rows)} file(s)")
        except Exception as e:
            print(f"✗ {p}: {e}", file=sys.stderr)

    if all_rows and args.manifest:
        write_manifest(all_rows, args.manifest)
        print(f"✅ wrote manifest: {args.manifest}")
    print(f"✅ finished: {len(all_rows)} output image(s)")

if __name__ == "__main__":
    main()