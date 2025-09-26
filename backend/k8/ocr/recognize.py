#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kb/ocr/recognize.py
-------------------
Unified OCR runner for EasyOCR / PaddleOCR / Tesseract.

- Reads an OCR YAML (easyocr.yaml / paddleocr.yaml / tesseract.yaml)
- Builds the corresponding loader (easyocr_loader / paddleocr_loader / tesseract_loader)
- Runs on a single file, a directory (recursive), or a CSV/JSONL manifest
- Lets the loader decide how/where to save results (JSONL/TXT) per YAML

Examples:
  # EasyOCR
  python kb/ocr/recognize.py --config kb/ocr/configs/easyocr.yaml file docs/invoice.png
  python kb/ocr/recognize.py --config kb/ocr/configs/easyocr.yaml dir  docs/

  # PaddleOCR (PDFs supported)
  python kb/ocr/recognize.py --config kb/ocr/configs/paddleocr.yaml file scans/bank.pdf

  # Tesseract (PDFs supported)
  python kb/ocr/recognize.py --config kb/ocr/configs/tesseract.yaml dir scans/ --glob "**/*.tif"

  # Manifest (CSV/JSONL with column/key 'image' or 'path')
  python kb/ocr/recognize.py --config kb/ocr/configs/paddleocr.yaml manifest data/ocr_manifest.csv
"""

from __future__ import annotations
import os
import sys
import csv
import json
import glob
import argparse
from typing import Any, Dict, List, Optional

import yaml

# Local loaders
from kb.ocr.easyocr_loader import EasyOCRLoader # type: ignore
from kb.ocr.paddleocr_loader import PaddleOCRLoader # type: ignore
from kb.ocr.tesseract_loader import TesseractLoader # type: ignore


# ------------------------------------------------------------
# Config helpers
# ------------------------------------------------------------

def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _detect_engine(cfg_path: str) -> str:
    cfg = _read_yaml(cfg_path)
    # keep structure similar to easyocr/paddle/tesseract yamls
    eng = (cfg.get("ocr", {}).get("engine") or "").strip().lower()
    if not eng:
        # try a permissive guess from filename
        base = os.path.basename(cfg_path).lower()
        if "easy" in base:
            eng = "easyocr"
        elif "paddle" in base:
            eng = "paddleocr"
        elif "tesseract" in base or "tess" in base:
            eng = "tesseract"
    if eng not in {"easyocr", "paddleocr", "tesseract"}:
        raise ValueError(f"Unsupported or missing engine in YAML: {cfg_path}")
    return eng

def _build_loader(engine: str, cfg_path: str):
    e = engine.lower()
    if e == "easyocr":
        return EasyOCRLoader(cfg_path)
    if e == "paddleocr":
        return PaddleOCRLoader(cfg_path)
    if e == "tesseract":
        return TesseractLoader(cfg_path)
    raise ValueError(f"Unknown engine: {engine}")


# ------------------------------------------------------------
# Inputs
# ------------------------------------------------------------

IMG_EXTS = {".png",".jpg",".jpeg",".tif",".tiff",".bmp",".gif",".webp",".pdf"}

def gather_files(target: str, pattern: Optional[str]) -> List[str]:
    if os.path.isfile(target):
        return [target]
    if os.path.isdir(target):
        pat = pattern or "**/*.*"
        return [p for p in glob.glob(os.path.join(target, pat), recursive=True)
                if os.path.isfile(p) and os.path.splitext(p.lower())[1] in IMG_EXTS]
    raise FileNotFoundError(target)

def read_manifest(path: str) -> List[str]:
    """
    Accept CSV (header) or JSONL. Recognized keys: 'image' or 'path'.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    items: List[str] = []
    if path.lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                obj = json.loads(s)
                p = obj.get("image") or obj.get("path")
                if p:
                    items.append(p)
    else:
        with open(path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                p = row.get("image") or row.get("path")
                if p:
                    items.append(p)
    if not items:
        raise ValueError(f"No 'image' or 'path' entries found in manifest: {path}")
    return items


# ------------------------------------------------------------
# Runners
# ------------------------------------------------------------

def run_file(loader, path: str, no_save: bool = False) -> List[Dict[str, Any]]:
    """
    Returns a list because PDFs (Paddle/Tesseract) yield one payload per page.
    """
    if path.lower().endswith(".pdf") and hasattr(loader, "transcribe_pdf"):
        payloads = loader.transcribe_pdf(path)
        if not no_save:
            for payload in payloads:
                loader.save_result(payload)
        else:
            print(json.dumps(payloads, ensure_ascii=False, indent=2))
        print(f"✅ processed {len(payloads)} page(s)")
        return payloads

    result = loader.transcribe_image(path)
    if not no_save:
        loader.save_result(result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    print("✅ processed 1 file")
    return [result]

def run_dir(loader, image_dir: str, pattern: str, no_save: bool = False) -> List[Dict[str, Any]]:
    paths = gather_files(image_dir, pattern)
    print(f"Found {len(paths)} path(s)")
    results = loader.process_batch(paths, save=not no_save) if hasattr(loader, "process_batch") \
               else loader.process_dir(image_dir, pattern, save=not no_save)
    print(f"✅ processed {len(results)} file/page payload(s)")
    return results

def run_manifest(loader, manifest_path: str, no_save: bool = False) -> List[Dict[str, Any]]:
    files = read_manifest(manifest_path)
    print(f"Manifest contains {len(files)} path(s)")
    results = loader.process_batch(files, save=not no_save)
    print(f"✅ processed {len(results)} file/page payload(s)")
    return results


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Unified OCR recognizer (EasyOCR / PaddleOCR / Tesseract)")
    ap.add_argument("--config", required=True, help="Path to OCR YAML (easyocr.yaml / paddleocr.yaml / tesseract.yaml)")
    ap.add_argument("--engine", default=None, help="Override YAML engine: easyocr|paddleocr|tesseract")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("file", help="OCR a single image/PDF")
    p1.add_argument("path", help="Path to image or PDF")
    p1.add_argument("--no-save", action="store_true")

    p2 = sub.add_parser("dir", help="OCR a directory (recursive)")
    p2.add_argument("image_dir", help="Directory path")
    p2.add_argument("--glob", default="**/*.*", help='Glob pattern (default="**/*.*")')
    p2.add_argument("--no-save", action="store_true")

    p3 = sub.add_parser("manifest", help="OCR from a CSV/JSONL manifest")
    p3.add_argument("path", help="Manifest with 'image' or 'path' column/key")
    p3.add_argument("--no-save", action="store_true")

    args = ap.parse_args()

    engine = (args.engine or _detect_engine(args.config)).lower()
    loader = _build_loader(engine, args.config)

    if args.cmd == "file":
        run_file(loader, args.path, no_save=args.no_save); return
    if args.cmd == "dir":
        run_dir(loader, args.image_dir, args.glob, no_save=args.no_save); return
    if args.cmd == "manifest":
        run_manifest(loader, args.path, no_save=args.no_save); return


if __name__ == "__main__":
    main()