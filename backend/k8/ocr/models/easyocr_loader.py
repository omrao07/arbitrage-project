#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kb/ocr/easyocr_loader.py
------------------------
Loader + thin wrapper around EasyOCR.

- Reads kb/ocr/configs/easyocr.yaml
- Preprocess: grayscale, denoise, binarize, deskew, resize, pad
- Inference: text/low_text/link thresholds, decoder, batch_size, workers
- Output: JSONL/TXT with boxes + confidence (one JSON per input image)
- CLI: file / dir

Install:
  pip install easyocr opencv-python-headless numpy pyyaml
  # Optional: for better deskew accuracy (not required)
  pip install scikit-image

Example:
  python kb/ocr/easyocr_loader.py file path/to/image.png
  python kb/ocr/easyocr_loader.py dir data/scans/ --glob "**/*.png"
"""

from __future__ import annotations
import os
import re
import io
import sys
import json
import time
import glob
import math
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml
import numpy as np

# Prefer headless OpenCV for servers
try:
    import cv2  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: opencv-python-headless (or opencv-python).") from e

try:
    import easyocr  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: easyocr. pip install easyocr") from e


DEFAULT_CFG_PATH = os.path.join("kb", "ocr", "configs", "easyocr.yaml")


# =========================================================
# Config types
# =========================================================

@dataclass
class EasyOCRConfig:
    # ocr
    languages: List[str]
    gpu: bool
    model_storage_dir: Optional[str]
    download_enabled: bool

    # preprocess
    grayscale: bool
    denoise: bool
    binarize: bool
    deskew: bool
    resize_factor: float
    pad: int

    # inference
    text_threshold: float
    low_text: float
    link_threshold: float
    decoder: str
    batch_size: int
    workers: int
    detail: int

    # postprocess
    spellcheck: bool
    correct_confusable: bool
    min_length: int
    uppercase: bool
    lowercase: bool

    # runtime
    device: str
    log_level: str

    # output
    results_dir: str
    save_format: str
    include_boxes: bool
    include_confidence: bool


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str = DEFAULT_CFG_PATH) -> EasyOCRConfig:
    cfg = _read_yaml(path) or {}
    ocr = cfg.get("ocr", {})
    pre = cfg.get("preprocess", {})
    inf = cfg.get("inference", {})
    post = cfg.get("postprocess", {})
    run = cfg.get("runtime", {})
    out = cfg.get("output", {})

    return EasyOCRConfig(
        languages=list(ocr.get("languages", ["en"])),
        gpu=bool(ocr.get("gpu", True)),
        model_storage_dir=ocr.get("model_storage_dir"),
        download_enabled=bool(ocr.get("download_enabled", True)),

        grayscale=bool(pre.get("grayscale", True)),
        denoise=bool(pre.get("denoise", True)),
        binarize=bool(pre.get("binarize", True)),
        deskew=bool(pre.get("deskew", True)),
        resize_factor=float(pre.get("resize_factor", 2.0)),
        pad=int(pre.get("pad", 10)),

        text_threshold=float(inf.get("text_threshold", 0.7)),
        low_text=float(inf.get("low_text", 0.4)),
        link_threshold=float(inf.get("link_threshold", 0.4)),
        decoder=str(inf.get("decoder", "greedy")),
        batch_size=int(inf.get("batch_size", 16)),
        workers=int(inf.get("workers", 2)),
        detail=int(inf.get("detail", 1)),

        spellcheck=bool(post.get("spellcheck", False)),
        correct_confusable=bool(post.get("correct_confusable", True)),
        min_length=int(post.get("min_length", 2)),
        uppercase=bool(post.get("uppercase", False)),
        lowercase=bool(post.get("lowercase", True)),

        device=str(run.get("device", "auto")),
        log_level=str(run.get("log_level", "INFO")),

        results_dir=str(out.get("results_dir", os.path.join("outputs", "ocr", "easyocr"))),
        save_format=str(out.get("save_format", "jsonl")),
        include_boxes=bool(out.get("include_boxes", True)),
        include_confidence=bool(out.get("include_confidence", True)),
    )


# =========================================================
# Preprocess
# =========================================================

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def _denoise(img: np.ndarray) -> np.ndarray:
    # median blur is safe for text edges
    k = 3 if min(img.shape[:2]) > 700 else 1
    k = max(1, k | 1)
    return cv2.medianBlur(img, k)

def _binarize(img: np.ndarray) -> np.ndarray:
    # adaptive thresholding to handle variable lighting
    if img.ndim != 2:
        img = _to_gray(img)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 5)

def _deskew(img: np.ndarray) -> np.ndarray:
    """
    Estimate skew angle via minAreaRect of text mask and rotate.
    """
    gray = _to_gray(img)
    # threshold for text mask
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if coords.size == 0:
        return img
    rect = cv2.minAreaRect(coords.astype(np.float32))
    angle = rect[-1]
    # OpenCV returns angle in [-90, 0)
    if angle < -45:
        angle = 90 + angle
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def _resize(img: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0:
        return img
    h, w = img.shape[:2]
    nh, nw = max(1, int(h * factor)), max(1, int(w * factor))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)

def _pad(img: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return img
    if img.ndim == 2:
        return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))

def preprocess_image(img: np.ndarray, cfg: EasyOCRConfig) -> np.ndarray:
    out = img.copy()
    if cfg.grayscale:
        out = _to_gray(out)
    if cfg.denoise:
        out = _denoise(out)
    if cfg.binarize:
        out = _binarize(out)
    if cfg.deskew:
        out = _deskew(out)
    if cfg.resize_factor and abs(cfg.resize_factor - 1.0) > 1e-6:
        out = _resize(out, cfg.resize_factor)
    if cfg.pad and cfg.pad > 0:
        out = _pad(out, cfg.pad)
    return out


# =========================================================
# Postprocess
# =========================================================

_CONFUSABLES = {
    "0": "O", "1": "l", "5": "S", "8": "B",
    "o": "0", "l": "1", "|": "I", "€": "E", "$": "S"
}

def _correct_confusables(text: str) -> str:
    return "".join(_CONFUSABLES.get(c, c) for c in text)

def _normalize_text(text: str, cfg: EasyOCRConfig) -> str:
    t = text.strip()
    if cfg.correct_confusable:
        t = _correct_confusables(t)
    if cfg.lowercase and not cfg.uppercase:
        t = t.lower()
    elif cfg.uppercase and not cfg.lowercase:
        t = t.upper()
    return t


# =========================================================
# Reader wrapper
# =========================================================

class EasyOCRLoader:
    def __init__(self, cfg_path: str = DEFAULT_CFG_PATH):
        self.cfg = load_config(cfg_path)
        _ensure_dir(self.cfg.results_dir)

        # Construct EasyOCR reader
        self.reader = easyocr.Reader(
            lang_list=self.cfg.languages,
            gpu=self.cfg.gpu,
            model_storage_directory=self.cfg.model_storage_dir,
            download_enabled=self.cfg.download_enabled,
            # You can pass detector/recognizer args via environment if needed
        )

    # ---------- I/O ----------
    @staticmethod
    def _imread(path: str) -> np.ndarray:
        data = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if data is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return data

    def _to_output_path(self, image_path: str) -> str:
        base = os.path.splitext(os.path.basename(image_path))[0]
        ext = "jsonl" if self.cfg.save_format.lower() == "jsonl" else "txt"
        return os.path.join(self.cfg.results_dir, f"{base}.{ext}")

    # ---------- Core ----------
    def _infer(self, img: np.ndarray) -> List[Tuple[List[List[int]], str, float]]:
        """
        Returns EasyOCR raw list: [ [box(xyxyxyxy), text, conf], ... ]
        """
        res = self.reader.readtext(
            img,
            detail=1 if self.cfg.include_boxes or self.cfg.include_confidence or self.cfg.detail else 0,
            paragraph=False,
            decoder=self.cfg.decoder,
            text_threshold=self.cfg.text_threshold,
            low_text=self.cfg.low_text,
            link_threshold=self.cfg.link_threshold,
            batch_size=self.cfg.batch_size,
            workers=self.cfg.workers,
            # contrast_ths / adjust_contrast can be tuned if needed
        )
        return res

    @staticmethod
    def _combine_text(res: List[Tuple[List[List[int]], str, float]], cfg: EasyOCRConfig) -> str:
        parts = []
        for item in res:
            # item: (box, text, conf) when detail=1; or text when detail=0
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                txt = str(item[1])
            else:
                txt = str(item)
            txt = txt.strip()
            if len(txt) >= cfg.min_length:
                parts.append(txt)
        return "\n".join(parts).strip()

    def transcribe_image(self, image_path: str) -> Dict[str, Any]:
        t0 = time.time()
        img = self._imread(image_path)
        h, w = img.shape[:2]
        proc = preprocess_image(img, self.cfg)
        raw = self._infer(proc)

        blocks = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                box = item[0] if isinstance(item[0], (list, tuple)) else None
                text = str(item[1] if len(item) > 1 else "")
                conf = float(item[2]) if len(item) > 2 else None
            else:
                box, text, conf = None, str(item), None

            text = _normalize_text(text, self.cfg)
            if len(text) < self.cfg.min_length:
                continue

            blk = {"text": text}
            if self.cfg.include_boxes and box is not None:
                # Ensure integer points
                blk["box"] = [[int(x), int(y)] for x, y in box] # type: ignore
            if self.cfg.include_confidence and conf is not None:
                blk["confidence"] = float(conf) # type: ignore
            blocks.append(blk)

        combined_text = "\n".join([b["text"] for b in blocks]) if blocks else self._combine_text(raw, self.cfg)

        payload: Dict[str, Any] = {
            "image_path": image_path,
            "engine": "easyocr",
            "languages": self.cfg.languages,
            "dims": [h, w],
            "text": combined_text,
            "blocks": blocks if self.cfg.detail or self.cfg.include_boxes or self.cfg.include_confidence else [],
            "latency_s": round(time.time() - t0, 3),
        }
        return payload

    def save_result(self, result: Dict[str, Any]) -> str:
        out_path = self._to_output_path(result["image_path"])
        if self.cfg.save_format.lower() == "jsonl":
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(result.get("text", ""))
        return out_path

    # ---------- Batch ----------
    def process_batch(self, image_paths: List[str], save: bool = True) -> List[Dict[str, Any]]:
        outs = []
        for p in image_paths:
            try:
                r = self.transcribe_image(p)
                if save:
                    self.save_result(r)
                outs.append(r)
                print(f"✓ {p}  →  {len(r.get('text',''))} chars, {len(r.get('blocks',[]))} boxes")
            except Exception as e:
                print(f"✗ {p}: {e}", file=sys.stderr)
        return outs

    def process_dir(self, image_dir: str, pattern: str = "**/*.*", save: bool = True) -> List[Dict[str, Any]]:
        paths = [p for p in glob.glob(os.path.join(image_dir, pattern), recursive=True)
                 if os.path.isfile(p) and os.path.splitext(p.lower())[1] in {".png",".jpg",".jpeg",".tif",".tiff",".bmp",".gif",".webp",".pdf"}]
        # Note: for PDFs, you should rasterize pages first; EasyOCR can sometimes handle PDFs via pdf2image.
        return self.process_batch(paths, save=save)


# =========================================================
# CLI
# =========================================================

def main():
    ap = argparse.ArgumentParser(description="EasyOCR loader")
    ap.add_argument("--config", default=DEFAULT_CFG_PATH, help="Path to kb/ocr/configs/easyocr.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("file", help="OCR a single image")
    s1.add_argument("image", help="Path to image")
    s1.add_argument("--no-save", action="store_true")

    s2 = sub.add_parser("dir", help="OCR a directory (recursive)")
    s2.add_argument("image_dir", help="Directory path")
    s2.add_argument("--glob", default="**/*.*", help='Glob pattern (default="**/*.*")')
    s2.add_argument("--no-save", action="store_true")

    args = ap.parse_args()

    loader = EasyOCRLoader(args.config)

    if args.cmd == "file":
        result = loader.transcribe_image(args.image)
        if not args.no_save:
            out = loader.save_result(result)
            print(f"✅ wrote {out}")
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.cmd == "dir":
        rs = loader.process_dir(args.image_dir, args.glob, save=not args.no_save)
        print(f"✅ processed {len(rs)} files")
        return


if __name__ == "__main__":
    main()