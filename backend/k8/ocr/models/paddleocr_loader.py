#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kb/ocr/paddleocr_loader.py
--------------------------
Loader + thin wrapper around PaddleOCR.

- Reads kb/ocr/configs/paddleocr.yaml
- Preprocess: grayscale, denoise, binarize, deskew, resize, pad
- Inference: configurable det/rec/cls models, thresholds, batch
- Output: JSONL/TXT with boxes + confidence (one JSON per input image)
- CLI: file / dir (with optional PDF rasterization when pdf2image is installed)

Install:
  pip install "paddleocr>=2.7.0" "opencv-python-headless" pyyaml numpy
  # Optional for PDFs:
  pip install pdf2image pillow

Example:
  python kb/ocr/paddleocr_loader.py file path/to/image.png
  python kb/ocr/paddleocr_loader.py dir data/scans/ --glob "**/*.png"
"""

from __future__ import annotations
import os
import sys
import re
import io
import json
import time
import glob
import math
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml
import numpy as np

# OpenCV (prefer headless for servers)
try:
    import cv2  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: opencv-python-headless (or opencv-python).") from e

# PaddleOCR
try:
    from paddleocr import PaddleOCR  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: paddleocr. pip install paddleocr") from e

# Optional PDF rasterization
_HAS_PDF = True
try:
    from pdf2image import convert_from_path  # type: ignore
    from PIL import Image  # noqa
except Exception:
    _HAS_PDF = False


DEFAULT_CFG_PATH = os.path.join("kb", "ocr", "configs", "paddleocr.yaml")


# =========================================================
# Config
# =========================================================

@dataclass
class PaddleOCRConfig:
    # ocr
    languages: List[str]
    use_angle_cls: bool
    det_model_dir: Optional[str]
    rec_model_dir: Optional[str]
    cls_model_dir: Optional[str]
    use_gpu: bool
    enable_mkldnn: bool
    gpu_mem: int
    precision: str  # fp32|fp16

    # preprocess
    grayscale: bool
    denoise: bool
    binarize: bool
    deskew: bool
    resize_long: int
    pad: int

    # inference
    det_algorithm: str
    rec_algorithm: str
    use_space_char: bool
    box_thresh: float
    text_thresh: float
    max_batch_size: int
    output_time: bool

    # postprocess
    spellcheck: bool
    correct_confusable: bool
    merge_boxes: bool
    min_length: int
    lowercase: bool
    uppercase: bool

    # runtime
    device: str
    num_threads: int
    log_level: str

    # output
    results_dir: str
    save_format: str
    include_boxes: bool
    include_confidence: bool


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str = DEFAULT_CFG_PATH) -> PaddleOCRConfig:
    cfg = _read_yaml(path) or {}
    ocr = cfg.get("ocr", {})
    pre = cfg.get("preprocess", {})
    inf = cfg.get("inference", {})
    post = cfg.get("postprocess", {})
    run = cfg.get("runtime", {})
    out = cfg.get("output", {})

    return PaddleOCRConfig(
        languages=list(ocr.get("languages", ["en"])),
        use_angle_cls=bool(ocr.get("use_angle_cls", True)),
        det_model_dir=ocr.get("det_model_dir"),
        rec_model_dir=ocr.get("rec_model_dir"),
        cls_model_dir=ocr.get("cls_model_dir"),
        use_gpu=bool(ocr.get("use_gpu", True)),
        enable_mkldnn=bool(ocr.get("enable_mkldnn", False)),
        gpu_mem=int(ocr.get("gpu_mem", 8000)),
        precision=str(ocr.get("precision", "fp32")),

        grayscale=bool(pre.get("grayscale", True)),
        denoise=bool(pre.get("denoise", True)),
        binarize=bool(pre.get("binarize", True)),
        deskew=bool(pre.get("deskew", True)),
        resize_long=int(pre.get("resize_long", 1280)),
        pad=int(pre.get("pad", 10)),

        det_algorithm=str(inf.get("det_algorithm", "DB")),
        rec_algorithm=str(inf.get("rec_algorithm", "CRNN")),
        use_space_char=bool(inf.get("use_space_char", True)),
        box_thresh=float(inf.get("box_thresh", 0.6)),
        text_thresh=float(inf.get("text_thresh", 0.5)),
        max_batch_size=int(inf.get("max_batch_size", 32)),
        output_time=bool(inf.get("output_time", False)),

        spellcheck=bool(post.get("spellcheck", False)),
        correct_confusable=bool(post.get("correct_confusable", True)),
        merge_boxes=bool(post.get("merge_boxes", True)),
        min_length=int(post.get("min_length", 2)),
        lowercase=bool(post.get("lowercase", True)),
        uppercase=bool(post.get("uppercase", False)),

        device=str(run.get("device", "auto")),
        num_threads=int(run.get("num_threads", 4)),
        log_level=str(run.get("log_level", "INFO")),

        results_dir=str(out.get("results_dir", os.path.join("outputs", "ocr", "paddleocr"))),
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
    k = 3 if min(img.shape[:2]) > 700 else 1
    k = max(1, k | 1)
    return cv2.medianBlur(img, k)

def _binarize(img: np.ndarray) -> np.ndarray:
    if img.ndim != 2:
        img = _to_gray(img)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 5)

def _deskew(img: np.ndarray) -> np.ndarray:
    gray = _to_gray(img)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if coords.size == 0:
        return img
    rect = cv2.minAreaRect(coords.astype(np.float32))
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def _resize_long(img: np.ndarray, target_long: int) -> np.ndarray:
    if target_long <= 0:
        return img
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side == target_long:
        return img
    scale = target_long / float(long_side)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)

def _pad(img: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return img
    if img.ndim == 2:
        return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))

def preprocess_image(img: np.ndarray, cfg: PaddleOCRConfig) -> np.ndarray:
    out = img.copy()
    if cfg.grayscale:
        out = _to_gray(out)
    if cfg.denoise:
        out = _denoise(out)
    if cfg.binarize:
        out = _binarize(out)
    if cfg.deskew:
        out = _deskew(out)
    if cfg.resize_long and cfg.resize_long > 0:
        out = _resize_long(out, cfg.resize_long)
    if cfg.pad and cfg.pad > 0:
        out = _pad(out, cfg.pad)
    return out


# =========================================================
# Postprocess
# =========================================================

_CONFUSABLES = {"0": "O", "1": "l", "5": "S", "8": "B", "o": "0", "l": "1", "|": "I"}

def _normalize_text(text: str, cfg: PaddleOCRConfig) -> str:
    t = (text or "").strip()
    if cfg.correct_confusable:
        t = "".join(_CONFUSABLES.get(c, c) for c in t)
    if cfg.lowercase and not cfg.uppercase:
        t = t.lower()
    elif cfg.uppercase and not cfg.lowercase:
        t = t.upper()
    return t

def _merge_overlaps(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Greedy merge of highly-overlapping boxes with same text (simple heuristic)."""
    if not blocks:
        return blocks
    def iou(b1, b2) -> float:
        # use axis-aligned bbox IoU on polygon min/max
        def rect(b):
            xs = [p[0] for p in b]; ys = [p[1] for p in b]
            return min(xs), min(ys), max(xs), max(ys)
        x1,y1,x2,y2 = rect(b1); X1,Y1,X2,Y2 = rect(b2)
        ix1, iy1 = max(x1, X1), max(y1, Y1)
        ix2, iy2 = min(x2, X2), min(y2, Y2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        a1 = (x2 - x1) * (y2 - y1); a2 = (X2 - X1) * (Y2 - Y1)
        union = a1 + a2 - inter + 1e-9
        return inter / union
    kept: List[Dict[str, Any]] = []
    used = [False]*len(blocks)
    for i, bi in enumerate(blocks):
        if used[i]:
            continue
        group = [i]
        for j in range(i+1, len(blocks)):
            if used[j]:
                continue
            if _normalize_text(blocks[j]["text"], cfg=None) == _normalize_text(bi["text"], cfg=None) and iou(blocks[i]["box"], blocks[j]["box"]) > 0.5:#type:ignore
                used[j] = True
                group.append(j)
        if len(group) == 1:
            kept.append(blocks[i]); continue
        # average confidence, keep first box
        confs = [blocks[g].get("confidence", None) for g in group if blocks[g].get("confidence") is not None]
        avg_conf = float(sum(confs) / len(confs)) if confs else None#type:ignore
        newb = dict(blocks[i])
        if avg_conf is not None:
            newb["confidence"] = avg_conf
        kept.append(newb)
    return kept


# =========================================================
# Loader
# =========================================================

class PaddleOCRLoader:
    def __init__(self, cfg_path: str = DEFAULT_CFG_PATH):
        self.cfg = load_config(cfg_path)
        _ensure_dir(self.cfg.results_dir)

        # Choose language (PaddleOCR expects a single 'lang' code; use first provided)
        lang = (self.cfg.languages[0] if self.cfg.languages else "en")

        # Precision
        use_fp16 = (self.cfg.precision.lower() == "fp16")

        # Build OCR engine
        self.ocr = PaddleOCR(
            use_angle_cls=self.cfg.use_angle_cls,
            lang=lang,
            det_model_dir=self.cfg.det_model_dir or None,
            rec_model_dir=self.cfg.rec_model_dir or None,
            cls_model_dir=self.cfg.cls_model_dir or None,
            use_gpu=self.cfg.use_gpu,
            enable_mkldnn=self.cfg.enable_mkldnn,
            use_space_char=self.cfg.use_space_char,
            gpu_mem=self.cfg.gpu_mem,
            precision="fp16" if use_fp16 else "fp32",
            # Note: det/rec algorithms are selected by chosen model dirs; explicit algorithm flags are limited
            # show_log = False  # could wire from cfg.log_level
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

    # ---------- PDF handling (optional) ----------
    @staticmethod
    def _rasterize_pdf(pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
        if not _HAS_PDF:
            raise RuntimeError("PDF support requires pdf2image + pillow. pip install pdf2image pillow")
        pages = convert_from_path(pdf_path, dpi=dpi)
        imgs = [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pages]
        return imgs

    # ---------- Core ----------
    def _infer(self, img: np.ndarray):
        """
        PaddleOCR.__call__(img) returns:
          [ [ [x1,y1],[x2,y2],[x3,y3],[x4,y4] ], (text, conf) ], ... ]
        """
        return self.ocr.ocr(img, cls=self.cfg.use_angle_cls, det=True, rec=True)

    def _post_blocks(self, raw, cfg: PaddleOCRConfig) -> Tuple[str, List[Dict[str, Any]]]:
        blocks = []
        for line in raw:
            try:
                box = line[0]  # 4 points
                txt, conf = line[1]
            except Exception:
                # Some versions return nested pages: [[line1,...], [line2,...]]
                if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
                    return self._post_blocks(raw[0], cfg)
                continue
            text = _normalize_text(str(txt), cfg)
            if len(text) < cfg.min_length:
                continue
            blk = {"text": text}
            if cfg.include_boxes and box is not None:
                blk["box"] = [[int(x), int(y)] for x, y in box]#type:ignore
            if cfg.include_confidence and conf is not None:
                try:
                    blk["confidence"] = float(conf)#type:ignore
                except Exception:
                    pass
            blocks.append(blk)

        if cfg.merge_boxes:
            blocks = _merge_overlaps(blocks)

        combined = "\n".join([b["text"] for b in blocks]).strip()
        return combined, blocks

    def transcribe_image(self, image_path: str) -> Dict[str, Any]:
        t0 = time.time()
        img = self._imread(image_path)
        h, w = img.shape[:2]
        proc = preprocess_image(img, self.cfg)

        # run OCR
        raw = self._infer(proc)
        # Some builds return list per page; standardize to first page
        if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list) and len(raw[0]) and isinstance(raw[0][0], list):
            raw = raw[0]

        text, blocks = self._post_blocks(raw, self.cfg)

        payload: Dict[str, Any] = {
            "image_path": image_path,
            "engine": "paddleocr",
            "languages": self.cfg.languages,
            "dims": [h, w],
            "text": text,
            "blocks": blocks,
            "latency_s": round(time.time() - t0, 3),
        }
        return payload

    def transcribe_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Rasterize each page and OCR. Returns a list of per-page payloads.
        Output filenames append _pNNN to page stems.
        """
        imgs = self._rasterize_pdf(pdf_path)
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        out_payloads = []
        for pi, img in enumerate(imgs):
            t0 = time.time()
            proc = preprocess_image(img, self.cfg)
            raw = self._infer(proc)
            # normalize page-wise structure
            if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
                raw = raw[0]
            text, blocks = self._post_blocks(raw, self.cfg)
            h, w = img.shape[:2]
            payload = {
                "image_path": f"{pdf_path}#page={pi+1}",
                "engine": "paddleocr",
                "languages": self.cfg.languages,
                "dims": [h, w],
                "text": text,
                "blocks": blocks,
                "latency_s": round(time.time() - t0, 3),
            }
            out_payloads.append(payload)
        return out_payloads

    def save_result(self, result: Dict[str, Any]) -> str:
        out_path = self._to_output_path(result["image_path"].split("#page=")[0])
        # If page-tagged (pdf), append suffix
        if "#page=" in result["image_path"]:
            stem, page = out_path.rsplit(".", 1)
            page_no = result["image_path"].split("#page=")[1]
            out_path = f"{stem}_p{int(page_no):03d}.{page}"

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
                if p.lower().endswith(".pdf"):
                    payloads = self.transcribe_pdf(p)
                    for payload in payloads:
                        if save:
                            self.save_result(payload)
                        outs.append(payload)
                    print(f"✓ {p}  →  {len(payloads)} page(s)")
                else:
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
                 if os.path.isfile(p) and os.path.splitext(p.lower())[1] in {
                     ".png",".jpg",".jpeg",".tif",".tiff",".bmp",".gif",".webp",".pdf"
                 }]
        return self.process_batch(paths, save=save)


# =========================================================
# CLI
# =========================================================

def main():
    ap = argparse.ArgumentParser(description="PaddleOCR loader")
    ap.add_argument("--config", default=DEFAULT_CFG_PATH, help="Path to kb/ocr/configs/paddleocr.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("file", help="OCR a single image/PDF")
    s1.add_argument("path", help="Path to image or PDF")
    s1.add_argument("--no-save", action="store_true")

    s2 = sub.add_parser("dir", help="OCR a directory (recursive)")
    s2.add_argument("image_dir", help="Directory path")
    s2.add_argument("--glob", default="**/*.*", help='Glob pattern (default="**/*.*")')
    s2.add_argument("--no-save", action="store_true")

    args = ap.parse_args()

    loader = PaddleOCRLoader(args.config)

    if args.cmd == "file":
        p = args.path
        if p.lower().endswith(".pdf"):
            payloads = loader.transcribe_pdf(p)
            for payload in payloads:
                if not args.no_save:
                    out = loader.save_result(payload)
                    print(f"✅ wrote {out}")
            if args.no_save:
                print(json.dumps(payloads, ensure_ascii=False, indent=2))
        else:
            result = loader.transcribe_image(p)
            if not args.no_save:
                out = loader.save_result(result)
                print(f"✅ wrote {out}")
            else:
                print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.cmd == "dir":
        rs = loader.process_dir(args.image_dir, args.glob, save=not args.no_save)
        print(f"✅ processed {len(rs)} file/page payload(s)")
        return


if __name__ == "__main__":
    main()