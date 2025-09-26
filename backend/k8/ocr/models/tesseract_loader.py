#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kb/ocr/tesseract_loader.py
--------------------------
Loader + thin wrapper around Tesseract (via pytesseract).

- Reads kb/ocr/configs/tesseract.yaml
- Preprocess: grayscale, denoise, binarize, deskew, resize, pad (OpenCV)
- Inference: pytesseract (TSV for boxes/conf), optional HOCR/PDF export
- Output: JSONL/TXT with text + boxes + confidences
- CLI: file / dir (recursive); PDF rasterization (pdf2image) optional

Install:
  sudo apt-get install tesseract-ocr
  pip install pytesseract opencv-python-headless pyyaml numpy
  # Optional for PDFs:
  pip install pdf2image pillow

Note:
  Ensure Tesseract's tessdata is reachable. If needed, set TESSDATA_PREFIX
  or set `ocr.tessdata_dir` in tesseract.yaml.
"""

from __future__ import annotations
import os
import sys
import io
import re
import json
import time
import glob
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml
import numpy as np

# OpenCV
try:
    import cv2  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: opencv-python-headless (or opencv-python).") from e

# pytesseract (wrapper for the tesseract binary)
try:
    import pytesseract  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: pytesseract. pip install pytesseract") from e

# Optional PDF support
_HAS_PDF = True
try:
    from pdf2image import convert_from_path  # type: ignore
    from PIL import Image  # noqa: F401
except Exception:
    _HAS_PDF = False


DEFAULT_CFG_PATH = os.path.join("kb", "ocr", "configs", "tesseract.yaml")


# =========================
# Config
# =========================

@dataclass
class TesseractConfig:
    # ocr
    languages: List[str]
    psm: int
    oem: int
    tessdata_dir: Optional[str]

    # preprocess
    grayscale: bool
    denoise: bool
    binarize: bool
    deskew: bool
    resize_factor: float
    pad: int

    # inference
    dpi: int
    config_extra: str
    timeout: int
    preserve_interword_spaces: int

    # postprocess
    spellcheck: bool
    correct_confusable: bool
    strip_punctuation: bool
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
    export_format: str  # plain|tsv|hocr|pdf


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str = DEFAULT_CFG_PATH) -> TesseractConfig:
    cfg = _read_yaml(path)
    ocr = cfg.get("ocr", {})
    pre = cfg.get("preprocess", {})
    inf = cfg.get("inference", {})
    post = cfg.get("postprocess", {})
    run = cfg.get("runtime", {})
    out = cfg.get("output", {})

    return TesseractConfig(
        languages=list(ocr.get("languages", ["eng"])),
        psm=int(ocr.get("psm", 3)),
        oem=int(ocr.get("oem", 3)),
        tessdata_dir=ocr.get("tessdata_dir"),

        grayscale=bool(pre.get("grayscale", True)),
        denoise=bool(pre.get("denoise", True)),
        binarize=bool(pre.get("binarize", True)),
        deskew=bool(pre.get("deskew", True)),
        resize_factor=float(pre.get("resize_factor", 2.0)),
        pad=int(pre.get("pad", 5)),

        dpi=int(inf.get("dpi", 300)),
        config_extra=str(inf.get("config_extra", "")),
        timeout=int(inf.get("timeout", 30)),
        preserve_interword_spaces=int(inf.get("preserve_interword_spaces", 1)),

        spellcheck=bool(post.get("spellcheck", False)),
        correct_confusable=bool(post.get("correct_confusable", True)),
        strip_punctuation=bool(post.get("strip_punctuation", False)),
        lowercase=bool(post.get("lowercase", True)),
        uppercase=bool(post.get("uppercase", False)),

        device=str(run.get("device", "cpu")),
        num_threads=int(run.get("num_threads", 4)),
        log_level=str(run.get("log_level", "INFO")),

        results_dir=str(out.get("results_dir", os.path.join("outputs", "ocr", "tesseract"))),
        save_format=str(out.get("save_format", "jsonl")),
        include_boxes=bool(out.get("include_boxes", True)),
        include_confidence=bool(out.get("include_confidence", True)),
        export_format=str(out.get("export_format", "tsv")),
    )


# =========================
# Preprocess
# =========================

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _imread(path: str) -> np.ndarray:
    # robust path reading (supports non-ascii)
    data = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return data

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

def _resize(img: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    nh, nw = max(1, int(round(h * factor))), max(1, int(round(w * factor)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)

def _pad(img: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return img
    if img.ndim == 2:
        return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))

def preprocess_image(img: np.ndarray, cfg: TesseractConfig) -> np.ndarray:
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


# =========================
# Postprocess (text cleanup)
# =========================

_CONFUSABLES = {"0":"O","1":"l","5":"S","8":"B","o":"0","l":"1","|":"I"}

def _clean_text(text: str, cfg: TesseractConfig) -> str:
    t = (text or "").strip()
    if cfg.correct_confusable:
        t = "".join(_CONFUSABLES.get(c, c) for c in t)
    if cfg.strip_punctuation:
        t = re.sub(r"[^\w\s\.\,\-\:/@%€$£₹()]+", "", t)
    if cfg.lowercase and not cfg.uppercase:
        t = t.lower()
    elif cfg.uppercase and not cfg.lowercase:
        t = t.upper()
    return t


# =========================
# Core OCR
# =========================

def _lang_str(langs: List[str]) -> str:
    # tesseract uses plus-separated codes: "eng+deu"
    return "+".join(langs) if langs else "eng"

def _build_config_str(cfg: TesseractConfig) -> str:
    parts = [
        f"--oem {cfg.oem}",
        f"--psm {cfg.psm}",
        f"-c preserve_interword_spaces={cfg.preserve_interword_spaces}",
    ]
    if cfg.config_extra:
        parts.append(str(cfg.config_extra))
    return " ".join(parts)

def _set_env_tessdata(cfg: TesseractConfig):
    if cfg.tessdata_dir:
        os.environ["TESSDATA_PREFIX"] = cfg.tessdata_dir

def _image_to_tsv(img: np.ndarray, lang: str, config: str, dpi: int, timeout: int) -> str:
    return pytesseract.image_to_data(
        img,
        lang=lang,
        config=f"{config} -c tessedit_create_tsv=1",
        output_type=pytesseract.Output.STRING,
        timeout=timeout,
        dpi=dpi,
    )

def _image_to_text(img: np.ndarray, lang: str, config: str, dpi: int, timeout: int) -> str:
    return pytesseract.image_to_string(
        img,
        lang=lang,
        config=config,
        timeout=timeout,
        dpi=dpi,
    )

def _image_to_hocr(img: np.ndarray, lang: str, config: str, dpi: int, timeout: int) -> bytes:
    return pytesseract.image_to_pdf_or_hocr(
        img,
        extension="hocr",
        lang=lang,
        config=config,
        timeout=timeout,
        dpi=dpi,
    )

def _image_to_pdf(img: np.ndarray, lang: str, config: str, dpi: int, timeout: int) -> bytes:
    return pytesseract.image_to_pdf_or_hocr(
        img,
        extension="pdf",
        lang=lang,
        config=config,
        timeout=timeout,
        dpi=dpi,
    )


def _parse_tsv(tsv: str, cfg: TesseractConfig) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Parse pytesseract TSV to (combined_text, blocks).
    Each token (level==5) yields a box; combine tokens into lines.
    """
    lines = tsv.strip().splitlines()
    if not lines:
        return "", []
    header = lines[0].split("\t")
    idx = {h: i for i, h in enumerate(header)}
    blocks: List[Dict[str, Any]] = []
    current_line_id = None
    line_text_parts: List[str] = []
    line_box = None
    line_conf_acc: List[float] = []

    def flush_line():
        nonlocal blocks, line_text_parts, line_box, line_conf_acc
        if line_text_parts:
            text = _clean_text(" ".join(line_text_parts), cfg)
            if text:
                blk = {"text": text}
                if cfg.include_boxes and line_box is not None:
                    blk["box"] = line_box#type:ignore
                if cfg.include_confidence and line_conf_acc:
                    blk["confidence"] = float(sum(line_conf_acc) / len(line_conf_acc))#type:ignore
                blocks.append(blk)
        line_text_parts, line_box, line_conf_acc = [], None, []

    for row in lines[1:]:
        cols = row.split("\t")
        if len(cols) != len(header):
            continue
        try:
            level = int(cols[idx["level"]])
            conf = float(cols[idx["conf"]]) if cols[idx["conf"]] not in ("", "-1") else None
            text = cols[idx["text"]].strip()
            left, top, width, height = [int(cols[idx[k]]) for k in ("left", "top", "width", "height")]
            line_num = (cols[idx.get("page_num", 0)], cols[idx.get("block_num", 0)], cols[idx.get("par_num", 0)], cols[idx.get("line_num", 0)])
        except Exception:
            continue

        if level == 5:  # word level
            # New line?
            if current_line_id != line_num:
                flush_line()
                current_line_id = line_num
                line_box = [[left, top], [left + width, top], [left + width, top + height], [left, top + height]]

            if text:
                line_text_parts.append(text)
                if conf is not None:
                    line_conf_acc.append(conf)

    # flush tail
    flush_line()

    combined_text = "\n".join([b["text"] for b in blocks]).strip()
    return combined_text, blocks


class TesseractLoader:
    def __init__(self, cfg_path: str = DEFAULT_CFG_PATH):
        self.cfg = load_config(cfg_path)
        _ensure_dir(self.cfg.results_dir)
        _set_env_tessdata(self.cfg)
        # threads: pytesseract respects OMP_NUM_THREADS for some ops
        os.environ.setdefault("OMP_NUM_THREADS", str(max(1, self.cfg.num_threads)))

    # ---------- I/O ----------
    def _to_output_path(self, image_path: str) -> str:
        base = os.path.splitext(os.path.basename(image_path))[0]
        ext = "jsonl" if self.cfg.save_format.lower() == "jsonl" else "txt"
        return os.path.join(self.cfg.results_dir, f"{base}.{ext}")

    @staticmethod
    def _rasterize_pdf(pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
        if not _HAS_PDF:
            raise RuntimeError("PDF support requires pdf2image + pillow. pip install pdf2image pillow")
        pages = convert_from_path(pdf_path, dpi=dpi)
        return [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pages]

    # ---------- Core ----------
    def transcribe_image(self, image_path: str) -> Dict[str, Any]:
        t0 = time.time()
        img = _imread(image_path)
        h, w = img.shape[:2]
        proc = preprocess_image(img, self.cfg)

        lang = _lang_str(self.cfg.languages)
        cfg_str = _build_config_str(self.cfg)

        # Prefer TSV when boxes/conf requested; fallback to plain text otherwise
        if self.cfg.include_boxes or self.cfg.include_confidence:
            tsv = _image_to_tsv(proc, lang, cfg_str, self.cfg.dpi, self.cfg.timeout)
            text, blocks = _parse_tsv(tsv, self.cfg)
        else:
            text = _clean_text(_image_to_text(proc, lang, cfg_str, self.cfg.dpi, self.cfg.timeout), self.cfg)
            blocks = []

        payload: Dict[str, Any] = {
            "image_path": image_path,
            "engine": "tesseract",
            "languages": self.cfg.languages,
            "dims": [h, w],
            "text": text,
            "blocks": blocks,
            "latency_s": round(time.time() - t0, 3),
        }

        # Optional export (tsv/hocr/pdf) next to JSON/TXT
        export = self.cfg.export_format.lower()
        try:
            stem = os.path.join(self.cfg.results_dir, os.path.splitext(os.path.basename(image_path))[0])
            if export == "tsv":
                # reuse TSV if we already computed it
                if not (self.cfg.include_boxes or self.cfg.include_confidence):
                    tsv = _image_to_tsv(proc, lang, cfg_str, self.cfg.dpi, self.cfg.timeout)
                with open(stem + ".tsv", "w", encoding="utf-8") as f:
                    f.write(tsv)
            elif export == "hocr":
                hocr = _image_to_hocr(proc, lang, cfg_str, self.cfg.dpi, self.cfg.timeout)
                with open(stem + ".hocr", "wb") as f:
                    f.write(hocr)
            elif export == "pdf":
                pdf = _image_to_pdf(proc, lang, cfg_str, self.cfg.dpi, self.cfg.timeout)
                with open(stem + ".pdf", "wb") as f:
                    f.write(pdf)
        except Exception as e:
            print(f"⚠️ export ({export}) failed: {e}", file=sys.stderr)

        return payload

    def transcribe_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        imgs = self._rasterize_pdf(pdf_path, dpi=self.cfg.dpi or 300)
        out = []
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        for i, img in enumerate(imgs, start=1):
            t0 = time.time()
            proc = preprocess_image(img, self.cfg)
            lang = _lang_str(self.cfg.languages)
            cfg_str = _build_config_str(self.cfg)

            if self.cfg.include_boxes or self.cfg.include_confidence:
                tsv = _image_to_tsv(proc, lang, cfg_str, self.cfg.dpi, self.cfg.timeout)
                text, blocks = _parse_tsv(tsv, self.cfg)
            else:
                text = _clean_text(_image_to_text(proc, lang, cfg_str, self.cfg.dpi, self.cfg.timeout), self.cfg)
                blocks = []

            h, w = img.shape[:2]
            payload = {
                "image_path": f"{pdf_path}#page={i}",
                "engine": "tesseract",
                "languages": self.cfg.languages,
                "dims": [h, w],
                "text": text,
                "blocks": blocks,
                "latency_s": round(time.time() - t0, 3),
            }
            out.append(payload)

            # Optional export per page
            try:
                stem = os.path.join(self.cfg.results_dir, f"{base}_p{i:03d}")
                export = self.cfg.export_format.lower()
                if export == "tsv":
                    with open(stem + ".tsv", "w", encoding="utf-8") as f:
                        f.write(tsv)
                elif export == "hocr":
                    hocr = _image_to_hocr(proc, lang, cfg_str, self.cfg.dpi, self.cfg.timeout)
                    with open(stem + ".hocr", "wb") as f:
                        f.write(hocr)
                elif export == "pdf":
                    pdf_bytes = _image_to_pdf(proc, lang, cfg_str, self.cfg.dpi, self.cfg.timeout)
                    with open(stem + ".pdf", "wb") as f:
                        f.write(pdf_bytes)
            except Exception as e:
                print(f"⚠️ export failed (page {i}): {e}", file=sys.stderr)

        return out

    def save_result(self, result: Dict[str, Any]) -> str:
        out_path = self._to_output_path(result["image_path"].split("#page=")[0])
        if "#page=" in result["image_path"]:
            stem, ext = os.path.splitext(out_path)
            page_no = int(result["image_path"].split("#page=")[1])
            out_path = f"{stem}_p{page_no:03d}{ext}"

        if self.cfg.save_format.lower() == "jsonl":
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(result.get("text", ""))
        return out_path

    # ---------- Batch ----------
    def process_batch(self, paths: List[str], save: bool = True) -> List[Dict[str, Any]]:
        outs = []
        for p in paths:
            try:
                if p.lower().endswith(".pdf"):
                    payloads = self.transcribe_pdf(p)
                    for payload in payloads:
                        if save:
                            self.save_result(payload)
                        outs.append(payload)
                    print(f"✓ {p} → {len(payloads)} page(s)")
                else:
                    r = self.transcribe_image(p)
                    if save:
                        self.save_result(r)
                    outs.append(r)
                    print(f"✓ {p} → {len(r.get('text',''))} chars, {len(r.get('blocks',[]))} lines")
            except Exception as e:
                print(f"✗ {p}: {e}", file=sys.stderr)
        return outs

    def process_dir(self, image_dir: str, pattern: str = "**/*.*", save: bool = True) -> List[Dict[str, Any]]:
        paths = [p for p in glob.glob(os.path.join(image_dir, pattern), recursive=True)
                 if os.path.isfile(p) and os.path.splitext(p.lower())[1] in {
                     ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp", ".pdf"
                 }]
        return self.process_batch(paths, save=save)


# =========================
# CLI
# =========================

def gather_files(target: str, pattern: Optional[str]) -> List[str]:
    if os.path.isfile(target):
        return [target]
    if os.path.isdir(target):
        pat = pattern or "**/*.*"
        exts = {".png",".jpg",".jpeg",".tif",".tiff",".bmp",".gif",".webp",".pdf"}
        return [p for p in glob.glob(os.path.join(target, pat), recursive=True)
                if os.path.isfile(p) and os.path.splitext(p.lower())[1] in exts]
    raise FileNotFoundError(target)

def main():
    ap = argparse.ArgumentParser(description="Tesseract OCR loader")
    ap.add_argument("--config", default=DEFAULT_CFG_PATH, help="Path to kb/ocr/configs/tesseract.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("file", help="OCR a single image/PDF")
    s1.add_argument("path", help="Path to image or PDF")
    s1.add_argument("--no-save", action="store_true")

    s2 = sub.add_parser("dir", help="OCR a directory (recursive)")
    s2.add_argument("image_dir", help="Directory path")
    s2.add_argument("--glob", default="**/*.*", help='Glob pattern (default="**/*.*")')
    s2.add_argument("--no-save", action="store_true")

    args = ap.parse_args()

    loader = TesseractLoader(args.config)

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
            r = loader.transcribe_image(p)
            if not args.no_save:
                out = loader.save_result(r)
                print(f"✅ wrote {out}")
            else:
                print(json.dumps(r, ensure_ascii=False, indent=2))
        return

    if args.cmd == "dir":
        rs = loader.process_dir(args.image_dir, args.glob, save=not args.no_save)
        print(f"✅ processed {len(rs)} file/page payload(s)")
        return


if __name__ == "__main__":
    main()