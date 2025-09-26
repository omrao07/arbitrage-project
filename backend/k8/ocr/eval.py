#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ocr/eval.py
-----------
Evaluate OCR hypotheses vs reference transcripts.

Supports:
  - References: .txt or .jsonl  (expects {"text": "..."}; optional {"blocks":[{"text":..., "box":[[x,y]x4]}]})
  - Hypotheses: .txt or .jsonl  (same schema as above; your OCR loaders write this)
  - Global text metrics: WER, CER, MER, WIL (jiwer) + RapidFuzz token/char F1
  - Box-level metrics (optional): IoU-based matching to compute det P/R/F1 and text scores on matched pairs
  - Per-file table + overall summary; optional CSV/JSON exports

Usage:
  # Basic (text-only scoring)
  python ocr/eval.py --refs data/ocr/refs --hyps outputs/ocr/easyocr --ref-ext txt --hyp-ext jsonl --out-csv ocr_eval.csv

  # Include box-level metrics when both refs/hyps JSONL have blocks with boxes
  python ocr/eval.py --refs refs/jsonl --hyps outputs/ocr/paddleocr --ref-ext jsonl --hyp-ext jsonl --with-boxes \
                     --iou 0.5 --text-thresh 0.7 --out-csv ocr_eval_boxes.csv

Dependencies:
  pip install jiwer pandas numpy rapidfuzz tabulate
"""

from __future__ import annotations
import os
import re
import json
import glob
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Pretty printing (optional)
try:
    from tabulate import tabulate  # type: ignore
    _HAS_TAB = True
except Exception:
    _HAS_TAB = False

# Text metrics
try:
    import jiwer  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: jiwer. Install with `pip install jiwer`.") from e

try:
    from rapidfuzz import fuzz  # type: ignore
    _HAS_RF = True
except Exception:
    _HAS_RF = False


# =========================
# I/O helpers
# =========================

def _read_txt(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    return {"text": text, "blocks": []}

def _read_jsonl(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s:
                obj = json.loads(s)
                break
        else:
            obj = {}
    return {
        "text": str(obj.get("text") or ""),
        "blocks": obj.get("blocks") or [],
        "dims": obj.get("dims"),
    }

def _load_item(path: str, ext_hint: Optional[str]) -> Dict[str, Any]:
    ext = (ext_hint or os.path.splitext(path)[1].lstrip(".")).lower()
    return _read_txt(path) if ext == "txt" else _read_jsonl(path)

def _gather(dirpath: str, ext: str) -> Dict[str, str]:
    patt = os.path.join(dirpath, f"**/*.{ext}")
    files = [p for p in glob.glob(patt, recursive=True) if os.path.isfile(p)]
    return {os.path.splitext(os.path.basename(p))[0]: p for p in files}


# =========================
# Normalization
# =========================

def _norm_words() -> jiwer.Compose:
    return jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ReduceToListOfListOfWords()])

def _norm_chars() -> jiwer.Compose:
    return jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])

def words(text: str) -> List[str]:
    arr = _norm_words()(text or "")
    return arr[0] if arr else []

def chars(text: str) -> str:
    return _norm_chars()(text or "")


# =========================
# Text metrics
# =========================

@dataclass
class TextScores:
    wer: float
    cer: float
    mer: float
    wil: float
    n_words_ref: int
    n_words_hyp: int
    n_chars_ref: int
    n_chars_hyp: int
    token_f1: Optional[float]
    char_f1: Optional[float]

def _f1_from_sim(sim: float) -> float:
    # Similarity 0..100 -> convert to 0..1 “F1-like” score (simple normalization)
    return max(0.0, min(1.0, sim / 100.0))

def score_text(ref_text: str, hyp_text: str) -> TextScores:
    wer = jiwer.wer(ref_text, hyp_text, truth_transform=_norm_words(), hypothesis_transform=_norm_words())
    mer = jiwer.mer(ref_text, hyp_text, truth_transform=_norm_words(), hypothesis_transform=_norm_words())
    wil = jiwer.wil(ref_text, hyp_text, truth_transform=_norm_words(), hypothesis_transform=_norm_words())

    ref_c = chars(ref_text); hyp_c = chars(hyp_text)
    cer = jiwer.cer(ref_c, hyp_c, truth_transform=jiwer.IdentityTransformation(), hypothesis_transform=jiwer.IdentityTransformation())

    nwr, nwh = len(words(ref_text)), len(words(hyp_text))
    ncr, nch = len(ref_c), len(hyp_c)

    if _HAS_RF:
        # Token-level: partial_ratio over whitespace-joined tokens approximates token F1
        token_sim = fuzz.token_set_ratio(ref_text, hyp_text)
        char_sim = fuzz.partial_ratio(ref_text, hyp_text)
        token_f1 = _f1_from_sim(token_sim)
        char_f1 = _f1_from_sim(char_sim)
    else:
        token_f1 = None
        char_f1 = None

    return TextScores(wer=float(wer), cer=float(cer), mer=float(mer), wil=float(wil),
                      n_words_ref=nwr, n_words_hyp=nwh, n_chars_ref=ncr, n_chars_hyp=nch,
                      token_f1=token_f1, char_f1=char_f1)


# =========================
# Box utilities
# =========================

def _rect_from_poly(poly: List[List[int]]) -> Tuple[int,int,int,int]:
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)

def _iou(b1: List[List[int]], b2: List[List[int]]) -> float:
    x1,y1,x2,y2 = _rect_from_poly(b1); X1,Y1,X2,Y2 = _rect_from_poly(b2)
    ix1, iy1 = max(x1, X1), max(y1, Y1)
    ix2, iy2 = min(x2, X2), min(y2, Y2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = (x2 - x1) * (y2 - y1); a2 = (X2 - X1) * (Y2 - Y1)
    return inter / float(a1 + a2 - inter + 1e-9)

@dataclass
class BoxMatchParams:
    iou_thresh: float = 0.5
    text_thresh: float = 0.7  # 0..1 after normalization (uses RapidFuzz token_set_ratio/100)

def _text_sim(a: str, b: str) -> float:
    if not _HAS_RF:
        # fallback: 1.0 if identical after normalization; else 0.0
        return 1.0 if chars(a) == chars(b) else 0.0
    return fuzz.token_set_ratio(a or "", b or "") / 100.0

def _extract_blocks(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    blocks = []
    for b in obj.get("blocks") or []:
        t = (b.get("text") or "").strip()
        box = b.get("box")
        if t and isinstance(box, (list, tuple)) and len(box) >= 4:
            # coerce to [[x,y],...]
            try:
                poly = [[int(p[0]), int(p[1])] for p in box][:4]
                blocks.append({"text": t, "box": poly})
            except Exception:
                continue
    return blocks

@dataclass
class BoxScores:
    det_precision: Optional[float]
    det_recall: Optional[float]
    det_f1: Optional[float]
    matched: int
    n_ref: int
    n_hyp: int
    text_acc_on_matches: Optional[float]  # fraction of matched pairs whose text_sim >= text_thresh

def match_boxes(ref_obj: Dict[str, Any], hyp_obj: Dict[str, Any], params: BoxMatchParams) -> Tuple[BoxScores, List[Tuple[int,int,float,float]]]:
    """
    Greedy bipartite matching by IoU, breaking ties with text similarity.
    Returns BoxScores and matches as list of (idx_ref, idx_hyp, iou, text_sim).
    """
    refs = _extract_blocks(ref_obj)
    hyps = _extract_blocks(hyp_obj)
    n_ref, n_hyp = len(refs), len(hyps)
    if n_ref == 0 and n_hyp == 0:
        return BoxScores(None, None, None, 0, 0, 0, None), []

    # build all candidate pairs above IoU threshold
    cand = []
    for i, rb in enumerate(refs):
        for j, hb in enumerate(hyps):
            iou = _iou(rb["box"], hb["box"])
            if iou >= params.iou_thresh:
                ts = _text_sim(rb["text"], hb["text"])
                cand.append((i, j, iou, ts))

    # sort by IoU then text similarity
    cand.sort(key=lambda x: (x[2], x[3]), reverse=True)

    used_r = set(); used_h = set(); matches = []
    for i, j, iou, ts in cand:
        if i in used_r or j in used_h:
            continue
        used_r.add(i); used_h.add(j)
        matches.append((i, j, iou, ts))

    tp = len(matches)
    fp = n_hyp - tp
    fn = n_ref - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2*prec*rec / (prec+rec)) if (prec+rec) > 0 else 0.0

    # text accuracy on matched pairs
    good_text = sum(1 for (_,_,_,ts) in matches if ts >= params.text_thresh)
    text_acc = (good_text / tp) if tp > 0 else None

    scores = BoxScores(det_precision=prec, det_recall=rec, det_f1=f1, matched=tp, n_ref=n_ref, n_hyp=n_hyp, text_acc_on_matches=text_acc)
    return scores, matches


# =========================
# Main evaluation
# =========================

def evaluate(ref_dir: str, hyp_dir: str, ref_ext: str, hyp_ext: str,
             with_boxes: bool, iou_thresh: float, text_thresh: float) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    refs = _gather(ref_dir, ref_ext)
    hyps = _gather(hyp_dir, hyp_ext)

    common = sorted(set(refs.keys()) & set(hyps.keys()))
    rows = []
    match_rows = []

    for base in common:
        ref_obj = _load_item(refs[base], ref_ext)
        hyp_obj = _load_item(hyps[base], hyp_ext)

        # Text scores
        ts = score_text(ref_obj["text"], hyp_obj["text"])

        row = {
            "file": base,
            "wer": ts.wer, "cer": ts.cer, "mer": ts.mer, "wil": ts.wil,
            "n_words_ref": ts.n_words_ref, "n_words_hyp": ts.n_words_hyp,
            "n_chars_ref": ts.n_chars_ref, "n_chars_hyp": ts.n_chars_hyp,
            "token_f1": ts.token_f1, "char_f1": ts.char_f1,
        }

        # Box scores (optional)
        if with_boxes:
            bs, matches = match_boxes(ref_obj, hyp_obj, BoxMatchParams(iou_thresh=iou_thresh, text_thresh=text_thresh))
            row.update({
                "det_precision": bs.det_precision,
                "det_recall": bs.det_recall,
                "det_f1": bs.det_f1,
                "boxes_ref": bs.n_ref,
                "boxes_hyp": bs.n_hyp,
                "boxes_matched": bs.matched,
                "text_acc_on_matches": bs.text_acc_on_matches,
            })
            for (ri, hj, iou, tsim) in matches:
                match_rows.append({
                    "file": base,
                    "ref_idx": ri,
                    "hyp_idx": hj,
                    "iou": iou,
                    "text_sim": tsim
                })

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("file")
    match_df = pd.DataFrame(match_rows).sort_values(["file","ref_idx","hyp_idx"]) if with_boxes and match_rows else None
    return df, match_df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    # Micro averages (weighted by ref length)
    w_words = df["n_words_ref"].replace(0, np.nan)
    w_chars = df["n_chars_ref"].replace(0, np.nan)

    def wavg(col, w):
        x = df[col].astype(float)
        return float(np.nansum(x * w) / np.nansum(w)) if np.nansum(w) > 0 else np.nan

    out = {
        "files": len(df),
        "WER_micro": wavg("wer", w_words),
        "CER_micro": wavg("cer", w_chars),
        "MER_micro": wavg("mer", w_words),
        "WIL_micro": wavg("wil", w_words),
        "WER_macro": float(df["wer"].mean()),
        "CER_macro": float(df["cer"].mean()),
        "MER_macro": float(df["mer"].mean()),
        "WIL_macro": float(df["wil"].mean()),
    }
    if "det_f1" in df.columns:
        out.update({
            "DetF1_macro": float(df["det_f1"].mean()),
            "DetP_macro": float(df["det_precision"].mean()),
            "DetR_macro": float(df["det_recall"].mean()),
            "TextAcc@matches_macro": float(df["text_acc_on_matches"].mean()),
        })
    if "token_f1" in df.columns and df["token_f1"].notna().any():
        out["TokenF1_macro"] = float(df["token_f1"].mean())
    if "char_f1" in df.columns and df["char_f1"].notna().any():
        out["CharF1_macro"] = float(df["char_f1"].mean())

    return pd.DataFrame([out])


# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser(description="Evaluate OCR outputs vs references")
    ap.add_argument("--refs", required=True, help="Directory of reference transcripts (txt or jsonl)")
    ap.add_argument("--hyps", required=True, help="Directory of hypothesis outputs (txt or jsonl)")
    ap.add_argument("--ref-ext", default="txt", choices=["txt","jsonl"], help="Reference extension")
    ap.add_argument("--hyp-ext", default="jsonl", choices=["txt","jsonl"], help="Hypothesis extension")
    ap.add_argument("--with-boxes", action="store_true", help="Compute box-level metrics when both sides have blocks/boxes")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold for box matching (default 0.5)")
    ap.add_argument("--text-thresh", type=float, default=0.7, help="Text similarity threshold (0..1) to count matched text as correct")
    ap.add_argument("--out-csv", default=None, help="Write per-file metrics to CSV")
    ap.add_argument("--out-json", default=None, help="Write per-file metrics to JSON")
    ap.add_argument("--out-matches-csv", default=None, help="Write per-match table (when --with-boxes)")
    ap.add_argument("--out-matches-json", default=None, help="Write per-match JSON (when --with-boxes)")
    args = ap.parse_args()

    df, mdf = evaluate(args.refs, args.hyps, args.ref_ext, args.hyp_ext, args.with_boxes, args.iou, args.text_thresh)
    summ = summarize(df)

    if not summ.empty:
        print("\n=== Summary ===")
        if _HAS_TAB:
            print(tabulate(summ, headers="keys", floatfmt=".4f", tablefmt="github")) # type: ignore
        else:
            print(summ.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if not df.empty:
        print("\n=== Per-file (head) ===")
        if _HAS_TAB:
            print(tabulate(df.head(10), headers="keys", floatfmt=".4f", tablefmt="github", showindex=False)) # type: ignore
        else:
            print(df.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))

    if args.out_csv:
        df.to_csv(args.out_csv, index=False); print(f"✅ wrote {args.out_csv}")
    if args.out_json:
        df.to_json(args.out_json, orient="records", force_ascii=False, indent=2); print(f"✅ wrote {args.out_json}")

    if args.with_boxes and mdf is not None and not mdf.empty:
        print(f"\n=== Box matches (head, {len(mdf)} total) ===")
        if _HAS_TAB:
            print(tabulate(mdf.head(10), headers="keys", floatfmt=".4f", tablefmt="github", showindex=False)) # type: ignore
        else:
            print(mdf.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
        if args.out_matches_csv:
            mdf.to_csv(args.out_matches_csv, index=False); print(f"✅ wrote {args.out_matches_csv}") # type: ignore
        if args.out_matches_json:
            mdf.to_json(args.out_matches_json, orient="records", force_ascii=False, indent=2); print(f"✅ wrote {args.out_matches_json}") # type: ignore

if __name__ == "__main__":
    main()