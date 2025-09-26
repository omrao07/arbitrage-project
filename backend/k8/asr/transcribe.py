#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
asr/transcribe.py
-----------------
Unified entrypoint for ASR transcription across engines:
  - Whisper (openai-whisper or HF fallback) via kb/asr/whisper_loader.py
  - Wav2Vec 2.0 (HF) via kb/asr/wav2vec_loader.py
  - DeepSpeech (optional) via kb/asr/deepspeech_loader.py (if you add it)

You can:
  - Transcribe a single file
  - Transcribe a directory (glob)
  - Transcribe a list from a CSV/JSONL manifest (col 'audio' or 'path')

Examples:
  python asr/transcribe.py --config kb/asr/configs/whisper.yaml file data/clean/call_001.wav
  python asr/transcribe.py --config kb/asr/configs/wav2vec.yaml dir data/clean/ --glob "**/*.wav"
  python asr/transcribe.py --config kb/asr/configs/whisper.yaml manifest out/clean_manifest.csv

Notes:
  - Output path/format is controlled by each engine's YAML (transcripts_dir, save_format)
  - This script just dispatches; engine-specific options live in their YAML/loader
"""

from __future__ import annotations
import os
import sys
import json
import glob
import argparse
from typing import Any, Dict, List, Optional

import yaml

# Repo-local imports (engine loaders)
# Ensure this script sits under kb/asr/ and you run from repo root or have PYTHONPATH set.
from kb.asr.whisper_loader import WhisperLoader # type: ignore
from kb.asr.wav2vec_loader import Wav2VecLoader # type: ignore

# Optional: if/when you add it
try:
    from kb.asr.deepspeech_loader import DeepSpeechLoader  # you can implement similar to others #type:ignore
    _HAS_DS = True
except Exception:
    _HAS_DS = False


# ------------------------------------------------------------
# Config dispatch
# ------------------------------------------------------------

def read_engine_from_yaml(cfg_path: str) -> str:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    eng = ((cfg.get("asr") or {}).get("engine") or "").strip().lower()
    if not eng:
        raise ValueError(f"'engine' not found in YAML: {cfg_path}")
    return eng


def build_loader(engine: str, cfg_path: str):
    eng = engine.lower()
    if eng in ("whisper", "openai-whisper"):
        return WhisperLoader(cfg_path)
    if eng in ("wav2vec2", "wav2vec", "w2v2"):
        return Wav2VecLoader(cfg_path)
    if eng in ("deepspeech", "ds"):
        if not _HAS_DS:
            raise RuntimeError(
                "DeepSpeech loader not available. Create kb/asr/deepspeech_loader.py or install its deps."
            )
        return DeepSpeechLoader(cfg_path)
    raise ValueError(f"Unsupported engine: {engine}")


# ------------------------------------------------------------
# Inputs
# ------------------------------------------------------------

def gather_files(target: str, pattern: Optional[str]) -> List[str]:
    if os.path.isfile(target):
        return [target]
    if os.path.isdir(target):
        pat = pattern or "**/*"
        # support common audio extensions
        exts = (".wav", ".flac", ".mp3", ".m4a", ".ogg")
        files = [p for p in glob.glob(os.path.join(target, pat), recursive=True)
                 if os.path.splitext(p)[1].lower() in exts and os.path.isfile(p)]
        return files
    raise FileNotFoundError(target)


def read_manifest(path: str) -> List[str]:
    """
    Accepts CSV or JSONL. Expects a column/key named 'audio' or 'path'.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    out: List[str] = []
    if path.lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                obj = json.loads(s)
                p = obj.get("audio") or obj.get("path")
                if p:
                    out.append(p)
    else:
        # lightweight CSV reader
        import csv
        with open(path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                p = row.get("audio") or row.get("path")
                if p:
                    out.append(p)
    if not out:
        raise ValueError(f"No 'audio'/'path' entries found in manifest: {path}")
    return out


# ------------------------------------------------------------
# Run helpers
# ------------------------------------------------------------

def run_file(loader, audio: str, no_save: bool = False) -> Dict[str, Any]:
    tr = loader.transcribe_file(audio)
    if not no_save:
        out_path = loader.save_transcript(tr)
        print(f"✅ wrote {out_path}")
    else:
        print(json.dumps(tr, ensure_ascii=False, indent=2))
    return tr


def run_dir(loader, audio_dir: str, pattern: str, no_save: bool = False) -> List[Dict[str, Any]]:
    files = gather_files(audio_dir, pattern)
    print(f"Found {len(files)} file(s)")
    trs = loader.transcribe_batch(files, save=not no_save)
    print(f"✅ processed {len(trs)} file(s)")
    return trs


def run_manifest(loader, manifest_path: str, no_save: bool = False) -> List[Dict[str, Any]]:
    files = read_manifest(manifest_path)
    print(f"Manifest contains {len(files)} file(s)")
    trs = loader.transcribe_batch(files, save=not no_save)
    print(f"✅ processed {len(trs)} file(s)")
    return trs


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Unified ASR transcriber (Whisper/Wav2Vec2/DeepSpeech)")
    ap.add_argument("--config", required=True, help="Path to engine YAML (whisper.yaml / wav2vec.yaml / deepspeech.yaml)")
    ap.add_argument("--engine", default=None, help="Override engine in YAML (whisper|wav2vec2|deepspeech)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("file", help="Transcribe a single file")
    s1.add_argument("audio", help="Path to audio (wav/flac/mp3/m4a/ogg)")
    s1.add_argument("--no-save", action="store_true", help="Print JSON to stdout and skip file write")

    s2 = sub.add_parser("dir", help="Transcribe a directory (recursive)")
    s2.add_argument("audio_dir", help="Directory containing audio files")
    s2.add_argument("--glob", default="**/*", help='Glob pattern (default="**/*")')
    s2.add_argument("--no-save", action="store_true")

    s3 = sub.add_parser("manifest", help="Transcribe from a CSV/JSONL manifest")
    s3.add_argument("path", help="Manifest path (CSV or JSONL with column/key 'audio' or 'path')")
    s3.add_argument("--no-save", action="store_true")

    args = ap.parse_args()

    engine = (args.engine or read_engine_from_yaml(args.config)).lower()
    loader = build_loader(engine, args.config)

    if args.cmd == "file":
        run_file(loader, args.audio, no_save=args.no_save)
        return
    if args.cmd == "dir":
        run_dir(loader, args.audio_dir, args.glob, no_save=args.no_save)
        return
    if args.cmd == "manifest":
        run_manifest(loader, args.path, no_save=args.no_save)
        return


if __name__ == "__main__":
    main()