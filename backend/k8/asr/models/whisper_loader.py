#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
whisper_loader.py
-----------------
Loader + wrapper around OpenAI Whisper ASR with Transformers fallback.

- Reads kb/asr/configs/whisper.yaml
- Prefers openai-whisper; falls back to HF pipeline("automatic-speech-recognition")
- Methods: transcribe_file(), transcribe_batch(), transcribe_dir()
- Outputs JSONL (or TXT) with timestamps + confidence when available

Install:
  pip install openai-whisper soundfile librosa pyyaml --upgrade
  # optional fallback:
  pip install transformers torch --upgrade

Note:
  Whisper "hotwords" are not natively supported; config entries are ignored.

"""

from __future__ import annotations
import os
import sys
import json
import time
import glob
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml

# -------- Optional deps (prefer openai-whisper) --------
_HAS_WHISPER = True
try:
    import whisper  # openai-whisper#type:ignore
except Exception:
    _HAS_WHISPER = False

# Fallback: Hugging Face pipeline
_HAS_HF = True
try:
    from transformers import pipeline  # type: ignore
except Exception:
    _HAS_HF = False

try:
    import torch  # type: ignore
except Exception:
    torch = None

try:
    import soundfile as sf  # type: ignore
except Exception as e:
    raise RuntimeError("Missing dependency: soundfile. pip install soundfile") from e


DEFAULT_CFG_PATH = os.path.join("kb", "asr", "configs", "whisper.yaml")


# ==================== Config ====================

@dataclass
class WhisperConfig:
    model_size: str = "base"           # tiny|base|small|medium|large
    model_path: Optional[str] = "models/whisper"
    language: Optional[str] = "en"
    task: str = "transcribe"           # transcribe|translate
    sample_rate: int = 16000
    # inference
    fp16: bool = True
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    length_penalty: float = -0.05
    condition_on_previous_text: bool = True
    # runtime
    device: str = "auto"               # auto|cpu|cuda|<cuda index>
    num_threads: int = 4
    use_gpu: bool = True
    gpu_device: int = 0
    streaming: bool = False
    log_level: str = "INFO"
    # output
    transcripts_dir: str = os.path.join("outputs", "transcripts", "whisper")
    save_format: str = "jsonl"         # jsonl|txt
    include_timestamps: bool = True
    save_confidence: bool = True


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str = DEFAULT_CFG_PATH) -> WhisperConfig:
    cfg = _read_yaml(path)
    asr = cfg.get("asr", {})
    infer = cfg.get("inference", {})
    runtime = cfg.get("runtime", {})
    out = cfg.get("output", {})

    return WhisperConfig(
        model_size=asr.get("model_size", "base"),
        model_path=asr.get("model_path") or "models/whisper",
        language=asr.get("language") or None,
        task=asr.get("task", "transcribe"),
        sample_rate=asr.get("sample_rate", 16000),
        fp16=bool(infer.get("fp16", True)),
        beam_size=int(infer.get("beam_size", 5)),
        best_of=int(infer.get("best_of", 5)),
        patience=float(infer.get("patience", 1.0)),
        length_penalty=float(infer.get("length_penalty", -0.05)),
        condition_on_previous_text=bool(infer.get("condition_on_previous_text", True)),
        device=runtime.get("device", "auto"),
        num_threads=int(runtime.get("num_threads", 4)),
        use_gpu=bool(runtime.get("use_gpu", True)),
        gpu_device=int(runtime.get("gpu_device", 0)),
        streaming=bool(runtime.get("streaming", False)),
        log_level=runtime.get("log_level", "INFO"),
        transcripts_dir=out.get("transcripts_dir", os.path.join("outputs", "transcripts", "whisper")),
        save_format=out.get("save_format", "jsonl"),
        include_timestamps=bool(out.get("include_timestamps", True)),
        save_confidence=bool(out.get("save_confidence", True)),
    )


# ==================== Loader ====================

def _pick_device_index(device: str, prefer_gpu: bool, gpu_idx: int) -> str:
    """
    For openai-whisper: return "cuda" or "cpu".
    For HF pipeline: returns -1 (CPU) or gpu index.
    """
    # If user specifies numeric string (e.g., "1"), treat as CUDA index
    if device not in ("auto", "cpu", "cuda"):
        try:
            idx = int(device)
            if torch is not None and torch.cuda.is_available():
                torch.cuda.set_device(idx)
                return "cuda"
        except Exception:
            return "cpu"

    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    # auto
    if prefer_gpu and torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_device(gpu_idx)
        except Exception:
            pass
        return "cuda"
    return "cpu"


class WhisperLoader:
    """
    Wrapper that prefers openai-whisper; falls back to HF pipeline if needed.
    """

    def __init__(self, cfg_path: str = DEFAULT_CFG_PATH):
        self.cfg = load_config(cfg_path)
        os.makedirs(self.cfg.transcripts_dir, exist_ok=True)

        self.runtime_device = _pick_device_index(self.cfg.device, self.cfg.use_gpu, self.cfg.gpu_device)

        # Threading (CPU)
        try:
            if torch is not None and self.runtime_device == "cpu":
                torch.set_num_threads(max(1, self.cfg.num_threads))
        except Exception:
            pass

        self.backend = None
        self.model = None
        self.asr_pipe = None

        if _HAS_WHISPER:
            # openai-whisper model
            name = self.cfg.model_size
            # local cache dir if provided
            model_dir = self.cfg.model_path if self.cfg.model_path else None
            self.model = whisper.load_model(name, device=self.runtime_device, download_root=model_dir) #type:ignore
            self.backend = "whisper"
        elif _HAS_HF:
            # HF fallback (use an appropriate whisper checkpoint)
            model_name = f"openai/whisper-{self.cfg.model_size}"
            device_idx = -1 if self.runtime_device == "cpu" else 0
            kwargs = dict(model=model_name, generate_kwargs={
                "task": self.cfg.task,
                "language": self.cfg.language or "en",
                "num_beams": self.cfg.beam_size,
                "length_penalty": self.cfg.length_penalty,
            })
            try:
                self.asr_pipe = pipeline("automatic-speech-recognition", device=device_idx, **kwargs)#type:ignore
            except TypeError:
                kwargs.pop("generate_kwargs", None)
                self.asr_pipe = pipeline("automatic-speech-recognition", device=device_idx, **kwargs)#type:ignore
            self.backend = "hf"
        else:
            raise RuntimeError(
                "No ASR backend available. Install either 'openai-whisper' or 'transformers+torch'."
            )

    # --------------- I/O helpers ---------------

    @staticmethod
    def _read_audio(path: str, target_sr: int) -> Tuple[List[float], int]:
        """
        Read audio to mono float32, resample if needed with librosa.
        """
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if hasattr(audio, "ndim") and audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            try:
                import librosa  # type: ignore
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            except Exception as e:
                raise RuntimeError(
                    f"Audio sample rate {sr} != {target_sr}. Install librosa for resampling."
                ) from e
        return audio, sr#type:ignore

    def _to_output_path(self, audio_path: str) -> str:
        base = os.path.splitext(os.path.basename(audio_path))[0]
        ext = "jsonl" if self.cfg.save_format.lower() == "jsonl" else "txt"
        return os.path.join(self.cfg.transcripts_dir, f"{base}.{ext}")

    # --------------- Core transcription ---------------

    def _transcribe_openai_whisper(self, audio: List[float], sr: int) -> Dict[str, Any]:
        """
        Use openai-whisper's transcribe; returns {text, segments:[...]}
        """
        options = dict(
            task=self.cfg.task,
            language=self.cfg.language,  # None -> auto
            beam_size=self.cfg.beam_size,
            best_of=self.cfg.best_of,
            patience=self.cfg.patience,
            condition_on_previous_text=self.cfg.condition_on_previous_text,
            fp16=(self.cfg.fp16 and self.runtime_device == "cuda"),
        )
        # whisper accepts filepath or numpy array; we pass numpy
        import numpy as np
        audio_np = np.asarray(audio, dtype=np.float32)
        result = self.model.transcribe(audio_np, **options)#type:ignore
        # result has 'text' and 'segments' with timestamps + avg_logprob/no_speech_prob
        return result

    def _transcribe_hf(self, audio: List[float], sr: int) -> Dict[str, Any]:
        """
        HF pipeline transcription with timestamps when supported.
        """
        if self.asr_pipe is None:
            raise RuntimeError("HF pipeline not initialized")
        result = self.asr_pipe({"array": audio, "sampling_rate": sr}, return_timestamps=True)
        if isinstance(result, dict) and "text" in result:
            return result
        return {"text": str(result), "chunks": []}

    def transcribe_file(self, audio_path: str) -> Dict[str, Any]:
        t0 = time.time()
        audio, sr = self._read_audio(audio_path, self.cfg.sample_rate)

        if self.backend == "whisper":
            out = self._transcribe_openai_whisper(audio, sr)
            text = out.get("text", "")
            # Convert segments to uniform schema
            chunks = []
            for seg in out.get("segments", []):
                item = {
                    "text": seg.get("text", ""),
                    "timestamp": [seg.get("start"), seg.get("end")],
                }
                if self.cfg.save_confidence:
                    # whisper doesn't give a single "confidence", but we can use avg_logprob
                    if "avg_logprob" in seg and seg["avg_logprob"] is not None:
                        item["confidence"] = float(seg["avg_logprob"])
                chunks.append(item)
        else:
            out = self._transcribe_hf(audio, sr)
            text = out.get("text", "")
            # HF returns "chunks" with "timestamp" and sometimes "score"
            chunks = []
            for ch in out.get("chunks", []):
                item = {
                    "text": ch.get("text", ""),
                    "timestamp": ch.get("timestamp") or ch.get("timestamps"),
                }
                if self.cfg.save_confidence and "score" in ch:
                    item["confidence"] = ch["score"]
                chunks.append(item)

        payload: Dict[str, Any] = {
            "audio_path": audio_path,
            "backend": self.backend,
            "model": (self.cfg.model_size if self.backend == "whisper" else f"openai/whisper-{self.cfg.model_size}"),
            "language": self.cfg.language,
            "task": self.cfg.task,
            "duration_s": len(audio) / float(sr),
            "text": text,
            "chunks": chunks if self.cfg.include_timestamps else [],
            "latency_s": round(time.time() - t0, 3),
        }
        return payload

    def save_transcript(self, transcript: Dict[str, Any]) -> str:
        out_path = self._to_output_path(transcript["audio_path"])
        if self.cfg.save_format.lower() == "jsonl":
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(transcript, ensure_ascii=False) + "\n")
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(transcript.get("text", ""))
        return out_path

    # --------------- Batch helpers ---------------

    def transcribe_batch(self, audio_paths: List[str], save: bool = True) -> List[Dict[str, Any]]:
        outs = []
        for p in audio_paths:
            try:
                tr = self.transcribe_file(p)
                if save:
                    self.save_transcript(tr)
                outs.append(tr)
                print(f"✓ {p}  →  {len(tr.get('text',''))} chars")
            except Exception as e:
                print(f"✗ {p}: {e}", file=sys.stderr)
        return outs

    def transcribe_dir(self, audio_dir: str, pattern: str = "**/*.wav", save: bool = True) -> List[Dict[str, Any]]:
        paths = [p for p in glob.glob(os.path.join(audio_dir, pattern), recursive=True) if os.path.isfile(p)]
        return self.transcribe_batch(paths, save=save)


# ==================== CLI ====================

def main():
    ap = argparse.ArgumentParser(description="Whisper ASR loader/transcriber")
    ap.add_argument("--config", default=DEFAULT_CFG_PATH, help="Path to kb/asr/configs/whisper.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("file", help="Transcribe a single audio file")
    s1.add_argument("audio", help="Path to audio (wav/flac/mp3)")
    s1.add_argument("--no-save", action="store_true", help="Do not write transcript file")

    s2 = sub.add_parser("dir", help="Transcribe a directory of audio files")
    s2.add_argument("audio_dir", help="Directory path")
    s2.add_argument("--glob", default="**/*.wav", help="Glob pattern (default **/*.wav)")
    s2.add_argument("--no-save", action="store_true")

    args = ap.parse_args()

    loader = WhisperLoader(args.config)

    if args.cmd == "file":
        tr = loader.transcribe_file(args.audio)
        if not args.no_save:
            out = loader.save_transcript(tr)
            print(f"✅ wrote {out}")
        else:
            print(json.dumps(tr, ensure_ascii=False, indent=2))
        return

    if args.cmd == "dir":
        trs = loader.transcribe_dir(args.audio_dir, pattern=args.glob, save=not args.no_save)
        print(f"✅ processed {len(trs)} files")

if __name__ == "__main__":
    main()