# backend/api/voice_interface.py
from __future__ import annotations

import io
import os
import json
import time
import tempfile
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import StreamingResponse, JSONResponse

# ---------------- Optional deps (all gracefully optional) --------------------
HAVE_VOSK = True
HAVE_PYDUB = True
HAVE_TTS = True
HAVE_REDIS = True

try:
    import vosk  # type: ignore # offline speech-to-text
except Exception:
    HAVE_VOSK = False

try:
    from pydub import AudioSegment  # type: ignore # format normalization
except Exception:
    HAVE_PYDUB = False

try:
    import pyttsx3  # type: ignore # offline TTS
except Exception:
    HAVE_TTS = False

try:
    from redis.asyncio import Redis as AsyncRedis
except Exception:
    HAVE_REDIS = False
    AsyncRedis = None  # type: ignore

router = APIRouter()

# ---------------- Environment / defaults ------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
VOICE_CMD_STREAM = os.getenv("VOICE_CMD_STREAM", "voice.commands")
SWARM_TASKS_STREAM = os.getenv("SWARM_TASKS_STREAM", "swarm.tasks")
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "")  # e.g., /models/vosk-model-small-en-us-0.15
LANG = os.getenv("VOICE_LANG", "en")
TTS_RATE = int(os.getenv("TTS_RATE", "175"))  # words per minute
TTS_VOLUME = float(os.getenv("TTS_VOLUME", "1.0"))  # 0..1

# ---------------- Lazy singletons -------------------------------------------
_vosk_model = None
_tts_engine = None
_redis: Optional[AsyncRedis] = None # type: ignore

async def get_redis() -> Optional[AsyncRedis]: # type: ignore
    global _redis
    if not HAVE_REDIS:
        return None
    if _redis is not None:
        return _redis
    try:
        _redis = AsyncRedis.from_url(REDIS_URL, decode_responses=True) # type: ignore
        await _redis.ping() # type: ignore
        return _redis
    except Exception:
        _redis = None
        return None

def get_vosk() -> Optional[Any]:
    global _vosk_model
    if not HAVE_VOSK:
        return None
    if _vosk_model is not None:
        return _vosk_model
    # Load model lazily; allow env path
    path = VOSK_MODEL_PATH
    try:
        if path and os.path.isdir(path):
            _vosk_model = vosk.Model(path)
        else:
            # last resort: small English model if packaged (may not exist)
            _vosk_model = vosk.Model(lang=LANG)  # uses built-in small model if available
        return _vosk_model
    except Exception:
        return None

def get_tts() -> Optional[Any]:
    global _tts_engine
    if not HAVE_TTS:
        return None
    if _tts_engine is not None:
        return _tts_engine
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", TTS_RATE)
        engine.setProperty("volume", TTS_VOLUME)
        _tts_engine = engine
        return _tts_engine
    except Exception:
        return None

# ---------------- Audio utils -----------------------------------------------
TARGET_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2  # 16-bit PCM

def _normalize_to_wav16le(data: bytes, mime: str | None = None) -> bytes:
    """
    Convert arbitrary audio bytes to 16k mono 16-bit PCM WAV (vosk preferred).
    Requires pydub+ffmpeg for formats other than WAV PCM.
    """
    # fast path: WAV already
    if HAVE_PYDUB:
        try:
            seg = AudioSegment.from_file(io.BytesIO(data), format=_infer_fmt(mime))
            seg = seg.set_channels(TARGET_CHANNELS).set_frame_rate(TARGET_RATE).set_sample_width(TARGET_SAMPLE_WIDTH)
            out = io.BytesIO()
            seg.export(out, format="wav")
            return out.getvalue()
        except Exception:
            pass
    # fallback: return as-is (may fail in STT if not correct)
    return data

def _infer_fmt(mime: Optional[str]) -> Optional[str]:
    if not mime:
        return None
    if "wav" in mime: return "wav"
    if "mpeg" in mime or "mp3" in mime: return "mp3"
    if "aac" in mime or "m4a" in mime: return "mp4"
    if "ogg" in mime or "opus" in mime: return "ogg"
    if "webm" in mime: return "webm"
    return None

# ---------------- Core STT (file) -------------------------------------------
def _stt_bytes_wav(wav_bytes: bytes) -> Dict[str, Any]:
    model = get_vosk()
    if not model:
        raise RuntimeError("STT unavailable: Vosk model not loaded and no fallback configured.")
    try:
        rec = vosk.KaldiRecognizer(model, TARGET_RATE)
        rec.SetWords(True)
        # stream in chunks
        chunk = 3200  # ~0.1s at 16k mono 16-bit
        bio = io.BytesIO(wav_bytes)
        # skip header if WAV (simple sniff)
        if wav_bytes[:4] == b"RIFF":
            # let vosk parse anyway; reading all bytes is fine
            pass
        while True:
            b = bio.read(chunk)
            if not b:
                break
            rec.AcceptWaveform(b)
        res = json.loads(rec.FinalResult() or "{}")
        text = res.get("text", "").strip()
        return {"text": text, "result": res}
    except Exception as e:
        raise RuntimeError(f"STT error: {e}")

@router.post("/voice/stt")
async def post_stt(file: UploadFile = File(...)):
    data = await file.read()
    wav = _normalize_to_wav16le(data, file.content_type)
    try:
        out = _stt_bytes_wav(wav)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse({"ok": True, "text": out.get("text", ""), "raw": out.get("result", {}), "lang": LANG})

# ---------------- Core TTS (file) -------------------------------------------
def _tts_wav_bytes(text: str) -> bytes:
    engine = get_tts()
    if not engine:
        raise RuntimeError("TTS unavailable: pyttsx3 not initialized.")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        engine.save_to_file(text, path)
        engine.runAndWait()
        with open(path, "rb") as f:
            audio = f.read()
        return audio
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass

@router.post("/voice/tts")
async def post_tts(payload: Dict[str, Any]):
    text = str(payload.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    try:
        audio = _tts_wav_bytes(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return StreamingResponse(io.BytesIO(audio), media_type="audio/wav")

# ---------------- WS: streaming STT -----------------------------------------
@router.websocket("/ws/voice")
async def ws_voice(ws: WebSocket, mode: str = Query("stt"), lang: str = Query(LANG)):
    await ws.accept()
    if mode != "stt":
        await ws.send_json({"error": "unsupported mode", "mode": mode})
        await ws.close(code=1003)
        return

    model = get_vosk()
    if not model:
        await ws.send_json({"error": "STT unavailable (vosk not loaded). Set VOSK_MODEL_PATH or install vosk model."})
        await ws.close(code=1011)
        return

    # incremental recognizer
    rec = vosk.KaldiRecognizer(model, TARGET_RATE)
    rec.SetWords(True)

    await ws.send_json({"ready": True, "sample_rate": TARGET_RATE, "channels": TARGET_CHANNELS})
    try:
        while True:
            # Expect BINARY audio chunks in 16k mono 16-bit PCM WAV or raw PCM
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                b = msg["bytes"]
                # if WAV with RIFF header, just feed bytes; vosk will handle
                ok = rec.AcceptWaveform(b)
                if ok:
                    res = json.loads(rec.Result() or "{}")
                    await ws.send_json({"partial": False, "result": res, "ts_ms": int(time.time() * 1000)})
                else:
                    part = json.loads(rec.PartialResult() or "{}")
                    if part.get("partial"):
                        await ws.send_json({"partial": True, "text": part.get("partial"), "ts_ms": int(time.time() * 1000)})
            else:
                # text control messages (e.g., {"flush":true})
                data = msg.get("text")
                if not data:
                    continue
                try:
                    j = json.loads(data)
                except Exception:
                    j = {}
                if j.get("flush"):
                    final = json.loads(rec.FinalResult() or "{}")
                    await ws.send_json({"partial": False, "final": True, "result": final})
                    # reset for next utterance
                    rec = vosk.KaldiRecognizer(model, TARGET_RATE)
                    rec.SetWords(True)
                if j.get("close"):
                    await ws.close()
                    return
    except WebSocketDisconnect:
        return
    except Exception:
        try:
            await ws.close(code=1011)
        except Exception:
            pass

# ---------------- Optional: publish to Redis / handoff to swarm -------------
async def publish_voice_command(text: str, meta: Dict[str, Any] | None = None) -> None:
    r = await get_redis()
    if not r:
        return
    payload = {"ts_ms": int(time.time() * 1000), "text": text, "meta": meta or {}}
    try:
        await r.xadd(VOICE_CMD_STREAM, {"json": json.dumps(payload)}, maxlen=2000, approximate=True)
    except Exception:
        pass

async def submit_swarm_query(text: str) -> None:
    """Optional: push a natural-language query to your Query Copilot capability."""
    r = await get_redis()
    if not r:
        return
    task = {
        "id": f"voice-{int(time.time()*1000)}",
        "kind": "rpc",
        "capability": "pnl_by_strategy" if "pnl" in text.lower() else "risk_summary",
        "params": {"from": "voice", "q": text},
        "ttl_ms": 30000,
        "submit_ts": int(time.time() * 1000),
    }
    try:
        await r.xadd(SWARM_TASKS_STREAM, {"json": json.dumps(task)}, maxlen=20000, approximate=True)
    except Exception:
        pass