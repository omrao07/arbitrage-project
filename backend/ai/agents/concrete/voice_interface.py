# backend/ai/agents/concrete/voice_interface.py
from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ------------------------------------------------------------
# BaseAgent shim
# ------------------------------------------------------------
try:
    from ..core.base_agent import BaseAgent  # type: ignore
except Exception:
    class BaseAgent:
        name: str = "voice_interface"
        def plan(self, *a, **k): ...
        def act(self, *a, **k): ...
        def explain(self, *a, **k): ...
        def heartbeat(self, *a, **k): return {"ok": True}

# ------------------------------------------------------------
# Optional downstream router / copilot
# ------------------------------------------------------------
try:
    # best effort NL → intents router
    from .query_copilot import QueryCopilotAgent  # type: ignore
except Exception:
    class QueryCopilotAgent(BaseAgent): # type: ignore
        name = "query_copilot"
        def act(self, req):
            # dumb echo for fallback
            return {"intents": ["unknown"], "symbol": None,
                    "results": {"echo": req.get("query")}, "generated_at": int(time.time()*1000)}

# ------------------------------------------------------------
# Optional STT/TTS connectors (with safe fallbacks)
# ------------------------------------------------------------
# STT contract: stt_transcribe(audio_bytes: bytes, language: Optional[str]) -> str
try:
    # e.g., ..connectors/llm/openai.py or local_gguf.py could expose STT/TTS
    from ..connectors.llm.openai import stt_transcribe, tts_speak  # type: ignore
except Exception:
    def stt_transcribe(audio_bytes: bytes, language: Optional[str] = None) -> str:
        # Fallback: pretend we heard nothing; caller should provide text
        return "[no-stt: provide 'text' in request]"
    def tts_speak(text: str, voice: str = "neutral") -> bytes:
        # Fallback: return small WAV-like placeholder bytes (not a real audio file)
        return f"FAKE-TTS::{voice}::{text}".encode("utf-8")

# ------------------------------------------------------------
# Data models
# ------------------------------------------------------------
@dataclass
class VoiceRequest:
    # One of these must be provided
    wav_path: Optional[str] = None          # path to a WAV/MP3/OGG file (handled externally)
    audio_b64: Optional[str] = None         # base64-encoded audio bytes
    text: Optional[str] = None              # bypass STT and send text directly

    # STT config
    language: Optional[str] = None          # "en", "hi", etc.
    wake_word: Optional[str] = None         # e.g., "bolt" (if provided, require it)
    push_to_talk: bool = False              # if True, skip wake-word logic

    # Routing preferences
    target: Optional[str] = None            # "copilot" | future: "router"
    notes: Optional[str] = None

    # TTS config
    tts_voice: str = "neutral"              # "neutral"|"female"|"male"|custom

@dataclass
class VoiceResponse:
    recognized_text: str
    intents: List[str]
    symbol: Optional[str]
    result: Dict[str, Any]
    tts_audio_b64: Optional[str]
    generated_at: int

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def _read_audio_bytes(path: str) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return b""

def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii") if data else ""

# ------------------------------------------------------------
# Voice Interface Agent
# ------------------------------------------------------------
class VoiceInterfaceAgent(BaseAgent): # type: ignore
    """
    Voice → Intent → Result → Voice:
      1) Accept audio (file path or base64) OR plain text.
      2) STT (if audio) with optional wake-word guard.
      3) Route to QueryCopilot to fetch price/orderbook/news/risk.
      4) Return results + TTS (base64) for playback in the UI.
    """

    name = "voice_interface"

    def __init__(self):
        super().__init__()
        self.copilot = QueryCopilotAgent()

    # -------- Planning --------
    def plan(self, req: VoiceRequest | Dict[str, Any]) -> VoiceRequest:
        if isinstance(req, VoiceRequest):
            return req
        return VoiceRequest(
            wav_path=req.get("wav_path"),
            audio_b64=req.get("audio_b64"),
            text=req.get("text"),
            language=req.get("language"),
            wake_word=req.get("wake_word"),
            push_to_talk=bool(req.get("push_to_talk", False)),
            target=req.get("target", "copilot"),
            notes=req.get("notes"),
            tts_voice=req.get("tts_voice", "neutral"),
        )

    # -------- Acting --------
    def act(self, request: VoiceRequest | Dict[str, Any]) -> VoiceResponse:
        req = self.plan(request)

        # 1) Acquire text (STT or direct)
        recognized = (req.text or "").strip()

        if not recognized:
            audio_bytes = b""
            if req.audio_b64:
                try:
                    audio_bytes = base64.b64decode(req.audio_b64)
                except Exception:
                    audio_bytes = b""
            elif req.wav_path:
                audio_bytes = _read_audio_bytes(req.wav_path)

            if audio_bytes:
                recognized = (stt_transcribe(audio_bytes, language=req.language) or "").strip()
            else:
                recognized = "[empty-input]"

        # 2) Wake-word / push-to-talk gate
        if not req.push_to_talk and req.wake_word:
            ww = req.wake_word.lower()
            if ww not in recognized.lower():
                # Not addressed to us; return neutral response (no TTS)
                return VoiceResponse(
                    recognized_text=recognized,
                    intents=[],
                    symbol=None,
                    result={"info": "wake-word not detected; ignoring"},
                    tts_audio_b64=None,
                    generated_at=int(time.time()*1000),
                )
            # Strip wake-word from query for nicer UX
            recognized = recognized.lower().replace(ww, "", 1).strip()

        # 3) Route (default: copilot)
        intents: List[str] = []
        symbol: Optional[str] = None
        result: Dict[str, Any] = {}

        if (req.target or "copilot") == "copilot":
            cp_resp = self.copilot.act({"query": recognized})
            # cp_resp could be dataclass or dict (support both)
            intents = list(getattr(cp_resp, "intents", None) or cp_resp.get("intents", []) or [])
            symbol = getattr(cp_resp, "symbol", None) or cp_resp.get("symbol")
            result = getattr(cp_resp, "results", None) or cp_resp.get("results", {}) or {}
        else:
            # future: send to router
            result = {"error": f"unknown target '{req.target}'"}
            intents = ["unknown"]

        # 4) TTS the essential summary
        spoken = self._summarize_for_tts(recognized, intents, symbol, result)
        tts_bytes = tts_speak(spoken, voice=req.tts_voice)
        audio_b64 = _b64(tts_bytes)

        return VoiceResponse(
            recognized_text=recognized,
            intents=intents,
            symbol=symbol,
            result=result,
            tts_audio_b64=audio_b64,
            generated_at=int(time.time()*1000),
        )

    # -------- Helpers --------
    def _summarize_for_tts(self, text: str, intents: List[str], symbol: Optional[str], result: Dict[str, Any]) -> str:
        if not intents:
            return "I didn't catch a valid request."
        intent = intents[0]
        sym = symbol or "the symbol"
        if intent == "price":
            candles = (result.get("price") or result.get("candles") or [])
            last = candles[-1]["c"] if candles else None
            return f"{sym} last price is {round(float(last),2) if last else 'unavailable'}."
        if intent == "orderbook":
            ob = result.get("orderbook") or {}
            b0 = (ob.get("bids") or [{}])[0].get("px")
            a0 = (ob.get("asks") or [{}])[0].get("px")
            if b0 and a0:
                return f"Top of book for {sym}: bid {b0}, ask {a0}."
            return f"Order book for {sym} unavailable."
        if intent == "news":
            headlines = (result.get("news") or result.get("headlines") or [])[:2]
            if headlines and isinstance(headlines[0], dict):
                titles = "; ".join(h.get("headline","") for h in headlines if h.get("headline"))
                return f"Latest headlines for {sym}: {titles}."
            return f"No recent headlines for {sym}."
        if intent == "risk":
            r = result.get("risk") or {}
            varv = r.get("VaR")
            return f"Estimated Value at Risk for {sym} is {round(float(varv)*100,2) if varv is not None else 'unavailable'} percent."
        # unknown or multi-intent
        return "Request processed."

    def explain(self) -> str:
        return (
            "VoiceInterfaceAgent converts audio to text (STT), parses the request via QueryCopilot, "
            "and synthesizes a spoken answer (TTS). It supports wake-word gating and push-to-talk."
        )

    def heartbeat(self) -> Dict[str, Any]:
        return {"ok": True, "agent": self.name, "ts": int(time.time())}


# ------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    agent = VoiceInterfaceAgent()
    # Text path
    r = agent.act({"text": "Show me AAPL price and news", "tts_voice": "neutral"})
    print("recognized:", r.recognized_text)
    print("intents:", r.intents, "symbol:", r.symbol)
    print("tts bytes (b64) len:", len(r.tts_audio_b64 or ""))