// frontend/components/VoiceCommand.tsx
import React, { useEffect, useRef, useState } from "react";

declare global {
  interface Window {
    webkitSpeechRecognition: any;
  }
}

export default function VoiceCommand() {
  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [log, setLog] = useState<string[]>([]);
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    if (!("webkitSpeechRecognition" in window)) {
      console.warn("Web Speech API not supported in this browser.");
      return;
    }

    const recognition = new window.webkitSpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onstart = () => setListening(true);
    recognition.onend = () => setListening(false);
    recognition.onerror = (e: any) => console.error("Speech error:", e);

    recognition.onresult = (event: any) => {
      let finalTranscript = "";
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        const res = event.results[i];
        if (res.isFinal) {
          finalTranscript += res[0].transcript;
        }
      }
      if (finalTranscript) {
        setTranscript(finalTranscript);
        handleCommand(finalTranscript);
      }
    };

    recognitionRef.current = recognition;
  }, []);

  const toggleListening = () => {
    if (!recognitionRef.current) return;
    if (listening) {
      recognitionRef.current.stop();
    } else {
      recognitionRef.current.start();
    }
  };

  const handleCommand = async (cmd: string) => {
    setLog((prev) => [`> ${cmd}`, ...prev]);

    try {
      const res = await fetch("/api/voice/command", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ command: cmd }),
      });
      const data = await res.json();
      if (data?.reply) {
        setLog((prev) => [`AI: ${data.reply}`, ...prev]);
      }
    } catch (err) {
      console.error("Voice command error:", err);
    }
  };

return (
    <div className="rounded-2xl shadow-md p-4 bg-white dark:bg-gray-900">
        <header className="mb-3 flex items-center justify-between">
            <h2 className="text-lg font-semibold">Voice Command</h2>
            <button
                onClick={toggleListening}
                className={`px-3 py-1.5 rounded-lg text-sm ${
                    listening ? "bg-red-500 text-white" : "bg-blue-600 text-white"
                }`}
            >
                {listening ? "Stop" : "Start"}
            </button>
        </header>

        <div className="text-sm mb-2">
            {listening ? "🎙 Listening…" : "Idle"}
        </div>

        {transcript && (
            <div className="mb-2 p-2 border rounded bg-gray-50 dark:bg-gray-800 text-sm">
                Last: {transcript}
            </div>
        )}

        <div className="h-40 overflow-auto border rounded-lg p-2 text-sm bg-gray-50 dark:bg-gray-800">
            {log.map((l, i) => (
                <div key={i}>{l}</div>
            ))}
            {log.length === 0 && <div className="opacity-60">No commands yet.</div>}
        </div>
    </div>
);
}