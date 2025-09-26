"use client";

import React, { useState } from "react";
import { Play, StopCircle, Activity } from "lucide-react";

export default function ExecutionPage() {
  const [running, setRunning] = useState(false);

  const handleStart = () => {
    setRunning(true);
    // here you could trigger API call / websocket to start execution
  };

  const handleStop = () => {
    setRunning(false);
    // here you could trigger API call to stop execution
  };

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 flex items-center justify-center">
      <div className="max-w-md w-full rounded-xl border border-neutral-800 bg-neutral-900 p-6 shadow-lg">
        <div className="flex items-center gap-3 mb-4">
          <Activity className="h-6 w-6 text-emerald-500" />
          <h1 className="text-lg font-semibold">Execution Console</h1>
        </div>

        <p className="text-sm text-neutral-400 mb-6">
          Control the execution of your trading strategies. Use the buttons
          below to start or stop live execution.
        </p>

        <div className="flex gap-3">
          <button
            onClick={handleStart}
            disabled={running}
            className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium 
              ${running ? "bg-neutral-800 text-neutral-500 cursor-not-allowed" : "bg-emerald-600 hover:bg-emerald-500 text-white"}`}
          >
            <Play className="h-4 w-4" />
            Start
          </button>

          <button
            onClick={handleStop}
            disabled={!running}
            className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium 
              ${!running ? "bg-neutral-800 text-neutral-500 cursor-not-allowed" : "bg-rose-600 hover:bg-rose-500 text-white"}`}
          >
            <StopCircle className="h-4 w-4" />
            Stop
          </button>
        </div>
      </div>
    </div>
  );
}