// exec.tsx
import React, { useEffect, useState } from "react";

/**
 * Exec Console Panel
 * - A lightweight terminal-like component for running and showing command outputs.
 * - Takes commands from props or user input (if enabled).
 * - Styled with Tailwind for a dark Bloomberg-style console look.
 */

export interface ExecLog {
  id: string;
  cmd: string;
  ts: Date;
  status: "running" | "success" | "error";
  output: string[];
}

export interface ExecProps {
  /** Optional initial logs */
  initialLogs?: ExecLog[];
  /** Called when user submits a command */
  onRun?: (cmd: string) => Promise<{ status: "success" | "error"; output: string[] }>;
  /** Allow user to input commands interactively */
  interactive?: boolean;
  /** Max number of logs to retain */
  maxLogs?: number;
  /** Panel title */
  title?: string;
}

const Exec: React.FC<ExecProps> = ({
  initialLogs = [],
  onRun,
  interactive = true,
  maxLogs = 200,
  title = "Execution Console",
}) => {
  const [logs, setLogs] = useState<ExecLog[]>(initialLogs);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);

  const addLog = (log: ExecLog) => {
    setLogs((prev) => {
      const next = [...prev, log];
      if (next.length > maxLogs) next.shift();
      return next;
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !onRun) return;
    const id = `${Date.now()}`;
    const cmd = input.trim();
    setInput("");
    const log: ExecLog = { id, cmd, ts: new Date(), status: "running", output: [] };
    addLog(log);
    setBusy(true);
    try {
      const res = await onRun(cmd);
      setLogs((prev) =>
        prev.map((l) => (l.id === id ? { ...l, status: res.status, output: res.output } : l))
      );
    } catch (err: any) {
      setLogs((prev) =>
        prev.map((l) =>
          l.id === id
            ? { ...l, status: "error", output: [String(err?.message || err || "Unknown error")] }
            : l
        )
      );
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    const bottom = document.getElementById("exec-bottom");
    if (bottom) bottom.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  return (
    <div className="flex flex-col w-full h-full bg-zinc-950 rounded-2xl border border-zinc-800 overflow-hidden">
      <div className="px-3 py-2 border-b border-zinc-800 text-sm font-medium text-zinc-200">
        {title}
      </div>
      <div className="flex-1 overflow-auto font-mono text-sm px-3 py-2 space-y-2">
        {logs.map((l) => (
          <div key={l.id}>
            <div className="flex items-center gap-2">
              <span className="text-zinc-400">$</span>
              <span className="text-zinc-200">{l.cmd}</span>
              <span
                className={
                  l.status === "running"
                    ? "text-amber-400"
                    : l.status === "success"
                    ? "text-emerald-400"
                    : "text-rose-400"
                }
              >
                [{l.status}]
              </span>
            </div>
            <div className="pl-6 text-zinc-300 whitespace-pre-wrap">
              {l.output.length > 0
                ? l.output.map((line, i) => <div key={i}>{line}</div>)
                : l.status === "running"
                ? "…"
                : null}
            </div>
          </div>
        ))}
        <div id="exec-bottom" />
      </div>
      {interactive && (
        <form
          onSubmit={handleSubmit}
          className="border-t border-zinc-800 flex items-center gap-2 px-3 py-2"
        >
          <span className="text-zinc-500 font-mono">$</span>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={busy}
            placeholder="Type a command…"
            className="flex-1 bg-transparent outline-none text-zinc-100 font-mono text-sm"
          />
        </form>
      )}
    </div>
  );
};

export default Exec;