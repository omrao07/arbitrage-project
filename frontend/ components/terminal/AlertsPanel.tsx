// frontend/components/AlertsPanel.tsx
import React, { useEffect, useMemo, useState } from "react";

/* ---------------------------- Types ---------------------------- */

export type AlertLevel = "info" | "warn" | "error" | "success";

export interface Alert {
  id: string;
  time: number;            // epoch ms
  level: AlertLevel;
  message: string;
  context?: Record<string, any>;
}

interface Props {
  /** If provided, no fetch. */
  alerts?: Alert[];
  /** Polling endpoint returning Alert[] */
  endpoint?: string;       // default: /api/alerts
  pollMs?: number;         // default: 10s
  title?: string;
}

/* -------------------------- Component -------------------------- */

export default function AlertsPanel({
  alerts,
  endpoint = "/api/alerts",
  pollMs = 10_000,
  title = "Alerts",
}: Props) {
  const [rows, setRows] = useState<Alert[]>(alerts ?? []);
  const [err, setErr] = useState<string | null>(null);
  const [filter, setFilter] = useState<AlertLevel | "all">("all");

  /* polling if no alerts passed */
  useEffect(() => {
    if (alerts) { setRows(alerts); return; }
    let ignore = false;
    async function fetchAlerts() {
      try {
        const res = await fetch(endpoint);
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        const json = (await res.json()) as Alert[];
        if (!ignore) setRows(json);
      } catch (e: any) {
        if (!ignore) setErr(e?.message || "Failed to load alerts");
      }
    }
    fetchAlerts();
    const t = setInterval(fetchAlerts, pollMs);
    return () => { ignore = true; clearInterval(t); };
  }, [alerts, endpoint, pollMs]);

  const filtered = useMemo(() => {
    if (filter === "all") return rows;
    return rows.filter((a) => a.level === filter);
  }, [rows, filter]);

  /* actions */
  function clearAll() {
    setRows([]);
  }
  function exportJSON() {
    download(`alerts_${Date.now()}.json`, JSON.stringify(rows, null, 2), "application/json");
  }
  function exportCSV() {
    const headers = ["id","time","level","message"];
    const lines = rows.map((a) =>
      [a.id, new Date(a.time).toISOString(), a.level, JSON.stringify(a.message)].join(",")
    );
    download(`alerts_${Date.now()}.csv`, [headers.join(","), ...lines].join("\n"), "text/csv;charset=utf-8;");
  }

  /* ----------------------------- UI ----------------------------- */

  return (
    <div className="rounded-2xl shadow-md bg-white dark:bg-gray-900 p-4">
      <header className="flex items-center justify-between mb-3">
        <div>
          <h2 className="text-lg font-semibold">{title}</h2>
          <p className="text-xs opacity-70">{rows.length} alerts loaded</p>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value as any)}
            className="px-2 py-1.5 rounded-md border text-sm dark:border-gray-800"
          >
            <option value="all">All</option>
            <option value="info">Info</option>
            <option value="warn">Warnings</option>
            <option value="error">Errors</option>
            <option value="success">Success</option>
          </select>
          <button onClick={exportJSON} className="px-2 py-1 text-sm rounded-md border dark:border-gray-800">Export JSON</button>
          <button onClick={exportCSV} className="px-2 py-1 text-sm rounded-md border dark:border-gray-800">Export CSV</button>
          <button onClick={clearAll} className="px-2 py-1 text-sm rounded-md border dark:border-gray-800">Clear</button>
        </div>
      </header>

      {err && <div className="text-sm text-red-600 mb-2">Error: {err}</div>}

      <div className="max-h-96 overflow-y-auto divide-y dark:divide-gray-800">
        {filtered.map((a) => (
          <div key={a.id} className="py-2 px-1 text-sm flex items-start gap-2">
            <span className={`mt-1 w-2 h-2 rounded-full flex-shrink-0 ${dotColor(a.level)}`} />
            <div className="flex-1">
              <div className="flex justify-between">
                <span className="font-medium">{a.message}</span>
                <span className="opacity-60 text-xs">{new Date(a.time).toLocaleString()}</span>
              </div>
              {a.context && (
                <pre className="mt-1 text-[11px] bg-gray-50 dark:bg-gray-800 p-2 rounded-md overflow-x-auto">
                  {JSON.stringify(a.context, null, 2)}
                </pre>
              )}
            </div>
          </div>
        ))}
        {filtered.length === 0 && (
          <div className="p-4 text-sm opacity-60 text-center">No alerts</div>
        )}
      </div>
    </div>
  );
}

/* -------------------------- Helpers -------------------------- */

function dotColor(level: AlertLevel) {
  switch (level) {
    case "info": return "bg-blue-500";
    case "warn": return "bg-yellow-500";
    case "error": return "bg-red-600";
    case "success": return "bg-green-500";
    default: return "bg-gray-400";
  }
}
function download(name: string, body: string, type: string) {
  const blob = new Blob([body], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click(); URL.revokeObjectURL(url);
}