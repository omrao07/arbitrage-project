"use client";

/**
 * components/connections.tsx
 * Zero-import panel to manage data/API connections.
 * Tailwind-only. No external imports.
 *
 * Features:
 * - List of connections (name, type, status)
 * - Toggle on/off
 * - Add new connection placeholder
 */

export type Connection = {
  id: string;
  name: string;
  type: "API" | "DB" | "File" | "Other";
  status: "connected" | "disconnected";
};

export default function ConnectionsPanel({
  connections = [],
  onToggle,
  onAdd,
}: {
  connections: Connection[];
  onToggle?: (id: string) => void;
  onAdd?: () => void;
}) {
  return (
    <div className="w-full rounded-xl border border-neutral-800 bg-neutral-900 p-4 text-neutral-100">
      {/* Header */}
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-sm font-semibold">Connections</h2>
        {onAdd && (
          <button
            onClick={onAdd}
            className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-xs hover:bg-neutral-800"
          >
            + Add
          </button>
        )}
      </div>

      {/* List */}
      <ul className="space-y-2">
        {connections.length === 0 && (
          <li className="text-xs text-neutral-500">No connections yet.</li>
        )}
        {connections.map((c) => (
          <li
            key={c.id}
            className="flex items-center justify-between rounded-lg border border-neutral-800 bg-neutral-950 px-3 py-2"
          >
            <div className="min-w-0">
              <p className="truncate text-sm font-medium">{c.name}</p>
              <p className="text-xs text-neutral-400">{c.type}</p>
            </div>
            <div className="flex items-center gap-2">
              <span
                className={`h-2 w-2 rounded-full ${
                  c.status === "connected" ? "bg-emerald-400" : "bg-neutral-500"
                }`}
              />
              {onToggle && (
                <button
                  onClick={() => onToggle(c.id)}
                  className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 text-xs hover:bg-neutral-800"
                >
                  {c.status === "connected" ? "Disconnect" : "Connect"}
                </button>
              )}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

/* ---------------------- Ambient React (no imports used) --------------------- */
declare const React: any;