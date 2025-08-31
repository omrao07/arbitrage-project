// frontend/components/ExplainFlow.tsx
import React, { useMemo, useState } from "react";

/** ---------- Types ---------- */
export type FlowStatus = "ok" | "warn" | "fail" | "pending";

export interface FlowStep {
  id: string;
  title: string;              // e.g., "Risk Check"
  desc?: string;              // quick summary shown in card
  status: FlowStatus;
  latencyMs?: number;         // time spent in this step
  meta?: Record<string, any>; // arbitrary details (rendered in side panel)
}

export interface FlowEdge {
  from: string;  // step id
  to: string;    // step id
  label?: string; // e.g., "Order(vwap) qty=100"
}

export interface ExplainFlowData {
  steps: FlowStep[];
  edges: FlowEdge[];
  startedAt?: number;   // ms epoch
  finishedAt?: number;  // ms epoch
  symbol?: string;
  orderId?: string;
  side?: "buy" | "sell";
  qty?: number;
}

/** ---------- Component ---------- */
interface Props {
  data: ExplainFlowData;
  compact?: boolean;       // tighter layout (for sidebars)
}

export default function ExplainFlow({ data, compact = false }: Props) {
  const [activeId, setActiveId] = useState<string | null>(null);

  const { stepsById, columns, totalLatency } = useMemo(() => {
    const byId = new Map<string, FlowStep>();
    data.steps.forEach((s) => byId.set(s.id, s));
    // naive column layout: ordered as provided
    const cols = data.steps.map((s) => s.id);
    const latency = data.steps.reduce((acc, s) => acc + (s.latencyMs ?? 0), 0);
    return { stepsById: byId, columns: cols, totalLatency: latency };
  }, [data]);

  const active = activeId ? stepsById.get(activeId) || null : null;

  return (
    <div className="rounded-2xl shadow-md bg-white dark:bg-gray-900 p-4">
      {/* Header */}
      <header className="flex items-center justify-between mb-3">
        <div className="space-y-0.5">
          <h2 className="text-lg font-semibold">Explain Flow</h2>
          <p className="text-xs opacity-70">
            {data.symbol ? `${data.symbol} — ` : ""}
            {data.side ? data.side.toUpperCase() : ""}{data.qty ? ` ${fmtNum(data.qty)}` : ""}{data.orderId ? ` • ${data.orderId}` : ""}
          </p>
        </div>
        <div className="text-xs opacity-70">
          {totalLatency ? <>Total latency: <b>{fmtMs(totalLatency)}</b></> : "—"}
        </div>
      </header>

      <div className={`grid gap-4 ${compact ? "grid-cols-1" : "grid-cols-12"}`}>
        {/* Flow lane */}
        <section className={`${compact ? "" : "col-span-8"} relative`}>
          <div className="overflow-x-auto">
            {/* columns as cards */}
            <div className="flex items-stretch gap-3 min-w-max">
              {columns.map((id, idx) => {
                const s = stepsById.get(id)!;
                const isActive = activeId === id;
                return (
                  <div key={id} className="flex flex-col items-center">
                    <button
                      onClick={() => setActiveId(isActive ? null : id)}
                      className={[
                        "w-56 text-left rounded-xl border p-3 transition",
                        "hover:shadow-sm focus:outline-none",
                        statusBorder(s.status),
                        isActive ? "ring-2 ring-blue-500" : "",
                      ].join(" ")}
                      title={s.desc}
                    >
                      <div className="flex items-center justify-between">
                        <div className="font-medium">{s.title}</div>
                        <StatusPill status={s.status} />
                      </div>
                      {s.desc && <div className="text-xs opacity-70 mt-1 line-clamp-2">{s.desc}</div>}
                      {typeof s.latencyMs === "number" && (
                        <div className="text-[11px] mt-2">
                          Latency: <span className="font-medium">{fmtMs(s.latencyMs)}</span>
                        </div>
                      )}
                    </button>

                    {/* arrow to next */}
                    {idx < columns.length - 1 && (
                      <Arrow label={edgeLabel(data.edges, id, columns[idx + 1])} />
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* timeline summary */}
          <div className="mt-4">
            <LatencyBar steps={columns.map((id) => stepsById.get(id)!)} />
          </div>
        </section>

        {/* Side panel */}
        <aside className={`${compact ? "" : "col-span-4"} rounded-xl border dark:border-gray-800 p-3`}>
          <h3 className="text-sm font-semibold mb-2">Details</h3>
          {active ? (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="font-medium">{active.title}</div>
                <StatusPill status={active.status} />
              </div>
              {active.desc && <p className="text-sm opacity-80">{active.desc}</p>}
              {typeof active.latencyMs === "number" && (
                <div className="text-xs opacity-80">Step latency: <b>{fmtMs(active.latencyMs)}</b></div>
              )}
              {active.meta ? (
                <div className="mt-2">
                  <KeyValueTable obj={active.meta} />
                </div>
              ) : (
                <div className="text-xs opacity-60">No extra metadata.</div>
              )}
            </div>
          ) : (
            <div className="text-xs opacity-60">Click a step to inspect its metadata, checks and decisions.</div>
          )}
        </aside>
      </div>
    </div>
  );
}

/** ---------- Subcomponents ---------- */

function Arrow({ label }: { label?: string }) {
  return (
    <div className="h-10 flex items-center">
      <svg width="120" height="24" className="opacity-80">
        <defs>
          <marker id="arrow" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto">
            <polygon points="0 0, 10 3, 0 6" fill="#6B7280" />
          </marker>
        </defs>
        <line x1="0" y1="12" x2="100" y2="12" stroke="#6B7280" strokeWidth="2" markerEnd="url(#arrow)" />
        {label && (
          <text x="50" y="9" textAnchor="middle" fontSize="10" fill="#6B7280">{label}</text>
        )}
      </svg>
    </div>
  );
}

function LatencyBar({ steps }: { steps: FlowStep[] }) {
  const total = Math.max(1, steps.reduce((a, s) => a + (s.latencyMs ?? 0), 0));
  return (
    <div>
      <div className="text-xs font-medium mb-1 opacity-80">Latency breakdown</div>
      <div className="w-full h-3 rounded-full overflow-hidden bg-gray-200 dark:bg-gray-800 flex">
        {steps.map((s) => {
          const w = ((s.latencyMs ?? 0) / total) * 100;
          return (
            <div
              key={s.id}
              className={`${statusBg(s.status)} h-full`}
              style={{ width: `${w}%` }}
              title={`${s.title}: ${fmtMs(s.latencyMs ?? 0)}`}
            />
          );
        })}
      </div>
      <div className="text-[11px] mt-1 opacity-70 flex flex-wrap gap-2">
        {steps.map((s) => (
          <span key={s.id} className="inline-flex items-center gap-1">
            <span className={`w-3 h-3 rounded-sm ${statusBg(s.status)}`} />
            {s.title}
          </span>
        ))}
      </div>
    </div>
  );
}

function StatusPill({ status }: { status: FlowStatus }) {
  const cls =
    status === "ok" ? "bg-green-100 text-green-700" :
    status === "warn" ? "bg-yellow-100 text-yellow-800" :
    status === "fail" ? "bg-red-100 text-red-700" :
    "bg-gray-100 text-gray-700";
  const text =
    status === "ok" ? "OK" :
    status === "warn" ? "WARN" :
    status === "fail" ? "FAIL" : "PENDING";
  return <span className={`px-2 py-0.5 rounded-md text-[11px] font-semibold ${cls}`}>{text}</span>;
}

function KeyValueTable({ obj }: { obj: Record<string, any> }) {
  const keys = Object.keys(obj);
  return (
    <table className="w-full text-xs">
      <tbody>
        {keys.map((k) => (
          <tr key={k} className="border-t dark:border-gray-800">
            <td className="py-1 pr-2 font-medium align-top">{k}</td>
            <td className="py-1 text-gray-700 dark:text-gray-200 break-all">
              {renderVal(obj[k])}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function renderVal(v: any): React.ReactNode {
  if (v === null || v === undefined) return <span className="opacity-70">—</span>;
  if (typeof v === "object") return <pre className="text-[11px] whitespace-pre-wrap">{JSON.stringify(v, null, 2)}</pre>;
  if (typeof v === "number") return <span>{fmtNum(v)}</span>;
  return String(v);
}

/** ---------- Utilities & styling helpers ---------- */

function statusBorder(s: FlowStatus) {
  return s === "ok" ? "border-green-300 dark:border-green-700"
    : s === "warn" ? "border-yellow-300 dark:border-yellow-700"
    : s === "fail" ? "border-red-300 dark:border-red-700"
    : "border-gray-300 dark:border-gray-700";
}
function statusBg(s: FlowStatus) {
  return s === "ok" ? "bg-green-500"
    : s === "warn" ? "bg-yellow-500"
    : s === "fail" ? "bg-red-500"
    : "bg-gray-500";
}
function edgeLabel(edges: FlowEdge[], from: string, to: string) {
  return edges.find((e) => e.from === from && e.to === to)?.label;
}
function fmtNum(x: number) {
  try { return x.toLocaleString(); } catch { return String(x); }
}
function fmtMs(ms: number) {
  if (!ms) return "0 ms";
  if (ms < 1000) return `${ms} ms`;
  const s = ms / 1000;
  return `${s.toFixed(s >= 10 ? 0 : 1)} s`;
}