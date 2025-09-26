"use client";

import React, { useMemo, useState } from "react";

/* ================================
 * Types
 * ================================ */

export type ShockType = "abs" | "rel"; // abs = add value; rel = % change

export type Shock = {
  variable: string;   // e.g., "Revenue", "COGS", "EV/EBITDA"
  type: ShockType;    // "abs" or "rel"
  value: number;      // if rel, 0.10 = +10%; if abs, additive in same units
};

export type Scenario = {
  id: string;
  name: string;
  tag?: string;       // e.g., "Macro", "Company", "Downside"
  date?: string;      // ISO date string
  notes?: string;
  base: Record<string, number>;   // baseline metrics for all variables
  shocks: Shock[];                // overrides/perturbations
};

export interface ScenariosPanelProps {
  scenarios: Scenario[];
  title?: string;
  /** restrict displayed metrics (order matters) */
  metrics?: string[];
  /** called when a scenario is (re)computed */
  onRun?: (scenario: Scenario, result: Record<string, number>) => void;
}

/* ================================
 * Local UI (no deps)
 * ================================ */

const Card: React.FC<React.PropsWithChildren<{ className?: string }>> = ({ className, children }) => (
  <div className={`rounded-2xl border border-neutral-200/70 bg-white shadow ${className ?? ""}`}>{children}</div>
);

const CardHeader: React.FC<React.PropsWithChildren<{ className?: string }>> = ({ className, children }) => (
  <div className={`flex flex-wrap items-center justify-between gap-3 border-b px-4 py-3 ${className ?? ""}`}>
    {children}
  </div>
);

const CardTitle: React.FC<React.PropsWithChildren> = ({ children }) => (
  <h2 className="text-lg font-semibold">{children}</h2>
);

const CardContent: React.FC<React.PropsWithChildren<{ className?: string }>> = ({ className, children }) => (
  <div className={`px-4 py-3 ${className ?? ""}`}>{children}</div>
);

const Badge: React.FC<React.PropsWithChildren<{ tone?: "neutral" | "green" | "red" | "indigo" }>> = ({
  children,
  tone = "neutral",
}) => {
  const tones: Record<string, string> = {
    neutral: "bg-neutral-100 text-neutral-800",
    green: "bg-green-100 text-green-800",
    red: "bg-red-100 text-red-800",
    indigo: "bg-indigo-100 text-indigo-800",
  };
  return (
    <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${tones[tone]}`}>
      {children}
    </span>
  );
};

/* ================================
 * Helpers / math
 * ================================ */

const fmt = (x: number, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : "–");
const fmtPct = (x: number, d = 1) => (Number.isFinite(x) ? `${(x * 100).toFixed(d)}%` : "–");
const toDate = (s?: string) => (s ? new Date(s) : undefined);

function applyShocks(base: Record<string, number>, shocks: Shock[]): Record<string, number> {
  const out: Record<string, number> = { ...base };
  for (const s of shocks) {
    const cur = Number.isFinite(out[s.variable]) ? out[s.variable] : 0;
    out[s.variable] = s.type === "rel" ? cur * (1 + s.value) : cur + s.value;
  }
  return out;
}

// Simple horizontal bar pair (Base vs Scenario)
function PairBars({
  base,
  alt,
  maxWidth = 220,
  height = 12,
  colorBase = "#a3a3a3",
  colorAlt = "#0ea5e9",
}: {
  base: number;
  alt: number;
  maxWidth?: number;
  height?: number;
  colorBase?: string;
  colorAlt?: string;
}) {
  const max = Math.max(Math.abs(base), Math.abs(alt), 1e-9);
  const wb = Math.min(maxWidth, (Math.abs(base) / max) * maxWidth);
  const wa = Math.min(maxWidth, (Math.abs(alt) / max) * maxWidth);
  return (
    <svg width={maxWidth} height={height * 2 + 6}>
      <rect x={0} y={0} width={wb} height={height} fill={colorBase} opacity={0.6} />
      <rect x={0} y={height + 6} width={wa} height={height} fill={colorAlt} />
    </svg>
  );
}

/* ================================
 * Component
 * ================================ */

const ScenariosPanel: React.FC<ScenariosPanelProps> = ({
  scenarios,
  title = "Scenarios",
  metrics,
  onRun,
}) => {
  const [q, setQ] = useState("");
  const [sortKey, setSortKey] = useState<"name" | "tag" | "date" | "shocks">("date");
  const [sortAsc, setSortAsc] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(scenarios[0]?.id ?? null);

  // Derive the metric list in display order
  const metricKeys = useMemo(() => {
    if (metrics && metrics.length) return metrics;
    // otherwise union of all base keys, stable order by frequency then name
    const counts = new Map<string, number>();
    scenarios.forEach((s) => {
      Object.keys(s.base).forEach((k) => counts.set(k, (counts.get(k) ?? 0) + 1));
    });
    return Array.from(counts.entries())
      .sort((a, b) => (b[1] - a[1]) || a[0].localeCompare(b[0]))
      .map(([k]) => k)
      .slice(0, 12); // keep it compact
  }, [scenarios, metrics]);

  // Filter & sort scenario list
  const list = useMemo(() => {
    const fq = q.trim().toLowerCase();
    const arr = scenarios.filter(
      (s) =>
        !fq ||
        s.name.toLowerCase().includes(fq) ||
        (s.tag ?? "").toLowerCase().includes(fq) ||
        (s.notes ?? "").toLowerCase().includes(fq)
    );
    arr.sort((a, b) => {
      if (sortKey === "name") return sortAsc ? a.name.localeCompare(b.name) : b.name.localeCompare(a.name);
      if (sortKey === "tag") return sortAsc ? (a.tag ?? "").localeCompare(b.tag ?? "") : (b.tag ?? "").localeCompare(a.tag ?? "");
      if (sortKey === "shocks") return sortAsc ? a.shocks.length - b.shocks.length : b.shocks.length - a.shocks.length;
      // date (default)
      const ad = toDate(a.date)?.getTime() ?? 0;
      const bd = toDate(b.date)?.getTime() ?? 0;
      return sortAsc ? ad - bd : bd - ad;
    });
    return arr;
  }, [scenarios, q, sortKey, sortAsc]);

  const selected = useMemo(() => list.find((s) => s.id === selectedId) ?? list[0], [list, selectedId]);

  // Compute result for selected scenario
  const result = useMemo(() => {
    if (!selected) return {};
    const out = applyShocks(selected.base, selected.shocks);
    onRun?.(selected, out);
    return out;
  }, [selected, onRun]);

  const toggleSort = (k: typeof sortKey) => {
    if (k === sortKey) setSortAsc((s) => !s);
    else {
      setSortKey(k);
      setSortAsc(true);
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <input
          className="h-9 w-72 rounded-md border border-neutral-300 bg-white px-3 text-sm outline-none focus:border-neutral-500"
          placeholder="Search scenarios, tags, notes…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />
      </CardHeader>

      <CardContent className="grid grid-cols-1 gap-4 md:grid-cols-3">
        {/* List */}
        <div className="md:col-span-1 rounded-lg border border-neutral-200">
          <div className="flex items-center justify-between border-b px-3 py-2 text-xs font-semibold uppercase tracking-wide text-neutral-600">
            <button className="hover:underline" onClick={() => toggleSort("date")}>
              Date {sortKey === "date" && <span className="text-[10px]">{sortAsc ? "▲" : "▼"}</span>}
            </button>
            <button className="hover:underline" onClick={() => toggleSort("name")}>
              Name {sortKey === "name" && <span className="text-[10px]">{sortAsc ? "▲" : "▼"}</span>}
            </button>
            <button className="hover:underline" onClick={() => toggleSort("tag")}>
              Tag {sortKey === "tag" && <span className="text-[10px]">{sortAsc ? "▲" : "▼"}</span>}
            </button>
            <button className="hover:underline" onClick={() => toggleSort("shocks")}>
              Shocks {sortKey === "shocks" && <span className="text-[10px]">{sortAsc ? "▲" : "▼"}</span>}
            </button>
          </div>
          <ul className="max-h-[420px] divide-y overflow-auto">
            {list.map((s) => (
              <li
                key={s.id}
                className={`cursor-pointer px-3 py-2 text-sm hover:bg-neutral-50 ${
                  s.id === selected?.id ? "bg-neutral-50" : ""
                }`}
                onClick={() => setSelectedId(s.id)}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="truncate font-medium">{s.name}</div>
                  {s.tag && <Badge tone="indigo">{s.tag}</Badge>}
                </div>
                <div className="mt-0.5 flex items-center justify-between text-xs text-neutral-500">
                  <span>{s.date ?? "—"}</span>
                  <span>{s.shocks.length} shocks</span>
                </div>
              </li>
            ))}
            {list.length === 0 && (
              <li className="px-3 py-4 text-sm text-neutral-500">No scenarios match.</li>
            )}
          </ul>
        </div>

        {/* Details */}
        <div className="md:col-span-2 rounded-lg border border-neutral-200">
          {!selected ? (
            <div className="p-4 text-sm text-neutral-500">Select a scenario to view details.</div>
          ) : (
            <>
              <div className="border-b px-3 py-2">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="text-sm font-semibold">{selected.name}</div>
                  <div className="flex items-center gap-2">
                    {selected.tag && <Badge tone="indigo">{selected.tag}</Badge>}
                    <span className="text-xs text-neutral-500">{selected.date ?? "—"}</span>
                  </div>
                </div>
                {selected.notes && <div className="mt-1 text-sm text-neutral-700">{selected.notes}</div>}
              </div>

              <div className="grid grid-cols-1 gap-4 p-3 lg:grid-cols-2">
                {/* Metrics table */}
                <div className="rounded-md border p-2">
                  <div className="mb-1 text-xs uppercase text-neutral-500">Metrics (Base vs Scenario)</div>
                  <table className="min-w-full border-collapse text-sm">
                    <thead>
                      <tr className="border-b bg-neutral-50 text-left text-xs font-semibold uppercase tracking-wide text-neutral-600">
                        <th className="px-2 py-1">Metric</th>
                        <th className="px-2 py-1 text-right">Base</th>
                        <th className="px-2 py-1 text-right">Scenario</th>
                        <th className="px-2 py-1 text-right">Δ</th>
                        <th className="px-2 py-1">Chart</th>
                      </tr>
                    </thead>
                    <tbody>
                      {metricKeys.map((k) => {
                        const b = selected.base[k];
                        const a = (result as any)[k];
                        const delta = Number.isFinite(b) && Number.isFinite(a) ? a - b : NaN;
                        const rel = Number.isFinite(delta) && Math.abs(b) > 1e-12 ? delta / b : NaN;
                        return (
                          <tr key={k} className="border-b last:border-0">
                            <td className="px-2 py-1">{k}</td>
                            <td className="px-2 py-1 text-right font-mono tabular-nums">{fmt(b)}</td>
                            <td className="px-2 py-1 text-right font-mono tabular-nums">{fmt(a)}</td>
                            <td className="px-2 py-1 text-right font-mono tabular-nums">
                              {fmt(delta)} {Number.isFinite(rel) ? ` (${fmtPct(rel)})` : ""}
                            </td>
                            <td className="px-2 py-1">
                              {Number.isFinite(b) && Number.isFinite(a) ? (
                                <PairBars base={b} alt={a} />
                              ) : (
                                <span className="text-xs text-neutral-400">n/a</span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                {/* Shocks list */}
                <div className="rounded-md border p-2">
                  <div className="mb-1 text-xs uppercase text-neutral-500">Shocks</div>
                  <table className="min-w-full border-collapse text-sm">
                    <thead>
                      <tr className="border-b bg-neutral-50 text-left text-xs font-semibold uppercase tracking-wide text-neutral-600">
                        <th className="px-2 py-1">Variable</th>
                        <th className="px-2 py-1">Type</th>
                        <th className="px-2 py-1 text-right">Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selected.shocks.length === 0 ? (
                        <tr>
                          <td colSpan={3} className="px-2 py-4 text-center text-neutral-500">
                            No shocks defined.
                          </td>
                        </tr>
                      ) : (
                        selected.shocks.map((s, i) => (
                          <tr key={`${s.variable}-${i}`} className="border-b last:border-0">
                            <td className="px-2 py-1">{s.variable}</td>
                            <td className="px-2 py-1">
                              <Badge tone={s.type === "rel" ? "green" : "neutral"}>
                                {s.type === "rel" ? "Relative" : "Absolute"}
                              </Badge>
                            </td>
                            <td className="px-2 py-1 text-right font-mono tabular-nums">
                              {s.type === "rel" ? fmtPct(s.value) : fmt(s.value)}
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default ScenariosPanel;