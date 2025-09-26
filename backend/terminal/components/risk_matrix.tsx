// risk-matrix.tsx
import React, { useMemo, useRef, useState } from "react";

/**
 * Risk Matrix (Impact × Likelihood) — React + TypeScript + Tailwind
 * - Rows = Likelihood levels (e.g., Rare … Almost Certain)
 * - Columns = Impact levels (e.g., Minor … Catastrophic)
 * - Drag risks between cells; keyboard move with arrows
 * - Color-coded severity (green → amber → red)
 * - Search/filter, grouping chips, CSV export
 *
 * No external deps required.
 */

export type Risk = {
  id: string;
  title: string;
  owner?: string;
  tag?: string; // category / area
  likelihood: number; // row index (0..rows-1)
  impact: number; // col index (0..cols-1)
  notes?: string;
  status?: "open" | "mitigating" | "closed";
};

export interface RiskMatrixProps {
  rows: string[]; // Likelihood labels (top→bottom)
  cols: string[]; // Impact labels (left→right)
  risks: Risk[];
  /** Called when risks are moved/edited */
  onChange?: (next: Risk[]) => void;
  /** Optional id prefix for accessibility / deterministic keys */
  idBase?: string;
  /** Show counts in cell corners */
  showCounts?: boolean;
  /** Optional cell size (px) — matrix scales with labels dynamically */
  cellSize?: number;
  /** Optional filter presets (chips) */
  tags?: string[];
}

const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));

/** Smooth color ramp green → yellow → red (0..1) */
function severityColor(t: number) {
  // piecewise: green(0,153,74) -> yellow(255,193,7) -> red(239,68,68)
  const stops = [
    { t: 0, c: [0, 153, 74] },
    { t: 0.5, c: [255, 193, 7] },
    { t: 1, c: [239, 68, 68] },
  ];
  const a = stops.find((s) => t >= s.t) || stops[0];
  const b = stops.find((s) => s.t >= t) || stops[stops.length - 1];
  const u = (t - a.t) / Math.max(1e-9, b.t - a.t);
  const lerp = (x: number, y: number) => Math.round(x + (y - x) * u);
  const [r, g, bch] = [lerp(a.c[0], b.c[0]), lerp(a.c[1], b.c[1]), lerp(a.c[2], b.c[2])];
  return `rgb(${r}, ${g}, ${bch})`;
}

const pillCls = "inline-flex items-center gap-1 rounded-lg px-2 py-1 text-xs text-zinc-100 bg-zinc-800 border border-zinc-700 hover:bg-zinc-700 transition";

const RiskMatrix: React.FC<RiskMatrixProps> = ({
  rows,
  cols,
  risks,
  onChange,
  idBase = "risk-matrix",
  showCounts = true,
  cellSize = 120,
  tags,
}) => {
  const [query, setQuery] = useState("");
  const [activeTag, setActiveTag] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<"all" | "open" | "mitigating" | "closed">("all");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const rootRef = useRef<HTMLDivElement | null>(null);

  const R = rows.length;
  const C = cols.length;
  const maxScore = R * C;

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    return risks.filter((r) => {
      const matchesQ = !q || r.title.toLowerCase().includes(q) || r.owner?.toLowerCase().includes(q) || r.tag?.toLowerCase().includes(q);
      const matchesTag = !activeTag || r.tag === activeTag;
      const matchesStatus = statusFilter === "all" || r.status === statusFilter;
      return matchesQ && matchesTag && matchesStatus;
    });
  }, [risks, query, activeTag, statusFilter]);

  // bucketize by (row, col)
  const buckets = useMemo(() => {
    const m: Record<string, Risk[]> = {};
    for (const r of visible) {
      const key = `${r.likelihood}:${r.impact}`;
      (m[key] ||= []).push(r);
    }
    return m;
  }, [visible]);

  const moveRisk = (id: string, li: number, cj: number) => {
    const next = risks.map((r) => (r.id === id ? { ...r, likelihood: clamp(li, 0, R - 1), impact: clamp(cj, 0, C - 1) } : r));
    onChange?.(next);
  };

  const onDragStart = (risk: Risk) => (e: React.DragEvent) => {
    e.dataTransfer.setData("text/plain", risk.id);
    e.dataTransfer.effectAllowed = "move";
    setSelectedId(risk.id);
  };
  const onDropCell = (li: number, cj: number) => (e: React.DragEvent) => {
    e.preventDefault();
    const id = e.dataTransfer.getData("text/plain");
    if (!id) return;
    moveRisk(id, li, cj);
  };

  const onKeyDownRisk = (r: Risk) => (e: React.KeyboardEvent) => {
    // Arrow keys to move within grid
    let di = 0, dj = 0;
    if (e.key === "ArrowUp") di = -1;
    else if (e.key === "ArrowDown") di = 1;
    else if (e.key === "ArrowLeft") dj = -1;
    else if (e.key === "ArrowRight") dj = 1;
    else return;
    e.preventDefault();
    moveRisk(r.id, r.likelihood + di, r.impact + dj);
  };

  const exportCSV = () => {
    const header = ["id", "title", "owner", "tag", "status", "likelihood", "impact", "likelihood_label", "impact_label"].join(",");
    const lines = risks.map((r) =>
      [
        r.id,
        `"${r.title.replace(/"/g, '""')}"`,
        r.owner ?? "",
        r.tag ?? "",
        r.status ?? "",
        r.likelihood,
        r.impact,
        rows[r.likelihood] ?? "",
        cols[r.impact] ?? "",
      ].join(",")
    );
    const blob = new Blob([header + "\n" + lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "risk_matrix.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  const counts = (li: number, cj: number) => buckets[`${li}:${cj}`]?.length ?? 0;
  const cellScore = (li: number, cj: number) => (li + 1) * (cj + 1);
  const cellT = (li: number, cj: number) => (cellScore(li, cj) - 1) / (maxScore - 1 || 1);

  // Layout sizes
  const labelColW = Math.max(140, Math.min(220, Math.max(...rows.map((s) => s.length)) * 8 + 24));
  const labelRowH = 40;
  const gridW = C * cellSize;
  const gridH = R * cellSize;

  return (
    <div ref={rootRef} className="w-full text-zinc-100">
      {/* Controls */}
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search risks…"
          className="bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-amber-400/50"
        />
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value as any)}
          className="bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-2 text-sm"
        >
          <option value="all">All statuses</option>
          <option value="open">Open</option>
          <option value="mitigating">Mitigating</option>
          <option value="closed">Closed</option>
        </select>
        {tags && tags.length > 0 && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-400">Tags:</span>
            <button
              onClick={() => setActiveTag(null)}
              className={`px-2 py-1 rounded-md border ${activeTag ? "border-zinc-800 text-zinc-400" : "border-amber-400/40 text-amber-300"}`}
            >
              All
            </button>
            {tags.map((t) => (
              <button
                key={t}
                onClick={() => setActiveTag((x) => (x === t ? null : t))}
                className={`px-2 py-1 rounded-md border ${activeTag === t ? "border-amber-400/60 text-amber-200" : "border-zinc-800 text-zinc-400"}`}
              >
                {t}
              </button>
            ))}
          </div>
        )}
        <div className="ml-auto flex items-center gap-2">
          <button onClick={exportCSV} className="px-3 py-2 rounded-lg bg-zinc-800 border border-zinc-700 hover:bg-zinc-700 text-sm">
            Export CSV
          </button>
        </div>
      </div>

      {/* Matrix frame */}
      <div className="relative overflow-x-auto">
        {/* Column labels */}
        <div
          className="sticky top-0 z-10 grid"
          style={{
            marginLeft: labelColW,
            gridTemplateColumns: `repeat(${C}, ${cellSize}px)`,
            height: labelRowH,
          }}
        >
          {cols.map((c, j) => (
            <div key={`col-${j}`} className="flex items-end justify-center text-xs text-zinc-300 pb-1">
              {c}
            </div>
          ))}
        </div>

        <div className="flex">
          {/* Row labels */}
          <div style={{ width: labelColW }} className="select-none">
            <div style={{ height: labelRowH }} />
            {rows.map((r, i) => (
              <div key={`row-${i}`} style={{ height: cellSize }} className="flex items-center justify-end pr-3 text-xs text-zinc-300">
                {r}
              </div>
            ))}
          </div>

          {/* Grid */}
          <div
            className="rounded-xl border border-zinc-800 bg-zinc-950"
            style={{ width: gridW, height: gridH + labelRowH }}
          >
            <div style={{ height: labelRowH }} />
            <div
              className="grid"
              style={{ gridTemplateColumns: `repeat(${C}, ${cellSize}px)` }}
            >
              {rows.map((_, li) =>
                cols.map((__, cj) => {
                  const key = `${li}:${cj}`;
                  const bag = buckets[key] || [];
                  const t = cellT(li, cj);
                  const bg = severityColor(t);
                  return (
                    <div
                      key={`cell-${key}`}
                      onDragOver={(e) => e.preventDefault()}
                      onDrop={onDropCell(li, cj)}
                      className="relative border border-zinc-800/80"
                      style={{ width: cellSize, height: cellSize, background: `linear-gradient(180deg, ${bg} 0%, rgba(0,0,0,0.24) 100%)` }}
                    >
                      {/* counts */}
                      {showCounts && (
                        <div className="absolute top-1 left-1 text-[10px] font-medium text-zinc-100/90 px-1 rounded bg-zinc-900/40">
                          {counts(li, cj)}
                        </div>
                      )}
                      {/* severity badge */}
                      <div className="absolute top-1 right-1 text-[10px] text-zinc-100/90 px-1 rounded bg-zinc-900/40">
                        {(li + 1) * (cj + 1)}
                      </div>

                      {/* risk pills */}
                      <div className="absolute inset-0 p-2 overflow-auto">
                        <div className="flex flex-col gap-1">
                          {bag
                            .sort((a, b) => a.title.localeCompare(b.title))
                            .map((r) => (
                              <button
                                key={r.id}
                                draggable
                                onDragStart={onDragStart(r)}
                                onKeyDown={onKeyDownRisk(r)}
                                onClick={() => setSelectedId(r.id)}
                                className={`${pillCls} ${selectedId === r.id ? "ring-2 ring-amber-400/50" : ""}`}
                                title={`${r.title}${r.owner ? ` — ${r.owner}` : ""}`}
                                aria-label={`Risk ${r.title}`}
                              >
                                <span className="truncate max-w-[calc(${cellSize}px-80px)]">{r.title}</span>
                                {r.tag && <span className="rounded bg-zinc-700/70 px-1 text-[10px]">{r.tag}</span>}
                                {r.status && (
                                  <span
                                    className={`rounded px-1 text-[10px] ${
                                      r.status === "open"
                                        ? "bg-red-600/50"
                                        : r.status === "mitigating"
                                        ? "bg-amber-500/50"
                                        : "bg-emerald-600/50"
                                    }`}
                                  >
                                    {r.status}
                                  </span>
                                )}
                              </button>
                            ))}
                        </div>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="mt-3 flex items-center gap-3">
        <span className="text-xs text-zinc-400">Severity</span>
        <div className="h-2 w-48 rounded" style={{ background: "linear-gradient(90deg, rgb(0,153,74), rgb(255,193,7), rgb(239,68,68))" }} />
        <div className="text-[10px] text-zinc-400">low</div>
        <div className="text-[10px] text-zinc-400 ml-auto">
          Tip: drag pills between cells, or select a pill and use arrow keys to move it.
        </div>
      </div>
    </div>
  );
};

export default RiskMatrix;