"use client";

import React, { useMemo, useState } from "react";

/** Inputs: labels.length === matrix.length === matrix[i].length  */
export type CorrMatrixProps = {
  title?: string;
  labels: string[];
  /** values in [-1, 1] */
  matrix: number[][];
  /** show numeric values in cells */
  showValues?: boolean;
  /** fixed cell size in px; otherwise auto based on container */
  cellSize?: number; // e.g., 22
  /** callback on cell click */
  onSelect?: (rowIndex: number, colIndex: number, value: number) => void;
};

type OrderMode = "original" | "avg-corr" | "variance" | "alphabetical";

/* ---------------- Helpers ---------------- */

const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));

/** -1..+1 → color ramp (blue → white → red) */
function heatColor(x: number) {
  const t = (clamp(x, -1, 1) + 1) / 2; // 0..1
  const r = Math.round(255 * t);
  const b = Math.round(255 * (1 - t));
  const g = 235 - Math.round(90 * Math.abs(x)); // slight desat
  return `rgb(${r},${g},${b})`;
}

function formatVal(x: number, d = 2) {
  if (!Number.isFinite(x)) return "–";
  return x.toFixed(d);
}

function computeAvgRow(m: number[][]) {
  return m.map((row, i) => {
    const s = row.reduce((a, v, j) => a + (i === j ? 0 : v), 0);
    return s / Math.max(1, row.length - 1);
  });
}

function computeVarianceRow(m: number[][]) {
  return m.map((row, i) => {
    const vals = row.filter((_, j) => j !== i);
    const mu = vals.reduce((a, v) => a + v, 0) / Math.max(1, vals.length);
    const v = vals.reduce((a, v) => a + (v - mu) ** 2, 0) / Math.max(1, vals.length);
    return v;
  });
}

function reorderMatrix(labels: string[], m: number[][], mode: OrderMode) {
  const n = labels.length;
  const idxs = Array.from({ length: n }, (_, i) => i);

  if (mode === "alphabetical") {
    idxs.sort((a, b) => labels[a].localeCompare(labels[b]));
  } else if (mode === "avg-corr") {
    const scores = computeAvgRow(m);
    idxs.sort((a, b) => scores[b] - scores[a]); // higher avg corr first (cluster-ish)
  } else if (mode === "variance") {
    const scores = computeVarianceRow(m);
    idxs.sort((a, b) => scores[b] - scores[a]); // more varied rows first
  } // original = identity

  const L = idxs.map((i) => labels[i]);
  const M = idxs.map((i) => idxs.map((j) => m[i][j]));
  return { labels: L, matrix: M, order: idxs };
}

function toCSV(labels: string[], m: number[][]) {
  const header = ["", ...labels].join(",");
  const rows = labels.map((lab, i) => [lab, ...m[i].map((v) => formatVal(v, 4))].join(","));
  return [header, ...rows].join("\n");
}

/* ---------------- Component ---------------- */

const CorrelationMatrix: React.FC<CorrMatrixProps> = ({
  title = "Correlation Matrix",
  labels,
  matrix,
  showValues = false,
  cellSize,
  onSelect,
}) => {
  const [orderMode, setOrderMode] = useState<OrderMode>("original");
  const [hover, setHover] = useState<{ r: number; c: number; v: number } | null>(null);

  // Validate shape
  const valid = useMemo(() => {
    const n = labels.length;
    return n > 0 && matrix.length === n && matrix.every((row) => row.length === n);
  }, [labels, matrix]);

  const { labels: L, matrix: M } = useMemo(
    () => (valid ? reorderMatrix(labels, matrix, orderMode) : { labels, matrix, order: [] }),
    [labels, matrix, orderMode, valid]
  );

  const n = L.length;
  const size = cellSize ?? (n <= 18 ? 26 : n <= 28 ? 22 : n <= 40 ? 18 : 14);
  const pad = 8;
  const gridW = n * size;
  const gridH = n * size;

  const legend = useMemo(
    () => Array.from({ length: 11 }, (_, i) => -1 + (i * 2) / 10), // [-1,-0.8,...,1]
    []
  );

  const downloadCSV = () => {
    const csv = toCSV(L, M);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${title.replace(/\s+/g, "_").toLowerCase()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="w-full rounded-2xl border border-neutral-200 bg-white shadow">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b px-4 py-3">
        <h2 className="text-lg font-semibold">{title}</h2>
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <label className="flex items-center gap-2">
            Order:
            <select
              value={orderMode}
              onChange={(e) => setOrderMode(e.target.value as OrderMode)}
              className="h-8 rounded-md border border-neutral-300 px-2"
            >
              <option value="original">Original</option>
              <option value="avg-corr">By Avg Corr</option>
              <option value="variance">By Variance</option>
              <option value="alphabetical">Alphabetical</option>
            </select>
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={showValues}
              onChange={() => {
                // not controlled externally: quick toggle
                const ev = new Event("toggleValues");
                window.dispatchEvent(ev);
              }}
              onClick={(e) => e.preventDefault()}
            />
            <span className="text-neutral-600">Values</span>
          </label>
          <button
            onClick={downloadCSV}
            className="rounded-md border border-neutral-300 bg-neutral-50 px-3 py-1.5 text-neutral-700 hover:bg-neutral-100"
          >
            Export CSV
          </button>
        </div>
      </div>

      {/* Matrix */}
      {valid ? (
        <div className="flex w-full overflow-auto">
          {/* Y labels */}
          <div className="sticky left-0 z-10 bg-white/80 backdrop-blur supports-[backdrop-filter]:bg-white/60">
            <table className="border-collapse">
              <tbody>
                {L.map((lab, r) => (
                  <tr key={`yl-${r}`} style={{ height: size }}>
                    <td className="pr-2 text-right text-[11px] text-neutral-600 whitespace-nowrap">{lab}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Grid + top labels in one scrollable table */}
          <div className="overflow-auto">
            {/* Top labels */}
            <div className="sticky top-0 z-10 bg-white/80 backdrop-blur supports-[backdrop-filter]:bg-white/60">
              <table className="border-collapse">
                <thead>
                  <tr>
                    {L.map((lab, c) => (
                      <th
                        key={`xl-${c}`}
                        style={{ width: size }}
                        className="px-1 pb-1 text-center text-[11px] text-neutral-600 whitespace-nowrap"
                      >
                        {lab}
                      </th>
                    ))}
                  </tr>
                </thead>
              </table>
            </div>

            {/* Grid */}
            <div>
              <table className="border-collapse">
                <tbody>
                  {M.map((row, r) => (
                    <tr key={`r-${r}`} style={{ height: size }}>
                      {row.map((v, c) => {
                        const color = heatColor(v);
                        const isDiag = r === c;
                        const showNum = showValues || false;
                        return (
                          <td key={`c-${r}-${c}`} style={{ width: size }}>
                            <div
                              title={`${L[r]} × ${L[c]} = ${formatVal(v, 3)}`}
                              onMouseEnter={() => setHover({ r, c, v })}
                              onMouseLeave={() => setHover(null)}
                              onClick={() => onSelect?.(r, c, v)}
                              className={`flex h-full w-full items-center justify-center rounded ${isDiag ? "outline outline-1 outline-neutral-300" : ""}`}
                              style={{ background: color, height: size }}
                            >
                              {showNum && (
                                <span className="select-none text-[10px] font-medium" style={{ color: Math.abs(v) > 0.6 ? "#fff" : "#0f172a" }}>
                                  {formatVal(v, 2)}
                                </span>
                              )}
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Legend */}
            <div className="flex items-center gap-2 px-3 py-2 text-[11px] text-neutral-600">
              <span>-1</span>
              <div className="flex h-2 w-40 overflow-hidden rounded">
                {legend.map((x, i) => (
                  <div key={i} className="h-2 flex-1" style={{ background: heatColor(x) }} />
                ))}
              </div>
              <span>+1</span>
              {hover && (
                <span className="ml-4 text-neutral-800">
                  {L[hover.r]} × {L[hover.c]} = <span className="font-mono">{formatVal(hover.v, 3)}</span>
                </span>
              )}
            </div>
          </div>
        </div>
      ) : (
        <div className="px-4 py-6 text-sm text-red-600">Invalid matrix shape: labels and matrix must be square and aligned.</div>
      )}
    </div>
  );
};

export default CorrelationMatrix;