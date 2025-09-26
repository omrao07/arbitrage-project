"use client";

import React, { useMemo } from "react";

/** Input: map of ticker -> numeric series (same length for all). */
export type SeriesMap = Record<string, number[]>;

export type CorrelationMatrixProps = {
  series: SeriesMap;
  decimals?: number;          // default 2
  height?: number;            // default 480
  onCellClick?: (row: string, col: string, rho: number) => void;
};

/* ---------- math helpers (pure TS) ---------- */
function pearson(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  if (n === 0) return NaN;

  let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i], y = b[i];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    sx += x; sy += y;
    sxx += x * x; syy += y * y;
    sxy += x * y;
  }
  const cov = sxy - (sx * sy) / n;
  const vx  = sxx - (sx * sx) / n;
  const vy  = syy - (sy * sy) / n;
  const denom = Math.sqrt(vx * vy);
  if (!Number.isFinite(denom) || denom === 0) return NaN;
  return Math.max(-1, Math.min(1, cov / denom));
}

/** Build a symmetric correlation matrix from series map. */
function buildMatrix(series: SeriesMap): { keys: string[]; values: number[][] } {
  const keys = Object.keys(series);
  const m = keys.length;
  const values: number[][] = Array.from({ length: m }, () => Array(m).fill(0));
  for (let i = 0; i < m; i++) {
    values[i][i] = 1;
    for (let j = i + 1; j < m; j++) {
      const rho = pearson(series[keys[i]], series[keys[j]]);
      values[i][j] = rho;
      values[j][i] = rho;
    }
  }
  return { keys, values };
}

/* ---------- color scale (-1 blue → 0 gray → +1 red) ---------- */
function colorFor(rho: number): string {
  if (!Number.isFinite(rho)) return "#2a2a2a";
  const t = (rho + 1) / 2; // 0..1
  const r = Math.round(255 * t);
  const b = Math.round(255 * (1 - t));
  const g = Math.round(255 * (0.6 - Math.abs(rho) * 0.6)); // desaturate mid
  return `rgb(${r},${g},${b})`;
}

/* ---------- component ---------- */
export default function CorrelationMatrix({
  series,
  decimals = 2,
  height = 480,
  onCellClick,
}: CorrelationMatrixProps) {
  const { keys, values } = useMemo(() => buildMatrix(series), [series]);
  const size = keys.length;
  const cell = Math.max(24, Math.min(64, Math.floor((height - 80) / Math.max(1, size)))); // px
  const gridW = cell * size;

  return (
    <div className="w-full" style={{ color: "#ddd" }}>
      <div className="flex items-end justify-between mb-2">
        <div className="text-sm opacity-80">Correlation Matrix (ρ)</div>
        <Legend />
      </div>

      {/* header labels */}
      <div className="overflow-auto rounded border border-[#333]" style={{ maxHeight: height }}>
        <div style={{ width: gridW + 140 }} className="relative">
          {/* top labels */}
          <div className="sticky top-0 z-10 pl-[140px] bg-[#0b0b0b]">
            <div className="grid" style={{ gridTemplateColumns: `repeat(${size}, ${cell}px)` }}>
              {keys.map(k => (
                <div key={`top-${k}`} className="text-xs text-center p-1 truncate">{k}</div>
              ))}
            </div>
          </div>

          {/* left labels + grid */}
          <div className="flex">
            {/* left labels */}
            <div className="sticky left-0 z-10 w-[140px] bg-[#0b0b0b]">
              {keys.map((k, i) => (
                <div
                  key={`left-${k}`}
                  style={{ height: cell }}
                  className="text-xs flex items-center pl-2 border-b border-[#222] truncate"
                >
                  {k}
                </div>
              ))}
            </div>

            {/* heatmap grid */}
            <div>
              {values.map((row, i) => (
                <div
                  key={`row-${i}`}
                  className="grid"
                  style={{ gridTemplateColumns: `repeat(${size}, ${cell}px)` }}
                >
                  {row.map((rho, j) => (
                    <button
                      key={`cell-${i}-${j}`}
                      title={`${keys[i]} × ${keys[j]} — ρ=${Number(rho).toFixed(decimals)}`}
                      onClick={() => onCellClick?.(keys[i], keys[j], rho)}
                      className="border border-[#222] focus:outline-none"
                      style={{
                        height: cell,
                        background: colorFor(rho),
                        cursor: onCellClick ? "pointer" : "default",
                      }}
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* numbers overlay toggle-able: uncomment if you want values printed in cells */}
      {/* <NumbersOverlay keys={keys} values={values} cell={cell} decimals={decimals} /> */}
    </div>
  );
}

/* ---------- optional legend ---------- */
function Legend() {
  const stops = [-1, -0.5, 0, 0.5, 1];
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="opacity-70">-1</span>
      <div className="h-3 w-32 rounded overflow-hidden flex">
        {Array.from({ length: 32 }).map((_, i) => {
          const t = i / 31;
          const rho = t * 2 - 1;
          return <div key={i} style={{ width: "3.125%", background: colorFor(rho) }} />;
        })}
      </div>
      <span className="opacity-70">+1</span>
    </div>
  );
}

/* ---------- (optional) render numeric labels inside each cell ----------
function NumbersOverlay({
  keys, values, cell, decimals
}: { keys: string[]; values: number[][]; cell: number; decimals: number; }) {
  return (
    <div className="pointer-events-none -mt-[calc(100%)] relative">
      {values.map((row, i) => (
        <div key={`nr-${i}`} className="grid" style={{ gridTemplateColumns: `repeat(${keys.length}, ${cell}px)` }}>
          {row.map((rho, j) => (
            <div key={`nrc-${i}-${j}`} style={{ height: cell }} className="text-[10px] flex items-center justify-center">
              {Number(rho).toFixed(decimals)}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
----------------------------------------------------------------------- */