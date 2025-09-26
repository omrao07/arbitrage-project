"use client";

import React, { useMemo, useState } from "react";

export type HeatmapProps = {
  title?: string;
  rows: string[];
  cols: string[];
  data: number[][]; // shape = rows.length × cols.length
  showValues?: boolean;
  cellSize?: number; // px size of each square (default auto)
  diverging?: boolean; // true = -1..1 scale; false = 0..max scale
  onSelect?: (r: number, c: number, v: number) => void;
};

/* ---------- Helpers ---------- */

const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));

function interpolateColor(t: number, diverging: boolean) {
  if (diverging) {
    // diverging [-1,1] → blue→white→red
    const tt = (t + 1) / 2;
    const r = Math.round(255 * tt);
    const g = Math.round(255 * (1 - Math.abs(t)) * 0.9 + 180 * Math.abs(t));
    const b = Math.round(255 * (1 - tt));
    return `rgb(${r},${g},${b})`;
  } else {
    // sequential [0,1] → white→blue
    const r = Math.round(240 - 140 * t);
    const g = Math.round(240 - 140 * t);
    const b = Math.round(255 - 200 * t);
    return `rgb(${r},${g},${b})`;
  }
}

function formatVal(x: number, d = 2) {
  if (!Number.isFinite(x)) return "–";
  return x.toFixed(d);
}

/* ---------- Component ---------- */

const Heatmap: React.FC<HeatmapProps> = ({
  title = "Heatmap",
  rows,
  cols,
  data,
  showValues = false,
  cellSize,
  diverging = false,
  onSelect,
}) => {
  const [hover, setHover] = useState<{ r: number; c: number; v: number } | null>(null);

  const { min, max } = useMemo(() => {
    let lo = Infinity,
      hi = -Infinity;
    data.forEach((row) =>
      row.forEach((v) => {
        if (Number.isFinite(v)) {
          lo = Math.min(lo, v);
          hi = Math.max(hi, v);
        }
      })
    );
    if (!Number.isFinite(lo)) lo = 0;
    if (!Number.isFinite(hi)) hi = 1;
    return { min: lo, max: hi };
  }, [data]);

  const nR = rows.length,
    nC = cols.length;
  const size = cellSize ?? (nC <= 15 ? 28 : nC <= 25 ? 22 : 16);

  const legendVals = useMemo(() => {
    if (diverging) return [-1, -0.5, 0, 0.5, 1];
    return [min, (min + max) / 2, max];
  }, [diverging, min, max]);

  return (
    <div className="w-full rounded-2xl border border-neutral-200 bg-white shadow">
      {/* Header */}
      <div className="flex items-center justify-between border-b px-4 py-3">
        <h2 className="text-lg font-semibold">{title}</h2>
        {hover && (
          <div className="text-xs text-neutral-700">
            {rows[hover.r]} × {cols[hover.c]} ={" "}
            <span className="font-mono">{formatVal(hover.v)}</span>
          </div>
        )}
      </div>

      {/* Grid */}
      <div className="overflow-auto">
        <table className="border-collapse">
          <thead>
            <tr>
              <th />
              {cols.map((c, j) => (
                <th
                  key={j}
                  className="px-1 text-center text-[11px] text-neutral-600 whitespace-nowrap"
                  style={{ height: size }}
                >
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, i) => (
              <tr key={i}>
                <td className="pr-2 text-right text-[11px] text-neutral-600 whitespace-nowrap">
                  {rows[i]}
                </td>
                {row.map((v, j) => {
                  const t = diverging
                    ? clamp(v, -1, 1)
                    : clamp((v - min) / Math.max(1e-9, max - min), 0, 1);
                  const color = interpolateColor(t, diverging);
                  return (
                    <td key={j}>
                      <div
                        className="flex items-center justify-center rounded cursor-pointer"
                        style={{ width: size, height: size, background: color }}
                        onClick={() => onSelect?.(i, j, v)}
                        onMouseEnter={() => setHover({ r: i, c: j, v })}
                        onMouseLeave={() => setHover(null)}
                        title={`${rows[i]} × ${cols[j]} = ${formatVal(v)}`}
                      >
                        {showValues && (
                          <span
                            className="text-[10px] font-medium select-none"
                            style={{
                              color:
                                diverging && Math.abs(v) > 0.6
                                  ? "#fff"
                                  : "hsl(222, 47%, 11%)",
                            }}
                          >
                            {formatVal(v, 1)}
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
      <div className="flex items-center gap-2 px-4 py-2 text-[11px] text-neutral-600">
        {legendVals.map((lv, i) => (
          <React.Fragment key={i}>
            <span>{formatVal(lv, diverging ? 1 : 2)}</span>
            {i < legendVals.length - 1 && (
              <div
                className="h-2 flex-1 rounded"
                style={{
                  background: `linear-gradient(to right, ${interpolateColor(
                    diverging ? legendVals[i] : (lv - min) / (max - min),
                    diverging
                  )}, ${interpolateColor(
                    diverging ? legendVals[i + 1] : (legendVals[i + 1] - min) / (max - min),
                    diverging
                  )})`,
                }}
              />
            )}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};

export default Heatmap;