"use client";

import React, { useMemo, useRef, useState } from "react";

/* ================================
 * Types
 * ================================ */

export type PnlPoint = {
  date: string | Date; // ISO or Date
  pnl: number;         // cumulative PnL (use same unit across series)
};

type Props = {
  data: PnlPoint[];
  title?: string;
  height?: number;          // default 260
  showDrawdown?: boolean;   // shade drawdowns vs running peak
  asPercent?: boolean;      // format y as %
  baseline?: number;        // zero-line baseline (default 0)
  benchmark?: PnlPoint[];   // optional comparison line
};

/* ================================
 * Small helpers
 * ================================ */

const toDate = (d: string | Date) => (d instanceof Date ? d : new Date(d));
const fmtPct = (x: number, d = 2) => `${(x * 100).toFixed(d)}%`;
const fmt = (x: number, d = 2) => x.toFixed(d);

function extent<T>(arr: T[], acc: (t: T) => number): [number, number] {
  let lo = Number.POSITIVE_INFINITY, hi = Number.NEGATIVE_INFINITY;
  for (const v of arr) {
    const x = acc(v);
    if (!Number.isFinite(x)) continue;
    if (x < lo) lo = x;
    if (x > hi) hi = x;
  }
  if (lo === Number.POSITIVE_INFINITY) lo = 0, hi = 1;
  if (lo === hi) { lo -= 1; hi += 1; }
  return [lo, hi];
}

function linePath(points: { x: number; y: number }[]) {
  if (!points.length) return "";
  let d = `M ${points[0].x} ${points[0].y}`;
  for (let i = 1; i < points.length; i++) d += ` L ${points[i].x} ${points[i].y}`;
  return d;
}

function niceTicks(min: number, max: number, count = 5) {
  // simple nice ticks (not d3-nice, but good enough)
  const span = max - min;
  const step = Math.pow(10, Math.floor(Math.log10(span / count)));
  const err = (count * step) / span;
  const mult = err <= 0.15 ? 10 : err <= 0.35 ? 5 : err <= 0.75 ? 2 : 1;
  const niceStep = step * mult;
  const niceMin = Math.floor(min / niceStep) * niceStep;
  const niceMax = Math.ceil(max / niceStep) * niceStep;
  const ticks: number[] = [];
  for (let v = niceMin; v <= niceMax + 1e-12; v += niceStep) ticks.push(Number(v.toFixed(12)));
  return { ticks, niceMin, niceMax, niceStep };
}

function bisectDate(xs: number[], x: number) {
  // return index of closest point to x (left-biased)
  let lo = 0, hi = xs.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs(xs[lo] - x) > Math.abs(xs[lo - 1] - x)) return lo - 1;
  return lo;
}

function computeDrawdown(series: number[]) {
  const dd: number[] = [];
  let peak = -Infinity;
  for (const v of series) {
    peak = Math.max(peak, v);
    dd.push(v - peak); // negative or zero
  }
  return dd;
}

/* ================================
 * Component
 * ================================ */

const StrategyPnlChart: React.FC<Props> = ({
  data,
  title = "Strategy PnL",
  height = 260,
  showDrawdown = true,
  asPercent = false,
  baseline = 0,
  benchmark,
}) => {
  const margin = { top: 28, right: 18, bottom: 28, left: 56 };
  const [w, setW] = useState<number>(720);
  const ref = useRef<HTMLDivElement>(null);

  // Resize observer (simple)
  React.useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver((entries) => {
      const cr = entries[0].contentRect;
      setW(Math.max(320, cr.width));
    });
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);

  // Normalize data
  const main = useMemo(() => data
    .map(d => ({ t: +toDate(d.date), v: d.pnl }))
    .sort((a, b) => a.t - b.t), [data]);

  const bench = useMemo(() => (benchmark ?? [])
    .map(d => ({ t: +toDate(d.date), v: d.pnl }))
    .sort((a, b) => a.t - b.t), [benchmark]);

  const innerW = Math.max(10, w - margin.left - margin.right);
  const innerH = Math.max(10, height - margin.top - margin.bottom);

  // Domains
  const xDomain: [number, number] = useMemo(() => {
    const xs = [...main.map(p => p.t), ...bench.map(p => p.t)];
    if (xs.length === 0) return [Date.now() - 86400000 * 30, Date.now()];
    return [Math.min(...xs), Math.max(...xs)];
  }, [main, bench]);

  const yDomain: [number, number] = useMemo(() => {
    const ys = [...main.map(p => p.v), ...bench.map(p => p.v), baseline];
    const [lo, hi] = extent(ys, x => x);
    // pad 5%
    const pad = (hi - lo) * 0.05;
    return [lo - pad, hi + pad];
  }, [main, bench, baseline]);

  // Scales
  const xScale = (t: number) =>
    margin.left + ((t - xDomain[0]) / Math.max(1, xDomain[1] - xDomain[0])) * innerW;
  const yScale = (v: number) =>
    margin.top + innerH - ((v - yDomain[0]) / Math.max(1e-9, yDomain[1] - yDomain[0])) * innerH;

  // Paths
  const mainPts = main.map(p => ({ x: xScale(p.t), y: yScale(p.v) }));
  const benchPts = bench.map(p => ({ x: xScale(p.t), y: yScale(p.v) }));
  const ddSeries = showDrawdown ? computeDrawdown(main.map(m => m.v)) : [];
  const ddAreaPath = useMemo(() => {
    if (!showDrawdown || main.length === 0) return "";
    // area from peak line (0 drawdown) down to dd (negative values)
    const baseY = yScale(0); // drawdown relative to peak -> base at 0
    const ptsTop = main.map((m) => ({ x: xScale(m.t), y: yScale(0) }));
    const ptsDown = main.map((m, i) => ({ x: xScale(m.t), y: yScale(ddSeries[i]) }));
    let d = "";
    for (let i = 0; i < ptsTop.length; i++) {
      d += i === 0 ? `M ${ptsTop[i].x} ${ptsTop[i].y}` : ` L ${ptsTop[i].x} ${ptsTop[i].y}`;
    }
    for (let i = ptsDown.length - 1; i >= 0; i--) {
      d += ` L ${ptsDown[i].x} ${ptsDown[i].y}`;
    }
    d += " Z";
    return d;
  }, [showDrawdown, main, ddSeries, xScale, yScale]);

  // Ticks
  const { ticks: yTicks } = niceTicks(yDomain[0], yDomain[1], 5);

  // Hover
  const [hoverX, setHoverX] = useState<number | null>(null);
  const [hoverIdx, hoverData] = useMemo(() => {
    if (hoverX == null || main.length === 0) return [null, null] as const;
    const idx = bisectDate(main.map(p => p.t), hoverX);
    return [idx, main[idx]] as const;
  }, [hoverX, main]);

  const formatY = (val: number) => (asPercent ? fmtPct(val) : fmt(val));

  return (
    <div ref={ref} className="w-full rounded-2xl border border-neutral-200 bg-white shadow">
      {/* Header */}
      <div className="flex items-center justify-between border-b px-4 py-3">
        <h2 className="text-lg font-semibold">{title}</h2>
        <div className="text-xs text-neutral-600">
          {main.length ? new Date(main[0].t).toLocaleDateString() : "—"} —{" "}
          {main.length ? new Date(main[main.length - 1].t).toLocaleDateString() : "—"}
        </div>
      </div>

      {/* Chart */}
      <div className="relative">
        <svg width={w} height={height}>
          {/* Grid / axes */}
          {/* Horizontal grid lines */}
          {yTicks.map((yt, i) => (
            <g key={i}>
              <line
                x1={margin.left}
                x2={margin.left + innerW}
                y1={yScale(yt)}
                y2={yScale(yt)}
                stroke="#e5e7eb"
                strokeDasharray="3,3"
              />
              <text
                x={margin.left - 8}
                y={yScale(yt)}
                textAnchor="end"
                dominantBaseline="middle"
                className="fill-neutral-500 text-[11px]"
              >
                {formatY(yt)}
              </text>
            </g>
          ))}
          {/* X-axis ticks (quarterly-ish by spacing) */}
          {main.length > 1 && (() => {
            const ticks = 6;
            const step = (xDomain[1] - xDomain[0]) / ticks;
            return Array.from({ length: ticks + 1 }, (_, i) => xDomain[0] + i * step).map((t, i) => (
              <g key={i}>
                <line
                  x1={xScale(t)} x2={xScale(t)}
                  y1={margin.top + innerH} y2={margin.top + innerH + 4}
                  stroke="#9ca3af"
                />
                <text
                  x={xScale(t)} y={margin.top + innerH + 16}
                  textAnchor="middle" className="fill-neutral-500 text-[11px]"
                >
                  {new Date(t).toLocaleDateString(undefined, { year: "2-digit", month: "short" })}
                </text>
              </g>
            ));
          })()}

          {/* Baseline (0 or provided) */}
          <line
            x1={margin.left}
            x2={margin.left + innerW}
            y1={yScale(baseline)}
            y2={yScale(baseline)}
            stroke="#d1d5db"
          />

          {/* Drawdown area (relative to running peak) */}
          {showDrawdown && ddAreaPath && (
            <path d={ddAreaPath} fill="#fecaca" fillOpacity={0.45} />
          )}

          {/* Benchmark line */}
          {benchPts.length > 1 && (
            <path
              d={linePath(benchPts)}
              fill="none"
              stroke="#9ca3af"
              strokeWidth={1.25}
              strokeDasharray="4 3"
            />
          )}

          {/* Main line */}
          {mainPts.length > 1 && (
            <path d={linePath(mainPts)} fill="none" stroke="#0ea5e9" strokeWidth={1.8} />
          )}

          {/* Hover vertical + marker */}
          {hoverIdx != null && hoverData && (
            <>
              <line
                x1={xScale(hoverData.t)}
                x2={xScale(hoverData.t)}
                y1={margin.top}
                y2={margin.top + innerH}
                stroke="#94a3b8"
                strokeDasharray="3 3"
              />
              <circle cx={xScale(hoverData.t)} cy={yScale(hoverData.v)} r={3.2} fill="#0ea5e9" />
            </>
          )}
        </svg>

        {/* Hover overlay & tooltip */}
        <div
          className="absolute inset-0"
          onMouseMove={(e) => {
            const rect = (e.currentTarget as HTMLDivElement).getBoundingClientRect();
            const x = e.clientX - rect.left;
            // map screen x to domain time
            const xt = xDomain[0] + ((x - margin.left) / Math.max(1, innerW)) * (xDomain[1] - xDomain[0]);
            setHoverX(xt);
          }}
          onMouseLeave={() => setHoverX(null)}
        />

        {hoverIdx != null && hoverData && (
          <div
            className="pointer-events-none absolute rounded-md border border-neutral-300 bg-white px-2 py-1 text-xs shadow"
            style={{
              left: Math.min(
                Math.max(margin.left, xScale(hoverData.t) + 8),
                w - 160
              ),
              top: margin.top + 8,
            }}
          >
            <div className="font-medium">
              {new Date(hoverData.t).toLocaleDateString(undefined, {
                year: "numeric",
                month: "short",
                day: "2-digit",
              })}
            </div>
            <div className="flex items-center gap-2">
              <span className="inline-block h-2 w-2 rounded-full" style={{ background: "#0ea5e9" }} />
              PnL: <span className="font-mono">{formatY(hoverData.v)}</span>
            </div>
            {bench.length > 0 && (() => {
              const idxB = bisectDate(bench.map(b => b.t), hoverData.t);
              const bv = bench[idxB]?.v;
              if (Number.isFinite(bv)) {
                return (
                  <div className="mt-1 flex items-center gap-2">
                    <span className="inline-block h-2 w-2 rounded-full bg-neutral-400" />
                    Bench: <span className="font-mono">{formatY(bv!)}</span>
                  </div>
                );
              }
              return null;
            })()}
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 px-4 py-2 text-xs text-neutral-600">
        <span className="inline-flex items-center gap-2">
          <span className="inline-block h-2 w-4 rounded-sm" style={{ background: "#0ea5e9" }} />
          Strategy
        </span>
        {bench.length > 0 && (
          <span className="inline-flex items-center gap-2">
            <span
              className="inline-block h-2 w-4 rounded-sm"
              style={{ background: "#9ca3af" }}
            />
            Benchmark
          </span>
        )}
        {showDrawdown && (
          <span className="inline-flex items-center gap-2">
            <span className="inline-block h-2 w-4 rounded-sm bg-red-300" />
            Drawdown (vs peak)
          </span>
        )}
      </div>
    </div>
  );
};

export default StrategyPnlChart;