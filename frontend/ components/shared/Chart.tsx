"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";

/* ========= Types ========= */
export type XY = { x: number | string | Date; y: number | null };

export type Series = {
  id: string;
  type?: "line" | "area" | "bar";
  color?: string;
  data: XY[];
};

export type ChartProps = {
  title?: string;
  height?: number;            // default 280
  series: Series[];           // at least one
  xType?: "time" | "category";
  grid?: boolean;             // default true
  yFormat?: (v: number) => string;
  xFormat?: (x: number | string | Date) => string;
  baseline?: number;          // draw horizontal line at value
};

/* ========= Utils ========= */
const isDateLike = (v: any) =>
  v instanceof Date || (typeof v === "string" && !isNaN(Date.parse(v)));

const toNum = (v: number | string | Date) =>
  v instanceof Date ? +v : isDateLike(v) ? +new Date(v) : (v as number);

function extent(vals: number[]) {
  const xs = vals.filter((n) => Number.isFinite(n));
  if (!xs.length) return [0, 1];
  let lo = Math.min(...xs), hi = Math.max(...xs);
  if (lo === hi) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.05;
  return [lo - pad, hi + pad] as [number, number];
}

function fmtCompact(n: number) {
  return Math.abs(n) >= 1e9 ? (n / 1e9).toFixed(2) + "B" :
         Math.abs(n) >= 1e6 ? (n / 1e6).toFixed(2) + "M" :
         Math.abs(n) >= 1e3 ? (n / 1e3).toFixed(2) + "K" :
         n.toFixed(2);
}

/* ========= Component ========= */
const Chart: React.FC<ChartProps> = ({
  title,
  height = 280,
  series,
  xType,
  grid = true,
  yFormat = fmtCompact,
  xFormat,
  baseline,
}) => {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(720);

  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver((e) =>
      setWidth(Math.max(320, e[0].contentRect.width))
    );
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);

  const margin = { top: title ? 36 : 16, right: 32, bottom: 28, left: 48 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  // detect x type
  const isTime = useMemo(() => {
    if (xType) return xType === "time";
    const xs = series.flatMap((s) => s.data.map((d) => d.x));
    return xs.filter(isDateLike).length >= xs.length * 0.6;
  }, [series, xType]);

  // x domain
  const xDomain = useMemo<[number, number]>(() => {
    const xs = series.flatMap((s) => s.data.map((d, i) => isTime ? toNum(d.x) : i));
    if (!xs.length) return [0, 1];
    return [Math.min(...xs), Math.max(...xs)];
  }, [series, isTime]);

  // y domain
  const ys = series.flatMap((s) => s.data.map((d) => d.y ?? NaN));
  const yDomain = extent(ys as number[]);

  const xScale = (x: number | string | Date, i: number) =>
    margin.left +
    (isTime
      ? ((toNum(x) - xDomain[0]) / (xDomain[1] - xDomain[0])) * innerW
      : (i / Math.max(1, (series[0]?.data.length ?? 1) - 1)) * innerW);

  const yScale = (y: number) =>
    margin.top + innerH - ((y - yDomain[0]) / (yDomain[1] - yDomain[0])) * innerH;

  const palette = ["#0ea5e9", "#22c55e", "#a78bfa", "#f59e0b", "#ef4444", "#10b981"];
  const colorOf = (s: Series, i: number) => s.color || palette[i % palette.length];

  return (
    <div ref={ref} className="w-full rounded-2xl border border-neutral-200 bg-white shadow">
      {title && <div className="border-b px-4 py-2 font-semibold">{title}</div>}

      <svg width={width} height={height}>
        {/* grid + axes */}
        {grid &&
          [0.25, 0.5, 0.75, 1].map((f, i) => (
            <line
              key={i}
              x1={margin.left}
              x2={margin.left + innerW}
              y1={margin.top + innerH * f}
              y2={margin.top + innerH * f}
              stroke="#e5e7eb"
              strokeDasharray="3 3"
            />
          ))}

        {/* baseline */}
        {baseline != null && (
          <line
            x1={margin.left}
            x2={margin.left + innerW}
            y1={yScale(baseline)}
            y2={yScale(baseline)}
            stroke="#9ca3af"
          />
        )}

        {/* y axis labels */}
        {yDomain &&
          [yDomain[0], (yDomain[0] + yDomain[1]) / 2, yDomain[1]].map((t, i) => (
            <text
              key={i}
              x={margin.left - 6}
              y={yScale(t)}
              textAnchor="end"
              dominantBaseline="middle"
              className="fill-neutral-600 text-[11px]"
            >
              {yFormat(t)}
            </text>
          ))}

        {/* series */}
        {series.map((s, si) => {
          const color = colorOf(s, si);
          if (s.type === "bar") {
            const bw = innerW / s.data.length * 0.7;
            return (
              <g key={s.id}>
                {s.data.map((d, i) => {
                  if (d.y == null) return null;
                  const cx = xScale(d.x, i);
                  const y0 = yScale(0);
                  const yv = yScale(d.y);
                  return (
                    <rect
                      key={i}
                      x={cx - bw / 2}
                      y={Math.min(y0, yv)}
                      width={bw}
                      height={Math.abs(yv - y0)}
                      fill={color}
                      opacity={0.85}
                    />
                  );
                })}
              </g>
            );
          }
          // line/area
          const pts: { x: number; y: number }[] = [];
          s.data.forEach((d, i) => {
            if (d.y == null) return;
            pts.push({ x: xScale(d.x, i), y: yScale(d.y) });
          });
          if (!pts.length) return null;
          let d = `M ${pts[0].x} ${pts[0].y}`;
          for (let i = 1; i < pts.length; i++) d += ` L ${pts[i].x} ${pts[i].y}`;
          if (s.type === "area") {
            d += ` L ${pts[pts.length - 1].x} ${yScale(0)} L ${pts[0].x} ${yScale(0)} Z`;
            return <path key={s.id} d={d} fill={color + "55"} stroke="none" />;
          }
          return <path key={s.id} d={d} fill="none" stroke={color} strokeWidth={1.8} />;
        })}
      </svg>
    </div>
  );
};

export default Chart;