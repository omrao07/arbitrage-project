"use client";

import React, { useMemo, useState } from "react";

type Props = {
  data: number[];
  width?: number;          // px, default 160
  height?: number;         // px, default 40
  strokeWidth?: number;    // default 1.5
  fill?: boolean;          // area fill
  smooth?: boolean;        // curve smoothing
  showLastValue?: boolean; // badge at the end
  baseline?: number;       // y=0 reference
  positiveColor?: string;  // "#22c55e"
  negativeColor?: string;  // "#ef4444"
  neutralColor?: string;   // "#9ca3af"
  bg?: string;             // container bg
  format?: (v: number) => string;
  onPoint?: (idx: number, value: number) => void;
};

export default function PNLSparkline({
  data,
  width = 160,
  height = 40,
  strokeWidth = 1.5,
  fill = true,
  smooth = false,
  showLastValue = true,
  baseline = 0,
  positiveColor = "#22c55e",
  negativeColor = "#ef4444",
  neutralColor = "#9ca3af",
  bg = "transparent",
  format = (v) => (v >= 0 ? `+${v.toFixed(2)}` : v.toFixed(2)),
  onPoint,
}: Props) {
  const n = data?.length ?? 0;
  if (!n) return <div style={{ width, height }} />;

  const padX = 2;
  const padY = 2;
  const w = Math.max(10, width);
  const h = Math.max(10, height);

  const { min, max, range, xAt, yAt, points, color, d, dFill } = useMemo(() => {
    const mn = Math.min(...data, baseline);
    const mx = Math.max(...data, baseline);
    const rg = mx - mn || 1;

    const xAtFn = (i: number) => padX + (i * (w - padX * 2)) / Math.max(1, n - 1);
    const yAtFn = (v: number) => h - padY - ((v - mn) / rg) * (h - padY * 2);

    const pts = data.map((v, i) => [xAtFn(i), yAtFn(v)] as const);
    const col =
      data[n - 1] > data[0] ? positiveColor :
      data[n - 1] < data[0] ? negativeColor : neutralColor;

    const path = smooth ? pathSmooth(pts) : pathLine(pts);
    const pathFill = `${path} L ${xAtFn(n - 1)} ${yAtFn(mn)} L ${xAtFn(0)} ${yAtFn(mn)} Z`;

    return { min: mn, max: mx, range: rg, xAt: xAtFn, yAt: yAtFn, points: pts, color: col, d: path, dFill: pathFill };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data.join(","), width, height, smooth, baseline, positiveColor, negativeColor, neutralColor]);

  const [hover, setHover] = useState<number | null>(null);

  function handleMove(e: React.MouseEvent<SVGSVGElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const rx = e.clientX - rect.left;
    const ratio = Math.min(1, Math.max(0, (rx - padX) / (w - padX * 2)));
    const i = Math.round(ratio * (n - 1));
    setHover(i);
    onPoint?.(i, data[i]);
  }

  function handleLeave() {
    setHover(null);
    onPoint?.(-1, NaN);
  }

  const lastX = xAt(n - 1);
  const lastY = yAt(data[n - 1]);
  const hoverX = hover != null ? xAt(hover) : null;
  const hoverY = hover != null ? yAt(data[hover]) : null;

  return (
    <div style={{ width: w, height: h, background: bg, position: "relative" }}>
      <svg width={w} height={h} onMouseMove={handleMove} onMouseLeave={handleLeave}>
        {/* baseline */}
        <line
          x1={padX}
          x2={w - padX}
          y1={yAt(baseline)}
          y2={yAt(baseline)}
          stroke="#2a2a2a"
          strokeWidth={1}
          strokeDasharray="3 3"
        />

        {/* area fill */}
        {fill && <path d={dFill} fill={hexToRgba(color, 0.18)} stroke="none" />}

        {/* main line */}
        <path d={d} fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinejoin="round" strokeLinecap="round" />

        {/* end dot */}
        <circle cx={lastX} cy={lastY} r={2.2} fill={color} />

        {/* hover cursor */}
        {hoverX != null && hoverY != null && (
          <>
            <line x1={hoverX} x2={hoverX} y1={padY} y2={h - padY} stroke="rgba(255,255,255,0.2)" strokeWidth={1} />
            <circle cx={hoverX} cy={hoverY} r={2.4} fill="#fff" />
          </>
        )}
      </svg>

      {/* last value badge */}
      {showLastValue && (
        <div
          style={{
            position: "absolute",
            top: Math.max(0, lastY - 14),
            left: Math.min(w - 64, Math.max(0, lastX + 6)),
            background: "#0b0b0b",
            color,
            border: "1px solid #222",
            fontSize: 10,
            padding: "2px 6px",
            borderRadius: 4,
            lineHeight: 1.2,
          }}
        >
          {format(data[n - 1])}
        </div>
      )}

      {/* hover tooltip */}
      {hoverX != null && hoverY != null && (
        <div
          style={{
            position: "absolute",
            top: Math.max(0, hoverY - 22),
            left: Math.min(w - 64, Math.max(0, hoverX + 6)),
            background: "#0b0b0b",
            color: "#d1d5db",
            border: "1px solid #222",
            fontSize: 10,
            padding: "2px 6px",
            borderRadius: 4,
            lineHeight: 1.2,
            pointerEvents: "none",
          }}
        >
          {format(data[hover!])}
        </div>
      )}
    </div>
  );
}

/* ---------------- helpers ---------------- */

function pathLine(pts: readonly (readonly [number, number])[]): string {
  return pts.reduce((acc, [x, y], i) => (i === 0 ? `M ${x} ${y}` : `${acc} L ${x} ${y}`), "");
}

function pathSmooth(pts: readonly (readonly [number, number])[], tension = 0.2): string {
  if (pts.length < 3) return pathLine(pts);
  let d = `M ${pts[0][0]} ${pts[0][1]}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i === 0 ? 0 : i - 1];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[i + 2] || p2;

    const cp1x = p1[0] + ((p2[0] - p0[0]) / 6) * (tension * 3);
    const cp1y = p1[1] + ((p2[1] - p0[1]) / 6) * (tension * 3);
    const cp2x = p2[0] - ((p3[0] - p1[0]) / 6) * (tension * 3);
    const cp2y = p2[1] - ((p3[1] - p1[1]) / 6) * (tension * 3);

    d += ` C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${p2[0]} ${p2[1]}`;
  }
  return d;
}

function hexToRgba(hex: string, a: number) {
  const m = hex.replace("#", "");
  const full = m.length === 3 ? m.split("").map((c) => c + c).join("") : m;
  const bigint = parseInt(full, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return `rgba(${r},${g},${b},${a})`;
}