"use client";

/**
 * cryptochart.tsx
 * Zero-import, dependency-free price chart for crypto.
 * - Pure React + SVG + Tailwind (no Recharts, no icons)
 * - Price line with min/max autoscale
 * - Optional volume bars
 * - Hover crosshair + tooltip
 * - Range filter (ALL, 1M, 1W, 1D) based on timestamps
 */

type CryptoPoint = {
  ts: string;      // ISO timestamp, e.g. "2025-09-24T06:15:00Z"
  price: number;   // last price
  volume?: number; // optional volume (base or notional)
};

type RangeKey = "ALL" | "1M" | "1W" | "1D";

export type CryptoChartProps = {
  data: CryptoPoint[];
  title?: string;
  currency?: string;          // e.g., "USD", "INR"
  height?: number;            // px
  showVolume?: boolean;
  defaultRange?: RangeKey;
  className?: string;
};

export default function CryptoChart({
  data,
  title = "Crypto Price",
  currency = "USD",
  height = 280,
  showVolume = true,
  defaultRange = "ALL",
  className = "",
}: CryptoChartProps) {
  // ---- state
  const [range, setRange] = useState<RangeKey>(defaultRange);
  const [hoverX, setHoverX] = useState<number | null>(null);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [bounds, setBounds] = useState({ w: 0, h: 0 });

  const wrapRef = useRef<HTMLDivElement>(null);

  // ---- effects
  useEffect(() => {
    const observer = new ResizeObserver(([entry]) => {
      const r = entry.contentRect;
      setBounds({ w: Math.max(320, r.width), h: height });
    });
    if (wrapRef.current) observer.observe(wrapRef.current);
    return () => observer.disconnect();
  }, [height]);

  // ---- helpers
  const fmtNum = (n: number, d = 2) =>
    n.toLocaleString(currency === "INR" ? "en-IN" : "en-US", {
      maximumFractionDigits: d,
    });

  const shortTs = (iso: string) => {
    const dt = new Date(iso);
    return dt.toLocaleString(undefined, {
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const filterByRange = (rows: CryptoPoint[], key: RangeKey) => {
    if (key === "ALL") return rows;
    const now = rows.length ? new Date(rows[rows.length - 1].ts).getTime() : Date.now();
    const ms =
      key === "1D" ? 86400000 :
      key === "1W" ? 7 * 86400000 :
      key === "1M" ? 30 * 86400000 : Infinity;
    return rows.filter((r) => new Date(r.ts).getTime() >= now - ms);
  };

  const rows = useMemo(() => {
    const sorted = data.slice().sort((a, b) => +new Date(a.ts) - +new Date(b.ts));
    return filterByRange(sorted, range);
  }, [data, range]);

  // ---- layout + scales
  const pad = { left: 36, right: 12, top: 12, bottom: showVolume ? 56 : 24 };
  const w = Math.max(320, bounds.w);
  const h = Math.max(200, bounds.h);
  const innerW = Math.max(1, w - pad.left - pad.right);
  const innerH = Math.max(1, h - pad.top - pad.bottom);
  const volH = showVolume ? Math.floor(innerH * 0.28) : 0;
  const priceH = innerH - volH - (showVolume ? 8 : 0);

  const prices = rows.map((r) => r.price);
  const vols = rows.map((r) => r.volume ?? 0);
  const minP = prices.length ? Math.min(...prices) : 0;
  const maxP = prices.length ? Math.max(...prices) : 1;
  const padP = (maxP - minP) * 0.08 || maxP * 0.08;
  const yMin = Math.max(0, minP - padP);
  const yMax = maxP + padP;

  const minV = vols.length ? 0 : 0;
  const maxV = vols.length ? Math.max(...vols, 1) : 1;

  const x = (i: number) => pad.left + (i * innerW) / Math.max(1, rows.length - 1);
  const yP = (p: number) =>
    pad.top + (priceH - ((p - yMin) / (yMax - yMin || 1)) * priceH);
  const yV = (v: number) =>
    pad.top + priceH + 8 + (volH - (v / (maxV || 1)) * volH);

  const pathD = useMemo(() => {
    if (rows.length === 0) return "";
    return rows
      .map((r, i) => `${i ? "L" : "M"} ${x(i).toFixed(2)} ${yP(r.price).toFixed(2)}`)
      .join(" ");
  }, [rows, w, h, range]);

  // axis ticks
  const yTicks = 4;
  const yTickVals = Array.from({ length: yTicks + 1 }, (_, i) => yMin + ((yMax - yMin) * i) / yTicks);

  // hover logic
  const onMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = (e.target as SVGElement).closest("svg")!.getBoundingClientRect();
    const px = e.clientX - rect.left;
    setHoverX(px);
    if (!rows.length) return;
    // nearest index by x
    const t = Math.round(((px - pad.left) / innerW) * (rows.length - 1));
    const idx = Math.max(0, Math.min(rows.length - 1, t));
    setHoverIdx(idx);
  };

  const onMouseLeave = () => {
    setHoverX(null);
    setHoverIdx(null);
  };

  const active = hoverIdx != null ? rows[hoverIdx] : rows.at(-1) || null;
  const activeX = hoverIdx != null ? x(hoverIdx) : rows.length ? x(rows.length - 1) : 0;

  return (
    <div ref={wrapRef} className={`w-full rounded-xl border border-neutral-800 bg-neutral-900 ${className}`}>
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800 px-4 py-3">
        <div className="space-y-0.5">
          <h3 className="text-sm font-semibold text-neutral-200">{title}</h3>
          <p className="text-xs text-neutral-400">
            {rows.length ? `${new Date(rows[0].ts).toLocaleDateString()} → ${new Date(rows[rows.length - 1].ts).toLocaleDateString()}` : "No data"}
          </p>
        </div>
        <div className="flex items-center gap-1 text-xs">
          {(["ALL", "1M", "1W", "1D"] as RangeKey[]).map((r) => (
            <button
              key={r}
              onClick={() => setRange(r)}
              className={[
                "rounded-md border px-2 py-1",
                range === r
                  ? "border-emerald-600 bg-emerald-600/20 text-emerald-300"
                  : "border-neutral-800 bg-neutral-900 text-neutral-300 hover:bg-neutral-800/60",
              ].join(" ")}
            >
              {r}
            </button>
          ))}
        </div>
      </div>

      {/* Chart */}
      <div className="px-2 py-2">
        <svg
          width="100%"
          viewBox={`0 0 ${w} ${h}`}
          className="block"
          onMouseMove={onMouseMove}
          onMouseLeave={onMouseLeave}
        >
          {/* grid + y ticks */}
          {yTickVals.map((val, i) => {
            const yy = yP(val);
            return (
              <g key={i}>
                <line x1={pad.left} y1={yy} x2={w - pad.right} y2={yy} stroke="#27272a" strokeDasharray="3 3" />
                <text x={4} y={yy + 4} fill="#9ca3af" fontSize="10">
                  {currency === "INR" ? `₹ ${fmtNum(val, 0)}` : `${currency === "USD" ? "$" : ""}${fmtNum(val, 0)}`}
                </text>
              </g>
            );
          })}

          {/* price path */}
          <path d={pathD} fill="none" stroke="#10b981" strokeWidth="2" />

          {/* price dots (sparse) */}
          {rows.map((r, i) =>
            (rows.length <= 80 || i % Math.ceil(rows.length / 80) === 0) ? (
              <circle key={i} cx={x(i)} cy={yP(r.price)} r="1.8" fill="#10b981" />
            ) : null
          )}

          {/* volume bars */}
          {showVolume && volH > 0 && rows.map((r, i) => {
            const barW = Math.max(1, innerW / Math.max(1, rows.length - 1)) * 0.66;
            const vx = x(i) - barW / 2;
            const vy = yV(r.volume ?? 0);
            const vh = pad.top + priceH + 8 + volH - vy;
            return (
              <rect
                key={`v-${i}`}
                x={vx}
                y={vy}
                width={barW}
                height={vh}
                fill="#334155"
                opacity={0.85}
                rx="2"
              />
            );
          })}

          {/* x-axis time labels (sparse) */}
          {rows.map((r, i) =>
            (i === 0 || i === rows.length - 1 || i % Math.ceil(rows.length / 6 || 1) === 0) ? (
              <text
                key={`x-${i}`}
                x={x(i)}
                y={h - 6}
                textAnchor="middle"
                fill="#9ca3af"
                fontSize="10"
              >
                {labelForRange(r.ts, range)}
              </text>
            ) : null
          )}

          {/* hover crosshair + marker */}
          {active && hoverX != null && (
            <>
              <line x1={activeX} y1={pad.top} x2={activeX} y2={pad.top + priceH} stroke="#6b7280" strokeDasharray="4 4" />
              <circle cx={activeX} cy={yP(active.price)} r="3" fill="#10b981" stroke="#0a0a0a" strokeWidth="1" />
            </>
          )}
        </svg>
      </div>

      {/* Tooltip / footer */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-neutral-800 px-4 py-3 text-xs">
        <div className="text-neutral-300">
          {active ? (
            <>
              <span className="text-neutral-400">{shortTs(active.ts)}</span>{" · "}
              <span className="font-medium">
                {currency === "INR" ? "₹ " : currency === "USD" ? "$" : ""}
                {fmtNum(active.price)}
              </span>
              {typeof active.volume === "number" && (
                <>
                  {" · "}
                  <span className="text-neutral-400">Vol</span> {fmtNum(active.volume, 0)}
                </>
              )}
            </>
          ) : (
            "Hover over the chart"
          )}
        </div>
        <div className="text-neutral-400">
          {rows.length} points · {range}
        </div>
      </div>
    </div>
  );
}

/* ------------------------------ Utilities ------------------------------ */

function labelForRange(iso: string, r: RangeKey) {
  const d = new Date(iso);
  if (r === "1D") {
    return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
  }
  if (r === "1W") {
    return d.toLocaleDateString(undefined, { weekday: "short" });
  }
  return d.toLocaleDateString(undefined, { month: "short", day: "2-digit" });
}

/* -------------------------- React (no imports) -------------------------- */
/* This file assumes React types are globally available in your setup.
   If not, add at the top:
   import React, { useEffect, useMemo, useRef, useState } from "react";
   …but you asked for zero imports, so we rely on ambient types. */

// Minimal ambient declarations to quiet TS if needed (remove in real app)
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useEffect(cb: () => void | (() => void), deps?: any[]): void;
declare function useMemo<T>(cb: () => T, deps: any[]): T;
declare function useRef<T>(v: T | null): { current: T | null };

/* ----------------------------- Example usage -----------------------------
const points: CryptoPoint[] = Array.from({ length: 200 }).map((_, i) => {
  const ts = new Date(Date.now() - (200 - i) * 60 * 60 * 1000);
  return {
    ts: ts.toISOString(),
    price: 60000 + Math.sin(i / 9) * 800 + Math.random() * 300,
    volume: 100 + Math.max(0, Math.sin(i / 5)) * 400 + Math.random() * 150,
  };
});

<CryptoChart data={points} title="BTC/USDT" currency="USD" defaultRange="1W" />
--------------------------------------------------------------------------- */