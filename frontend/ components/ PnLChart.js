// frontend/components/pnlchart.js
// A sleek, production-ready P&L chart using Recharts.
// - Shows cumulative PnL line, daily PnL bars, and drawdown area
// - Works with provided `data` prop OR auto-fetches from `/api/pnl`
// - Responsive, dark-mode friendly, and keyboard accessible

import React, { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  Area,
  Bar,
  Line,
  Brush,
  ReferenceLine,
} from "recharts";

// ----- Types (JSDoc)
// @typedef {Object} PnLPoint
// @property {number|string|Date} ts   // timestamp (ms, ISO string, or Date)
// @property {number} [pnl]            // daily PnL
// @property {number} [cum]            // cumulative PnL (optional; will be computed if missing)
// @property {number} [dd]             // drawdown (negative numbers; will be computed if missing)

/**
 * Format date labels nicely (local timezone).
 * @param {number} t
 * @returns {string}
 */
function fmtDate(t) {
  try {
    return new Intl.DateTimeFormat(undefined, {
      year: "2-digit",
      month: "short",
      day: "2-digit",
    }).format(new Date(t));
  } catch {
    return String(t);
  }
}

/**
 * Format currency compactly; change currency if you prefer.
 * @param {number} v
 * @returns {string}
 */
function fmtMoney(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "";
  try {
    return new Intl.NumberFormat(undefined, {
      style: "currency",
      currency: "USD",
      notation: "compact",
      maximumFractionDigits: 2,
    }).format(v);
  } catch {
    return `${v.toFixed?.(2) ?? v}`;
  }
}

/**
 * Compute cumulative PnL & drawdown if not present.
 * @param {PnLPoint[]} rows
 * @returns {PnLPoint[]}
 */
function withDerived(rows) {
  let cum = 0;
  let peak = -Infinity;
  return rows
    .map((r) => {
      const ts =
        r.ts instanceof Date
          ? r.ts.getTime()
          : typeof r.ts === "string"
          ? Date.parse(r.ts)
          : Number(r.ts);
      const pnl = Number(r.pnl ?? 0);
      const c = Number.isFinite(r.cum) ? Number(r.cum) : (cum += pnl);
      peak = Math.max(peak, c);
      const dd = Number.isFinite(r.dd) ? Number(r.dd) : c - peak; // negative or 0
      return { ts, pnl, cum: c, dd };
    })
    .sort((a, b) => a.ts - b.ts);
}

/**
 * Nice min/max padding for axes.
 */
function padDomain(min, max, pad = 0.05) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return ["dataMin", "dataMax"];
  if (min === max) return [min - 1, max + 1];
  const span = Math.max(1e-9, max - min);
  return [min - span * pad, max + span * pad];
}

/**
 * Tooltip renderer.
 */
function PnLTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null;
  const row = payload.reduce((acc, p) => ({ ...acc, [p.dataKey]: p.value }), {});
  return (
    <div
      role="dialog"
      aria-label="PnL tooltip"
      className="rounded-xl shadow-lg border border-black/10 dark:border-white/10 bg-white/90 dark:bg-zinc-900/90 backdrop-blur p-3 text-sm"
    >
      <div className="font-medium mb-1">{fmtDate(label)}</div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1">
        <span className="opacity-70">Daily PnL</span>
        <span className={row.pnl >= 0 ? "text-emerald-600" : "text-rose-600"}>{fmtMoney(row.pnl)}</span>

        <span className="opacity-70">Cumulative</span>
        <span className={row.cum >= 0 ? "text-emerald-600" : "text-rose-600"}>{fmtMoney(row.cum)}</span>

        <span className="opacity-70">Drawdown</span>
        <span className={row.dd < 0 ? "text-rose-600" : "text-zinc-500"}>{fmtMoney(row.dd)}</span>
      </div>
    </div>
  );
}

/**
 * Legend item names.
 */
const LEGEND_NAMES = {
  pnl: "Daily PnL",
  cum: "Cumulative PnL",
  dd: "Drawdown",
};

/**
 * PnLChart Component.
 * @param {{ data?: PnLPoint[], height?: number, endpoint?: string }} props
 */
export default function PnLChart({ data, height = 380, endpoint = "/api/pnl" }) {
  const [remote, setRemote] = useState(/** @type {PnLPoint[]|null} */(null));
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  // Fetch if no data passed in
  useEffect(() => {
    let mounted = true;
    if (data?.length) return;
    (async () => {
      try {
        setLoading(true);
        setErr("");
        const res = await fetch(endpoint, { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        if (mounted) setRemote(json?.data ?? json ?? []);
      } catch (e) {
        if (mounted) setErr(e?.message ?? "Failed to load PnL");
      } finally {
        if (mounted) setLoading(false);
      }
    })();
    return () => {
      mounted = false;
    };
  }, [endpoint, data]);

  const rows = useMemo(() => withDerived((data && data.length ? data : remote) ?? []), [data, remote]);

  const domains = useMemo(() => {
    if (!rows.length) return { yLeft: ["auto", "auto"], yRight: ["auto", "auto"] };
    const minBar = Math.min(...rows.map((r) => r.pnl));
    const maxBar = Math.max(...rows.map((r) => r.pnl));
    const minLine = Math.min(...rows.map((r) => Math.min(r.cum, r.dd)));
    const maxLine = Math.max(...rows.map((r) => Math.max(r.cum, r.dd)));
    return {
      yLeft: padDomain(minBar, maxBar, 0.2),
      yRight: padDomain(minLine, maxLine, 0.1),
    };
  }, [rows]);

  if (loading && !rows.length) {
    return (
      <div
        className="w-full flex items-center justify-center h-40 text-sm text-zinc-500 dark:text-zinc-400"
        role="status"
        aria-live="polite"
      >
        Loading PnL…
      </div>
    );
  }

  if (err && !rows.length) {
    return (
      <div className="w-full rounded-xl border border-rose-300/40 bg-rose-50 dark:bg-rose-950/20 p-4 text-rose-600 dark:text-rose-300">
        Failed to load PnL: {err}
      </div>
    );
  }

  return (
    <div className="w-full rounded-2xl border border-black/5 dark:border-white/10 bg-white dark:bg-zinc-950 p-4 shadow-sm">
      {/* Gradients */}
      <svg width="0" height="0" className="absolute">
        <defs>
          <linearGradient id="ddFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgb(244 63 94)" stopOpacity="0.25" />
            <stop offset="100%" stopColor="rgb(244 63 94)" stopOpacity="0.02" />
          </linearGradient>
          <linearGradient id="cumStroke" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="rgb(34 197 94)" />
            <stop offset="100%" stopColor="rgb(59 130 246)" />
          </linearGradient>
        </defs>
      </svg>

      <div className="flex items-center justify-between mb-2">
        <h3 className="text-base md:text-lg font-semibold">P&L Overview</h3>
        <div className="text-xs md:text-sm text-zinc-500 dark:text-zinc-400">
          {rows.length ? `${fmtDate(rows[0].ts)} — ${fmtDate(rows[rows.length - 1].ts)}` : "—"}
        </div>
      </div>

      <div style={{ width: "100%", height }}>
        <ResponsiveContainer>
          <ComposedChart
            data={rows}
            margin={{ top: 10, right: 24, bottom: 0, left: 8 }}
            accessibilityLayer
            syncId="pnl-sync"
          >
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
            <XAxis
              dataKey="ts"
              tickFormatter={fmtDate}
              minTickGap={24}
              tick={{ fontSize: 11, fill: "currentColor", opacity: 0.7 }}
              axisLine={{ stroke: "currentColor", opacity: 0.2 }}
              tickLine={{ stroke: "currentColor", opacity: 0.2 }}
              type="number"
              domain={["dataMin", "dataMax"]}
              scale="time"
            />
            <YAxis
              yAxisId="left"
              domain={domains.yLeft}
              tickFormatter={fmtMoney}
              tick={{ fontSize: 11, fill: "currentColor", opacity: 0.7 }}
              axisLine={{ stroke: "currentColor", opacity: 0.2 }}
              tickLine={{ stroke: "currentColor", opacity: 0.2 }}
              width={60}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              domain={domains.yRight}
              tickFormatter={fmtMoney}
              tick={{ fontSize: 11, fill: "currentColor", opacity: 0.7 }}
              axisLine={{ stroke: "currentColor", opacity: 0.2 }}
              tickLine={{ stroke: "currentColor", opacity: 0.2 }}
              width={60}
            />

            <Tooltip content={<PnLTooltip />} />
            <Legend
              wrapperStyle={{ paddingTop: 8 }}
              formatter={(v) => LEGEND_NAMES[v] ?? v}
            />

            {/* Zero line for daily PnL bars */}
            <ReferenceLine yAxisId="left" y={0} stroke="currentColor" strokeOpacity={0.15} />

            {/* Daily PnL bars (left axis) */}
            <Bar
              name="pnl"
              dataKey="pnl"
              yAxisId="left"
              barSize={3}
              radius={[2, 2, 0, 0]}
              fill="currentColor"
              fillOpacity={0.35}
            />

            {/* Drawdown area (right axis) — negative values */}
            <Area
              name="dd"
              dataKey="dd"
              yAxisId="right"
              type="monotone"
              stroke="none"
              fill="url(#ddFill)"
              fillOpacity={1}
              isAnimationActive={false}
            />

            {/* Cumulative line (right axis) */}
            <Line
              name="cum"
              dataKey="cum"
              yAxisId="right"
              type="monotone"
              stroke="url(#cumStroke)"
              strokeWidth={2.2}
              dot={false}
              activeDot={{ r: 3 }}
            />

            {/* Brush for quick zoom */}
            {rows.length > 50 && (
              <Brush
                dataKey="ts"
                height={24}
                travellerWidth={8}
                tickFormatter={fmtDate}
                stroke="currentColor"
                fill="currentColor"
                fillOpacity={0.06}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}