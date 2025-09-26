"use client";

import React, { useMemo } from "react";

/* =========================================
 * Types
 * ========================================= */

export type RiskMetricPoint = { date: string | Date; value: number };
export type Exposure = { bucket: string; value: number }; // e.g., sector/region/asset
export type LimitKey = "gross" | "net" | "leverage" | "singleName" | "var" | "dd";
export type Limits = Partial<Record<LimitKey, number>>;

export type RiskMetrics = {
  // snapshot values
  gross?: number;       // × equity
  net?: number;         // × equity
  leverage?: number;    // assets/equity
  var95?: number;       // as fraction of equity (e.g., 0.04 = 4%)
  vol?: number;         // daily or annualized (document in label)
  beta?: number;
  maxDD?: number;       // realized to-date (fraction)
  // small time-series to render sparklines
  pnlSeries?: RiskMetricPoint[];
  ddSeries?: RiskMetricPoint[];
  varSeries?: RiskMetricPoint[];
};

export interface StrategyRiskViewProps {
  name: string;
  asOf?: string; // ISO
  metrics: RiskMetrics;
  limits?: Limits;
  exposures?: {
    byAsset?: Exposure[];
    bySector?: Exposure[];
    byRegion?: Exposure[];
    // values should sum to ~100 if in %
  };
  // Heatmap matrix (e.g., factor or book corr). -1..+1
  correlationMatrix?: {
    labels: string[];
    matrix: number[][]; // square
  };
}

/* =========================================
 * Small helpers
 * ========================================= */

const fmtPct = (x?: number, d = 1) =>
  Number.isFinite(x as number) ? `${(((x as number)) * 100).toFixed(d)}%` : "–";
const fmt = (x?: number, d = 2) =>
  Number.isFinite(x as number) ? (x as number).toFixed(d) : "–";

const toDateNum = (d: string | Date) => (d instanceof Date ? +d : +new Date(d));

function sparkPath(values: number[], w: number, h: number) {
  if (!values.length) return "";
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = Math.max(1e-9, max - min);
  const step = values.length > 1 ? w / (values.length - 1) : w;
  const y = (v: number) => h - ((v - min) / range) * h;

  let d = `M 0 ${y(values[0])}`;
  for (let i = 1; i < values.length; i++) d += ` L ${i * step} ${y(values[i])}`;
  return d;
}

function barRows(data: Exposure[]) {
  const max = Math.max(1e-9, ...data.map((d) => Math.abs(d.value)));
  return data.map((d) => ({ ...d, pct: Math.abs(d.value) / max }));
}

function clamp(n: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, n));
}

function heatColor(x: number) {
  // x in [-1,1] -> blue (neg) to white to red (pos)
  const t = (x + 1) / 2; // [0,1]
  const r = Math.round(255 * t);
  const b = Math.round(255 * (1 - t));
  const g = 235 - Math.round(70 * Math.abs(x)); // slightly desaturated
  return `rgb(${r},${g},${b})`;
}

function breach(actual?: number, limit?: number, mode: "lte" | "gte" = "lte") {
  if (!Number.isFinite(actual!) || !Number.isFinite(limit!)) return false;
  return mode === "lte" ? (actual as number) > (limit as number) : (actual as number) < (limit as number);
}

/* =========================================
 * Tiny sub-components
 * ========================================= */

const Stat: React.FC<{ label: string; value: string; warn?: boolean; help?: string }> = ({
  label,
  value,
  warn,
  help,
}) => (
  <div className={`rounded-lg border px-3 py-2 ${warn ? "border-red-300 bg-red-50" : "border-neutral-200"}`}>
    <div className="text-[11px] uppercase tracking-wide text-neutral-500">{label}</div>
    <div className={`mt-0.5 text-sm ${warn ? "text-red-700" : "text-neutral-900"}`} title={help}>
      {value}
    </div>
  </div>
);

const LimitRow: React.FC<{ label: string; actual?: number; limit?: number; fmtFn?: (x?: number) => string; mode?: "lte" | "gte" }> =
  ({ label, actual, limit, fmtFn = fmt, mode = "lte" }) => {
    const hit = breach(actual, limit, mode);
    const pct = Number.isFinite(actual!) && Number.isFinite(limit!)
      ? clamp((actual as number) / Math.max(1e-9, limit as number), 0, 1)
      : 0;
    return (
      <div className={`flex items-center justify-between gap-3 rounded-md border px-3 py-2 ${hit ? "border-red-300 bg-red-50" : "border-neutral-200"}`}>
        <div className="text-sm">{label}</div>
        <div className="flex-1">
          <div className="h-2 w-full rounded bg-neutral-100">
            <div
              className={`h-2 rounded ${hit ? "bg-red-500" : "bg-neutral-800"}`}
              style={{ width: `${pct * 100}%` }}
            />
          </div>
        </div>
        <div className="w-40 text-right text-xs text-neutral-600">
          <span className="font-mono">{fmtFn(actual)}</span>
          <span className="mx-1">/</span>
          <span className="font-mono">{fmtFn(limit)}</span>
        </div>
      </div>
    );
  };

const Spark: React.FC<{ points?: RiskMetricPoint[]; w?: number; h?: number; line?: string; fill?: string }> = ({
  points = [],
  w = 160,
  h = 40,
  line = "#0ea5e9",
  fill = "rgba(14,165,233,0.12)",
}) => {
  if (!points.length) return <div className="text-xs text-neutral-400">n/a</div>;
  const xs = points.map((p) => toDateNum(p.date));
  const ys = points.map((p) => p.value);
  const d = sparkPath(ys, w, h);
  const min = Math.min(...ys), max = Math.max(...ys);
  const baseline = Math.min(0, min), top = Math.max(0, max);

  return (
    <svg width={w} height={h}>
      {/* zero line */}
      {baseline <= 0 && top >= 0 && (
        <line x1={0} x2={w} y1={(1 - (0 - min) / Math.max(1e-9, max - min)) * h} y2={(1 - (0 - min) / Math.max(1e-9, max - min)) * h} stroke="#e5e7eb" />
      )}
      {/* area */}
      <path d={`${d} L ${w} ${h} L 0 ${h} Z`} fill={fill} stroke="none" />
      {/* line */}
      <path d={d} stroke={line} fill="none" strokeWidth={1.5} />
    </svg>
  );
};

const Bars: React.FC<{ data?: Exposure[]; title: string; unit?: "%" | "x" }> = ({ data = [], title, unit = "%" }) => {
  const rows = barRows(data);
  return (
    <div className="rounded-lg border border-neutral-200">
      <div className="border-b px-3 py-2 text-sm font-semibold">{title}</div>
      <div className="p-3 space-y-2">
        {rows.length === 0 && <div className="text-xs text-neutral-500">No data</div>}
        {rows.map((r) => (
          <div key={r.bucket} className="grid grid-cols-6 items-center gap-2">
            <div className="col-span-2 truncate text-sm">{r.bucket}</div>
            <div className="col-span-3 h-2 rounded bg-neutral-100">
              <div className="h-2 rounded bg-neutral-800" style={{ width: `${r.pct * 100}%` }} />
            </div>
            <div className="col-span-1 text-right text-xs font-mono">
              {unit === "%" ? fmtPct(r.value / 100) : fmt(r.value)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

const Heatmap: React.FC<{ labels: string[]; matrix: number[][]; title?: string }> = ({ labels, matrix, title = "Correlation" }) => {
  const n = labels.length;
  const size = Math.min(28, Math.max(16, 180 / Math.max(3, n))); // cell size
  return (
    <div className="rounded-lg border border-neutral-200">
      <div className="border-b px-3 py-2 text-sm font-semibold">{title}</div>
      <div className="px-3 py-2 overflow-auto">
        <table className="border-collapse">
          <thead>
            <tr>
              <th />
              {labels.map((l) => (
                <th key={l} className="px-1 text-[11px] text-neutral-600">{l}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => (
              <tr key={i}>
                <td className="pr-1 text-[11px] text-neutral-600">{labels[i]}</td>
                {row.map((v, j) => (
                  <td key={j} title={v.toFixed(2)}>
                    <div
                      style={{ width: size, height: size, background: heatColor(clamp(v, -1, 1)), borderRadius: 3 }}
                    />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

/* =========================================
 * Main component
 * ========================================= */

const StrategyRiskView: React.FC<StrategyRiskViewProps> = ({
  name,
  asOf,
  metrics,
  limits = {},
  exposures,
  correlationMatrix,
}) => {
  const breachGross = breach(metrics.gross, limits.gross, "lte");
  const breachNet = breach(metrics.net, limits.net, "lte");
  const breachLev = breach(metrics.leverage, limits.leverage, "lte");
  const breachVar = breach(metrics.var95, limits.var, "lte");
  const breachDD = breach(metrics.maxDD, limits.dd, "lte");

  const breaches = [breachGross, breachNet, breachLev, breachVar, breachDD].filter(Boolean).length;

  return (
    <div className="w-full rounded-2xl border border-neutral-200 bg-white shadow">
      {/* Header */}
      <div className="flex items-center justify-between border-b px-4 py-3">
        <div>
          <h2 className="text-lg font-semibold">Risk — {name}</h2>
          <div className="text-xs text-neutral-500">
            As of {asOf ? new Date(asOf).toLocaleString() : "—"} •{" "}
            {breaches > 0 ? (
              <span className="text-red-600">{breaches} breach{breaches > 1 ? "es" : ""}</span>
            ) : (
              <span className="text-green-600">0 breaches</span>
            )}
          </div>
        </div>
      </div>

      {/* Top row: snapshot stats + sparklines */}
      <div className="grid grid-cols-1 gap-3 p-3 lg:grid-cols-3">
        <div className="space-y-2">
          <Stat label="Gross" value={`${fmt(metrics.gross)}×`} warn={breachGross} help="Gross exposure / equity" />
          <Stat label="Net" value={`${fmt(metrics.net)}×`} warn={breachNet} help="Long - Short / equity" />
          <Stat label="Leverage" value={`${fmt(metrics.leverage)}×`} warn={breachLev} help="Assets / equity" />
        </div>

        <div className="space-y-2">
          <Stat label="VaR (95%)" value={fmtPct(metrics.var95)} warn={breachVar} help="1-day VaR as % of equity" />
          <Stat label="Volatility" value={fmtPct(metrics.vol)} help="Assumed annualized unless noted" />
          <Stat label="Beta" value={fmt(metrics.beta)} />
        </div>

        <div className="space-y-2">
          <Stat label="Max Drawdown" value={fmtPct(metrics.maxDD)} warn={breachDD} />
          <div className="rounded-lg border border-neutral-200 px-3 py-2">
            <div className="mb-1 text-[11px] uppercase tracking-wide text-neutral-500">PnL (spark)</div>
            <Spark points={metrics.pnlSeries} />
          </div>
        </div>
      </div>

      {/* Limits progress */}
      <div className="grid grid-cols-1 gap-3 px-3 pb-3 lg:grid-cols-2">
        <div className="rounded-lg border border-neutral-200 p-3">
          <div className="mb-2 text-xs uppercase text-neutral-500">Limits</div>
          <div className="space-y-2">
            <LimitRow label="Gross ×" actual={metrics.gross} limit={limits.gross} />
            <LimitRow label="Net ×" actual={metrics.net} limit={limits.net} />
            <LimitRow label="Leverage ×" actual={metrics.leverage} limit={limits.leverage} />
            <LimitRow label="VaR (95%)" actual={metrics.var95} limit={limits.var} fmtFn={fmtPct} />
            <LimitRow label="Drawdown" actual={metrics.maxDD} limit={limits.dd} fmtFn={fmtPct} />
          </div>
        </div>

        <div className="grid grid-cols-1 gap-3">
          <div className="rounded-lg border border-neutral-200 p-3">
            <div className="mb-2 text-xs uppercase text-neutral-500">VaR (spark)</div>
            <Spark points={metrics.varSeries} />
          </div>
          <div className="rounded-lg border border-neutral-200 p-3">
            <div className="mb-2 text-xs uppercase text-neutral-500">Drawdown (spark)</div>
            <Spark points={metrics.ddSeries} />
          </div>
        </div>
      </div>

      {/* Exposures */}
      <div className="grid grid-cols-1 gap-3 px-3 pb-3 lg:grid-cols-3">
        <Bars title="Exposure by Asset" data={exposures?.byAsset ?? []} unit="%" />
        <Bars title="Exposure by Sector" data={exposures?.bySector ?? []} unit="%" />
        <Bars title="Exposure by Region" data={exposures?.byRegion ?? []} unit="%" />
      </div>

      {/* Correlation heatmap */}
      {correlationMatrix && correlationMatrix.labels.length > 0 && (
        <div className="px-3 pb-3">
          <Heatmap
            labels={correlationMatrix.labels}
            matrix={correlationMatrix.matrix}
            title="Book / Factor Correlation"
          />
        </div>
      )}
    </div>
  );
};

export default StrategyRiskView;