"use client";

import React, { useMemo, useState, useEffect, useRef } from "react";

/* ==============================
 * Shared types
 * ============================== */
type RiskMetricPoint = { date: string | Date; value: number };
type Exposure = { bucket: string; value: number };

export type DashboardRiskProps = {
  asOf?: string;
  metrics: {
    gross: number;
    net: number;
    leverage: number;
    var95: number; // fraction (0.04 = 4%)
    vol: number;   // fraction annualized
    beta: number;
    maxDD: number; // fraction
    pnlSeries: RiskMetricPoint[];
    ddSeries: RiskMetricPoint[];
    varSeries: RiskMetricPoint[];
  };
  exposures?: {
    byAsset?: Exposure[];
    bySector?: Exposure[];
    byRegion?: Exposure[];
  };
  correlationMatrix?: {
    labels: string[];
    matrix: number[][];
  };
  title?: string;
};

/* ==============================
 * Small helpers
 * ============================== */
const fmtPct = (x?: number, d = 2) =>
  Number.isFinite(x as number) ? `${((x as number) * 100).toFixed(d)}%` : "–";
const fmt = (x?: number, d = 2) =>
  Number.isFinite(x as number) ? (x as number).toFixed(d) : "–";
const clamp = (n: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, n));
const toDateNum = (d: string | Date) => (d instanceof Date ? +d : +new Date(d));

/* ==============================
 * Tiny visual atoms (Spark, Bars)
 * ============================== */
const Spark: React.FC<{ points?: RiskMetricPoint[]; w?: number; h?: number; line?: string; fill?: string }> = ({
  points = [],
  w = 160,
  h = 40,
  line = "#0ea5e9",
  fill = "rgba(14,165,233,0.12)",
}) => {
  if (!points.length) return <div className="text-xs text-neutral-400">n/a</div>;
  const ys = points.map((p) => p.value);
  const min = Math.min(...ys), max = Math.max(...ys);
  const range = Math.max(1e-9, max - min);
  const step = points.length > 1 ? w / (points.length - 1) : w;
  const Y = (v: number) => h - ((v - min) / range) * h;

  let d = `M 0 ${Y(ys[0])}`;
  for (let i = 1; i < ys.length; i++) d += ` L ${i * step} ${Y(ys[i])}`;

  return (
    <svg width={w} height={h}>
      {/* zero line if spans 0 */}
      {min <= 0 && max >= 0 && (
        <line x1={0} x2={w} y1={Y(0)} y2={Y(0)} stroke="#e5e7eb" />
      )}
      <path d={`${d} L ${w} ${h} L 0 ${h} Z`} fill={fill} />
      <path d={d} stroke={line} fill="none" strokeWidth={1.5} />
    </svg>
  );
};

const Bars: React.FC<{ data?: Exposure[]; title: string; unit?: "%" | "x" }> = ({ data = [], title, unit = "%" }) => {
  const max = Math.max(1e-9, ...data.map((d) => Math.abs(d.value)));
  return (
    <div className="rounded-lg border border-neutral-200">
      <div className="border-b px-3 py-2 text-sm font-semibold">{title}</div>
      <div className="p-3 space-y-2">
        {data.length === 0 && <div className="text-xs text-neutral-500">No data</div>}
        {data.map((d) => {
          const pct = Math.abs(d.value) / max;
          return (
            <div key={d.bucket} className="grid grid-cols-6 items-center gap-2">
              <div className="col-span-2 truncate text-sm">{d.bucket}</div>
              <div className="col-span-3 h-2 rounded bg-neutral-100">
                <div className="h-2 rounded bg-neutral-800" style={{ width: `${pct * 100}%` }} />
              </div>
              <div className="col-span-1 text-right text-xs font-mono">
                {unit === "%" ? fmtPct(d.value / 100, 1) : fmt(d.value, 2)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

/* ==============================
 * Correlation Heatmap (inline)
 * ============================== */
function heatColor(x: number) {
  const t = (clamp(x, -1, 1) + 1) / 2; // 0..1
  const r = Math.round(255 * t);
  const b = Math.round(255 * (1 - t));
  const g = 235 - Math.round(90 * Math.abs(x));
  return `rgb(${r},${g},${b})`;
}

const Heatmap: React.FC<{ labels: string[]; matrix: number[][]; title?: string; showValues?: boolean }> = ({
  labels,
  matrix,
  title = "Correlation",
  showValues = false,
}) => {
  const valid = labels.length > 0 && matrix.length === labels.length && matrix.every((r) => r.length === labels.length);
  const size = labels.length <= 18 ? 26 : labels.length <= 28 ? 22 : labels.length <= 40 ? 18 : 14;
  const legend = useMemo(() => Array.from({ length: 11 }, (_, i) => -1 + (i * 2) / 10), []);

  if (!valid) {
    return <div className="rounded-lg border border-neutral-200 p-3 text-sm text-red-600">Invalid matrix shape.</div>;
  }

  return (
    <div className="rounded-lg border border-neutral-200">
      <div className="border-b px-3 py-2 text-sm font-semibold">{title}</div>
      <div className="flex w-full overflow-auto">
        {/* Y labels */}
        <div className="sticky left-0 z-10 bg-white/80 backdrop-blur supports-[backdrop-filter]:bg-white/60">
          <table className="border-collapse">
            <tbody>
              {labels.map((lab, r) => (
                <tr key={`yl-${r}`} style={{ height: size }}>
                  <td className="pr-2 text-right text-[11px] text-neutral-600 whitespace-nowrap">{lab}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Grid + X labels */}
        <div className="overflow-auto">
          <div className="sticky top-0 z-10 bg-white/80 backdrop-blur supports-[backdrop-filter]:bg-white/60">
            <table className="border-collapse">
              <thead>
                <tr>
                  {labels.map((lab, c) => (
                    <th key={`xl-${c}`} style={{ width: size }} className="px-1 pb-1 text-center text-[11px] text-neutral-600 whitespace-nowrap">
                      {lab}
                    </th>
                  ))}
                </tr>
              </thead>
            </table>
          </div>

          <table className="border-collapse">
            <tbody>
              {matrix.map((row, r) => (
                <tr key={`r-${r}`} style={{ height: size }}>
                  {row.map((v, c) => (
                    <td key={`c-${r}-${c}`} style={{ width: size }}>
                      <div
                        className={`flex h-full w-full items-center justify-center rounded ${r === c ? "outline outline-1 outline-neutral-300" : ""}`}
                        style={{ background: heatColor(v), height: size }}
                        title={`${labels[r]} × ${labels[c]} = ${v.toFixed(2)}`}
                      >
                        {showValues && (
                          <span className="select-none text-[10px] font-medium" style={{ color: Math.abs(v) > 0.6 ? "#fff" : "#0f172a" }}>
                            {v.toFixed(2)}
                          </span>
                        )}
                      </div>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>

          {/* Legend */}
          <div className="flex items-center gap-2 px-3 py-2 text-[11px] text-neutral-600">
            <span>-1</span>
            <div className="flex h-2 w-40 overflow-hidden rounded">
              {legend.map((x, i) => (
                <div key={i} className="h-2 flex-1" style={{ background: heatColor(x) }} />
              ))}
            </div>
            <span>+1</span>
          </div>
        </div>
      </div>
    </div>
  );
};

/* ==============================
 * Strategy Risk View (inline)
 * ============================== */
const Stat: React.FC<{ label: string; value: string; warn?: boolean; help?: string }> = ({ label, value, warn, help }) => (
  <div className={`rounded-lg border px-3 py-2 ${warn ? "border-red-300 bg-red-50" : "border-neutral-200"}`} title={help}>
    <div className="text-[11px] uppercase tracking-wide text-neutral-500">{label}</div>
    <div className={`mt-0.5 text-sm ${warn ? "text-red-700" : "text-neutral-900"}`}>{value}</div>
  </div>
);

function breach(actual?: number, limit?: number) {
  if (!Number.isFinite(actual!) || !Number.isFinite(limit!)) return false;
  return (actual as number) > (limit as number);
}

const StrategyRiskView: React.FC<{
  name: string;
  asOf?: string;
  metrics: {
    gross?: number;
    net?: number;
    leverage?: number;
    var95?: number;
    vol?: number;
    beta?: number;
    maxDD?: number;
    pnlSeries?: RiskMetricPoint[];
    ddSeries?: RiskMetricPoint[];
    varSeries?: RiskMetricPoint[];
  };
  limits?: { gross?: number; net?: number; leverage?: number; var?: number; dd?: number };
  exposures?: { byAsset?: Exposure[]; bySector?: Exposure[]; byRegion?: Exposure[] };
  correlationMatrix?: { labels: string[]; matrix: number[][] };
}> = ({ name, asOf, metrics, limits = {}, exposures, correlationMatrix }) => {
  const bGross = breach(metrics.gross, limits.gross);
  const bNet = breach(metrics.net, limits.net);
  const bLev = breach(metrics.leverage, limits.leverage);
  const bVar = breach(metrics.var95, limits.var);
  const bDD = breach(metrics.maxDD, limits.dd);
  const breaches = [bGross, bNet, bLev, bVar, bDD].filter(Boolean).length;

  return (
    <div className="w-full rounded-2xl border border-neutral-200 bg-white shadow">
      {/* Header */}
      <div className="flex items-center justify-between border-b px-4 py-3">
        <div>
          <h2 className="text-lg font-semibold">Risk — {name}</h2>
          <div className="text-xs text-neutral-500">
            As of {asOf ? new Date(asOf).toLocaleString() : "—"} •{" "}
            {breaches > 0 ? <span className="text-red-600">{breaches} breach{breaches > 1 ? "es" : ""}</span> : <span className="text-green-600">0 breaches</span>}
          </div>
        </div>
      </div>

      {/* Snapshot */}
      <div className="grid grid-cols-1 gap-3 p-3 lg:grid-cols-3">
        <div className="space-y-2">
          <Stat label="Gross" value={`${fmt(metrics.gross)}×`} warn={bGross} help="Gross exposure / equity" />
          <Stat label="Net" value={`${fmt(metrics.net)}×`} warn={bNet} help="Long - Short / equity" />
          <Stat label="Leverage" value={`${fmt(metrics.leverage)}×`} warn={bLev} help="Assets / equity" />
        </div>
        <div className="space-y-2">
          <Stat label="VaR (95%)" value={fmtPct(metrics.var95)} warn={bVar} />
          <Stat label="Volatility" value={fmtPct(metrics.vol)} />
          <Stat label="Beta" value={fmt(metrics.beta)} />
        </div>
        <div className="space-y-2">
          <Stat label="Max Drawdown" value={fmtPct(metrics.maxDD)} warn={bDD} />
          <div className="rounded-lg border border-neutral-200 px-3 py-2">
            <div className="mb-1 text-[11px] uppercase tracking-wide text-neutral-500">PnL (spark)</div>
            <Spark points={metrics.pnlSeries} />
          </div>
        </div>
      </div>

      {/* Limits + more sparks */}
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

      {/* Correlation */}
      {correlationMatrix && correlationMatrix.labels.length > 0 && (
        <div className="px-3 pb-3">
          <Heatmap labels={correlationMatrix.labels} matrix={correlationMatrix.matrix} title="Book / Factor Correlation" />
        </div>
      )}
    </div>
  );
};

const LimitRow: React.FC<{
  label: string;
  actual?: number;
  limit?: number;
  fmtFn?: (x?: number) => string;
}> = ({ label, actual, limit, fmtFn = fmt }) => {
  const hit = breach(actual, limit);
  const pct = Number.isFinite(actual!) && Number.isFinite(limit!)
    ? clamp((actual as number) / Math.max(1e-9, limit as number), 0, 1)
    : 0;
  return (
    <div className={`flex items-center justify-between gap-3 rounded-md border px-3 py-2 ${hit ? "border-red-300 bg-red-50" : "border-neutral-200"}`}>
      <div className="text-sm">{label}</div>
      <div className="flex-1">
        <div className="h-2 w-full rounded bg-neutral-100">
          <div className={`h-2 rounded ${hit ? "bg-red-500" : "bg-neutral-800"}`} style={{ width: `${pct * 100}%` }} />
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

/* ==============================
 * Main Dashboard
 * ============================== */
const RiskDashboard: React.FC<DashboardRiskProps> = ({
  asOf,
  metrics,
  exposures,
  correlationMatrix,
  title = "Risk Dashboard",
}) => {
  // Responsive container width (optional; not strictly needed here)
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState<number>(0);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver((entries) => {
      setW(entries[0].contentRect.width);
    });
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);

  return (
    <div ref={ref} className="w-full space-y-6">
      {/* Title */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">{title}</h1>
        <span className="text-xs text-neutral-500">
          As of {asOf ? new Date(asOf).toLocaleString() : "—"}
        </span>
      </div>

      {/* Portfolio block */}
      <StrategyRiskView
        name="Portfolio"
        asOf={asOf}
        metrics={metrics}
        limits={{ gross: 2.5, net: 1.2, leverage: 2.5, var: 0.05, dd: 0.2 }}
        exposures={exposures}
        correlationMatrix={correlationMatrix}
      />
    </div>
  );
};

export default RiskDashboard;