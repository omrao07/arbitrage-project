"use client";

import React, { useMemo } from "react";

/* =========================
 * Types
 * ========================= */
export type KPI = {
  label: string;
  value: string;
  hint?: string;
  warn?: boolean;
};

export type SeriesPoint = { date: string | Date; value: number };

export type Allocation = {
  bucket: string;     // e.g., "Equity", "Credit"
  weight: number;     // 0..1
};

export type RiskSnapshot = {
  gross?: number;     // × equity
  net?: number;       // × equity
  var95?: number;     // fraction (0.04 = 4%)
  vol?: number;       // fraction annualized
  beta?: number;
  maxDD?: number;     // fraction
};

export type Position = {
  id: string;
  name: string;       // ticker or instrument
  side: "Long" | "Short";
  notional: number;   // in base currency
  pnl?: number;       // realized+unrealized (same unit as notional)
  weight?: number;    // 0..1 of portfolio
  sector?: string;
  region?: string;
};

export type PortfolioOverviewProps = {
  title?: string;
  asOf?: string;                      // ISO
  kpis?: KPI[];
  performance?: SeriesPoint[];        // cumulative PnL (fraction) or currency
  asPercent?: boolean;                // format performance as % if true
  allocations?: Allocation[];         // by asset class / sleeve
  risk?: RiskSnapshot;
  topPositions?: Position[];
};

/* =========================
 * Helpers
 * ========================= */

const fmtPct = (x?: number, d = 2) =>
  Number.isFinite(x as number) ? `${((x as number) * 100).toFixed(d)}%` : "–";
const fmtNum = (x?: number, d = 2) =>
  Number.isFinite(x as number) ? (x as number).toFixed(d) : "–";
const toDateNum = (d: string | Date) => (d instanceof Date ? +d : +new Date(d));

function niceExtent(vals: number[]) {
  if (!vals.length) return [0, 1] as [number, number];
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (lo === hi) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.05;
  return [lo - pad, hi + pad] as [number, number];
}

/* =========================
 * Small visual components
 * ========================= */

const Card: React.FC<React.PropsWithChildren<{ title?: string; right?: React.ReactNode }>> = ({ title, right, children }) => (
  <div className="rounded-2xl border border-neutral-200 bg-white shadow">
    {(title || right) && (
      <div className="flex items-center justify-between border-b px-4 py-3">
        <h3 className="text-lg font-semibold">{title}</h3>
        {right}
      </div>
    )}
    <div className="p-4">{children}</div>
  </div>
);

const KPIGrid: React.FC<{ items: KPI[] }> = ({ items }) => (
  <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
    {items.map((k, i) => (
      <div key={i} className={`rounded-xl border px-3 py-2 ${k.warn ? "border-red-300 bg-red-50" : "border-neutral-200"}`}>
        <div className="text-[11px] uppercase tracking-wide text-neutral-500">{k.label}</div>
        <div className={`mt-0.5 text-sm ${k.warn ? "text-red-700" : "text-neutral-900"}`}>{k.value}</div>
        {k.hint && <div className="mt-0.5 text-[11px] text-neutral-500">{k.hint}</div>}
      </div>
    ))}
  </div>
);

const SparkLine: React.FC<{ data: SeriesPoint[]; height?: number; stroke?: string; fill?: string; asPercent?: boolean }> = ({
  data, height = 160, stroke = "#0ea5e9", fill = "rgba(14,165,233,0.12)", asPercent = false
}) => {
  const w = 720; // responsive container will clip; width comes from parent
  const pts = useMemo(() => data.map(d => ({ t: toDateNum(d.date), v: d.value })).sort((a,b)=>a.t-b.t), [data]);
  const xs = pts.map(p => p.t);
  const ys = pts.map(p => p.v);
  const xMin = Math.min(...xs, Date.now()), xMax = Math.max(...xs, Date.now());
  const [yMin, yMax] = niceExtent(ys.length ? ys : [0]);

  const x = (t: number) => ((t - xMin) / Math.max(1, xMax - xMin)) * (w - 48) + 40; // side margins
  const y = (v: number) => (height - 30) - ((v - yMin) / Math.max(1e-9, yMax - yMin)) * (height - 50);

  const d = pts.length
    ? `M ${x(pts[0].t)} ${y(pts[0].v)}` + pts.slice(1).map(p => ` L ${x(p.t)} ${y(p.v)}`).join("")
    : "";

  return (
    <svg width="100%" viewBox={`0 0 ${w} ${height}`} className="overflow-visible">
      {/* axes */}
      <line x1={40} x2={w-8} y1={height-30} y2={height-30} stroke="#e5e7eb"/>
      <line x1={40} x2={40} y1={10} y2={height-30} stroke="#e5e7eb"/>
      {/* zero baseline if percent */}
      {asPercent && yMin < 0 && yMax > 0 && (
        <line x1={40} x2={w-8} y1={y(0)} y2={y(0)} stroke="#d1d5db" />
      )}
      {/* area */}
      {pts.length > 1 && (
        <path d={`${d} L ${x(pts[pts.length-1].t)} ${height-30} L ${x(pts[0].t)} ${height-30} Z`} fill={fill} />
      )}
      {/* line */}
      {pts.length > 1 && <path d={d} stroke={stroke} fill="none" strokeWidth={1.8} />}
      {/* labels */}
      <text x={40} y={12} className="fill-neutral-500 text-[11px]">{asPercent ? `${(yMax*100).toFixed(1)}%` : yMax.toFixed(2)}</text>
      <text x={40} y={height-16} className="fill-neutral-500 text-[11px]">{asPercent ? `${(yMin*100).toFixed(1)}%` : yMin.toFixed(2)}</text>
    </svg>
  );
};

const StackedBar: React.FC<{ data: Allocation[] }> = ({ data }) => {
  const total = Math.max(1e-9, data.reduce((a, d) => a + Math.max(0, d.weight), 0));
  return (
    <div className="flex h-6 w-full overflow-hidden rounded-md border border-neutral-200">
      {data.map((d) => (
        <div
          key={d.bucket}
          title={`${d.bucket}: ${(d.weight * 100).toFixed(1)}%`}
          className="h-full"
          style={{
            width: `${(Math.max(0, d.weight) / total) * 100}%`,
            background: hashColor(d.bucket),
          }}
        />
      ))}
    </div>
  );
};

function hashColor(s: string) {
  // deterministic pleasant color from string
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
  const hue = Math.abs(h) % 360;
  return `hsl(${hue} 65% 60%)`;
}

const PositionsTable: React.FC<{ rows: Position[] }> = ({ rows }) => (
  <div className="overflow-auto">
    <table className="min-w-full border-collapse text-sm">
      <thead>
        <tr className="border-b bg-neutral-50 text-left text-xs font-semibold uppercase tracking-wide text-neutral-600">
          <th className="px-3 py-2">Name</th>
          <th className="px-3 py-2">Side</th>
          <th className="px-3 py-2">Sector</th>
          <th className="px-3 py-2">Region</th>
          <th className="px-3 py-2 text-right">Weight</th>
          <th className="px-3 py-2 text-right">Notional</th>
          <th className="px-3 py-2 text-right">PnL</th>
        </tr>
      </thead>
      <tbody>
        {rows.length === 0 ? (
          <tr>
            <td colSpan={7} className="px-3 py-6 text-center text-neutral-500">No positions</td>
          </tr>
        ) : (
          rows.map((p) => (
            <tr key={p.id} className="border-b last:border-0 hover:bg-neutral-50/60">
              <td className="px-3 py-2">{p.name}</td>
              <td className={`px-3 py-2 ${p.side === "Long" ? "text-green-700" : "text-red-700"}`}>{p.side}</td>
              <td className="px-3 py-2">{p.sector ?? "—"}</td>
              <td className="px-3 py-2">{p.region ?? "—"}</td>
              <td className="px-3 py-2 text-right tabular-nums">{p.weight != null ? fmtPct(p.weight) : "—"}</td>
              <td className="px-3 py-2 text-right tabular-nums">{p.notional != null ? fmtNum(p.notional, 0) : "—"}</td>
              <td className={`px-3 py-2 text-right tabular-nums ${Number(p.pnl) < 0 ? "text-red-700" : "text-green-700"}`}>
                {p.pnl != null ? fmtNum(p.pnl, 0) : "—"}
              </td>
            </tr>
          ))
        )}
      </tbody>
    </table>
  </div>
);

/* =========================
 * Main component
 * ========================= */

const PortfolioOverview: React.FC<PortfolioOverviewProps> = ({
  title = "Portfolio Overview",
  asOf,
  kpis = [],
  performance = [],
  asPercent = false,
  allocations = [],
  risk = {},
  topPositions = [],
}) => {
  const right = (
    <div className="text-xs text-neutral-500">
      As of {asOf ? new Date(asOf).toLocaleString() : "—"}
    </div>
  );

  return (
    <div className="w-full space-y-4">
      {/* Header card with KPIs */}
      <Card title={title} right={right}>
        {kpis.length > 0 ? <KPIGrid items={kpis} /> : <div className="text-sm text-neutral-500">Add KPIs to see snapshot metrics.</div>}
      </Card>

      {/* Row: Performance + Allocations + Risk */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Card title="Performance">
          <SparkLine data={performance} asPercent={asPercent} />
        </Card>

        <Card title="Allocations">
          {allocations.length ? (
            <div className="space-y-3">
              <StackedBar data={allocations} />
              <div className="grid grid-cols-2 gap-2 text-sm">
                {allocations.map((a) => (
                  <div key={a.bucket} className="flex items-center justify-between rounded-md border border-neutral-200 px-2 py-1">
                    <div className="flex items-center gap-2">
                      <span className="inline-block h-2 w-4 rounded-sm" style={{ background: hashColor(a.bucket) }} />
                      {a.bucket}
                    </div>
                    <div className="font-mono">{fmtPct(a.weight)}</div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="text-sm text-neutral-500">No allocation data</div>
          )}
        </Card>

        <Card title="Risk Snapshot">
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <div className="text-[11px] uppercase text-neutral-500">Gross</div>
              <div className="font-medium">{fmtNum(risk.gross)}×</div>
            </div>
            <div>
              <div className="text-[11px] uppercase text-neutral-500">Net</div>
              <div className="font-medium">{fmtNum(risk.net)}×</div>
            </div>
            <div>
              <div className="text-[11px] uppercase text-neutral-500">VaR (95%)</div>
              <div className="font-medium">{fmtPct(risk.var95)}</div>
            </div>
            <div>
              <div className="text-[11px] uppercase text-neutral-500">Vol</div>
              <div className="font-medium">{fmtPct(risk.vol)}</div>
            </div>
            <div>
              <div className="text-[11px] uppercase text-neutral-500">Beta</div>
              <div className="font-medium">{fmtNum(risk.beta)}</div>
            </div>
            <div>
              <div className="text-[11px] uppercase text-neutral-500">Max DD</div>
              <div className="font-medium">{fmtPct(risk.maxDD)}</div>
            </div>
          </div>
        </Card>
      </div>

      {/* Top positions */}
      <Card title="Top Positions">
        <PositionsTable rows={topPositions} />
      </Card>
    </div>
  );
};

export default PortfolioOverview;