"use client";

import React, { useMemo, useState } from "react";

/* ================================
 * Types
 * ================================ */

export type RegimeLabel =
  | "Risk-On"
  | "Risk-Off"
  | "Inflationary"
  | "Disinflation"
  | "Tightening"
  | "Easing"
  | "Stagflation"
  | "Growth";

export type RegimePoint = {
  /** ISO date (YYYY-MM-DD) or Date */
  date: string | Date;
  /** Regime label for that day */
  regime: RegimeLabel;
  /** Composite score (higher = more risk-on by convention) */
  score?: number;
  /** Realized / implied volatility (annualized %) */
  vol?: number;
  /** Peak-to-trough drawdown over a rolling window (fraction, e.g., -0.12) */
  drawdown?: number;
  /** Carry / term premium proxy (bps or %) */
  carry?: number;
  /** Growth proxy (PMI z-score, EPS breadth, etc.) */
  growth?: number;
  /** Inflation proxy (breakevens z-score, CPI surprise, etc.) */
  inflation?: number;
};

export interface RegimePanelProps {
  series: RegimePoint[];   // daily (or periodic) time-series
  title?: string;
}

/* ================================
 * Small local UI bits (no deps)
 * ================================ */

const Card: React.FC<React.PropsWithChildren<{ className?: string }>> = ({ className, children }) => (
  <div className={`rounded-2xl border border-neutral-200/70 bg-white shadow ${className ?? ""}`}>{children}</div>
);

const CardHeader: React.FC<React.PropsWithChildren<{ className?: string }>> = ({ className, children }) => (
  <div className={`flex flex-wrap items-center justify-between gap-3 border-b px-4 py-3 ${className ?? ""}`}>
    {children}
  </div>
);

const CardTitle: React.FC<React.PropsWithChildren> = ({ children }) => (
  <h2 className="text-lg font-semibold">{children}</h2>
);

const CardContent: React.FC<React.PropsWithChildren<{ className?: string }>> = ({ className, children }) => (
  <div className={`px-4 py-3 ${className ?? ""}`}>{children}</div>
);

const Badge: React.FC<React.PropsWithChildren<{ tone?: "neutral" | "green" | "red" | "amber" }>> = ({
  children,
  tone = "neutral",
}) => {
  const tones: Record<string, string> = {
    neutral: "bg-neutral-100 text-neutral-800",
    green: "bg-green-100 text-green-800",
    red: "bg-red-100 text-red-800",
    amber: "bg-amber-100 text-amber-800",
  };
  return (
    <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${tones[tone]}`}>
      {children}
    </span>
  );
};

/* ================================
 * Utilities
 * ================================ */

const toDate = (d: string | Date): Date => (d instanceof Date ? d : new Date(d));
const fmtDate = (d: Date) => d.toISOString().slice(0, 10);
const fmtPct = (x?: number, d = 1) =>
  x === undefined || !Number.isFinite(x) ? "–" : `${(x * 100).toFixed(d)}%`;
const fmt = (x?: number, d = 2) => (x === undefined || !Number.isFinite(x) ? "–" : x.toFixed(d));

function sparkPath(values: number[], w = 160, h = 40, pad = 4): string {
  if (values.length === 0) return "";
  const xs = values.map((v) => (Number.isFinite(v) ? v : NaN)).filter((v) => !Number.isNaN(v));
  if (xs.length === 0) return "";
  const min = Math.min(...xs);
  const max = Math.max(...xs);
  const n = values.length;
  const xStep = (w - 2 * pad) / Math.max(1, n - 1);
  const yScale = (v: number) =>
    max === min ? h / 2 : h - pad - ((v - min) / (max - min)) * (h - 2 * pad);
  let d = "";
  values.forEach((v, i) => {
    const x = pad + i * xStep;
    const y = yScale(v);
    d += i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`;
  });
  return d;
}

/* ================================
 * Main component
 * ================================ */

const RegimePanel: React.FC<RegimePanelProps> = ({ series, title = "Market Regimes" }) => {
  // Controls
  const [metric, setMetric] = useState<keyof RegimePoint>("score"); // which sparkline/column to show
  const [sortKey, setSortKey] = useState<"date" | "regime" | "metric">("date");
  const [sortAsc, setSortAsc] = useState<boolean>(false);
  const [from, setFrom] = useState<string>("");
  const [to, setTo] = useState<string>("");

  // Normalize + filter by date
  const rows = useMemo(() => {
    const normalized = series
      .map((r) => ({
        ...r,
        _date: toDate(r.date),
        _metric:
          (metric === "drawdown" ? r.drawdown :
           metric === "vol" ? r.vol :
           metric === "carry" ? r.carry :
           metric === "growth" ? r.growth :
           metric === "inflation" ? r.inflation :
           r.score) ?? NaN,
      }))
      .filter((r) => (from ? toDate(from) <= r._date : true) && (to ? r._date <= toDate(to) : true));
    // sort
    return normalized.sort((a, b) => {
      if (sortKey === "date") return sortAsc ? +a._date - +b._date : +b._date - +a._date;
      if (sortKey === "regime")
        return sortAsc
          ? String(a.regime).localeCompare(String(b.regime))
          : String(b.regime).localeCompare(String(a.regime));
      // metric
      const av = Number.isFinite(a._metric) ? a._metric : -Infinity;
      const bv = Number.isFinite(b._metric) ? b._metric : -Infinity;
      return sortAsc ? (av - bv) : (bv - av);
    });
  }, [series, metric, from, to, sortKey, sortAsc]);

  // Current regime (last valid row)
  const current = useMemo(() => {
    const valid = rows.filter((r) => Number.isFinite(r._metric) || r.regime);
    return valid.length ? valid[valid.length - 1] : undefined;
  }, [rows]);

  // Distribution by label
  const distribution = useMemo(() => {
    const counts = new Map<RegimeLabel, number>();
    rows.forEach((r) => counts.set(r.regime, (counts.get(r.regime) ?? 0) + 1));
    const total = rows.length || 1;
    return Array.from(counts.entries())
      .map(([regime, n]) => ({ regime, n, pct: n / total }))
      .sort((a, b) => b.n - a.n);
  }, [rows]);

  // Recent switches (where regime changes)
  const switches = useMemo(() => {
    const sw: { date: Date; from: RegimeLabel; to: RegimeLabel }[] = [];
    for (let i = 1; i < rows.length; i++) {
      if (rows[i].regime !== rows[i - 1].regime) {
        sw.push({ date: rows[i]._date, from: rows[i - 1].regime, to: rows[i].regime });
      }
    }
    return sw.slice(-10); // last 10 switches
  }, [rows]);

  // Sparkline values for selected metric
  const sparkVals = useMemo(() => rows.map((r) => (Number.isFinite(r._metric) ? (r._metric as number) : NaN)), [rows]);

  // Tone by regime (for badge color)
  const toneFor = (regime?: RegimeLabel): "neutral" | "green" | "red" | "amber" => {
    switch (regime) {
      case "Risk-On":
      case "Easing":
      case "Growth":
        return "green";
      case "Risk-Off":
      case "Tightening":
      case "Stagflation":
        return "red";
      case "Inflationary":
      case "Disinflation":
        return "amber";
      default:
        return "neutral";
    }
  };

  // Helpers
  const headers: { key: "date" | "regime" | "metric"; label: string }[] = [
    { key: "date", label: "Date" },
    { key: "regime", label: "Regime" },
    { key: "metric", label: metric.toUpperCase() },
  ];
  const headerClick = (k: typeof sortKey) => {
    if (k === sortKey) setSortAsc((s) => !s);
    else {
      setSortKey(k);
      setSortAsc(true);
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>{title}</CardTitle>

        <div className="flex flex-wrap items-center gap-2">
          <label className="text-sm text-neutral-600">Metric</label>
          <select
            value={metric}
            onChange={(e) => setMetric(e.target.value as keyof RegimePoint)}
            className="h-9 rounded-md border border-neutral-300 bg-white px-2 text-sm outline-none focus:border-neutral-500"
          >
            <option value="score">Score</option>
            <option value="vol">Vol</option>
            <option value="drawdown">Drawdown</option>
            <option value="carry">Carry</option>
            <option value="growth">Growth</option>
            <option value="inflation">Inflation</option>
          </select>

          <label className="ml-2 text-sm text-neutral-600">From</label>
          <input
            type="date"
            value={from}
            onChange={(e) => setFrom(e.target.value)}
            className="h-9 rounded-md border border-neutral-300 bg-white px-2 text-sm outline-none focus:border-neutral-500"
          />
          <label className="text-sm text-neutral-600">To</label>
          <input
            type="date"
            value={to}
            onChange={(e) => setTo(e.target.value)}
            className="h-9 rounded-md border border-neutral-300 bg-white px-2 text-sm outline-none focus:border-neutral-500"
          />
        </div>
      </CardHeader>

      {/* Top summary */}
      <CardContent className="grid grid-cols-1 gap-3 md:grid-cols-3">
        <div className="rounded-lg border border-neutral-200 p-3">
          <div className="text-xs uppercase text-neutral-500">Current Regime</div>
          <div className="mt-1 flex items-center gap-2">
            <Badge tone={toneFor(current?.regime)}>{current?.regime ?? "—"}</Badge>
            <div className="text-xs text-neutral-500">{current ? fmtDate(current._date) : ""}</div>
          </div>
          <div className="mt-2 text-sm text-neutral-700">
            {metric.toUpperCase()}:{" "}
            <span className="font-mono">
              {metric === "drawdown"
                ? fmtPct(current?._metric)
                : metric === "vol"
                ? fmt(current?._metric)
                : fmt(current?._metric)}
            </span>
          </div>
        </div>

        <div className="rounded-lg border border-neutral-200 p-3">
          <div className="text-xs uppercase text-neutral-500">Distribution</div>
          <div className="mt-2 grid grid-cols-2 gap-y-1 text-sm">
            {distribution.slice(0, 6).map((d) => (
              <div key={d.regime} className="flex items-center justify-between gap-2">
                <span className="truncate">{d.regime}</span>
                <span className="font-mono text-neutral-700">{(d.pct * 100).toFixed(1)}%</span>
              </div>
            ))}
            {distribution.length === 0 && <div className="text-neutral-500">No data</div>}
          </div>
        </div>

        <div className="rounded-lg border border-neutral-200 p-3">
          <div className="mb-2 flex items-center justify-between">
            <div className="text-xs uppercase text-neutral-500">Trend ({metric.toUpperCase()})</div>
            <div className="text-[10px] text-neutral-500">{rows.length} pts</div>
          </div>
          <svg viewBox="0 0 160 40" width="100%" height="40">
            <path
              d={sparkPath(sparkVals.filter((v) => Number.isFinite(v)) as number[])}
              stroke="#0ea5e9"
              strokeWidth="1.5"
              fill="none"
            />
          </svg>
        </div>
      </CardContent>

      {/* Table */}
      <CardContent className="overflow-x-auto">
        <table className="min-w-full border-collapse">
          <thead>
            <tr className="border-b bg-neutral-50 text-left text-xs font-semibold uppercase tracking-wide text-neutral-600">
              {headers.map((h) => (
                <th
                  key={h.key}
                  className="cursor-pointer px-3 py-2"
                  onClick={() => headerClick(h.key)}
                  title="Click to sort"
                >
                  <span className="inline-flex items-center gap-1">
                    {h.label}
                    {sortKey === h.key && (
                      <span className="text-[10px] text-neutral-500">{sortAsc ? "▲" : "▼"}</span>
                    )}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="text-sm">
            {rows.length === 0 ? (
              <tr>
                <td className="px-3 py-6 text-center text-neutral-500" colSpan={headers.length}>
                  No rows in range
                </td>
              </tr>
            ) : (
              rows.map((r, i) => (
                <tr key={`${fmtDate(r._date)}-${i}`} className="border-b last:border-0 hover:bg-neutral-50/60">
                  <td className="px-3 py-2">{fmtDate(r._date)}</td>
                  <td className="px-3 py-2">
                    <Badge tone={toneFor(r.regime)}>{r.regime}</Badge>
                  </td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums">
                    {metric === "drawdown"
                      ? fmtPct(r._metric)
                      : metric === "vol"
                      ? fmt(r._metric)
                      : fmt(r._metric)}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </CardContent>

      {/* Recent switches */}
      <CardContent>
        <div className="mb-2 text-xs uppercase text-neutral-500">Recent Regime Switches</div>
        {switches.length === 0 ? (
          <div className="text-sm text-neutral-500">No switches.</div>
        ) : (
          <ul className="space-y-1 text-sm">
            {switches.map((s, i) => (
              <li key={i} className="flex items-center gap-2">
                <span className="font-mono text-neutral-600">{fmtDate(s.date)}</span>
                <span className="text-neutral-700">•</span>
                <Badge tone={toneFor(s.from)}>{s.from}</Badge>
                <span className="text-neutral-500">→</span>
                <Badge tone={toneFor(s.to)}>{s.to}</Badge>
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
};

export default RegimePanel;