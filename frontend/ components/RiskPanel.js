// frontend/components/riskpanel.js
// Production‑ready Risk Panel for your dashboard.
// - Pulls (or accepts) portfolio risk metrics, exposures, limits, and correlations
// - Shows KPI cards, exposure bars, VaR time‑series, limit breaches, and a correlation heatmap
// - Dark‑mode friendly, responsive, no external state libs
//
// Data can be passed via props or fetched from `/api/risk`.
// Shapes documented below.

import React, { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  Bar,
  Line,
  ReferenceLine,
} from "recharts";

/**
 * ---- Expected data shapes ----
 * RiskPayload = {
 *   kpis: {
 *     net_exposure: number,    // % of NAV (e.g., 27.5 = 27.5%)
 *     gross_exposure: number,  // % of NAV
 *     leverage: number,        // gross / NAV
 *     var_1d_p99: number,      // $ (or base currency)
 *     drawdown: number,        // % (negative or positive number; negative preferred)
 *     margin_util: number      // % used
 *   },
 *   exposures: {
 *     by_asset: Array<{ name: string, long: number, short: number }>, // % NAV
 *     by_region: Array<{ name: string, long: number, short: number }>,
 *     by_strategy: Array<{ name: string, long: number, short: number }>
 *   },
 *   var_series: Array<{ ts: number|Date|string, var_p99: number, var_p95?: number }>,
 *   limits: Array<{
 *     key: string, label: string, value: number, limit: number, // value/limit in same units (% or abs)
 *     unit: "%" | "$",
 *     severity?: "low" | "med" | "high"
 *   }>,
 *   correlations: { symbols: string[], matrix: number[][] } // numbers in [-1, 1]
 * }
 */

// ---------- Formatters ----------
const fmtPct = (v) =>
  Number.isFinite(v) ? `${(v >= 0 ? "" : "−")}${Math.abs(v).toFixed(1)}%` : "—";
const fmtMoney = (v) => {
  if (!Number.isFinite(v)) return "—";
  return new Intl.NumberFormat(undefined, {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(v);
};
const fmtDate = (t) => {
  const d = new Date(t);
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "2-digit" }).format(d);
};

// ---------- KPI Card ----------
function Kpi({ label, value, hint, positiveGood = true }) {
  const isPct = typeof value === "string" && value.endsWith("%");
  const isNeg = (typeof value === "number" && value < 0) || (isPct && value.startsWith("−"));
  const color =
    positiveGood
      ? isNeg
        ? "text-rose-600 dark:text-rose-400"
        : "text-emerald-600 dark:text-emerald-400"
      : isNeg
      ? "text-emerald-600 dark:text-emerald-400"
      : "text-rose-600 dark:text-rose-400";
  return (
    <div className="rounded-2xl border border-black/5 dark:border-white/10 bg-white dark:bg-zinc-950 p-4">
      <div className="text-xs text-zinc-500 dark:text-zinc-400 mb-1">{label}</div>
      <div className={`text-xl font-semibold ${color}`}>{value}</div>
      {hint && <div className="text-xs text-zinc-500 dark:text-zinc-400 mt-1">{hint}</div>}
    </div>
  );
}

// ---------- Limit Bar ----------
function LimitRow({ label, value, limit, unit = "%", severity }) {
  const pct = Math.max(0, Math.min(1, (limit ? value / limit : 0)));
  const over = limit && value > limit;
  const barClass = over
    ? "bg-rose-500"
    : severity === "high"
    ? "bg-amber-500"
    : "bg-emerald-500";
  return (
    <div className="w-full">
      <div className="flex justify-between text-xs mb-1">
        <span className="text-zinc-600 dark:text-zinc-300">{label}</span>
        <span className={`font-medium ${over ? "text-rose-600 dark:text-rose-400" : ""}`}>
          {unit === "%" ? fmtPct(value) : fmtMoney(value)} / {unit === "%" ? fmtPct(limit) : fmtMoney(limit)}
        </span>
      </div>
      <div className="h-2.5 rounded-full bg-zinc-200 dark:bg-zinc-800 overflow-hidden">
        <div
          className={`h-full ${barClass}`}
          style={{ width: `${Math.min(100, Math.round(pct * 100))}%` }}
        />
      </div>
    </div>
  );
}

// ---------- Correlation Heat Cell ----------
function corrColor(v) {
  // blue (−1) → white (0) → red (+1)
  const x = Math.max(-1, Math.min(1, Number(v)));
  const r = x > 0 ? Math.round(255 * x) : 0;
  const b = x < 0 ? Math.round(255 * -x) : 0;
  const g = 255 - Math.round(155 * Math.abs(x));
  return `rgb(${r},${g},${b})`;
}

// ---------- Main Component ----------
export default function RiskPanel({ data, endpoint = "/api/risk", height = 240 }) {
  const [remote, setRemote] = useState(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);

  // Fetch if no data prop
  useEffect(() => {
    let mounted = true;
    if (data) return;
    (async () => {
      try {
        setLoading(true);
        setErr("");
        const res = await fetch(endpoint, { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        if (mounted) setRemote(json?.data ?? json);
      } catch (e) {
        if (mounted) setErr(e?.message ?? "Failed to load risk");
      } finally {
        if (mounted) setLoading(false);
      }
    })();
    return () => {
      mounted = false;
    };
  }, [endpoint, data]);

  const risk = data || remote;

  // Derived
  const varRows = useMemo(() => {
    const rows = (risk?.var_series ?? []).map((r) => ({
      ts: r.ts instanceof Date ? r.ts.getTime() : typeof r.ts === "string" ? Date.parse(r.ts) : Number(r.ts),
      var_p99: Number(r.var_p99 ?? 0),
      var_p95: Number(r.var_p95 ?? 0),
    }));
    return rows.sort((a, b) => a.ts - b.ts);
  }, [risk]);

  const exposureBars = useMemo(() => {
    const pick = (risk?.exposures?.by_asset?.length ? risk.exposures.by_asset : risk?.exposures?.by_region) ?? [];
    // recharts consumes positive values for stacked bars; we keep signed for tooltips and render abs on chart
    return pick.map((r) => ({ name: r.name, long: Math.max(0, r.long), short: Math.abs(Math.min(0, r.short)) }));
  }, [risk]);

  if (loading && !risk) {
    return (
      <div className="w-full h-40 flex items-center justify-center text-sm text-zinc-500 dark:text-zinc-400">
        Loading risk…
      </div>
    );
  }
  if (err && !risk) {
    return (
      <div className="w-full rounded-xl border border-rose-300/40 bg-rose-50 dark:bg-rose-950/20 p-4 text-rose-600 dark:text-rose-300">
        Failed to load risk: {err}
      </div>
    );
  }
  if (!risk) return null;

  const k = risk.kpis || {};
  const limits = risk.limits || [];
  const corr = risk.correlations || { symbols: [], matrix: [] };

  return (
    <div className="w-full grid gap-6">
      {/* KPI Row */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <Kpi label="Net Exposure" value={fmtPct(k.net_exposure)} hint="(% of NAV)" positiveGood={false} />
        <Kpi label="Gross Exposure" value={fmtPct(k.gross_exposure)} hint="(% of NAV)" positiveGood={false} />
        <Kpi label="Leverage" value={Number.isFinite(k.leverage) ? k.leverage.toFixed(2) + "×" : "—"} />
        <Kpi label="1‑Day VaR (99%)" value={fmtMoney(k.var_1d_p99)} hint="Parametric / Monte Carlo" positiveGood={false} />
        <Kpi label="Drawdown" value={fmtPct(k.drawdown)} hint="From peak" positiveGood />
        <Kpi label="Margin Util." value={fmtPct(k.margin_util)} hint="Clearing/Broker" positiveGood={false} />
      </div>

      {/* Exposure & VaR */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Exposure by Asset/Region */}
        <div className="rounded-2xl border border-black/5 dark:border-white/10 bg-white dark:bg-zinc-950 p-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-base md:text-lg font-semibold">Exposure Mix</h3>
            <div className="text-xs text-zinc-500 dark:text-zinc-400">(% NAV, long vs short)</div>
          </div>
          <div style={{ width: "100%", height }}>
            <ResponsiveContainer>
              <ComposedChart data={exposureBars} margin={{ top: 10, right: 24, bottom: 0, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                <XAxis dataKey="name" tick={{ fontSize: 11, fill: "currentColor", opacity: 0.7 }} />
                <YAxis
                  tickFormatter={(v) => `${v.toFixed(0)}%`}
                  tick={{ fontSize: 11, fill: "currentColor", opacity: 0.7 }}
                  axisLine={{ stroke: "currentColor", opacity: 0.2 }}
                  tickLine={{ stroke: "currentColor", opacity: 0.2 }}
                />
                <Tooltip
                  formatter={(v, k) => [`${v.toFixed(1)}%`, k === "long" ? "Long" : "Short"]}
                  labelFormatter={(n) => String(n)}
                />
                <Legend />
                <Bar name="Long" dataKey="long" stackId="x" fill="currentColor" fillOpacity={0.35} />
                <Bar name="Short" dataKey="short" stackId="x" fill="currentColor" fillOpacity={0.15} />
                <ReferenceLine y={0} stroke="currentColor" strokeOpacity={0.15} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* VaR series */}
        <div className="rounded-2xl border border-black/5 dark:border-white/10 bg-white dark:bg-zinc-950 p-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-base md:text-lg font-semibold">VaR Trend</h3>
            <div className="text-xs text-zinc-500 dark:text-zinc-400">1‑day VaR (USD)</div>
          </div>
          <div style={{ width: "100%", height }}>
            <ResponsiveContainer>
              <ComposedChart data={varRows} margin={{ top: 10, right: 24, bottom: 0, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.06)" />
                <XAxis
                  dataKey="ts"
                  tickFormatter={fmtDate}
                  type="number"
                  domain={["dataMin", "dataMax"]}
                  scale="time"
                  tick={{ fontSize: 11, fill: "currentColor", opacity: 0.7 }}
                />
                <YAxis
                  tickFormatter={fmtMoney}
                  tick={{ fontSize: 11, fill: "currentColor", opacity: 0.7 }}
                  width={70}
                />
                <Tooltip
                  formatter={(v, k) => [fmtMoney(v), k === "var_p99" ? "VaR 99%" : "VaR 95%"]}
                  labelFormatter={(ts) => fmtDate(ts)}
                />
                <Legend />
                <Line name="VaR 99%" dataKey="var_p99" stroke="currentColor" strokeWidth={2} dot={false} />
                {varRows.some((r) => Number.isFinite(r.var_p95)) && (
                  <Line name="VaR 95%" dataKey="var_p95" stroke="currentColor" strokeOpacity={0.4} dot={false} />
                )}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Limits & Correlations */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Limits */}
        <div className="rounded-2xl border border-black/5 dark:border-white/10 bg-white dark:bg-zinc-950 p-4">
          <h3 className="text-base md:text-lg font-semibold mb-3">Limits & Breaches</h3>
          <div className="grid gap-3">
            {limits.length ? (
              limits.map((l) => (
                <LimitRow key={l.key} label={l.label} value={l.value} limit={l.limit} unit={l.unit} severity={l.severity} />
              ))
            ) : (
              <div className="text-sm text-zinc-500 dark:text-zinc-400">No limits provided.</div>
            )}
          </div>
        </div>

        {/* Correlation Heatmap */}
        <div className="rounded-2xl border border-black/5 dark:border-white/10 bg-white dark:bg-zinc-950 p-4 overflow-auto">
          <h3 className="text-base md:text-lg font-semibold mb-3">Correlation Heatmap</h3>
          {corr.symbols?.length ? (
            <table className="text-xs">
              <thead>
                <tr>
                  <th className="sticky left-0 bg-white dark:bg-zinc-950 z-10"></th>
                  {corr.symbols.map((s) => (
                    <th key={s} className="px-2 py-1 text-right whitespace-nowrap">{s}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {corr.symbols.map((rowSym, i) => (
                  <tr key={rowSym}>
                    <th className="sticky left-0 bg-white dark:bg-zinc-950 pr-2 py-1 text-right whitespace-nowrap z-10">
                      {rowSym}
                    </th>
                    {corr.matrix[i]?.map((v, j) => (
                      <td key={`${i}-${j}`} className="w-8 h-6 text-center align-middle">
                        <div
                          className="w-8 h-6 rounded-sm border border-black/5 dark:border-white/10"
                          title={`${rowSym}↔${corr.symbols[j]}: ${v?.toFixed?.(2) ?? "—"}`}
                          style={{ background: corrColor(v), color: "rgba(0,0,0,0.65)" }}
                        >
                          {/* optional numbers: <span className="text-[10px]">{(v??0).toFixed(1)}</span> */}
                        </div>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="text-sm text-zinc-500 dark:text-zinc-400">No correlation matrix.</div>
          )}
        </div>
      </div>
    </div>
  );
}