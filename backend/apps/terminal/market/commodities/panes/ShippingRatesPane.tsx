"use client";

import React, { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";

/** One daily/weekly freight rate print (e.g., spot container rate per FEU) */
export type ShippingRatePoint = {
  date: string;        // YYYY-MM-DD
  lane: string;        // e.g., "CN->USWC", "CN->USEC", "EU->US"
  rateUsd: number;     // spot rate (USD / FEU or per chosen unit)
  change?: number;     // day/week change in USD
  index?: number;      // optional normalized index (100 = base)
};

type Props = {
  title?: string;
  unit?: string;            // e.g., "USD/FEU"
  data?: ShippingRatePoint[];
  fetchUrl?: string;        // API returning { data: ShippingRatePoint[] }
  defaultLane?: string;     // preselect lane
  height?: number;          // chart height
};

const fmt = (n: number, d = 0) => n.toLocaleString("en-US", { maximumFractionDigits: d });
const short = (iso: string) => {
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(y, m - 1, d).toLocaleDateString(undefined, { month: "short", day: "2-digit" });
};

export default function ShippingRatesPane({
  title = "Shipping Rates",
  unit = "USD/FEU",
  data,
  fetchUrl,
  defaultLane,
  height = 260,
}: Props) {
  const [remote, setRemote] = useState<ShippingRatePoint[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // Fetch remote (optional)
  useEffect(() => {
    if (!fetchUrl) return;
    (async () => {
      try {
        setLoading(true);
        setErr(null);
        const res = await fetch(fetchUrl, { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        setRemote(json.data ?? json ?? []);
      } catch (e: any) {
        setErr(e?.message || "Failed to fetch shipping rates");
      } finally {
        setLoading(false);
      }
    })();
  }, [fetchUrl]);

  // Rows & lanes
  const rows = useMemo(
    () => (data ?? remote ?? []).slice().sort((a, b) => a.date.localeCompare(b.date)),
    [data, remote]
  );

  const lanes = useMemo(() => Array.from(new Set(rows.map((r) => r.lane))).sort(), [rows]);
  const [lane, setLane] = useState<string>(defaultLane || lanes[0] || "");

  useEffect(() => {
    // reset lane when lanes list changes
    if (!lane && lanes.length) setLane(defaultLane && lanes.includes(defaultLane) ? defaultLane : lanes[0]);
  }, [lanes, lane, defaultLane]);

  const laneRows = useMemo(() => rows.filter((r) => (lane ? r.lane === lane : true)), [rows, lane]);
  const last = laneRows.at(-1);
  const prev = laneRows.at(-2);

  // Stats
  const stats = useMemo(() => {
    const latest = last?.rateUsd ?? 0;
    const wow = (last?.change ?? (last && prev ? last.rateUsd - prev.rateUsd : 0)) ?? 0;
    // 30-day avg (or 30 points)
    const tail = laneRows.slice(-30);
    const avg30 = tail.length ? tail.reduce((s, r) => s + r.rateUsd, 0) / tail.length : 0;
    const idx = last?.index ?? (avg30 ? (latest / avg30) * 100 : undefined);
    return { latest, wow, avg30, idx };
  }, [laneRows, last, prev]);

  // Chart rows
  const chartRows = useMemo(
    () =>
      laneRows.map((r) => ({
        x: short(r.date),
        rateUsd: +r.rateUsd.toFixed(0),
      })),
    [laneRows]
  );

  return (
    <div className="w-full rounded-xl border border-neutral-800 bg-neutral-900">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800 px-4 py-3">
        <div className="space-y-0.5">
          <h3 className="text-sm font-semibold text-neutral-200">{title}</h3>
          <p className="text-xs text-neutral-400">
            {lane || "All lanes"} · {laneRows.length} points
            {loading ? " · loading…" : ""}
            {err ? ` · error: ${err}` : ""}
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs">
          <select
            value={lane}
            onChange={(e) => setLane(e.target.value)}
            className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-200"
          >
            {lanes.length === 0 ? <option value="">No lanes</option> : null}
            {lanes.map((ln) => (
              <option key={ln} value={ln}>
                {ln}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Summary tiles */}
      <div className="grid grid-cols-1 gap-3 px-4 py-3 text-sm md:grid-cols-3">
        <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-3">
          <div className="text-neutral-400">Latest</div>
          <div className="mt-1 text-lg font-semibold text-neutral-100">
            ${fmt(stats.latest)} <span className="text-xs font-normal text-neutral-400">{unit}</span>
          </div>
        </div>
        <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-3">
          <div className="text-neutral-400">WoW Change</div>
          <div className={`mt-1 text-lg font-semibold ${stats.wow >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
            {stats.wow >= 0 ? "+" : ""}${fmt(stats.wow)}
          </div>
        </div>
        <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-3">
          <div className="text-neutral-400">30-Day Avg</div>
          <div className="mt-1 text-lg font-semibold text-neutral-100">
            ${fmt(stats.avg30)} <span className="text-xs font-normal text-neutral-400">{unit}</span>
          </div>
        </div>
      </div>

      {/* Chart */}
      <div style={{ height }} className="px-2 pb-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartRows} margin={{ left: 12, right: 18, top: 8, bottom: 8 }}>
            <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
            <XAxis dataKey="x" tick={{ fill: "#a3a3a3", fontSize: 12 }} axisLine={false} tickLine={false} />
            <YAxis
              tick={{ fill: "#a3a3a3", fontSize: 12 }}
              axisLine={false}
              tickLine={false}
              width={70}
              tickFormatter={(v) => `$${fmt(v)}`}
            />
            <Tooltip
              contentStyle={{ background: "#0a0a0a", border: "1px solid #262626" }}
              labelStyle={{ color: "#a3a3a3" }}
              formatter={(v: any) => [`$${fmt(Number(v))} ${unit}`, "Rate"]}
            />
            {stats.avg30 ? <ReferenceLine y={stats.avg30} stroke="#6b7280" strokeDasharray="4 4" /> : null}
            <Line type="monotone" dataKey="rateUsd" name="Rate" dot={false} stroke="#10b981" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Table */}
      <div className="border-t border-neutral-800">
        <table className="min-w-full text-sm">
          <thead className="bg-neutral-800/60 text-neutral-400">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Date</th>
              <th className="px-3 py-2 text-left font-medium">Lane</th>
              <th className="px-3 py-2 text-right font-medium">Rate ({unit})</th>
              <th className="px-3 py-2 text-right font-medium">Δ (USD)</th>
              <th className="px-3 py-2 text-right font-medium">Index</th>
            </tr>
          </thead>
          <tbody>
            {laneRows.slice(-20).reverse().map((r) => (
              <tr key={`${r.date}-${r.lane}`} className="border-t border-neutral-800">
                <td className="px-3 py-2">{short(r.date)}</td>
                <td className="px-3 py-2">{r.lane}</td>
                <td className="px-3 py-2 text-right">${fmt(r.rateUsd)}</td>
                <td className={`px-3 py-2 text-right ${((r.change ?? 0) >= 0 ? "text-emerald-400" : "text-rose-400")}`}>
                  {r.change != null ? `${r.change >= 0 ? "+" : ""}$${fmt(r.change)}` : "—"}
                </td>
                <td className="px-3 py-2 text-right">{r.index != null ? fmt(r.index, 2) : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}