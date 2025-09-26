"use client";

import React, { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";

/** One weekly inventory print for a commodity/region */
export type InventoryPoint = {
  date: string;        // YYYY-MM-DD (week ending)
  level: number;       // inventory level (e.g., million bbl / tonnes)
  change: number;      // WoW change (+build / -draw) in same units
  daysCover?: number;  // optional coverage metric
};

type Props = {
  title?: string;
  units?: string;          // e.g., "mn bbl", "kt"
  data?: InventoryPoint[];
  fetchUrl?: string;       // API returning { data: InventoryPoint[] }
  height?: number;         // chart height
};

const fmt = (n: number, d = 2) => n.toLocaleString("en-IN", { maximumFractionDigits: d });
const short = (iso: string) => {
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(y, m - 1, d).toLocaleDateString(undefined, { month: "short", day: "2-digit" });
};

export default function InventoryFlowsPane({
  title = "Inventory Flows",
  units = "mn bbl",
  data,
  fetchUrl,
  height = 260,
}: Props) {
  const [remote, setRemote] = useState<InventoryPoint[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

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
        setErr(e?.message || "Failed to fetch inventory");
      } finally {
        setLoading(false);
      }
    })();
  }, [fetchUrl]);

  const rows = useMemo(() => (data ?? remote ?? []).slice().sort((a, b) => a.date.localeCompare(b.date)), [data, remote]);

  const last = rows.at(-1);
  const prev = rows.at(-2);

  const summary = useMemo(() => {
    const level = last?.level ?? 0;
    const wow = last?.change ?? 0;
    const cover = last?.daysCover ?? undefined;
    const mtd = rows
      .filter((r) => last && r.date.slice(0, 7) === last.date.slice(0, 7))
      .reduce((acc, r) => acc + r.change, 0);
    return { level, wow, cover, mtd };
  }, [rows, last]);

  return (
    <div className="w-full rounded-xl border border-neutral-800 bg-neutral-900">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-neutral-800 px-4 py-3">
        <div className="space-y-0.5">
          <h3 className="text-sm font-semibold text-neutral-200">{title}</h3>
          <p className="text-xs text-neutral-400">
            {rows.length ? `Latest: ${short(last!.date)} • ${fmt(last!.level)} ${units}` : "No data"}
            {loading ? " • loading…" : ""}
            {err ? ` • error: ${err}` : ""}
          </p>
        </div>
      </div>

      {/* Summary tiles */}
      <div className="grid grid-cols-1 gap-3 px-4 py-3 text-sm md:grid-cols-3">
        <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-3">
          <div className="text-neutral-400">Inventory Level</div>
          <div className="mt-1 text-lg font-semibold text-neutral-100">
            {fmt(summary.level)} <span className="text-xs font-normal text-neutral-400">{units}</span>
          </div>
        </div>
        <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-3">
          <div className="text-neutral-400">WoW Change</div>
          <div className={`mt-1 text-lg font-semibold ${summary.wow >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
            {summary.wow >= 0 ? "+" : ""}
            {fmt(summary.wow)} <span className="text-xs font-normal text-neutral-400">{units}</span>
          </div>
        </div>
        <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-3">
          <div className="text-neutral-400">MTD Build/Draw</div>
          <div className={`mt-1 text-lg font-semibold ${summary.mtd >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
            {summary.mtd >= 0 ? "+" : ""}
            {fmt(summary.mtd)} <span className="text-xs font-normal text-neutral-400">{units}</span>
          </div>
        </div>
      </div>

      {/* Chart */}
      <div style={{ height }} className="px-2 pb-2">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows.map((r) => ({ ...r, x: short(r.date) }))} margin={{ left: 12, right: 18, top: 8, bottom: 8 }}>
            <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
            <XAxis dataKey="x" tick={{ fill: "#a3a3a3", fontSize: 12 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: "#a3a3a3", fontSize: 12 }} axisLine={false} tickLine={false} width={56} />
            <Tooltip
              contentStyle={{ background: "#0a0a0a", border: "1px solid #262626" }}
              labelStyle={{ color: "#a3a3a3" }}
              formatter={(v: any, name: string) =>
                name === "change" ? [`${fmt(Number(v))} ${units}`, "WoW Change"] : [`${fmt(Number(v))} ${units}`, "Level"]
              }
            />
            <ReferenceLine y={0} stroke="#6b7280" />
            <Bar dataKey="change" name="WoW Change" fill="#10b981" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Table */}
      <div className="border-t border-neutral-800">
        <table className="min-w-full text-sm">
          <thead className="bg-neutral-800/60 text-neutral-400">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Week</th>
              <th className="px-3 py-2 text-right font-medium">Level ({units})</th>
              <th className="px-3 py-2 text-right font-medium">WoW Δ</th>
              <th className="px-3 py-2 text-right font-medium">Days Cover</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(-12).reverse().map((r) => (
              <tr key={r.date} className="border-t border-neutral-800">
                <td className="px-3 py-2">{short(r.date)}</td>
                <td className="px-3 py-2 text-right">{fmt(r.level)}</td>
                <td className={`px-3 py-2 text-right ${r.change >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                  {r.change >= 0 ? "+" : ""}
                  {fmt(r.change)}
                </td>
                <td className="px-3 py-2 text-right">{r.daysCover != null ? fmt(r.daysCover) : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}