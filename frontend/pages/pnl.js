// frontend/pages/pnl.js
// P&L page: loads PnL from /api/pnl (via api.js), shows chart + table, export CSV.
// Works with the PnLChart component you added earlier.

import React, { useEffect, useMemo, useState } from "react";
import * as api from "@/lib/api";
import PnLChart from "@/components/pnlchart";

function fmtMoney(v) {
  if (!Number.isFinite(v)) return "—";
  return new Intl.NumberFormat(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(v);
}
function fmtDate(ts) {
  const d = new Date(ts);
  return new Intl.DateTimeFormat(undefined, { year: "2-digit", month: "short", day: "2-digit" }).format(d);
}
function derive(rows) {
  // ensure numeric ts and compute cum/dd
  let cum = 0, peak = -Infinity;
  return rows
    .map((r) => {
      const ts =
        r.ts instanceof Date ? r.ts.getTime() : typeof r.ts === "string" ? Date.parse(r.ts) : Number(r.ts);
      const pnl = Number(r.pnl ?? 0);
      cum += pnl;
      peak = Math.max(peak, cum);
      const dd = cum - peak;
      return { ts, pnl, cum, dd };
    })
    .sort((a, b) => a.ts - b.ts);
}

export default function PnLPage() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [range, setRange] = useState(90); // days
  const [showTable, setShowTable] = useState(true);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        setLoading(true);
        setErr("");
        const data = await api.getPnL(); // expects {data:[{ts,pnl},...]} or array
        const arr = Array.isArray(data?.data) ? data.data : Array.isArray(data) ? data : [];
        if (mounted) setRows(arr);
      } catch (e) {
        if (mounted) setErr(e?.message ?? "Failed to load PnL");
      } finally {
        if (mounted) setLoading(false);
      }
    })();
    return () => { mounted = false; };
  }, []);

  const derived = useMemo(() => derive(rows), [rows]);

  const filtered = useMemo(() => {
    if (!derived.length) return [];
    const cutoff = Date.now() - range * 86400000;
    return derived.filter((r) => r.ts >= cutoff);
  }, [derived, range]);

  const kpis = useMemo(() => {
    if (!filtered.length) return { cum: 0, avg: 0, best: 0, worst: 0, dd: 0 };
    const cum = filtered[filtered.length - 1].cum - (filtered[0].cum - filtered[0].pnl);
    const avg = filtered.reduce((a, r) => a + r.pnl, 0) / filtered.length;
    const best = Math.max(...filtered.map((r) => r.pnl));
    const worst = Math.min(...filtered.map((r) => r.pnl));
    const dd = Math.min(...filtered.map((r) => r.dd));
    return { cum, avg, best, worst, dd };
  }, [filtered]);

  function exportCSV() {
    const header = "date,pnl,cumulative,drawdown\n";
    const lines = filtered.map((r) => [new Date(r.ts).toISOString().slice(0, 10), r.pnl, r.cum, r.dd].join(","));
    const blob = new Blob([header + lines.join("\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "pnl.csv"; a.click();
    URL.revokeObjectURL(url);
  }

  if (loading && !rows.length) {
    return <div className="p-6 text-sm text-zinc-500">Loading PnL…</div>;
  }
  if (err && !rows.length) {
    return <div className="p-6 text-rose-600">Failed to load PnL: {err}</div>;
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <h2 className="text-lg font-semibold">P&L</h2>
        <select
          className="px-3 py-1.5 rounded-xl border border-black/10 dark:border-white/10 bg-white/60 dark:bg-zinc-900/60 text-sm"
          value={range}
          onChange={(e) => setRange(Number(e.target.value))}
        >
          {[7, 30, 90, 180, 365].map((d) => (
            <option key={d} value={d}>{d} days</option>
          ))}
        </select>
        <button onClick={() => setShowTable((s) => !s)} className="px-3 py-1.5 rounded-xl border text-sm">
          {showTable ? "Hide" : "Show"} Table
        </button>
        <button onClick={exportCSV} className="px-3 py-1.5 rounded-xl border text-sm">
          Export CSV
        </button>
        <div className="ml-auto text-xs text-zinc-500 dark:text-zinc-400">
          {filtered.length ? `${fmtDate(filtered[0].ts)} — ${fmtDate(filtered[filtered.length - 1].ts)}` : "—"}
        </div>
      </div>

      {/* Chart */}
      <PnLChart data={filtered} />

      {/* KPI cards */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <Kpi label="Cumulative" value={fmtMoney(kpis.cum)} upGood />
        <Kpi label="Avg / Day" value={fmtMoney(kpis.avg)} upGood />
        <Kpi label="Best Day" value={fmtMoney(kpis.best)} upGood />
        <Kpi label="Worst Day" value={fmtMoney(kpis.worst)} upGood={false} />
        <Kpi label="Max Drawdown" value={fmtMoney(kpis.dd)} upGood={false} />
      </div>

      {/* Table */}
      {showTable && (
        <div className="overflow-x-auto rounded-2xl border border-black/5 dark:border-white/10 bg-white dark:bg-zinc-950">
          <table className="min-w-full text-sm">
            <thead className="bg-zinc-50/70 dark:bg-zinc-900/40 text-zinc-600 dark:text-zinc-300">
              <tr>
                <Th>Date</Th>
                <Th align="right">Daily PnL</Th>
                <Th align="right">Cumulative</Th>
                <Th align="right">Drawdown</Th>
              </tr>
            </thead>
            <tbody>
              {filtered
                .slice()
                .reverse()
                .map((r, i) => (
                  <tr key={i} className="odd:bg-zinc-50/30 dark:odd:bg-zinc-900/20">
                    <Td>{fmtDate(r.ts)}</Td>
                    <Td align="right" className={r.pnl >= 0 ? "text-emerald-600" : "text-rose-600"}>
                      {fmtMoney(r.pnl)}
                    </Td>
                    <Td align="right" className={r.cum >= 0 ? "text-emerald-600" : "text-rose-600"}>
                      {fmtMoney(r.cum)}
                    </Td>
                    <Td align="right" className={r.dd < 0 ? "text-rose-600" : "text-zinc-500"}>
                      {fmtMoney(r.dd)}
                    </Td>
                  </tr>
                ))}
              {!filtered.length && (
                <tr>
                  <Td colSpan={4} className="text-center text-zinc-500 py-6">
                    No data.
                  </Td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function Kpi({ label, value, upGood = true }) {
  const neg = typeof value === "string" ? value.startsWith("-") || value.startsWith("−") : false;
  const cls =
    upGood
      ? neg
        ? "text-rose-600 dark:text-rose-400"
        : "text-emerald-600 dark:text-emerald-400"
      : neg
      ? "text-emerald-600 dark:text-emerald-400"
      : "text-rose-600 dark:text-rose-400";
  return (
    <div className="rounded-2xl border border-black/5 dark:border-white/10 bg-white dark:bg-zinc-950 p-4">
      <div className="text-xs text-zinc-500 dark:text-zinc-400 mb-1">{label}</div>
      <div className={`text-xl font-semibold ${cls}`}>{value}</div>
    </div>
  );
}

function Th({ children, align = "left" }) {
  return <th className={`px-3 py-2 font-medium ${align === "right" ? "text-right" : "text-left"}`}>{children}</th>;
}
function Td({ children, align = "left", colSpan, className = "" }) {
  return (
    <td
      colSpan={colSpan}
      className={`px-3 py-2 ${align === "right" ? "text-right" : "text-left"} whitespace-nowrap ${className}`}
    >
      {children}
    </td>
  );
}