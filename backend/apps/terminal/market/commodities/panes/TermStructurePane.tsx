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
  Legend,
} from "recharts";

/** Keep this in sync with fetchTermStruct.server.ts */
export type TermPoint = {
  symbol: string;
  expiry: string;             // YYYY-MM-DD
  daysToExp: number;
  futPrice: number;
  spot: number;
  basisPct: number;
  annualizedCarryPct: number;
  state: "CONTANGO" | "BACKWARDATION";
};

type Props = {
  title?: string;
  data?: TermPoint[];
  fetchUrl?: string;                // API returning { data: TermPoint[] } or raw array
  defaultCadence?: "WEEKLY" | "MONTHLY";
  height?: number;
};

const fmtN = (n: number, d = 2) => n.toLocaleString("en-IN", { maximumFractionDigits: d });
const shortDate = (iso: string) => {
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(y, m - 1, d).toLocaleDateString(undefined, { month: "short", day: "2-digit" });
};
const cx = (...c: Array<string | false | null | undefined>) => c.filter(Boolean).join(" ");

export default function TermStructurePane({
  title = "Futures Term Structure",
  data,
  fetchUrl,
  defaultCadence = "WEEKLY",
  height = 320,
}: Props) {
  const [remote, setRemote] = useState<TermPoint[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // UI controls
  const [scale, setScale] = useState<"linear" | "log">("linear");
  const [cadence, setCadence] = useState<"WEEKLY" | "MONTHLY">(defaultCadence);

  // Optional fetch
  useEffect(() => {
    if (!fetchUrl) return;
    (async () => {
      try {
        setLoading(true);
        setErr(null);
        const url = new URL(fetchUrl, typeof window !== "undefined" ? window.location.origin : "http://localhost");
        // If your API accepts cadence, pass it through:
        if (!url.searchParams.get("cadence")) url.searchParams.set("cadence", cadence);
        const res = await fetch(url.toString(), { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        setRemote(json.data ?? json ?? []);
      } catch (e: any) {
        setErr(e?.message || "Failed to fetch term structure");
      } finally {
        setLoading(false);
      }
    })();
  }, [fetchUrl, cadence]);

  // Source rows
  const rows = useMemo(
    () =>
      (data ?? remote ?? [])
        .slice()
        .sort((a, b) => a.daysToExp - b.daysToExp),
    [data, remote]
  );

  const symbol = rows[0]?.symbol ?? "—";
  const spot = rows[0]?.spot ?? 0;

  // Stats
  const stats = useMemo(() => {
    if (!rows.length) return { front: null as TermPoint | null, back: null as TermPoint | null, slopePct: 0 };
    const front = rows[0];
    const back = rows[rows.length - 1];
    const slopePct = front && back ? ((back.futPrice / front.futPrice - 1) * 100) : 0;
    return { front, back, slopePct };
  }, [rows]);

  // Recharts rows
  const chartRows = useMemo(
    () =>
      rows.map((p) => ({
        x: shortDate(p.expiry),
        futPrice: Number(p.futPrice.toFixed(2)),
        basisPct: p.basisPct,
        carryPct: p.annualizedCarryPct,
        state: p.state,
      })),
    [rows]
  );

  return (
    <div className="w-full rounded-xl border border-neutral-800 bg-neutral-900">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800 px-4 py-3">
        <div className="space-y-0.5">
          <h3 className="text-sm font-semibold text-neutral-200">{title}</h3>
          <p className="text-xs text-neutral-400">
            {symbol} · {rows.length} expiries
            {loading ? " · loading…" : ""}
            {err ? ` · error: ${err}` : ""}
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs">
          <select
            value={cadence}
            onChange={(e) => setCadence(e.target.value as any)}
            className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-200"
          >
            <option value="WEEKLY">Weekly</option>
            <option value="MONTHLY">Monthly</option>
          </select>

          <div className="ml-1 rounded-md border border-neutral-700 p-0.5">
            <button
              className={cx(
                "rounded-sm px-2 py-1",
                scale === "linear" ? "bg-neutral-800 text-neutral-200" : "text-neutral-400 hover:text-neutral-200"
              )}
              onClick={() => setScale("linear")}
            >
              Linear
            </button>
            <button
              className={cx(
                "rounded-sm px-2 py-1",
                scale === "log" ? "bg-neutral-800 text-neutral-200" : "text-neutral-400 hover:text-neutral-200"
              )}
              onClick={() => setScale("log")}
            >
              Log
            </button>
          </div>
        </div>
      </div>

      {/* Summary tiles */}
      <div className="grid grid-cols-1 gap-3 px-4 py-3 text-sm md:grid-cols-3">
        <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-3">
          <div className="text-neutral-400">Front</div>
          <div className="mt-1 text-neutral-100">
            {stats.front
              ? <>
                  <span className="font-semibold">₹ {fmtN(stats.front.futPrice)}</span>{" "}
                  <span className="text-xs text-neutral-400">
                    {shortDate(stats.front.expiry)} · {stats.front.basisPct.toFixed(2)}%
                  </span>
                </>
              : "—"}
          </div>
        </div>

        <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-3">
          <div className="text-neutral-400">Back</div>
          <div className="mt-1 text-neutral-100">
            {stats.back
              ? <>
                  <span className="font-semibold">₹ {fmtN(stats.back.futPrice)}</span>{" "}
                  <span className="text-xs text-neutral-400">
                    {shortDate(stats.back.expiry)} · {stats.back.basisPct.toFixed(2)}%
                  </span>
                </>
              : "—"}
          </div>
        </div>

        <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-3">
          <div className="text-neutral-400">Curve Slope</div>
          <div
            className={cx(
              "mt-1 text-lg font-semibold",
              stats.slopePct >= 0 ? "text-emerald-400" : "text-rose-400"
            )}
          >
            {stats.slopePct >= 0 ? "+" : ""}
            {fmtN(stats.slopePct, 2)}%
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
              domain={["auto", "auto"]}
              width={72}
              scale={scale}
              tickFormatter={(v) => `₹ ${fmtN(Number(v), 0)}`}
            />
            <Tooltip
              contentStyle={{ background: "#0a0a0a", border: "1px solid #262626" }}
              labelStyle={{ color: "#a3a3a3" }}
              formatter={(v: any, name: string, entry: any) => {
                if (name === "Futures") return [`₹ ${fmtN(Number(v))}`, "Futures"];
                if (name === "Basis %") return [`${Number(v).toFixed(2)}%`, "Basis %"];
                if (name === "Carry % (ann)") return [`${Number(v).toFixed(2)}%`, "Carry % (ann)"];
                return [v, name];
              }}
            />
            <Legend wrapperStyle={{ color: "#d4d4d4" }} />
            {/* Futures curve */}
            <Line type="monotone" dataKey="futPrice" name="Futures" dot={{ r: 2 }} stroke="#10b981" strokeWidth={2} />
            {/* Spot reference */}
            {spot ? (
              <ReferenceLine
                y={spot}
                stroke="#6b7280"
                strokeDasharray="4 4"
                ifOverflow="extendDomain"
                label={{ value: "Spot", fill: "#9ca3af", fontSize: 12, position: "right" }}
              />
            ) : null}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Table */}
      <div className="border-t border-neutral-800">
        <table className="min-w-full text-sm">
          <thead className="bg-neutral-800/60 text-neutral-400">
            <tr>
              <th className="px-3 py-2 text-left font-medium">Expiry</th>
              <th className="px-3 py-2 text-right font-medium">Futures</th>
              <th className="px-3 py-2 text-right font-medium">Basis %</th>
              <th className="px-3 py-2 text-right font-medium">Carry % (ann)</th>
              <th className="px-3 py-2 text-left font-medium">State</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.expiry} className="border-t border-neutral-800">
                <td className="px-3 py-2">{shortDate(r.expiry)}</td>
                <td className="px-3 py-2 text-right">₹ {fmtN(r.futPrice)}</td>
                <td className={cx("px-3 py-2 text-right", r.basisPct >= 0 ? "text-emerald-400" : "text-rose-400")}>
                  {r.basisPct.toFixed(2)}%
                </td>
                <td className={cx("px-3 py-2 text-right", r.annualizedCarryPct >= 0 ? "text-emerald-400" : "text-rose-400")}>
                  {r.annualizedCarryPct.toFixed(2)}%
                </td>
                <td
                  className={cx(
                    "px-3 py-2",
                    r.state === "CONTANGO" ? "text-emerald-300" : "text-rose-300"
                  )}
                >
                  {r.state}
                </td>
              </tr>
            ))}
            {rows.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-3 py-6 text-center text-neutral-400">
                  {loading ? "Loading…" : "No data"}
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ---------------- Example usage ----------------
import TermStructurePane, { TermPoint } from "./termstructure-pane";
import { fetchTermStructure } from "../data/fetchTermStruct.server";

// Server-side:
const data: TermPoint[] = await fetchTermStructure({ symbol: "NIFTY", cadence: "WEEKLY" });

// Client:
<TermStructurePane title="NIFTY Term Structure" data={data} />

// Or have it fetch on the client:
<TermStructurePane fetchUrl="/api/fetchTermStruct?symbol=NIFTY" defaultCadence="WEEKLY" />
------------------------------------------------- */