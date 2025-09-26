"use client";

import React, { useMemo, useState } from "react";
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

// Types match fetchTermStruct.server.ts
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
  data: TermPoint[];
  height?: number;            // px
  title?: string;
  showSpot?: boolean;
};

function formatINR(n: number) {
  return n.toLocaleString("en-IN", { maximumFractionDigits: 2 });
}
function shortDate(s: string) {
  // expects YYYY-MM-DD
  const [y, m, d] = s.split("-").map(Number);
  return new Date(y, m - 1, d).toLocaleDateString(undefined, {
    month: "short",
    day: "2-digit",
  });
}

export default function CommodChart({ data, height = 320, title, showSpot = true }: Props) {
  const [scale, setScale] = useState<"linear" | "log">("linear");

  // Recharts-friendly rows
  const rows = useMemo(
    () =>
      data
        .slice()
        .sort((a, b) => a.daysToExp - b.daysToExp)
        .map((p) => ({
          x: shortDate(p.expiry),
          expiry: p.expiry,
          futPrice: Number(p.futPrice.toFixed(2)),
          spot: Number(p.spot.toFixed(2)),
          basisPct: p.basisPct,
          carryPct: p.annualizedCarryPct,
          state: p.state,
        })),
    [data]
  );

  const spot = rows[0]?.spot ?? 0;

  return (
    <div className="w-full rounded-xl border border-neutral-800 bg-neutral-900">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-neutral-800 px-4 py-3">
        <div className="space-y-0.5">
          <h3 className="text-sm font-semibold text-neutral-200">
            {title ?? `${data?.[0]?.symbol ?? "COMMOD"} · Term Structure`}
          </h3>
          <p className="text-xs text-neutral-400">
            Prices across expiries · {rows.length} points {showSpot && spot ? `· Spot ₹ ${formatINR(spot)}` : ""}
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <button
            onClick={() => setScale("linear")}
            className={`rounded-md px-2 py-1 ${
              scale === "linear" ? "bg-neutral-800 text-neutral-200" : "text-neutral-400 hover:text-neutral-200"
            }`}
          >
            Linear
          </button>
          <button
            onClick={() => setScale("log")}
            className={`rounded-md px-2 py-1 ${
              scale === "log" ? "bg-neutral-800 text-neutral-200" : "text-neutral-400 hover:text-neutral-200"
            }`}
          >
            Log
          </button>
        </div>
      </div>

      {/* Chart */}
      <div style={{ height }} className="px-2 py-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ left: 12, right: 18, top: 10, bottom: 8 }}>
            <CartesianGrid stroke="#27272a" strokeDasharray="3 3" />
            <XAxis dataKey="x" tick={{ fill: "#a3a3a3", fontSize: 12 }} tickMargin={8} axisLine={false} tickLine={false} />
            <YAxis
              tick={{ fill: "#a3a3a3", fontSize: 12 }}
              axisLine={false}
              tickLine={false}
              domain={["auto", "auto"]}
              scale={scale}
              width={70}
            />
            <Tooltip
              contentStyle={{ background: "#0a0a0a", border: "1px solid #262626" }}
              labelStyle={{ color: "#a3a3a3" }}
              formatter={(v: any, name: string, entry: any) => {
                if (name === "Futures") return [`₹ ${formatINR(Number(v))}`, "Futures"];
                if (name === "Spot") return [`₹ ${formatINR(Number(v))}`, "Spot"];
                if (name === "Basis %") return [`${Number(v).toFixed(2)}%`, "Basis %"];
                if (name === "Carry % (ann)") return [`${Number(v).toFixed(2)}%`, "Carry % (ann)"];
                return [v, name];
              }}
            />
            <Legend wrapperStyle={{ color: "#d4d4d4" }} />
            <Line type="monotone" dataKey="futPrice" name="Futures" dot={{ r: 2 }} stroke="#10b981" strokeWidth={2} />
            {showSpot && spot ? (
              <ReferenceLine y={spot} stroke="#6b7280" strokeDasharray="4 4" ifOverflow="extendDomain" label={{ value: "Spot", fill: "#9ca3af", fontSize: 12, position: "right" }} />
            ) : null}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Footnotes / metrics */}
      <div className="grid grid-cols-1 gap-2 border-t border-neutral-800 px-4 py-3 text-xs md:grid-cols-3">
        <div className="text-neutral-400">
          <span className="text-neutral-300">Front:</span>{" "}
          {rows[0] ? `${rows[0].x} · ₹ ${formatINR(rows[0].futPrice)} · ${rows[0].basisPct.toFixed(2)}%` : "—"}
        </div>
        <div className="text-neutral-400">
          <span className="text-neutral-300">Back:</span>{" "}
          {rows.at(-1)
            ? `${rows.at(-1)!.x} · ₹ ${formatINR(rows.at(-1)!.futPrice)} · ${rows.at(-1)!.basisPct.toFixed(2)}%`
            : "—"}
        </div>
        <div className="text-neutral-400">
          <span className="text-neutral-300">State:</span>{" "}
          {rows[0] && rows[0].futPrice >= (rows[0].spot ?? 0) ? "Contango" : "Backwardation"}
        </div>
      </div>
    </div>
  );
}

/* -------------------------- Example usage --------------------------
import CommodChart, { TermPoint } from "./commodchart";
import { fetchTermStructure } from "@/backend/.../fetchTermStruct.server";

const data: TermPoint[] = await fetchTermStructure({ symbol: "NIFTY", cadence: "WEEKLY" });

<CommodChart data={data} title="NIFTY Futures Term Structure" />
------------------------------------------------------------------- */