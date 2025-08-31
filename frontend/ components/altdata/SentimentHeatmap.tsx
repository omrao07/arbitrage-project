// frontend/components/SentimentHeatmap.tsx
import React, { useEffect, useMemo, useState } from "react";

/**
 * Expected backend shape (GET /api/sentiment/heatmap):
 * {
 *   rows: ["AAPL","MSFT","NVDA", ...],               // y-axis (tickers/sectors)
 *   cols: ["News","Social","Reddit","India", ...],   // x-axis (sources/regions/factors)
 *   values: [                                        // values[i][j] ∈ [-1,1]
 *     [0.22, -0.18, 0.05, 0.31],
 *     [-0.41, -0.10, 0.02, 0.12],
 *     ...
 *   ],
 *   updated_at: "2025-08-26T12:00:00Z"
 * }
 */

type HeatmapPayload = {
  rows: string[];
  cols: string[];
  values: number[][];
  updated_at?: string;
};

interface Props {
  endpoint?: string;       // API endpoint to fetch heatmap data
  maxRows?: number;        // cap rows (for performance)
  title?: string;
}

export default function SentimentHeatmap({
  endpoint = "/api/sentiment/heatmap",
  maxRows = 60,
  title = "Sentiment Heatmap",
}: Props) {
  const [data, setData] = useState<HeatmapPayload | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // UI state
  const [rowFilter, setRowFilter] = useState("");
  const [sortByCol, setSortByCol] = useState<number | null>(null); // sort rows by a column's score desc
  const [reverse, setReverse] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        setLoading(true);
        const res = await fetch(endpoint);
        const json = (await res.json()) as HeatmapPayload;
        // basic shape guard
        if (!Array.isArray(json.rows) || !Array.isArray(json.cols) || !Array.isArray(json.values)) {
          throw new Error("Bad heatmap payload");
        }
        setData(json);
      } catch (e: any) {
        setErr(e?.message || "Failed to load heatmap");
      } finally {
        setLoading(false);
      }
    })();
  }, [endpoint]);

  const view = useMemo(() => {
    if (!data) return null;
    // Filter rows
    const idxs = data.rows
      .map((r, i) => [r, i] as const)
      .filter(([r]) => r.toLowerCase().includes(rowFilter.toLowerCase()));

    // Sort by selected column (if any)
    if (sortByCol !== null && data.values.length > 0 && data.cols[sortByCol] !== undefined) {
      idxs.sort((a, b) => {
        const va = data.values[a[1]]?.[sortByCol!] ?? 0;
        const vb = data.values[b[1]]?.[sortByCol!] ?? 0;
        return reverse ? va - vb : vb - va;
      });
    }

    const limited = idxs.slice(0, maxRows);
    return {
      rows: limited.map(([r]) => r),
      rowIdx: limited.map(([, i]) => i),
      cols: data.cols,
      values: limited.map(([, i]) => data.values[i]),
    };
  }, [data, rowFilter, sortByCol, reverse, maxRows]);

  // Color scale: -1 → red, 0 → gray, +1 → green
  function colorFor(v: number) {
    const x = Math.max(-1, Math.min(1, v));
    if (x >= 0) {
      // gray → green
      const g = Math.round(128 + x * 127);
      return `rgb(${200 - x * 80}, ${g}, ${200 - x * 140})`;
    } else {
      // gray → red
      const r = Math.round(128 + -x * 127);
      return `rgb(${r}, ${200 + x * 80}, ${200 + x * 140})`;
    }
  }

  function fmt(v?: number) {
    if (v === undefined || v === null || Number.isNaN(v)) return "—";
    return (v >= 0 ? "+" : "") + v.toFixed(2);
  }

  return (
    <div className="rounded-2xl shadow-md p-4 bg-white dark:bg-gray-900">
      <header className="mb-3 flex items-center justify-between gap-3">
        <div>
          <h2 className="text-xl font-semibold">{title}</h2>
          {!!data?.updated_at && (
            <div className="text-xs opacity-70">Updated: {new Date(data.updated_at).toLocaleString()}</div>
          )}
        </div>
        <div className="flex items-center gap-2">
          <input
            className="border rounded-lg px-3 py-1.5 text-sm dark:bg-gray-800"
            placeholder="Filter rows (e.g. AAPL, Banks)"
            value={rowFilter}
            onChange={(e) => setRowFilter(e.target.value)}
          />
          <select
            className="border rounded-lg px-2 py-1.5 text-sm dark:bg-gray-800"
            value={String(sortByCol ?? "")}
            onChange={(e) => setSortByCol(e.target.value === "" ? null : Number(e.target.value))}
          >
            <option value="">Sort by…</option>
            {data?.cols.map((c, j) => (
              <option key={j} value={j}>{c}</option>
            ))}
          </select>
          <button
            onClick={() => setReverse((x) => !x)}
            className="px-3 py-1.5 rounded-lg text-sm border dark:border-gray-700"
            title="Toggle sort order"
          >
            {reverse ? "⬇️ asc" : "⬆️ desc"}
          </button>
        </div>
      </header>

      {loading && <div className="text-sm opacity-70">Loading heatmap…</div>}
      {err && !loading && <div className="text-sm text-red-500">Error: {err}</div>}

      {!loading && !err && view && (
        <div className="overflow-auto">
          {/* Column headers */}
          <div className="grid" style={{ gridTemplateColumns: `180px repeat(${view.cols.length}, 1fr)` }}>
            <div className="sticky left-0 bg-white dark:bg-gray-900 z-10 p-2 font-medium border-b dark:border-gray-700">Row</div>
            {view.cols.map((c, j) => (
              <div key={j} className="p-2 text-xs font-medium border-b dark:border-gray-700 text-center">{c}</div>
            ))}

            {/* Rows */}
            {view.rows.map((r, i) => (
              <React.Fragment key={r}>
                <div className="sticky left-0 bg-white dark:bg-gray-900 z-10 p-2 text-sm border-b dark:border-gray-800">
                  {r}
                </div>
                {view.values[i].map((v, k) => (
                  <Cell key={k} value={v} color={colorFor(v)} />
                ))}
              </React.Fragment>
            ))}
          </div>

          {/* Legend */}
          <div className="mt-4 flex items-center gap-2 text-xs">
            <span className="opacity-70">Legend:</span>
            <Swatch label="-1" color={colorFor(-1)} />
            <Swatch label="-0.5" color={colorFor(-0.5)} />
            <Swatch label="0" color={colorFor(0)} />
            <Swatch label="+0.5" color={colorFor(0.5)} />
            <Swatch label="+1" color={colorFor(1)} />
            <span className="ml-3 opacity-70">(scores in −1..+1)</span>
          </div>
        </div>
      )}
    </div>
  );

  function Cell({ value, color }: { value: number; color: string }) {
    return (
      <div
        className="h-8 border-b border-r dark:border-gray-800 flex items-center justify-center text-xs"
        style={{ backgroundColor: color }}
        title={fmt(value)}
      >
        <span className="px-1 font-medium drop-shadow-sm">{fmt(value)}</span>
      </div>
    );
  }
}

function Swatch({ label, color }: { label: string; color: string }) {
  return (
    <span className="inline-flex items-center">
      <span className="w-5 h-3 rounded-sm mr-1 border" style={{ backgroundColor: color }} />
      {label}
    </span>
  );
}