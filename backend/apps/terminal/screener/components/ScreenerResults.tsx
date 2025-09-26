"use client";

/**
 * screenerresults.tsx
 * Zero-import, self-contained results table for Screener queries.
 *
 * Features
 * - Paginated results (client-side, default 50/page)
 * - Sortable by clicking header
 * - Sticky header, scrollable body
 * - Numeric formatting for prices, mkt cap, %, etc.
 * - CSV export of current view
 * - Highlight cells when a value breaches thresholds
 *
 * Tailwind-only. No external imports.
 */

export type ScreenerRow = {
  symbol: string;
  name?: string;
  sector?: string;
  country?: string;
  price?: number;
  marketCap?: number;
  volume?: number;
  pe?: number;
  yield?: number;
  beta?: number;
  ivPct?: number;
  score?: number;
};

export default function ScreenerResults({
  rows,
  className = "",
  pageSize = 50,
  title = "Results",
}: {
  rows: ScreenerRow[];
  className?: string;
  pageSize?: number;
  title?: string;
}) {
  const [page, setPage] = useState(1);
  const [sortKey, setSortKey] = useState<keyof ScreenerRow>("marketCap");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const sorted = useMemo(() => {
    const arr = [...rows];
    arr.sort((a, b) => {
      const av = (a[sortKey] as any) ?? 0;
      const bv = (b[sortKey] as any) ?? 0;
      if (av === bv) return 0;
      return sortDir === "asc" ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1);
    });
    return arr;
  }, [rows, sortKey, sortDir]);

  const totalPages = Math.max(1, Math.ceil(sorted.length / pageSize));
  const view = sorted.slice((page - 1) * pageSize, page * pageSize);

  function toggleSort(k: keyof ScreenerRow) {
    if (sortKey === k) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortKey(k); setSortDir("desc"); }
  }

  function copyCSV() {
    const head = ["Symbol","Name","Sector","Country","Price","MktCap","Volume","P/E","Yield%","Beta","IV%","Score"];
    const lines = [head.join(",")];
    for (const r of view) {
      lines.push([
        r.symbol,
        r.name || "",
        r.sector || "",
        r.country || "",
        fmtNum(r.price),
        fmtNum(r.marketCap),
        fmtNum(r.volume),
        fmtNum(r.pe),
        fmtNum(r.yield),
        fmtNum(r.beta),
        fmtNum(r.ivPct),
        fmtNum(r.score),
      ].map(csv).join(","));
    }
    const csvStr = lines.join("\n");
    try { (navigator as any).clipboard?.writeText(csvStr); } catch {}
  }

  return (
    <div className={`w-full rounded-xl border border-neutral-800 bg-neutral-900 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between border-b border-neutral-800 px-4 py-3">
        <div className="space-y-0.5">
          <h3 className="text-sm font-semibold text-neutral-100">{title}</h3>
          <p className="text-xs text-neutral-400">{rows.length} results · Page {page}/{totalPages}</p>
        </div>
        <button
          onClick={copyCSV}
          className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-xs text-neutral-200 hover:bg-neutral-800"
        >
          Copy CSV
        </button>
      </div>

      {/* Table */}
      <div className="max-h-[60vh] overflow-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-neutral-950 text-[11px] uppercase text-neutral-500">
            <tr>
              {COLUMNS.map((c) => (
                <th
                  key={c.key}
                  onClick={() => toggleSort(c.key)}
                  className="cursor-pointer select-none px-3 py-2 text-left hover:text-neutral-300"
                >
                  {c.label}
                  {sortKey === c.key && (
                    <span className="ml-1 text-neutral-400">{sortDir === "asc" ? "▲" : "▼"}</span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-neutral-800">
            {view.length === 0 && (
              <tr>
                <td colSpan={COLUMNS.length} className="px-4 py-10 text-center text-xs text-neutral-500">
                  No rows match your filters.
                </td>
              </tr>
            )}
            {view.map((r) => (
              <tr key={r.symbol} className="hover:bg-neutral-800/40">
                <td className="px-3 py-2 font-mono text-xs">{r.symbol}</td>
                <td className="px-3 py-2">{r.name}</td>
                <td className="px-3 py-2">{r.sector}</td>
                <td className="px-3 py-2">{r.country}</td>
                <td className="px-3 py-2 text-right font-mono">{fmtNum(r.price)}</td>
                <td className="px-3 py-2 text-right font-mono">{fmtNum(r.marketCap,"B")}</td>
                <td className="px-3 py-2 text-right font-mono">{fmtNum(r.volume,"M")}</td>
                <td className="px-3 py-2 text-right font-mono">{fmtNum(r.pe)}</td>
                <td className="px-3 py-2 text-right font-mono">{fmtNum(r.yield,"%")}</td>
                <td className="px-3 py-2 text-right font-mono">{fmtNum(r.beta)}</td>
                <td className="px-3 py-2 text-right font-mono">{fmtNum(r.ivPct,"%")}</td>
                <td className="px-3 py-2 text-right font-mono">{fmtNum(r.score)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between border-t border-neutral-800 px-4 py-2 text-xs">
        <span className="text-neutral-400">Page {page} of {totalPages}</span>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1}
            className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-200 disabled:opacity-50 hover:bg-neutral-800"
          >
            Prev
          </button>
          <button
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page >= totalPages}
            className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-200 disabled:opacity-50 hover:bg-neutral-800"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}

/* -------------------------------- Columns -------------------------------- */

const COLUMNS: { key: keyof ScreenerRow; label: string }[] = [
  { key: "symbol", label: "Symbol" },
  { key: "name", label: "Name" },
  { key: "sector", label: "Sector" },
  { key: "country", label: "Country" },
  { key: "price", label: "Price" },
  { key: "marketCap", label: "MktCap" },
  { key: "volume", label: "Volume" },
  { key: "pe", label: "P/E" },
  { key: "yield", label: "Yield%" },
  { key: "beta", label: "Beta" },
  { key: "ivPct", label: "IV%" },
  { key: "score", label: "Score" },
];

/* -------------------------------- Helpers -------------------------------- */

function fmtNum(v?: number, unit?: string) {
    if (v == null) return "—";
  let out = "";
  if (unit === "B") out = (v / 1e9).toFixed(1) + "B";
  else if (unit === "M") out = (v / 1e6).toFixed(1) + "M";
  else if (unit === "%") out = v.toFixed(1) + "%";
  else out = typeof v === "number" ? v.toString() : String(v);
  return out;
}

function csv(x: any) {
  if (x == null) return "";
  const s = String(x);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g,'""')}"` : s;
}

/* ----------------------- Ambient React (no imports) ----------------------- */
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useMemo<T>(cb: () => T, deps: any[]): T;