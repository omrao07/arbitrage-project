"use client";

import React, { useEffect, useMemo, useState } from "react";

// ---- Types (align with your fetchFuturesChain.server.ts) ----
export type FuturesContract = {
  symbol: string;
  expiry: string;                // ISO date
  strike: number;
  type: "CALL" | "PUT" | "FUT";
  last: number;
  change: number;
  volume: number;
  oi: number;
};

type SortKey = "symbol" | "expiry" | "type" | "strike" | "last" | "change" | "volume" | "oi";

type Props = {
  /** If provided, component will render this data. */
  data?: FuturesContract[];
  /** Optional: symbol to fetch (used only if fetchUrl provided). */
  symbol?: string;                       // e.g., "NIFTY"
  /** Optional: fetch URL that returns { data: FuturesContract[] } */
  fetchUrl?: string;                     // e.g., /api/fetchFuturesChain?symbol=NIFTY
  /** Optional callback on row click (e.g., to open ticket) */
  onSelect?: (row: FuturesContract) => void;
  /** Rows per page */
  pageSize?: number;
  /** Pane title */
  title?: string;
};

// ---- Utils ----
const fmt = (n: number, d = 2) => n.toLocaleString("en-IN", { maximumFractionDigits: d });
const shortDate = (iso: string) => {
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(y, m - 1, d).toLocaleDateString(undefined, { month: "short", day: "2-digit" });
};
const cx = (...cls: Array<string | false | null | undefined>) => cls.filter(Boolean).join(" ");

// ---- Component ----
export default function FutChainPane({
  data,
  symbol = "NIFTY",
  fetchUrl,
  onSelect,
  pageSize = 20,
  title = "Futures / Options Chain",
}: Props) {
  const [remote, setRemote] = useState<FuturesContract[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // filters
  const [expiry, setExpiry] = useState<string>("ALL");
  const [kind, setKind] = useState<"ALL" | "FUT" | "CALL" | "PUT">("ALL");
  const [q, setQ] = useState(""); // quick search (symbol contains or strike exact)
  // sorting
  const [sortKey, setSortKey] = useState<SortKey>("expiry");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");
  // paging
  const [page, setPage] = useState(1);

  // fetch (optional)
  useEffect(() => {
    if (!fetchUrl) return;
    (async () => {
      try {
        setLoading(true);
        setErr(null);
        const url = new URL(fetchUrl, typeof window !== "undefined" ? window.location.origin : "http://localhost");
        if (symbol && !url.searchParams.get("symbol")) url.searchParams.set("symbol", symbol);
        const res = await fetch(url.toString(), { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        const arr: FuturesContract[] = json.data ?? json ?? [];
        setRemote(arr);
      } catch (e: any) {
        setErr(e?.message || "Failed to fetch chain");
      } finally {
        setLoading(false);
      }
    })();
  }, [fetchUrl, symbol]);

  const rows = useMemo(() => (data ?? remote ?? []), [data, remote]);

  const expiries = useMemo(() => {
    const set = new Set(rows.map((r) => r.expiry));
    return ["ALL", ...Array.from(set).sort()];
  }, [rows]);

  const filtered = useMemo(() => {
    let res = rows;
    if (expiry !== "ALL") res = res.filter((r) => r.expiry === expiry);
    if (kind !== "ALL") res = res.filter((r) => r.type === kind);

    if (q.trim()) {
      const needle = q.trim().toUpperCase();
      const strikeNum = Number(needle);
      res = res.filter(
        (r) =>
          r.symbol.toUpperCase().includes(needle) ||
          (!Number.isNaN(strikeNum) && r.strike === strikeNum)
      );
    }
    return res;
  }, [rows, expiry, kind, q]);

  const sorted = useMemo(() => {
    const out = filtered.slice().sort((a, b) => {
      const dir = sortDir === "asc" ? 1 : -1;
      const A: any = a[sortKey];
      const B: any = b[sortKey];
      if (typeof A === "number" && typeof B === "number") return (A - B) * dir;
      return String(A).localeCompare(String(B)) * dir;
    });
    return out;
  }, [filtered, sortKey, sortDir]);

  const totalPages = Math.max(1, Math.ceil(sorted.length / pageSize));
  const pageRows = useMemo(() => {
    const start = (page - 1) * pageSize;
    return sorted.slice(start, start + pageSize);
  }, [sorted, page, pageSize]);

  // reset page when filters/sort change
  useEffect(() => setPage(1), [expiry, kind, q, sortKey, sortDir, rows.length]);

  const toggleSort = (k: SortKey) => {
    if (k === sortKey) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(k);
      setSortDir("asc");
    }
  };

  return (
    <div className="w-full rounded-xl border border-neutral-800 bg-neutral-900">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800 px-4 py-3">
        <div className="space-y-0.5">
          <h3 className="text-sm font-semibold text-neutral-200">{title}</h3>
          <p className="text-xs text-neutral-400">
            {symbol} · {rows.length} contracts
            {loading && " · loading…"}
            {err && ` · error: ${err}`}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 text-xs">
          {/* Expiry dropdown */}
          <select
            value={expiry}
            onChange={(e) => setExpiry(e.target.value)}
            className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-200"
          >
            {expiries.map((x) => (
              <option key={x} value={x}>
                {x === "ALL" ? "All expiries" : shortDate(x)}
              </option>
            ))}
          </select>

          {/* Type filter */}
          <select
            value={kind}
            onChange={(e) => setKind(e.target.value as any)}
            className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-200"
          >
            <option value="ALL">All</option>
            <option value="FUT">Futures</option>
            <option value="CALL">Calls</option>
            <option value="PUT">Puts</option>
          </select>

          {/* Quick search */}
          <input
            placeholder="Search symbol / strike"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            className="w-44 rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-neutral-200 placeholder:text-neutral-500"
          />
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead className="bg-neutral-800/60 text-neutral-400">
            <tr>
              {([
                ["symbol", "Symbol"],
                ["expiry", "Expiry"],
                ["type", "Type"],
                ["strike", "Strike"],
                ["last", "Last"],
                ["change", "Δ"],
                ["volume", "Vol"],
                ["oi", "OI"],
              ] as Array<[SortKey, string]>).map(([k, label]) => (
                <th
                  key={k}
                  onClick={() => toggleSort(k)}
                  className={cx(
                    "select-none px-3 py-2 text-left font-medium hover:text-neutral-200",
                    sortKey === k && "text-neutral-200"
                  )}
                >
                  {label}
                  {sortKey === k ? (sortDir === "asc" ? " ▲" : " ▼") : ""}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageRows.length === 0 ? (
              <tr>
                <td colSpan={8} className="px-3 py-6 text-center text-neutral-400">
                  {loading ? "Loading…" : "No contracts"}
                </td>
              </tr>
            ) : (
              pageRows.map((r) => (
                <tr
                  key={`${r.symbol}-${r.expiry}-${r.type}-${r.strike}`}
                  onClick={() => onSelect?.(r)}
                  className="border-t border-neutral-800 hover:bg-neutral-900/60"
                >
                  <td className="px-3 py-2 font-mono text-xs text-neutral-300">{r.symbol}</td>
                  <td className="px-3 py-2">{shortDate(r.expiry)}</td>
                  <td className="px-3 py-2">{r.type}</td>
                  <td className="px-3 py-2 text-right">{fmt(r.strike, 0)}</td>
                  <td className="px-3 py-2 text-right">{fmt(r.last)}</td>
                  <td
                    className={cx(
                      "px-3 py-2 text-right",
                      r.change >= 0 ? "text-emerald-400" : "text-rose-400"
                    )}
                  >
                    {fmt(r.change)}
                  </td>
                  <td className="px-3 py-2 text-right">{fmt(r.volume, 0)}</td>
                  <td className="px-3 py-2 text-right">{fmt(r.oi, 0)}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Footer / pagination */}
      <div className="flex items-center justify-between border-t border-neutral-800 px-4 py-3 text-xs text-neutral-400">
        <div>
          Page {page} / {totalPages} · Rows {pageRows.length} of {sorted.length}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            className="rounded-md border border-neutral-700 px-2 py-1 hover:bg-neutral-800"
            disabled={page <= 1}
          >
            Prev
          </button>
          <button
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            className="rounded-md border border-neutral-700 px-2 py-1 hover:bg-neutral-800"
            disabled={page >= totalPages}
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}