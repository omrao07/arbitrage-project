"use client";

import React, { useMemo, useState } from "react";

/** ---------- types ---------- */
export type CatalogItem = {
  id: string;                 // platform.category.code
  platform: "Bloomberg" | "Koyfin" | "Hammer" | string;
  category: string;           // Baseline | Extra | Asset-Specific | Unique | Other
  assetClass: string;         // Equities | FI | FX | Commodities | Derivatives | ...
  subClass?: string;
  type: "function" | "feature" | "ticker" | "screen" | string;
  code: string;               // e.g., EASY, CL1, EURUSD
  name: string;               // human friendly
  description: string;
  aliases?: string[];
  tags?: string[];
  region?: string;
  overlaps?: string[];        // if present & length>0 => not unique
  source?: string;
};

export type CatalogResultsProps = {
  items: CatalogItem[];
  search?: string;                      // free text to highlight
  pageSize?: number;                    // default 20
  onSelect?: (item: CatalogItem) => void;
};

/** ---------- helpers ---------- */
function hl(text: string, query: string) {
  if (!query) return text;
  const q = query.trim();
  if (!q) return text;
  const idx = text.toLowerCase().indexOf(q.toLowerCase());
  if (idx < 0) return text;
  const before = text.slice(0, idx);
  const hit = text.slice(idx, idx + q.length);
  const after = text.slice(idx + q.length);
  return (
    <>
      {before}
      <mark className="bg-yellow-600/50 text-yellow-100 rounded px-[2px]">{hit}</mark>
      {after}
    </>
  );
}

function badgePlatform(p: string) {
  const map: Record<string, string> = {
    Bloomberg: "bg-purple-700/30 text-purple-200 border-purple-700/60",
    Koyfin: "bg-blue-700/30 text-blue-200 border-blue-700/60",
    Hammer: "bg-emerald-700/30 text-emerald-200 border-emerald-700/60",
  };
  return map[p] || "bg-gray-700/30 text-gray-200 border-gray-700/60";
}

function badgeCategory(c: string) {
  return "bg-gray-700/30 text-gray-300 border-gray-700/60";
}

type SortKey = "relevance" | "platform" | "assetClass" | "category" | "code" | "name";

/** naive relevance score using search string */
function score(item: CatalogItem, q: string) {
  if (!q) return 0;
  const hay = (
    item.code +
    " " +
    item.name +
    " " +
    item.description +
    " " +
    (item.tags || []).join(" ") +
    " " +
    (item.aliases || []).join(" ")
  ).toLowerCase();
  const t = q.toLowerCase().trim();
  if (!t) return 0;
  let s = 0;
  if (hay.includes(t)) s += 5;
  if (item.code.toLowerCase().startsWith(t)) s += 5;
  if ((item.aliases || []).some(a => a.toLowerCase() === t)) s += 3;
  if ((item.tags || []).some(tag => tag.toLowerCase().includes(t))) s += 1;
  return s;
}

/** ---------- component ---------- */
export default function CatalogResults({
  items,
  search = "",
  pageSize = 20,
  onSelect,
}: CatalogResultsProps) {
  const [sortKey, setSortKey] = useState<SortKey>("relevance");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [page, setPage] = useState(1);

  const rows = useMemo(() => {
    const withScore = items.map((it) => ({ it, s: score(it, search) }));
    const arr = withScore.sort((a, b) => {
      if (sortKey === "relevance") {
        return sortDir === "asc" ? a.s - b.s : b.s - a.s;
      }
      const A = String((a.it as any)[sortKey] || "").toLowerCase();
      const B = String((b.it as any)[sortKey] || "").toLowerCase();
      if (A < B) return sortDir === "asc" ? -1 : 1;
      if (A > B) return sortDir === "asc" ? 1 : -1;
      return 0;
    });
    return arr.map((x) => x.it);
  }, [items, search, sortKey, sortDir]);

  const totalPages = Math.max(1, Math.ceil(rows.length / pageSize));
  const pageRows = rows.slice((page - 1) * pageSize, page * pageSize);

  function setSort(k: SortKey) {
    if (k === sortKey) setSortDir(sortDir === "asc" ? "desc" : "asc");
    else {
      setSortKey(k);
      setSortDir(k === "relevance" ? "desc" : "asc");
    }
    setPage(1);
  }

  return (
    <div className="bg-[#0b0b0b] border border-[#222] rounded-lg overflow-hidden">
      {/* header + controls */}
      <div className="flex items-center justify-between p-3 border-b border-[#222]">
        <div className="text-sm text-gray-300">
          {rows.length} results
          {search ? <span className="text-gray-500"> · for “{search}”</span> : null}
        </div>
        <div className="flex items-center gap-2 text-xs">
          <button
            className={`px-2 py-1 rounded border ${sortKey === "relevance" ? "border-yellow-600 text-yellow-300" : "border-[#333] text-gray-300"}`}
            onClick={() => setSort("relevance")}
          >
            Relevance {sortKey === "relevance" ? (sortDir === "asc" ? "↑" : "↓") : ""}
          </button>
          <button
            className={`px-2 py-1 rounded border ${sortKey === "platform" ? "border-[#666] text-gray-200" : "border-[#333] text-gray-300"}`}
            onClick={() => setSort("platform")}
          >
            Platform {sortKey === "platform" ? (sortDir === "asc" ? "↑" : "↓") : ""}
          </button>
          <button
            className={`px-2 py-1 rounded border ${sortKey === "assetClass" ? "border-[#666] text-gray-200" : "border-[#333] text-gray-300"}`}
            onClick={() => setSort("assetClass")}
          >
            Asset {sortKey === "assetClass" ? (sortDir === "asc" ? "↑" : "↓") : ""}
          </button>
          <button
            className={`px-2 py-1 rounded border ${sortKey === "category" ? "border-[#666] text-gray-200" : "border-[#333] text-gray-300"}`}
            onClick={() => setSort("category")}
          >
            Category {sortKey === "category" ? (sortDir === "asc" ? "↑" : "↓") : ""}
          </button>
          <button
            className={`px-2 py-1 rounded border ${sortKey === "code" ? "border-[#666] text-gray-200" : "border-[#333] text-gray-300"}`}
            onClick={() => setSort("code")}
          >
            Code {sortKey === "code" ? (sortDir === "asc" ? "↑" : "↓") : ""}
          </button>
          <button
            className={`px-2 py-1 rounded border ${sortKey === "name" ? "border-[#666] text-gray-200" : "border-[#333] text-gray-300"}`}
            onClick={() => setSort("name")}
          >
            Name {sortKey === "name" ? (sortDir === "asc" ? "↑" : "↓") : ""}
          </button>
        </div>
      </div>

      {/* table */}
      <div className="divide-y divide-[#1f1f1f]">
        {pageRows.length === 0 ? (
          <div className="p-6 text-sm text-gray-400">No results.</div>
        ) : (
          pageRows.map((it) => {
            const unique = !(it.overlaps && it.overlaps.length > 0);
            return (
              <button
                key={it.id}
                onClick={() => onSelect?.(it)}
                className="w-full text-left p-3 hover:bg-[#101010] transition-colors"
              >
                <div className="flex items-start gap-3">
                  {/* left: code & name */}
                  <div className="min-w-[180px]">
                    <div className="text-sm font-semibold text-gray-100">
                      {hl(it.code, search)}
                    </div>
                    <div className="text-xs text-gray-300">{hl(it.name, search)}</div>
                  </div>

                  {/* middle: description */}
                  <div className="flex-1 text-xs text-gray-400">
                    {hl(it.description, search)}
                  </div>

                  {/* right: chips */}
                  <div className="flex flex-col items-end gap-1 min-w-[220px]">
                    <div className="flex gap-2">
                      <span className={`px-2 py-[2px] rounded border text-[11px] ${badgePlatform(it.platform)}`}>
                        {it.platform}
                      </span>
                      <span className={`px-2 py-[2px] rounded border text-[11px] ${badgeCategory(it.category)}`}>
                        {it.category}
                      </span>
                      <span className="px-2 py-[2px] rounded border text-[11px] border-[#444] text-gray-300">
                        {it.assetClass}
                      </span>
                      {it.subClass ? (
                        <span className="px-2 py-[2px] rounded border text-[11px] border-[#444] text-gray-400">
                          {it.subClass}
                        </span>
                      ) : null}
                    </div>
                    <div className="flex items-center gap-2">
                      <span
                        className={`px-2 py-[1px] rounded text-[11px] ${
                          unique
                            ? "bg-emerald-700/30 text-emerald-200 border border-emerald-700/60"
                            : "bg-gray-700/30 text-gray-200 border border-gray-700/60"
                        }`}
                        title={unique ? "Unique feature (no overlaps)" : "Overlaps with others"}
                      >
                        {unique ? "Unique" : "Overlap"}
                      </span>
                      {(it.tags || []).slice(0, 4).map((t) => (
                        <span
                          key={t}
                          className="px-2 py-[1px] rounded border border-[#333] text-[11px] text-gray-300"
                        >
                          {t}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </button>
            );
          })
        )}
      </div>

      {/* pagination */}
      <div className="flex items-center justify-between p-3 border-t border-[#222] text-xs text-gray-300">
        <div>
          Page {page} / {totalPages}
        </div>
        <div className="flex gap-2">
          <button
            className="px-2 py-1 rounded border border-[#333] disabled:opacity-40"
            onClick={() => setPage(1)}
            disabled={page <= 1}
          >
            « First
          </button>
          <button
            className="px-2 py-1 rounded border border-[#333] disabled:opacity-40"
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1}
          >
            ‹ Prev
          </button>
          <button
            className="px-2 py-1 rounded border border-[#333] disabled:opacity-40"
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page >= totalPages}
          >
            Next ›
          </button>
          <button
            className="px-2 py-1 rounded border border-[#333] disabled:opacity-40"
            onClick={() => setPage(totalPages)}
            disabled={page >= totalPages}
          >
            Last »
          </button>
        </div>
      </div>
    </div>
  );
}