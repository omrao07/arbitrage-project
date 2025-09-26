"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";

/** One symbol row */
export type SymbolRow = {
  symbol: string;      // e.g., "RELIANCE"
  name?: string;       // "Reliance Industries Ltd"
  exchange?: string;   // "NSE", "NYSE"
  type?: string;       // "EQ", "FUT", "FX", "CRYPTO"
  currency?: string;   // "INR", "USD"
  price?: number;      // optional last price (for display only)
};

export type SymbolSearchProps = {
  /** Local list to search (optional if using fetch) */
  items?: SymbolRow[];
  /** API endpoint that returns { data: SymbolRow[] } or raw SymbolRow[] (optional) */
  fetchUrl?: string;
  /** Custom fetcher (q) => Promise<SymbolRow[]> (optional) */
  fetcher?: (query: string) => Promise<SymbolRow[]>;
  /** Called when a symbol is chosen */
  onSelect?: (row: SymbolRow) => void;
  /** Placeholder text */
  placeholder?: string;
  /** Debounce ms for remote fetch */
  debounceMs?: number;
  /** Max suggestions to show */
  limit?: number;
  /** Use compact density (smaller paddings) */
  compact?: boolean;
  /** Prefill query value */
  defaultQuery?: string;
  /** Keep input focused after select (default false) */
  keepFocusOnSelect?: boolean;
  /** ClassName for wrapper */
  className?: string;
};

const clamp = (n: number, min: number, max: number) => Math.max(min, Math.min(max, n));

export default function SymbolSearch({
  items,
  fetchUrl,
  fetcher,
  onSelect,
  placeholder = "Search symbols, ISIN, company…",
  debounceMs = 150,
  limit = 10,
  compact = false,
  defaultQuery = "",
  keepFocusOnSelect = false,
  className = "",
}: SymbolSearchProps) {
  const [q, setQ] = useState(defaultQuery);
  const [open, setOpen] = useState(false);
  const [idx, setIdx] = useState(0);
  const [remote, setRemote] = useState<SymbolRow[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [recent, setRecent] = useState<SymbolRow[]>([]);
  const wrapRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const debRef = useRef<number | null>(null);

  // load recents (simple in-memory/localStorage)
  useEffect(() => {
    try {
      const raw = localStorage.getItem("sym:recent");
      if (raw) setRecent(JSON.parse(raw));
    } catch {}
  }, []);

  const saveRecent = (row: SymbolRow) => {
    try {
      const next = [row, ...recent.filter((r) => r.symbol !== row.symbol)].slice(0, 8);
      setRecent(next);
      localStorage.setItem("sym:recent", JSON.stringify(next));
    } catch {}
  };

  // Debounced remote fetch
  useEffect(() => {
    if (!(fetchUrl || fetcher)) {
      setRemote(null);
      return;
    }
    if (!q.trim()) {
      setRemote(null);
      setLoading(false);
      setErr(null);
      return;
    }
    if (debRef.current) window.clearTimeout(debRef.current);
    debRef.current = window.setTimeout(async () => {
      try {
        setLoading(true);
        setErr(null);
        let rows: SymbolRow[] = [];
        if (fetcher) {
          rows = await fetcher(q.trim());
        } else if (fetchUrl) {
          const url = new URL(fetchUrl, typeof window !== "undefined" ? window.location.origin : "http://localhost");
          url.searchParams.set("q", q.trim());
          const res = await fetch(url.toString(), { cache: "no-store" });
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const json = await res.json();
          rows = Array.isArray(json) ? json : json.data ?? [];
        }
        setRemote(rows);
      } catch (e: any) {
        setErr(e?.message || "Failed to fetch symbols");
      } finally {
        setLoading(false);
      }
    }, debounceMs) as unknown as number;

    return () => {
      if (debRef.current) window.clearTimeout(debRef.current);
    };
  }, [q, fetchUrl, fetcher, debounceMs]);

  // Local fuzzy filter (very lightweight)
  const filteredLocal = useMemo(() => {
    if (!items || !q.trim()) return [];
    const needle = q.trim().toLowerCase();
    const score = (row: SymbolRow) => {
      const hay1 = row.symbol?.toLowerCase() ?? "";
      const hay2 = row.name?.toLowerCase() ?? "";
      const exact = hay1 === needle ? 1000 : 0;
      const starts = hay1.startsWith(needle) ? 100 : 0;
      const contains = hay1.includes(needle) ? 60 : 0;
      const nameHit = hay2.includes(needle) ? 40 : 0;
      return exact + starts + contains + nameHit;
    };
    return items
      .map((r) => ({ r, s: score(r) }))
      .filter(({ s }) => s > 0)
      .sort((a, b) => b.s - a.s)
      .slice(0, limit)
      .map(({ r }) => r);
  }, [items, q, limit]);

  // Decide suggestion list
  const sugg: SymbolRow[] = useMemo(() => {
    if (loading) return [];
    if (q.trim()) {
      // prefer remote if present, else local
      if (remote && remote.length) return remote.slice(0, limit);
      if (filteredLocal.length) return filteredLocal.slice(0, limit);
      return [];
    }
    // no query: show recent
    return recent.slice(0, limit);
  }, [q, remote, filteredLocal, loading, recent, limit]);

  // open/close dropdown
  useEffect(() => {
    setOpen(Boolean(q.trim()) || (recent.length > 0 && document.activeElement === inputRef.current));
    setIdx(0);
  }, [q, recent.length]);

  // click-away to close
  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  // keyboard nav
  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!open) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setIdx((i) => clamp(i + 1, 0, Math.max(0, sugg.length - 1)));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setIdx((i) => clamp(i - 1, 0, Math.max(0, sugg.length - 1)));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (sugg[idx]) choose(sugg[idx]);
    } else if (e.key === "Escape") {
      setOpen(false);
      (e.target as HTMLInputElement).blur();
    }
  };

  const choose = (row: SymbolRow) => {
    onSelect?.(row);
    saveRecent(row);
    if (!keepFocusOnSelect) {
      setOpen(false);
      inputRef.current?.blur();
    }
  };

  const density = compact
    ? { pad: "px-2.5 py-1.5", text: "text-xs", row: "px-2.5 py-1.5" }
    : { pad: "px-3.5 py-2", text: "text-sm", row: "px-3 py-2" };

  return (
    <div ref={wrapRef} className={`relative ${className}`}>
      {/* Input */}
      <div className="relative">
        <input
          ref={inputRef}
          type="search"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onFocus={() => setOpen(true)}
          onKeyDown={onKeyDown}
          placeholder={placeholder}
          className={[
            "w-full rounded-md border border-neutral-800 bg-neutral-900",
            "text-neutral-100 placeholder:text-neutral-500 outline-none focus:ring-2 focus:ring-emerald-600",
            density.pad,
            density.text,
          ].join(" ")}
        />
        {/* Search icon */}
        <div className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-neutral-500">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
            <circle cx="11" cy="11" r="7" stroke="currentColor" strokeWidth="2" />
            <path d="M20 20L16.65 16.65" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
        </div>
      </div>

      {/* Dropdown */}
      {open && (
        <div className="absolute z-30 mt-1 w-full overflow-hidden rounded-md border border-neutral-800 bg-neutral-950 shadow-xl">
          {/* Status row */}
          <div className="flex items-center justify-between border-b border-neutral-800 px-3 py-1.5 text-[11px] text-neutral-400">
            <span>
              {q.trim()
                ? loading
                  ? "Searching…"
                  : err
                  ? `Error: ${err}`
                  : sugg.length
                  ? `${sugg.length} results`
                  : "No matches"
                : recent.length
                ? "Recent"
                : "Start typing to search"}
            </span>
            {q && (
              <button
                onClick={() => setQ("")}
                className="text-neutral-400 hover:text-neutral-200"
                title="Clear"
              >
                clear
              </button>
            )}
          </div>

          {/* List */}
          <ul role="listbox" className="max-h-72 overflow-auto">
            {sugg.map((row, i) => {
              const active = i === idx;
              return (
                <li
                  key={`${row.symbol}-${row.exchange ?? ""}`}
                  role="option"
                  aria-selected={active}
                  onMouseEnter={() => setIdx(i)}
                  onMouseDown={(e) => {
                    // prevent input blur before click
                    e.preventDefault();
                    choose(row);
                  }}
                  className={[
                    "cursor-pointer border-b border-neutral-900",
                    active ? "bg-neutral-800/60" : "hover:bg-neutral-900/60",
                  ].join(" ")}
                >
                  <div className={`flex items-center justify-between ${density.row}`}>
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="truncate font-mono text-[12px] text-neutral-200">{row.symbol}</span>
                        {row.exchange && (
                          <span className="rounded bg-neutral-800 px-1.5 py-0.5 text-[10px] text-neutral-300">
                            {row.exchange}
                          </span>
                        )}
                        {row.type && (
                          <span className="rounded bg-neutral-900 px-1.5 py-0.5 text-[10px] text-neutral-400">
                            {row.type}
                          </span>
                        )}
                      </div>
                      {row.name && (
                        <div className="truncate text-[11px] text-neutral-400">{row.name}</div>
                      )}
                    </div>
                    <div className="ml-3 text-right">
                      {row.price != null && (
                        <div className="text-[12px] text-neutral-200">
                          {row.currency ? `${row.currency} ` : ""}
                          {row.price.toLocaleString("en-IN")}
                        </div>
                      )}
                    </div>
                  </div>
                </li>
              );
            })}
            {(!sugg || sugg.length === 0) && !loading && !err && (
              <li className="px-3 py-3 text-center text-xs text-neutral-500">No results</li>
            )}
          </ul>

          {/* Footer hint */}
          <div className="flex items-center justify-between border-t border-neutral-800 px-3 py-1.5 text-[10px] text-neutral-500">
            <div>
              ↑/↓ to navigate • Enter to select • Esc to close
            </div>
            {fetchUrl && <div className="italic">remote</div>}
          </div>
        </div>
      )}
    </div>
  );
}

/* --------------------------- Example usage ---------------------------
<SymbolSearch
  items={[
    { symbol: "RELIANCE", name: "Reliance Industries", exchange: "NSE", type: "EQ", currency: "INR", price: 2890.25 },
    { symbol: "HDFCBANK", name: "HDFC Bank", exchange: "NSE", type: "EQ", currency: "INR", price: 1560.4 },
    { symbol: "AAPL", name: "Apple Inc.", exchange: "NASDAQ", type: "EQ", currency: "USD", price: 227.31 },
  ]}
  onSelect={(row) => console.log("selected", row)}
/>

OR remote:

<SymbolSearch
  fetchUrl="/api/symbols" // server should read ?q= and return { data: SymbolRow[] }
  onSelect={(row) => route.push(`/symbol/${row.symbol}`)}
/>

OR custom fetcher:

<SymbolSearch
  fetcher={async (q) => {
    const r = await fetch(`/api/symbols?q=${encodeURIComponent(q)}`);
    const j = await r.json();
    return j.data;
  }}
  compact
/>
--------------------------------------------------------------------- */