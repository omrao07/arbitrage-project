"use client";

/**
 * equity-peers.tsx
 * Zero-import, self-contained peers table for an equity symbol.
 *
 * Features:
 * - Sortable columns (Symbol, Name, Price, %Chg, Mkt Cap, P/E, Beta)
 * - Search box (filters by symbol/name/industry/sector)
 * - Sector/industry filters
 * - Compact pagination
 * - Keyboard: "/" to focus search, "Esc" to clear
 * - Tailwind + inline SVG only
 */

export type EquityPeer = {
  symbol: string;
  name: string;
  sector: string;
  industry: string;
  marketCap: number; // $
  price: number;     // last
  changePct: number; // daily %
  beta: number;
  pe: number;
};

type ColKey = "symbol" | "name" | "price" | "changePct" | "marketCap" | "pe" | "beta";

export default function EquityPeersPane({
  baseSymbol,
  peers = [],
  className = "",
  pageSize = 10,
}: {
  baseSymbol: string;
  peers: EquityPeer[];
  className?: string;
  pageSize?: number;
}) {
  const [q, setQ] = useState("");
  const [sector, setSector] = useState<string>("all");
  const [industry, setIndustry] = useState<string>("all");
  const [sort, setSort] = useState<{ key: ColKey; dir: "asc" | "desc" }>({ key: "marketCap", dir: "desc" });
  const [page, setPage] = useState(1);

  const searchRef = useRef<HTMLInputElement | null>(null);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      const inField = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
      if (!inField && e.key === "/") { e.preventDefault(); searchRef.current?.focus(); }
      if (inField && e.key === "Escape") { (e.target as HTMLInputElement).blur(); setQ(""); }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  const sectors = ["all", ...uniq(peers.map((p) => p.sector))];
  const industries = ["all", ...uniq(peers.map((p) => p.industry))];

  const filtered = useMemo(() => {
    const qlc = q.trim().toLowerCase();
    let arr = peers
      .filter((p) => (sector === "all" ? true : p.sector === sector))
      .filter((p) => (industry === "all" ? true : p.industry === industry))
      .filter((p) =>
        !qlc
          ? true
          : (p.symbol + " " + p.name + " " + p.sector + " " + p.industry).toLowerCase().includes(qlc)
      );

    arr.sort((a, b) => {
      const k = sort.key;
      const dir = sort.dir === "asc" ? 1 : -1;
      const av = (a as any)[k];
      const bv = (b as any)[k];
      if (typeof av === "string" && typeof bv === "string") return av.localeCompare(bv) * dir;
      return ((av as number) - (bv as number)) * dir;
    });
    return arr;
  }, [peers, q, sector, industry, sort]);

  const pages = Math.max(1, Math.ceil(filtered.length / pageSize));
  useEffect(() => { if (page > pages) setPage(pages); }, [pages]); // keep in range
  const pageRows = filtered.slice((page - 1) * pageSize, page * pageSize);

  return (
    <div className={`w-full rounded-xl border border-neutral-800 bg-neutral-900 ${className}`}>
      {/* Header / controls */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-800 px-4 py-3">
        <div className="space-y-0.5">
          <h3 className="text-sm font-semibold text-neutral-100">{baseSymbol} Peers</h3>
          <p className="text-xs text-neutral-400">
            {peers.length} total · showing {filtered.length} {sector !== "all" ? `in ${sector}` : ""}{industry !== "all" ? ` / ${industry}` : ""}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <SearchBox
            value={q}
            onChange={(v) => { setQ(v); setPage(1); }}
            placeholder="Search symbol, name…  (press /)"
            inputRef={searchRef}
            className="w-56"
          />
          <Select
            label="Sector"
            value={sector}
            onChange={(v) => { setSector(v); setPage(1); }}
            options={sectors.map((s) => ({ value: s, label: titleize(s) }))}
          />
          <Select
            label="Industry"
            value={industry}
            onChange={(v) => { setIndustry(v); setPage(1); }}
            options={industries.map((s) => ({ value: s, label: titleize(s) }))}
          />
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-neutral-950 text-[11px] uppercase text-neutral-500">
            <tr className="border-b border-neutral-800">
              <Th label="Symbol" k="symbol" sort={sort} onSort={setSort} />
              <Th label="Name" k="name" sort={sort} onSort={setSort} />
              <Th label="Price" k="price" sort={sort} onSort={setSort} right />
              <Th label="% Chg" k="changePct" sort={sort} onSort={setSort} right />
              <Th label="Mkt Cap" k="marketCap" sort={sort} onSort={setSort} right />
              <Th label="P/E" k="pe" sort={sort} onSort={setSort} right />
              <Th label="Beta" k="beta" sort={sort} onSort={setSort} right />
            </tr>
          </thead>
          <tbody className="divide-y divide-neutral-800">
            {pageRows.length === 0 && (
              <tr>
                <td colSpan={7} className="px-3 py-6 text-center text-xs text-neutral-500">No peers match your filters.</td>
              </tr>
            )}
            {pageRows.map((p) => {
              const pos = p.changePct >= 0;
              return (
                <tr key={p.symbol} className="hover:bg-neutral-800/40">
                  <td className="px-3 py-2 font-medium text-neutral-100">{p.symbol}</td>
                  <td className="px-3 py-2 text-neutral-300">{p.name}</td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums">${fmt(p.price, 2)}</td>
                  <td className={`px-3 py-2 text-right font-mono tabular-nums ${pos ? "text-emerald-400" : "text-rose-400"}`}>
                    {fmtPct(p.changePct)}
                  </td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums">{fmtInt(p.marketCap)}</td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums">{isFinite(p.pe) ? fmt(p.pe, 1) : "—"}</td>
                  <td className="px-3 py-2 text-right font-mono tabular-nums">{fmt(p.beta, 2)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between border-t border-neutral-800 px-3 py-2 text-xs text-neutral-400">
        <div>Page {page} / {pages}</div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 hover:bg-neutral-800"
            disabled={page <= 1}
          >
            ← Prev
          </button>
          <button
            onClick={() => setPage((p) => Math.min(pages, p + 1))}
            className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 hover:bg-neutral-800"
            disabled={page >= pages}
          >
            Next →
          </button>
        </div>
      </div>
    </div>
  );
}

/* ----------------------------- Table header ----------------------------- */

function Th({
  label, k, sort, onSort, right = false,
}: {
  label: string;
  k: ColKey;
  sort: { key: ColKey; dir: "asc" | "desc" };
  onSort: (s: { key: ColKey; dir: "asc" | "desc" }) => void;
  right?: boolean;
}) {
  const active = sort.key === k;
  const dir = active ? sort.dir : undefined;
  return (
    <th
      className={`cursor-pointer select-none px-3 py-2 text-left ${right ? "text-right" : "text-left"}`}
      onClick={() => onSort({ key: k, dir: active && dir === "desc" ? "asc" : "desc" })}
      title="Sort"
    >
      <span className="inline-flex items-center gap-1">
        {label}
        <svg width="10" height="10" viewBox="0 0 24 24" className={`opacity-70 ${active ? "" : "invisible"}`}>
          {dir === "asc" ? (
            <path d="M7 14l5-5 5 5" stroke="#9ca3af" strokeWidth="2" fill="none" strokeLinecap="round" />
          ) : (
            <path d="M7 10l5 5 5-5" stroke="#9ca3af" strokeWidth="2" fill="none" strokeLinecap="round" />
          )}
        </svg>
      </span>
    </th>
  );
}

/* -------------------------------- Controls -------------------------------- */

function SearchBox({
  value,
  onChange,
  placeholder,
  className = "",
  inputRef,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  className?: string;
  inputRef?: any;
}) {
  return (
    <label className={`relative ${className}`}>
      <input
        ref={inputRef}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full rounded-md border border-neutral-700 bg-neutral-950 pl-8 pr-2 py-1.5 text-xs text-neutral-200 placeholder:text-neutral-500"
      />
      <svg className="pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 opacity-70" width="14" height="14" viewBox="0 0 24 24">
        <circle cx="11" cy="11" r="7" stroke="#9ca3af" strokeWidth="2" fill="none" />
        <path d="M20 20l-3.5-3.5" stroke="#9ca3af" strokeWidth="2" strokeLinecap="round" />
      </svg>
    </label>
  );
}

function Select({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <label className="flex items-center gap-2">
      <span className="text-neutral-400">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-neutral-700 bg-neutral-950 px-2 py-1 text-xs text-neutral-200"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

/* --------------------------------- Utils --------------------------------- */

function uniq<T>(a: T[]) { return Array.from(new Set(a)); }
function titleize(s: string) { return s === "all" ? "All" : s; }
function fmt(n: number, d = 2) { return n.toLocaleString("en-US", { maximumFractionDigits: d }); }
function fmtInt(n: number) { return n.toLocaleString("en-US"); }
function fmtPct(x: number) { return (x * 100).toLocaleString("en-US", { maximumFractionDigits: 2 }) + "%"; }

/* --------------- Ambient React (keep zero imports; no imports) --------------- */
declare const React: any;
declare function useState<T>(i: T | (() => T)): [T, (v: T | ((p: T) => T)) => void];
declare function useMemo<T>(cb: () => T, deps: any[]): T;
declare function useEffect(cb: () => void | (() => void), deps?: any[]): void;
declare function useRef<T>(v: T | null): { current: T | null };

/* ------------------------------- Example --------------------------------
import { fetchPeers } from "@/server/fetchPeers.server";
const peers = await fetchPeers("AAPL");
<EquityPeersPane baseSymbol="AAPL" peers={peers} />
--------------------------------------------------------------------------- */