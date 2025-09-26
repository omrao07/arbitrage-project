"use client";

import React, { useEffect, useMemo, useState } from "react";

type Mover = {
  symbol: string;
  name?: string;
  last?: number;
  change?: number;      // absolute
  changePct?: number;   // percent (±)
  volume?: number;      // shares/contracts
  turnover?: number;    // notional
  market?: string;      // exchange
  assetClass?: string;  // Equity | ETF | FX | Futures | Crypto ...
  sector?: string;
};

type TabKey = "gainers" | "losers" | "active";

type Props = {
  /** provide your own list; if omitted and autoFetch=true, component fetches */
  items?: Mover[];
  /** auto-fetch from endpoint (querystring: tab, limit, universe) */
  autoFetch?: boolean;
  endpoint?: string; // default "/api/market/top-movers"
  limit?: number;    // default 25
  universe?: string; // e.g., "US", "NIFTY500", "CRYPTO"
  /** default visible tab */
  initialTab?: TabKey;
  /** optional filters */
  sectors?: string[];
  assetClasses?: string[];
  /** row click */
  onOpen?: (m: Mover) => void;
  /** title */
  title?: string;
};

export default function TopMovers({
  items,
  autoFetch = true,
  endpoint = "/api/market/top-movers",
  limit = 25,
  universe = "US",
  initialTab = "gainers",
  sectors,
  assetClasses,
  onOpen,
  title = "Top Movers",
}: Props) {
  const [tab, setTab] = useState<TabKey>(initialTab);
  const [data, setData] = useState<Mover[]>(items ?? []);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const [fltSector, setFltSector] = useState<string>("All");
  const [fltClass, setFltClass] = useState<string>("All");
  const [sort, setSort] = useState<{ key: keyof Mover | "changeAbs"; dir: "desc" | "asc" }>({
    key: "changePct",
    dir: "desc",
  });

  // pull from API if asked and no items provided
  useEffect(() => {
    if (!autoFetch) return;
    if (items && items.length) return;
    let alive = true;
    (async () => {
      try {
        setLoading(true);
        setErr(null);
        const url = new URL(endpoint, typeof window !== "undefined" ? window.location.origin : "http://localhost");
        url.searchParams.set("tab", tab);
        url.searchParams.set("limit", String(limit));
        url.searchParams.set("universe", universe);
        const res = await fetch(url.toString(), { cache: "no-store" });
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        const json = await res.json();
        const rows: Mover[] = Array.isArray(json) ? json : json.items ?? [];
        if (alive) setData(normalize(rows));
      } catch (e: any) {
        if (alive) setErr(e?.message || "Failed to load movers");
      } finally {
        alive && setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, [autoFetch, items, endpoint, tab, limit, universe]);

  // if caller passes items, update internal state on change
  useEffect(() => {
    if (items) setData(normalize(items));
  }, [JSON.stringify(items ?? [])]);

  // filter + search + tab partition
  const filtered = useMemo(() => {
    const base = data.filter((m) => {
      if (fltSector !== "All" && (m.sector ?? "Other") !== fltSector) return false;
      if (fltClass !== "All" && (m.assetClass ?? "Other") !== fltClass) return false;
      if (q) {
        const s = q.toLowerCase();
        const hay = `${m.symbol} ${m.name ?? ""} ${m.sector ?? ""} ${m.assetClass ?? ""}`.toLowerCase();
        if (!hay.includes(s)) return false;
      }
      return true;
    });

    if (tab === "gainers") return base.filter((m) => (m.changePct ?? 0) > 0);
    if (tab === "losers")  return base.filter((m) => (m.changePct ?? 0) < 0);
    // most active → highest turnover then volume
    return base;
  }, [data, q, fltSector, fltClass, tab]);

  const sorted = useMemo(() => {
    const arr = [...filtered];
    const key = tab === "active" ? "turnover" : sort.key;
    const dir = tab === "active" ? "desc" : sort.dir;
    arr.sort((a, b) => {
      const av = key === "changeAbs" ? Math.abs(a.change ?? 0) : ((a as any)[key] ?? 0);
      const bv = key === "changeAbs" ? Math.abs(b.change ?? 0) : ((b as any)[key] ?? 0);
      return dir === "desc" ? (bv - av) : (av - bv);
    });
    return arr.slice(0, limit);
  }, [filtered, sort, tab, limit]);

  // build dropdown options from current data
  const sectorOpts = useMemo(
    () => ["All", ...(sectors ?? uniq(data.map((d) => d.sector ?? "Other")))],
    [data, sectors]
  );
  const classOpts = useMemo(
    () => ["All", ...(assetClasses ?? uniq(data.map((d) => d.assetClass ?? "Other")))],
    [data, assetClasses]
  );

  return (
    <div className="bg-[#0b0b0b] border border-[#222] rounded-lg overflow-hidden">
      {/* header */}
      <div className="px-4 py-2 border-b border-[#222] flex items-center justify-between gap-3">
        <div className="text-sm font-semibold text-gray-100">{title} • {universe}</div>
        <div className="flex items-center gap-2">
          <Tabs value={tab} onChange={setTab} />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Search…"
            className="bg-[#0f0f0f] border border-[#1f1f1f] rounded px-2 py-1 text-[12px] text-gray-200 outline-none"
          />
          <Select value={fltSector} onChange={setFltSector} options={sectorOpts} />
          <Select value={fltClass} onChange={setFltClass} options={classOpts} />
        </div>
      </div>

      {/* body */}
      {loading ? (
        <SkeletonRows />
      ) : err ? (
        <div className="p-3 text-xs text-red-400">{err}</div>
      ) : sorted.length === 0 ? (
        <div className="p-3 text-xs text-gray-400">No results.</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full text-[12px]">
            <thead className="bg-[#0f0f0f] border-b border-[#1f1f1f] text-gray-400">
              <tr>
                <Th>#</Th>
                <Th>Symbol</Th>
                <Th>Name</Th>
                <Th onSort={() => toggleSort(setSort, "last", sort)} sortable>Last</Th>
                <Th onSort={() => toggleSort(setSort, "changePct", sort)} sortable>% Chg</Th>
                <Th onSort={() => toggleSort(setSort, "changeAbs", sort)} sortable>Δ</Th>
                <Th onSort={() => toggleSort(setSort, "turnover", sort)} sortable>Turnover</Th>
                <Th onSort={() => toggleSort(setSort, "volume", sort)} sortable>Volume</Th>
                <Th>Sector</Th>
                <Th>Class</Th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#101010]">
              {sorted.map((m, i) => {
                const tone =
                  (m.changePct ?? 0) > 0 ? "text-emerald-300" :
                  (m.changePct ?? 0) < 0 ? "text-red-300" : "text-gray-300";
                return (
                  <tr
                    key={`${m.symbol}-${i}`}
                    className="hover:bg-[#101010] cursor-pointer"
                    onClick={() => onOpen?.(m)}
                    title={`${m.symbol} • ${m.name ?? ""}`}
                  >
                    <Td className="text-gray-500">{i + 1}</Td>
                    <Td className="text-gray-100">{m.symbol}</Td>
                    <Td className="text-gray-300 truncate max-w-[220px]">{m.name ?? "—"}</Td>
                    <Td className="text-gray-200">{fmtNum(m.last)}</Td>
                    <Td className={`font-semibold ${tone}`}>{fmtPct(m.changePct)}</Td>
                    <Td className={tone}>{fmtNum(m.change)}</Td>
                    <Td className="text-gray-300">{fmtNum(m.turnover)}</Td>
                    <Td className="text-gray-300">{fmtNum(m.volume)}</Td>
                    <Td className="text-gray-400">{m.sector ?? "—"}</Td>
                    <Td className="text-gray-400">{m.assetClass ?? "—"}</Td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/* ---------- tiny UI bits ---------- */

function Tabs({ value, onChange }: { value: TabKey; onChange: (t: TabKey) => void }) {
  const tabs: { key: TabKey; label: string }[] = [
    { key: "gainers", label: "Gainers" },
    { key: "losers",  label: "Losers" },
    { key: "active",  label: "Most Active" },
  ];
  return (
    <div className="flex bg-[#0f0f0f] border border-[#1f1f1f] rounded">
      {tabs.map((t) => {
        const active = value === t.key;
        return (
          <button
            key={t.key}
            onClick={() => onChange(t.key)}
            className={`px-2 py-1 text-[12px] ${active ? "text-white bg-[#151515]" : "text-gray-400 hover:text-gray-200"}`}
          >
            {t.label}
          </button>
        );
      })}
    </div>
  );
}

function Select({
  value, onChange, options,
}: { value: string; onChange: (v: string) => void; options: string[] }) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="bg-[#0f0f0f] border border-[#1f1f1f] rounded px-2 py-1 text-[12px] text-gray-200 outline-none"
    >
      {options.map((o) => <option key={o} value={o}>{o}</option>)}
    </select>
  );
}

function Th({
  children, onSort, sortable,
}: { children: React.ReactNode; onSort?: () => void; sortable?: boolean }) {
  return (
    <th
      onClick={sortable ? onSort : undefined}
      className={`px-3 py-2 text-left ${sortable ? "cursor-pointer select-none hover:text-gray-200" : ""}`}
    >
      {children}
    </th>
  );
}

function Td({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <td className={`px-3 py-2 ${className}`}>{children}</td>;
}

function SkeletonRows() {
  return (
    <div className="p-3 animate-pulse text-[12px] text-gray-500">Loading movers…</div>
  );
}

/* ---------- helpers ---------- */

function normalize(arr: Mover[]): Mover[] {
  return arr.map((m) => ({
    symbol: String(m.symbol),
    name: m.name ?? "",
    last: num(m.last),
    change: num(m.change),
    changePct: num(m.changePct),
    volume: num(m.volume),
    turnover: num(m.turnover),
    market: m.market,
    assetClass: m.assetClass ?? "Equity",
    sector: m.sector ?? "—",
  }));
}

function uniq(arr: string[]) {
  return Array.from(new Set(arr)).sort((a, b) => a.localeCompare(b));
}

function toggleSort(
  set: React.Dispatch<React.SetStateAction<{ key: keyof Mover | "changeAbs"; dir: "desc" | "asc" }>>,
  key: keyof Mover | "changeAbs",
  current: { key: keyof Mover | "changeAbs"; dir: "desc" | "asc" }
) {
  set((s) => (s.key === key ? { key, dir: s.dir === "desc" ? "asc" : "desc" } : { key, dir: "desc" }));
}

function num(v: any): number | undefined {
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

function fmtNum(n?: number) {
  if (n == null) return "—";
  const a = Math.abs(n);
  const sign = n < 0 ? "-" : "";
  if (a >= 1_000_000_000) return `${sign}${(a / 1_000_000_000).toFixed(2)}B`;
  if (a >= 1_000_000) return `${sign}${(a / 1_000_000).toFixed(2)}M`;
  if (a >= 1_000) return `${sign}${(a / 1_000).toFixed(2)}K`;
  return `${sign}${a.toFixed(2)}`;
}

function fmtPct(n?: number) {
  if (n == null || !Number.isFinite(n)) return "—";
  const s = n >= 0 ? "+" : "";
  return `${s}${n.toFixed(2)}%`;
}